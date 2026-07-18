from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Body
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional
import json
import os
import requests
import datetime
import threading

from gg import (
    build_vectorstore, retrieve,
    get_llm_response, stream_llm_openrouter,
    OLLAMA_BASE_URL, LLM_MODEL, EMBED_MODEL,
    LLM_PROVIDER, OPENROUTER_MODEL,
    load_indexed_hashes, save_indexed_hashes,
    get_file_hash, index_pdf,
    generate_cards_with_retry, CardJsonError,
    random_chunks_from_source,
)
from backend import user_store

# ── SESSION MEMORY ────────────────────────────────────────
session_store: dict = {}

# ── VECTORSTORE ───────────────────────────────────────────
collection = None

# Cache per-user collections so we don't re-open ChromaDB on every request.
_user_collections: dict = {}

# ── UPLOAD PROGRESS ───────────────────────────────────────
upload_progress: dict = {}

UPLOAD_DIR = "demo_pdfs"


@asynccontextmanager
async def lifespan(app: FastAPI):
    global collection
    pdf_paths = [
        os.path.join("demo_pdfs", f)
        for f in os.listdir("demo_pdfs")
        if f.endswith(".pdf")
    ]
    _, collection = build_vectorstore(pdf_paths)
    user_store.init_user_store()
    yield


app = FastAPI(lifespan=lifespan)


# ── /health ───────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


# ── /docs-list ────────────────────────────────────────────
class DocEntry(BaseModel):
    filename: str
    timestamp: str


class DocsListResponse(BaseModel):
    documents: list[DocEntry]


# ── /progress response schema ────────────────────────────
# id / source_pdf / created_at are Optional because the /quiz-result fallback
# path (used by old app builds that don't send quiz_id) appends legacy rows of
# the shape {timestamp, total_questions, correct} with none of those fields.
# source_pdf and correct are also Optional in the current path — a quiz can be
# generated across all documents (no source_pdf), and correct is null until
# /quiz-result is posted for that quiz.
class QuizHistoryEntry(BaseModel):
    id: Optional[str] = None
    source_pdf: Optional[str] = None
    created_at: Optional[str] = None
    timestamp: Optional[str] = None
    total_questions: int
    correct: Optional[int] = None


class FlashcardSetEntry(BaseModel):
    id: str
    source_pdf: Optional[str] = None
    created_at: str
    card_count: int
    cards_revealed: int


class ProgressResponse(BaseModel):
    user_id: int
    pdfs_uploaded: list[DocEntry]
    quiz_history: list[QuizHistoryEntry]
    flashcard_sets: list[FlashcardSetEntry]
    questions_answered_total: int
    questions_correct_total: int
    flashcards_revealed_total: int


@app.get("/docs-list", response_model=DocsListResponse)
def docs_list(user_id: Optional[int] = None):
    # user_id is required — kept Optional in the signature only so we can
    # respond with 400 instead of FastAPI's default 422 when it's missing.
    if user_id is None:
        raise HTTPException(status_code=400, detail="user_id is required")

    user_data = user_store.get_user_data(user_id)
    if user_data is None:
        raise HTTPException(status_code=404, detail="User not found")

    # response_model=DocsListResponse enforces the shape on the way out —
    # FastAPI validates every response, so this route physically cannot
    # drift from its documented contract, and openapi.json is truthful.
    return {
        "documents": [
            {"filename": entry["filename"], "timestamp": entry.get("timestamp", "")}
            for entry in user_data.get("pdfs_uploaded", [])
        ]
    }


# ── REQUEST SCHEMA ────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None
    user_id: Optional[int] = None


class RegisterRequest(BaseModel):
    name: str
    email: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


class QuizRequest(BaseModel):
    user_id: Optional[int] = None
    num_questions: int = 5
    source_pdf: Optional[str] = None


class FlashcardRequest(BaseModel):
    user_id: int
    # source_pdf omitted / null -> generate from the user's whole collection,
    # matching /quiz's existing "all documents" behavior.
    source_pdf: Optional[str] = None
    count: int


class FlashcardRevealRequest(BaseModel):
    user_id: int
    set_id: str
    revealed_count: int


_ALLOWED_FLASHCARD_COUNTS = {5, 10, 15, 20}
_FLASHCARD_SAMPLE_BATCH = 15


# ── PDF OWNERSHIP HELPER ─────────────────────────────────
def _require_user_owns_pdf(user_id: int, source_pdf: str) -> None:
    """404 if the user doesn't own a PDF with this filename. Never trust a client-supplied filename to be one the caller uploaded — this stops user 1 from targeting user 2's documents by guessing a name."""
    user_data = user_store.get_user_data(user_id)
    if user_data is None:
        raise HTTPException(status_code=404, detail="User not found")
    owned = {e.get("filename") for e in user_data.get("pdfs_uploaded", [])}
    if source_pdf not in owned:
        raise HTTPException(status_code=404, detail="Source PDF not found for this user")


# ── PER-USER COLLECTION HELPER ───────────────────────────
def get_collection_for_user(user_id: Optional[int]):
    """Return the global collection when user_id is None; otherwise open (and cache) the user's per-user ChromaDB collection. Raises 404 for an unknown user_id."""
    if user_id is None:
        return collection

    cached = _user_collections.get(user_id)
    if cached is not None:
        return cached

    user_data = user_store.get_user_data(user_id)
    if user_data is None:
        raise HTTPException(status_code=404, detail="User not found")

    _, user_collection = build_vectorstore([], chroma_dir=user_data["chroma_db_path"])
    _user_collections[user_id] = user_collection
    return user_collection


# ── /register ─────────────────────────────────────────────
@app.post("/register")
def register(req: RegisterRequest):
    try:
        user = user_store.create_user(req.name, req.email, req.password)
    except ValueError as e:
        return JSONResponse(
            status_code=400,
            content={"success": False, "message": str(e)},
        )
    return {"success": True, "user": user}


# ── /login ────────────────────────────────────────────────
@app.post("/login")
def login(req: LoginRequest):
    user = user_store.authenticate_user(req.email, req.password)
    if user is None:
        return JSONResponse(
            status_code=401,
            content={"success": False, "message": "Invalid email or password"},
        )
    return {"success": True, "user": user}


# ── /chat ─────────────────────────────────────────────────
@app.post("/chat")
def chat(req: ChatRequest):
    user_collection = get_collection_for_user(req.user_id)
    context = retrieve(user_collection, req.question)

    # Build conversation history from last 3 exchanges
    history_section = ""
    if req.session_id and req.session_id in session_store:
        exchanges = session_store[req.session_id][-3:]
        lines = []
        for ex in exchanges:
            lines.append(f"Student: {ex['question']}")
            lines.append(f"Assistant: {ex['answer']}")
        history_section = "\n=== CONVERSATION HISTORY ===\n" + "\n".join(lines) + "\n"

    prompt = f"""You are a helpful study assistant. A student has asked a question.

You have access to:
1. Relevant text excerpts from the student's uploaded notes
2. Descriptions of diagrams and images found in those notes (labeled as Diagram)
3. Your own general knowledge

Instructions:
- Use the retrieved content first. Mention the source file and page number.
- If a diagram description is relevant, reference it clearly.
- If notes have partial info, supplement with your own knowledge and label it.
- If nothing relevant is found, say "I could not find this in your uploaded notes, but based on my own knowledge:" and answer concisely.
{history_section}
=== RETRIEVED CONTENT ===
{context}

=== STUDENT QUESTION ===
{req.question}

Format:
[From your notes]: ...
[From general knowledge]: ... (only if needed)"""

    answer = get_llm_response(prompt)

    if req.session_id is not None:
        if req.session_id not in session_store:
            session_store[req.session_id] = []
        session_store[req.session_id].append({
            "question": req.question,
            "answer": answer,
        })

    return {"answer": answer, "sources": []}


# ── /clear-session ────────────────────────────────────────
@app.delete("/clear-session/{session_id}")
def clear_session(session_id: str):
    if session_id not in session_store:
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": "Session not found"},
        )
    del session_store[session_id]
    return {"success": True, "message": "Session cleared"}


# ── /stats ───────────────────────────────────────────────
@app.get("/stats")
def stats():
    indexed = list(load_indexed_hashes().keys())
    return {
        "model_name": LLM_MODEL,
        "embed_model": EMBED_MODEL,
        "total_chunks": collection.count(),
        "indexed_documents": indexed,
        "document_count": len(indexed),
        "server_time": datetime.datetime.utcnow().isoformat() + "Z",
        "llm_provider": LLM_MODEL if LLM_PROVIDER == "ollama" else OPENROUTER_MODEL,
        "provider_type": LLM_PROVIDER,
    }


# ── UPLOAD HELPERS ───────────────────────────────────────
def _run_indexing(filepath: str, filename: str, user_id: Optional[int] = None):
    def on_progress(pct: int):
        upload_progress[filename] = pct

    target_collection = get_collection_for_user(user_id)

    indexed_hashes = load_indexed_hashes()
    current_hash = get_file_hash(filepath)
    chunk_id_start = target_collection.count()
    index_pdf(filepath, target_collection, chunk_id_start, progress_callback=on_progress)
    indexed_hashes[filename] = current_hash
    save_indexed_hashes(indexed_hashes)
    upload_progress.pop(filename, None)

    if user_id is not None:
        user_data = user_store.get_user_data(user_id)
        if user_data is not None:
            user_data.setdefault("pdfs_uploaded", []).append({
                "filename": filename,
                "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            })
            user_store.save_user_data(user_id, user_data)


# ── /upload ───────────────────────────────────────────────
@app.post("/upload")
async def upload_pdf(
    file: UploadFile = File(...),
    user_id: Optional[int] = Form(None),
):
    # Validate the user up front so we don't kick off a background thread for an unknown user_id.
    if user_id is not None:
        get_collection_for_user(user_id)

    filename = file.filename
    filepath = os.path.join(UPLOAD_DIR, filename)

    contents = await file.read()
    with open(filepath, "wb") as f:
        f.write(contents)

    upload_progress[filename] = 0
    threading.Thread(
        target=_run_indexing, args=(filepath, filename, user_id), daemon=True
    ).start()

    return {"success": True, "message": "Upload started", "filename": filename}


# ── /upload-progress ──────────────────────────────────────
@app.get("/upload-progress/{filename}")
def get_upload_progress(filename: str):
    if filename not in upload_progress:
        return JSONResponse(
            status_code=404,
            content={"filename": filename, "message": "Not found or already complete"},
        )
    return {"percent": upload_progress[filename], "complete": False}


# ── /docs/{filename} ─────────────────────────────────────
@app.delete("/docs/{filename}")
def delete_doc(filename: str, user_id: Optional[int] = None):
    target_collection = get_collection_for_user(user_id)

    try:
        results = target_collection.get(where={"source": filename})
        if results["ids"]:
            target_collection.delete(ids=results["ids"])
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete from index: {e}")

    if user_id is None:
        hashes = load_indexed_hashes()
        if filename in hashes:
            del hashes[filename]
            save_indexed_hashes(hashes)
        filepath = os.path.join(UPLOAD_DIR, filename)
        if os.path.exists(filepath):
            os.remove(filepath)
    else:
        user_data = user_store.get_user_data(user_id)
        if user_data is not None:
            user_data["pdfs_uploaded"] = [
                entry for entry in user_data.get("pdfs_uploaded", [])
                if entry.get("filename") != filename
            ]
            user_store.save_user_data(user_id, user_data)

    return {"success": True, "message": f"Deleted '{filename}' from index"}


# ── /documents/{user_id}/{filename} ──────────────────────
@app.delete("/documents/{user_id}/{filename}")
def delete_user_document(user_id: int, filename: str):
    """Delete a PDF for a specific user: drops the file's chunks from the
    user's ChromaDB collection and removes the pdfs_uploaded entry.

    Quizzes and flashcard sets already generated from this PDF are
    self-contained payload files and are INTENTIONALLY kept. History
    should survive document deletion — don't 'clean them up' here.
    """
    user_data = _load_user_or_404(user_id)
    owned = [e for e in user_data.get("pdfs_uploaded", []) if e.get("filename") == filename]
    if not owned:
        raise HTTPException(status_code=404, detail="Document not found for this user")

    target_collection = get_collection_for_user(user_id)
    try:
        target_collection.delete(where={"source": filename})
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to delete chunks: {e}")

    user_data["pdfs_uploaded"] = [
        e for e in user_data.get("pdfs_uploaded", []) if e.get("filename") != filename
    ]
    user_store.save_user_data(user_id, user_data)

    filepath = os.path.join(UPLOAD_DIR, filename)
    if os.path.exists(filepath):
        try:
            os.remove(filepath)
        except OSError:
            pass

    return {"success": True, "message": f"Deleted '{filename}' for user {user_id}"}


# ── /quiz ─────────────────────────────────────────────────
@app.post("/quiz")
def generate_quiz(req: Optional[QuizRequest] = Body(None)):
    user_id = req.user_id if req else None
    n = max(1, min(20, req.num_questions if req else 5))
    source_pdf = req.source_pdf if req else None
    target_collection = get_collection_for_user(user_id)

    if source_pdf:
        if user_id is None:
            raise HTTPException(status_code=400, detail="source_pdf requires user_id")
        _require_user_owns_pdf(user_id, source_pdf)

    if target_collection.count() == 0:
        raise HTTPException(status_code=400, detail="No documents indexed. Upload a PDF first.")

    context = retrieve(
        target_collection,
        "key facts important concepts definitions diagrams examples",
        source_pdf=source_pdf,
    )

    prompt = f"""Based on the following study material, create exactly {n} multiple choice quiz questions to test a student's understanding.

IMPORTANT: Respond with ONLY valid JSON. No explanations, no markdown code fences, no extra text before or after the JSON.

Required format:
{{"questions": [{{"question": "Question text?", "options": ["Option A", "Option B", "Option C", "Option D"], "correct_index": 0, "explanation": "Brief explanation why this answer is correct"}}]}}

Rules:
- Each question must have exactly 4 options
- correct_index must be 0, 1, 2, or 3 (zero-based index of the correct option)
- Questions must be answerable from the study material below
- Do not add any text outside the JSON object

Study Material:
{context}

JSON:"""

    try:
        data = generate_cards_with_retry(prompt, expected_count=n, key="questions")
    except CardJsonError:
        raise HTTPException(status_code=500, detail="LLM did not return valid JSON. Try again.")

    # Persist the full quiz payload only when we have a user to attach it to.
    # Anonymous /quiz calls (no user_id) keep working as before — the response
    # just doesn't carry a quiz_id.
    if user_id is not None:
        quiz_id = _new_set_id("q")
        created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
        payload = {
            "id": quiz_id,
            "user_id": user_id,
            "source_pdf": source_pdf,
            "created_at": created_at,
            "questions": data["questions"],
        }
        user_store.save_quiz_payload(user_id, quiz_id, payload)

        user_data = user_store.get_user_data(user_id)
        if user_data is not None:
            user_data.setdefault("quiz_history", []).append({
                "id": quiz_id,
                "source_pdf": source_pdf,
                "created_at": created_at,
                "total_questions": len(data["questions"]),
                "correct": None,
            })
            user_store.save_user_data(user_id, user_data)

        return {
            "id": quiz_id,
            "source_pdf": source_pdf,
            "created_at": created_at,
            "questions": data["questions"],
        }

    return data


# ── /quiz-result ──────────────────────────────────────────
class QuizResultWithIdRequest(BaseModel):
    user_id: int
    total_questions: int
    correct: int
    quiz_id: Optional[str] = None


@app.post("/quiz-result")
def record_quiz_result(req: QuizResultWithIdRequest):
    user_data = user_store.get_user_data(req.user_id)
    if user_data is None:
        return JSONResponse(
            status_code=404,
            content={"success": False, "message": "User not found"},
        )

    history = user_data.setdefault("quiz_history", [])
    updated_existing = False
    if req.quiz_id:
        for entry in history:
            if entry.get("id") == req.quiz_id:
                entry["total_questions"] = req.total_questions
                entry["correct"] = req.correct
                updated_existing = True
                break

    # Fall back to append-a-row if no id was sent, or if the id is unknown
    # (old app builds don't send quiz_id — do not break them mid-sprint).
    if not updated_existing:
        history.append({
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "total_questions": req.total_questions,
            "correct": req.correct,
        })

    user_data["questions_answered_total"] = (
        user_data.get("questions_answered_total", 0) + req.total_questions
    )
    user_data["questions_correct_total"] = (
        user_data.get("questions_correct_total", 0) + req.correct
    )

    user_store.save_user_data(req.user_id, user_data)
    return {"success": True}


# ── /flashcards ───────────────────────────────────────────
def _new_set_id(prefix: str) -> str:
    return (
        f"{prefix}_"
        + datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    )


@app.post("/flashcards")
def generate_flashcards(req: FlashcardRequest):
    if req.count not in _ALLOWED_FLASHCARD_COUNTS:
        raise HTTPException(
            status_code=422,
            detail=f"count must be one of {sorted(_ALLOWED_FLASHCARD_COUNTS)}",
        )

    # source_pdf=None means "all documents" — mirror /quiz. Only enforce
    # ownership when the client actually named a specific file.
    if req.source_pdf is not None:
        _require_user_owns_pdf(req.user_id, req.source_pdf)
    else:
        # Still need to 404 unknown users up front so the empty-batch error
        # below doesn't mask a bad user_id.
        _load_user_or_404(req.user_id)
    target_collection = get_collection_for_user(req.user_id)

    batch = random_chunks_from_source(
        target_collection, req.source_pdf, batch_size=_FLASHCARD_SAMPLE_BATCH
    )
    if not batch:
        detail = (
            f"No indexed content found for '{req.source_pdf}'. "
            "It may still be uploading or the PDF is empty."
        ) if req.source_pdf else (
            "You have no indexed documents yet. Upload a PDF first."
        )
        raise HTTPException(status_code=400, detail=detail)

    context = "\n\n---\n\n".join(
        f"[{c['meta'].get('source', '?')} p{c['meta'].get('page', '?')}]\n{c['text']}"
        for c in batch
    )

    prompt = f"""Create exactly {req.count} multiple-choice flashcards from the study material below.

IMPORTANT: Respond with ONLY valid JSON. No explanations, no markdown code fences, no extra text before or after the JSON.

Required format:
{{"cards": [{{"question": "Question text?", "options": ["Option A", "Option B", "Option C", "Option D"], "correct_index": 0, "explanation": "Brief explanation why this answer is correct"}}]}}

Rules:
- Produce exactly {req.count} cards
- Each card has exactly 4 options
- correct_index must be 0, 1, 2, or 3 (zero-based index of the correct option)
- Cards must be answerable from the study material below
- Do not add any text outside the JSON object

Study material:
{context}

JSON:"""

    try:
        data = generate_cards_with_retry(
            prompt, expected_count=req.count, key="cards"
        )
    except CardJsonError:
        raise HTTPException(
            status_code=500, detail="LLM did not return valid JSON. Try again."
        )

    set_id = _new_set_id("f")
    created_at = datetime.datetime.now(datetime.timezone.utc).isoformat()
    payload = {
        "id": set_id,
        "user_id": req.user_id,
        "source_pdf": req.source_pdf,
        "created_at": created_at,
        "cards": data["cards"],
    }
    user_store.save_flashcard_payload(req.user_id, set_id, payload)

    user_data = user_store.get_user_data(req.user_id)
    user_data.setdefault("flashcard_sets", []).append({
        "id": set_id,
        "source_pdf": req.source_pdf,
        "created_at": created_at,
        "card_count": len(data["cards"]),
        "cards_revealed": 0,
    })
    user_store.save_user_data(req.user_id, user_data)

    return {
        "id": set_id,
        "source_pdf": req.source_pdf,
        "created_at": created_at,
        "cards": data["cards"],
    }


# ── HISTORY, RETRIEVAL, DELETE ───────────────────────────
def _sort_newest_first(entries: list) -> list:
    """Return entries sorted by created_at (or legacy timestamp) desc.

    Legacy quiz_history rows written before quizzes had ids only carry a
    `timestamp` field — treat that as the sort key so old rows still show
    up in the right place in the history screen.
    """
    def key(e: dict) -> str:
        return e.get("created_at") or e.get("timestamp") or ""
    return sorted(entries, key=key, reverse=True)


def _load_user_or_404(user_id: int) -> dict:
    user_data = user_store.get_user_data(user_id)
    if user_data is None:
        raise HTTPException(status_code=404, detail="User not found")
    return user_data


@app.get("/quizzes/{user_id}")
def list_quizzes(user_id: int):
    user_data = _load_user_or_404(user_id)
    return {"quizzes": _sort_newest_first(user_data.get("quiz_history", []))}


@app.get("/quizzes/{user_id}/{quiz_id}")
def get_quiz(user_id: int, quiz_id: str):
    user_data = _load_user_or_404(user_id)
    # Cross-check the id against this user's own index — file existence
    # alone is not enough, since we never want user 2 to fetch user 1's
    # quiz by guessing the id.
    if not any(e.get("id") == quiz_id for e in user_data.get("quiz_history", [])):
        raise HTTPException(status_code=404, detail="Quiz not found")
    payload = user_store.load_quiz_payload(user_id, quiz_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Quiz not found")
    return payload


@app.delete("/quizzes/{user_id}/{quiz_id}")
def delete_quiz(user_id: int, quiz_id: str):
    user_data = _load_user_or_404(user_id)
    history = user_data.get("quiz_history", [])
    filtered = [e for e in history if e.get("id") != quiz_id]
    if len(filtered) == len(history):
        raise HTTPException(status_code=404, detail="Quiz not found")
    user_data["quiz_history"] = filtered
    user_store.save_user_data(user_id, user_data)
    user_store.delete_quiz_payload(user_id, quiz_id)
    return {"success": True}


@app.get("/flashcards/{user_id}")
def list_flashcard_sets(user_id: int):
    user_data = _load_user_or_404(user_id)
    return {"flashcard_sets": _sort_newest_first(user_data.get("flashcard_sets", []))}


@app.get("/flashcards/{user_id}/{set_id}")
def get_flashcard_set(user_id: int, set_id: str):
    user_data = _load_user_or_404(user_id)
    if not any(e.get("id") == set_id for e in user_data.get("flashcard_sets", [])):
        raise HTTPException(status_code=404, detail="Flashcard set not found")
    payload = user_store.load_flashcard_payload(user_id, set_id)
    if payload is None:
        raise HTTPException(status_code=404, detail="Flashcard set not found")
    return payload


@app.delete("/flashcards/{user_id}/{set_id}")
def delete_flashcard_set(user_id: int, set_id: str):
    user_data = _load_user_or_404(user_id)
    sets = user_data.get("flashcard_sets", [])
    filtered = [e for e in sets if e.get("id") != set_id]
    if len(filtered) == len(sets):
        raise HTTPException(status_code=404, detail="Flashcard set not found")
    user_data["flashcard_sets"] = filtered
    user_store.save_user_data(user_id, user_data)
    user_store.delete_flashcard_payload(user_id, set_id)
    return {"success": True}


@app.post("/flashcard-reveal")
def record_flashcard_reveal(req: FlashcardRevealRequest):
    user_data = _load_user_or_404(req.user_id)
    target = None
    for entry in user_data.get("flashcard_sets", []):
        if entry.get("id") == req.set_id:
            target = entry
            break
    if target is None:
        raise HTTPException(status_code=404, detail="Flashcard set not found")

    # Client sends the running total; server takes max so re-taps don't
    # double-count and network retries are idempotent.
    prev = int(target.get("cards_revealed", 0))
    new = max(prev, int(req.revealed_count))
    delta = new - prev
    target["cards_revealed"] = new
    user_data["flashcards_revealed_total"] = (
        user_data.get("flashcards_revealed_total", 0) + delta
    )
    user_store.save_user_data(req.user_id, user_data)
    return {"success": True, "cards_revealed": new}


# ── /progress/{user_id} ───────────────────────────────────
@app.get("/progress/{user_id}", response_model=ProgressResponse)
def get_progress(user_id: int):
    """Return the user's progress summary.

    Includes questions_answered_total / questions_correct_total for quizzes
    and flashcards_revealed_total for flashcards, plus the flashcard_sets
    and quiz_history indexes so the client can show a history screen from
    one call. chroma_db_path is stripped — it's an internal file path.
    """
    user_data = user_store.get_user_data(user_id)
    if user_data is None:
        raise HTTPException(status_code=404, detail="User not found")

    # get_user_data() backfills any missing keys via the schema migration,
    # so old accounts already report the new flashcard counters as 0 here.
    return {k: v for k, v in user_data.items() if k != "chroma_db_path"}


# ── /chat-stream (SSE) ────────────────────────────────────
@app.post("/chat-stream")
def chat_stream(req: ChatRequest):
    user_collection = get_collection_for_user(req.user_id)
    context = retrieve(user_collection, req.question)

    history_section = ""
    if req.session_id and req.session_id in session_store:
        exchanges = session_store[req.session_id][-3:]
        lines = []
        for ex in exchanges:
            lines.append(f"Student: {ex['question']}")
            lines.append(f"Assistant: {ex['answer']}")
        history_section = "\n=== CONVERSATION HISTORY ===\n" + "\n".join(lines) + "\n"

    prompt = f"""You are a helpful study assistant. A student has asked a question.

You have access to:
1. Relevant text excerpts from the student's uploaded notes
2. Descriptions of diagrams and images found in those notes (labeled as Diagram)
3. Your own general knowledge

Instructions:
- Use the retrieved content first. Mention the source file and page number.
- If a diagram description is relevant, reference it clearly.
- If notes have partial info, supplement with your own knowledge and label it.
- If nothing relevant is found, say "I could not find this in your uploaded notes, but based on my own knowledge:" and answer concisely.
{history_section}
=== RETRIEVED CONTENT ===
{context}

=== STUDENT QUESTION ===
{req.question}

Format:
[From your notes]: ...
[From general knowledge]: ... (only if needed)"""

    def generate():
        full_answer = []

        if LLM_PROVIDER == "openrouter":
            for token in stream_llm_openrouter(prompt):
                full_answer.append(token)
                yield f"data: {token}\n\n"
        else:
            payload = {
                "model": LLM_MODEL,
                "messages": [{"role": "user", "content": prompt}],
                "stream": True,
            }
            with requests.post(
                f"{OLLAMA_BASE_URL}/api/chat", json=payload, stream=True
            ) as r:
                for line in r.iter_lines():
                    if line:
                        chunk = json.loads(line)
                        token = chunk.get("message", {}).get("content", "")
                        if token:
                            full_answer.append(token)
                            yield f"data: {token}\n\n"

        if req.session_id is not None:
            answer = "".join(full_answer)
            if req.session_id not in session_store:
                session_store[req.session_id] = []
            session_store[req.session_id].append(
                {"question": req.question, "answer": answer}
            )

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        generate(),
        media_type="text/event-stream",
        # Nginx honours this per-response to skip buffering; belt-and-braces
        # with the location-level proxy_buffering off in the site config.
        headers={"X-Accel-Buffering": "no"},
    )
