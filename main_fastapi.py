from contextlib import asynccontextmanager
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel
from typing import Optional
import json
import os
import requests
import datetime
import threading

from gg import (
    build_vectorstore, retrieve, stream_llm,
    OLLAMA_BASE_URL, LLM_MODEL, EMBED_MODEL,
    load_indexed_hashes, save_indexed_hashes,
    get_file_hash, index_pdf,
)

# ── SESSION MEMORY ────────────────────────────────────────
session_store: dict = {}

# ── VECTORSTORE ───────────────────────────────────────────
collection = None

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
    yield


app = FastAPI(lifespan=lifespan)


# ── /health ───────────────────────────────────────────────
@app.get("/health")
def health():
    return {"status": "ok"}


# ── /docs-list ────────────────────────────────────────────
@app.get("/docs-list")
def docs_list():
    return {"documents": list(load_indexed_hashes().keys())}


# ── REQUEST SCHEMA ────────────────────────────────────────
class ChatRequest(BaseModel):
    question: str
    session_id: Optional[str] = None


# ── /chat ─────────────────────────────────────────────────
@app.post("/chat")
def chat(req: ChatRequest):
    context = retrieve(collection, req.question)

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

    answer = stream_llm(prompt)

    if req.session_id is not None:
        if req.session_id not in session_store:
            session_store[req.session_id] = []
        session_store[req.session_id].append({
            "question": req.question,
            "answer": answer,
        })

    return {"answer": answer}


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
    }


# ── UPLOAD HELPERS ───────────────────────────────────────
def _run_indexing(filepath: str, filename: str):
    def on_progress(pct: int):
        upload_progress[filename] = pct

    indexed_hashes = load_indexed_hashes()
    current_hash = get_file_hash(filepath)
    chunk_id_start = collection.count()
    index_pdf(filepath, collection, chunk_id_start, progress_callback=on_progress)
    indexed_hashes[filename] = current_hash
    save_indexed_hashes(indexed_hashes)
    upload_progress.pop(filename, None)


# ── /upload ───────────────────────────────────────────────
@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    filename = file.filename
    filepath = os.path.join(UPLOAD_DIR, filename)

    contents = await file.read()
    with open(filepath, "wb") as f:
        f.write(contents)

    upload_progress[filename] = 0
    threading.Thread(target=_run_indexing, args=(filepath, filename), daemon=True).start()

    return {"message": "Upload started", "filename": filename}


# ── /upload-progress ──────────────────────────────────────
@app.get("/upload-progress/{filename}")
def get_upload_progress(filename: str):
    if filename not in upload_progress:
        return JSONResponse(
            status_code=404,
            content={"filename": filename, "message": "Not found or already complete"},
        )
    return {"filename": filename, "progress": upload_progress[filename], "complete": False}


# ── /chat-stream (SSE) ────────────────────────────────────
@app.post("/chat-stream")
def chat_stream(req: ChatRequest):
    context = retrieve(collection, req.question)

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

    return StreamingResponse(generate(), media_type="text/event-stream")
