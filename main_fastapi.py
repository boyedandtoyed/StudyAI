from contextlib import asynccontextmanager
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional
import os

from gg import build_vectorstore, retrieve, stream_llm

# ── SESSION MEMORY ────────────────────────────────────────
session_store: dict = {}

# ── VECTORSTORE ───────────────────────────────────────────
collection = None


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
