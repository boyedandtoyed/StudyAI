# AI-Powered Study Assistant — RAG Pipeline
### CSE 4316 | Team StudyAI | University of Texas at Arlington
**Binod Tiwari · Saujan Parajuli · Pushpa Raj Adhikari**

---

## What This Project Does

This is a command-line AI study assistant that lets a student upload their own PDF notes and ask questions about them. The system finds the most relevant parts of the notes and generates a context-aware answer using a local LLM. It also understands diagrams and images inside the PDFs.

The core technique used is **Retrieval-Augmented Generation (RAG)** — instead of sending the entire document to the AI, the system finds only the most relevant chunks and sends those. This makes it faster and more accurate regardless of how large the document is.

---

## Models Used

All three models run locally on a remote HP laptop server over a local network. No internet or cloud API is required.

| Model | Purpose |
|---|---|
| `nomic-embed-text` | Converts text into vectors (embeddings) for similarity search |
| `moondream` | Reads images and diagrams and describes them in text |
| `llama3.2:1b` | Reads the retrieved chunks and generates the final answer |

---

## How It Works — Step by Step

### At Startup (runs once)

1. Each PDF is opened with PyMuPDF and text is extracted page by page
2. Text is split into overlapping chunks of ~500 characters
3. Each chunk is sent to `nomic-embed-text` which converts it into a list of 768 numbers (a vector) representing its meaning
4. Images on each page are extracted and sent to `moondream` which describes them in plain text
5. Those image descriptions are also embedded into vectors
6. All vectors plus their original text are stored in ChromaDB (runs in memory on your laptop)

### When You Ask a Question

1. Your question is sent to `nomic-embed-text` and converted into a vector
2. ChromaDB compares that vector against all stored chunk vectors using cosine similarity
3. The 5 most relevant chunks are returned (text chunks and/or image description chunks)
4. Those chunks are assembled into a prompt along with your question
5. The prompt is sent to `llama3.2:1b` which streams the answer back live

The LLM never sees the full PDF — only the 5 most relevant chunks.

---

## What Each File Does

- `study_assistant_rag_v3.py` — main script, run this
- `solar_system_overview.pdf` — text-only demo document
- `solar_system_planets.pdf` — demo document with planet size and orbital diagrams
- `solar_system_moons_rings.pdf` — demo document with moon and ring system diagrams
- `chroma_db/` — folder where ChromaDB stores vectors (auto-created)

---

## Imports Explained

```python
import fitz       # PyMuPDF — reads PDFs, extracts text and images
import os         # checks if file paths exist on your laptop
import requests   # sends HTTP requests to the Ollama server
import json       # parses responses from Ollama
import base64     # encodes image bytes as text for sending over HTTP
import chromadb   # in-memory vector database, runs on your laptop
```

---

## API Calls — When and Where

Every API call goes to `http://192.168.1.211:11434` (the HP laptop server running Ollama).

| When | Why | Endpoint | Model |
|---|---|---|---|
| Startup — per text chunk | Embed the chunk | `/api/embeddings` | nomic-embed-text |
| Startup — per image | Describe the image | `/api/chat` | moondream |
| Startup — per image description | Embed the description | `/api/embeddings` | nomic-embed-text |
| Every question | Embed the question | `/api/embeddings` | nomic-embed-text |
| Every question | Generate the answer | `/api/chat` | llama3.2:1b |

ChromaDB never makes a network call. It runs entirely in memory on your laptop.

---

## Key Concepts

**Embedding** — converting text into a list of numbers where similar meanings produce similar numbers. This is how the system finds relevant chunks without doing keyword search.

**Cosine similarity** — the math ChromaDB uses to compare vectors. It measures the angle between two vectors. A small angle means the texts are semantically similar.

**Chunk overlap** — each chunk shares 50 characters with the next one so sentences are never cut off at a boundary and lost.

**xref** — the internal ID number PyMuPDF uses to locate an image inside a PDF. `page.get_images()` returns tuples and `img[0]` gets the first element which is this ID. It is then passed to `doc.extract_image(xref)` to get the actual image bytes.

**RAG vs Stuffing** — the old approach (stuffing) sent the entire document to the LLM. RAG only sends the relevant chunks. Stuffing breaks on large documents because LLMs have a context window limit. RAG stays fast and accurate regardless of document size.

---

## Setup

**Requirements:**
- Python 3.11
- Ollama running on the server with these models pulled:
  ```
  ollama pull nomic-embed-text
  ollama pull moondream
  ollama pull llama3.2:1b
  ```

**Install dependencies:**
```bash
pip install chromadb pymupdf
```

**Run:**
```bash
python study_assistant_rag_v3.py
```

**Commands at runtime:**
- Type any question — get an answer from your notes
- Type `quiz` — generate 3 multiple choice questions
- Type `quit` — exit

---

## Sample Questions (for demo with solar system PDFs)

1. What is the surface temperature of Venus and why is it so hot?
2. Which moon has a subsurface ocean that might support life?
3. How long does it take Neptune to orbit the Sun?
4. What does the planet size comparison diagram show?
5. What are Saturn's rings made of?

---

## What the Answer Looks Like

```
[From your notes]: According to solar_system_moons_rings.pdf (Page 2),
Saturn's rings are made of billions of chunks of ice and rock ranging
from tiny grains to house-sized boulders...

[From general knowledge]: The rings are divided into groups labeled
A through G. The Cassini Division is the most prominent gap...
```

---

## Architecture Diagram

```
Your Laptop                          HP Server (192.168.1.211)
───────────────────────────          ──────────────────────────
PDF files
    ↓
PyMuPDF extracts text + images
    ↓
Text chunks ──────────────────────→  nomic-embed-text
Image bytes ──────────────────────→  moondream → description
Image descriptions ───────────────→  nomic-embed-text
    ↓
ChromaDB stores all vectors
    (in memory on your laptop)

When question asked:
Question ─────────────────────────→  nomic-embed-text → vector
ChromaDB similarity search
    ↓ top 5 chunks
Assembled prompt ─────────────────→  llama3.2:1b
    ↓
Answer streamed back to terminal
```


# AI-Powered Study Assistant
CSE 4316 — Team StudyAI
University of Texas at Arlington

## Team
- Binod Tiwari — Backend & AI Integration
- Saujan Parajuli — Backend Support & Sprint Coordination  
- Pushpa Raj Adhikari — UI/UX & Android Frontend

## Project
An Android mobile app where students upload notes and 
get AI-powered answers using a RAG pipeline.

## Stack
- Python 3.11, FastAPI, ChromaDB, LangChain
- Ollama (llama3.2:1b, nomic-embed-text, moondream)
- Android (Java/Kotlin)

## Sprint Status
- Sprint 1 ✅ — RAG pipeline complete
| Use Case |                   Status |
|---|---|
| UC-01 Upload Study Material | Done |
| UC-02 Ask a Question | Done |
| UC-03 Get an AI-Generated Answer | Done |
| UC-04 Process Multiple Documents | Done |
| UC-05 Run the System Locally | Done |

- Sprint 2 🔄 — Android app + FastAPI