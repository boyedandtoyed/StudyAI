# AI-Powered Study Assistant
> CSE 4316 — Senior Design I | Team StudyAI | University of Texas at Arlington | Spring 2026

---

## Team

| Name | Role |
|---|---|
| Binod Tiwari | Backend Lead & AI Integration |
| Saujan Parajuli | Backend Support & Sprint Coordination |
| Pushpa Raj Adhikari | UI/UX Design & Android Frontend |

---

## Project Overview

An Android mobile application where students upload their own PDF notes and interact with them using AI. The system finds the most relevant sections of the uploaded material and generates accurate, context-aware answers using a local LLM. Images and diagrams inside PDFs are also understood through a vision model.

The core technique is **Retrieval-Augmented Generation (RAG)** — instead of sending the entire document to the AI, only the most semantically relevant chunks are retrieved and passed to the model. This keeps responses fast and accurate regardless of document size.

---

## Sprint Status

| Sprint | Focus | Status |
|---|---|---|
| Sprint 1 | RAG pipeline, PDF ingestion, local server, CLI prototype | ✅ Complete |
| Sprint 2 | FastAPI backend, Android mobile app, first prototype on device | 🔄 In Progress |
| Sprint 3 | Quiz generation, LangChain integration, UI refinement | 📅 Planned |

### Sprint 1 — Use Cases

| ID | Use Case | Status |
|---|---|---|
| UC-01 | Upload Study Material (PDF) | ✅ Done |
| UC-02 | Ask a Question in Natural Language | ✅ Done |
| UC-03 | Get an AI-Generated Answer with Source Citation | ✅ Done |
| UC-04 | Process Multiple Documents Simultaneously | ✅ Done |
| UC-05 | Run Entirely on Local Hardware (No Cloud) | ✅ Done |

### Sprint 2 — Use Cases

| ID | Use Case | Status |
|---|---|---|
| UC-06 | Upload PDF from Android App | 📅 Planned |
| UC-07 | Ask Question via Mobile Chat Interface | 📅 Planned |
| UC-08 | Generate Quiz from Mobile | 📅 Planned |
| UC-09 | View Answer with Source Citation on Mobile | 📅 Planned |
| UC-10 | Multi-Document Support on Mobile (stretch) | 📅 Stretch |

---

## Technology Stack

| Layer | Technology | Version | Status |
|---|---|---|---|
| Language | Python | 3.11 | ✅ Required — do not use 3.12+ |
| PDF Parsing | PyMuPDF (fitz) | Latest | ✅ Sprint 1 |
| Vector Database | ChromaDB | Latest | ✅ Sprint 1 |
| Embeddings | nomic-embed-text (Ollama) | — | ✅ Sprint 1 |
| Vision Model | moondream (Ollama) | — | ✅ Sprint 1 |
| LLM — Dev | llama3.2:1b (Ollama) | — | ✅ Sprint 1 |
| LLM — Demo | OpenAI gpt-4o-mini | — | 🔄 Ready, swap in one line |
| AI Orchestration | LangChain | — | 📅 Sprint 3 |
| API Framework | FastAPI | — | 📅 Sprint 2 |
| Mobile Frontend | Android (Java/Kotlin) | — | 📅 Sprint 2 |

> **Python version note:** This project requires Python 3.11 strictly.
> Python 3.12, 3.13, and 3.15 fail to install `chromadb` due to missing
> pre-built wheels for Rust-compiled dependencies (`ormsgpack`, `orjson`).
> Always create your venv using:
> ```bash
> py -3.11 -m venv venv
> ```

---

## How It Communicates

### Sprint 1 — Direct Ollama Connection (CLI)

```
Windows Laptop (Python client)
        │
        │  HTTP over Tailscale VPN
        │  (works from any network)
        ▼
HP Ubuntu Server
        ├── Ollama :11434
        │     ├── nomic-embed-text  (embeddings)
        │     ├── moondream         (vision/image descriptions)
        │     └── llama3.2:1b       (answer generation)
        └── ChromaDB runs locally on Windows laptop (in-memory/persistent)
```

### Sprint 2 — FastAPI Bridge (coming)

```
Android Phone
        │
        │  HTTP POST via Tailscale VPN
        ▼
HP Ubuntu Server
        ├── FastAPI :8000
        │     ├── POST /upload  → ingest PDF into ChromaDB
        │     ├── POST /ask     → retrieve + generate answer
        │     └── POST /quiz    → generate quiz questions
        └── Ollama :11434
              ├── nomic-embed-text
              ├── moondream
              └── llama3.2:1b
```

### Network — Tailscale VPN

The project uses [Tailscale](https://tailscale.com) to create a private encrypted tunnel between all devices. This allows the server to be reached from any network — home WiFi, college WiFi, mobile hotspot — without port forwarding or exposing the server to the public internet.

> **Important:** The server Tailscale IP used in Sprint 1 is specific to the
> current development environment. This IP is intentionally excluded from the
> repository and must be configured locally by each developer.
> It will be replaced with an environment variable in Sprint 2.

To configure, edit the top of `gg.py`:
```python
OLLAMA_BASE_URL = "http://YOUR_TAILSCALE_IP:11434"
```

---

## RAG Pipeline — How It Works

### Indexing (runs once at startup)

```
PDF File
  └─► PyMuPDF — extract text page by page
        └─► Split into 500-char overlapping chunks
              └─► nomic-embed-text → 768-dimensional vector per chunk
                    └─► Stored in ChromaDB with source + page metadata

Images in PDF (if any)
  └─► PyMuPDF — extract raw image bytes
        └─► moondream → plain text description of diagram
              └─► nomic-embed-text → vector
                    └─► Stored in ChromaDB tagged as image_description
```

Persistent caching: a hash of each PDF is stored in `indexed_files.json`.
On subsequent runs, unchanged files are skipped entirely.

### Question Answering (every query)

```
Student question
  └─► nomic-embed-text → question vector
        └─► ChromaDB cosine similarity search
              └─► Top 5 most relevant chunks returned
                    └─► Assembled into prompt with question
                          └─► llama3.2:1b → answer streamed live
```

The LLM never reads the full PDF — only the 5 most relevant chunks per query.

---

## Key Concepts

**Embedding** — converting text into numbers where similar meanings produce similar numbers. Enables semantic search rather than keyword matching.

**Cosine similarity** — the math ChromaDB uses to compare vectors. Smaller angle between two vectors = more similar meaning.

**Chunk overlap** — each 500-char chunk shares 50 characters with the next so sentences are never cut off at a boundary.

**xref** — the internal ID PyMuPDF uses to locate an image inside a PDF. `page.get_images()` returns tuples and `img[0]` is this ID. Passed to `doc.extract_image(xref)` to get the actual bytes.

**RAG vs Stuffing** — stuffing sends the full document to the LLM (fails on large docs due to context window limits). RAG retrieves only relevant chunks (fast and accurate at any document size).

**Persistent ChromaDB** — vectors saved to `chroma_db/` folder on disk. Skips re-indexing unchanged files on every run.

---

## Project Structure

```
StudyAI/
├── gg.py                            ← Sprint 1 main script (RAG pipeline)
├── requirements.txt                 ← Python dependencies
├── README.md
├── .gitignore
├── demo_pdfs/
│   ├── solar_system_overview.pdf    ← text-only demo document
│   ├── solar_system_planets.pdf     ← demo with planet diagrams
│   └── solar_system_moons_rings.pdf ← demo with moon/ring diagrams
├── android/                         ← Sprint 2: Pushpa's Android app
├── docs/                            ← sprint reports, project charter
├── chroma_db/                       ← auto-created by ChromaDB, not committed
└── indexed_files.json               ← auto-created hash cache, not committed
```

---

## Setup

### 1. Requirements

- **Python 3.11** (strictly required)
- Ollama running on your server with models pulled:

```bash
ollama pull nomic-embed-text
ollama pull moondream
ollama pull llama3.2:1b
```

### 2. Clone & Install

```bash
git clone https://github.com/boyedandtoyed/StudyAI.git
cd StudyAI

# Create venv with Python 3.11
py -3.11 -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Install
pip install chromadb pymupdf requests
```

### 3. Configure Server

Open `gg.py` and set your server IP at the top:

```python
OLLAMA_BASE_URL = "http://YOUR_TAILSCALE_IP:11434"
```

### 4. Run

```bash
python gg.py
```

Automatically loads all PDFs from the `demo_pdfs/` folder.

### 5. Commands

| Input | Action |
|---|---|
| Any question | Search notes and stream an answer |
| `quiz` | Generate 3 multiple choice questions |
| `quit` | Exit |

---

## Demo Questions (Solar System PDFs)

```
What is the surface temperature of Venus and why is it so hot?
Which moon has a subsurface ocean that might support life?
How long does it take Neptune to orbit the Sun?
What does the planet size comparison diagram show?
What are Saturn's rings made of and how thick are they?
```

---

## Dependencies

```
pymupdf     — PDF text and image extraction
chromadb    — local vector database
requests    — HTTP calls to Ollama server
```

> **Sprint 3 addition:** LangChain will replace the custom text chunker
> with `RecursiveCharacterTextSplitter` and add more robust pipeline management.

---

## Roadmap

| Sprint | Deliverable |
|---|---|
| ✅ Sprint 1 | CLI RAG prototype with vision support |
| 🔄 Sprint 2 | FastAPI + Android app — first mobile prototype |
| 📅 Sprint 3 | LangChain integration, quiz on mobile, UI polish |
| 📅 Final | Performance improvements, full documentation, project defense |