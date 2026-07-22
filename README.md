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

## Deployment (Sprint 7)

The backend is published at **https://studyai.binodtiwari.com**. The Android app
talks to this URL directly, so no Tailscale VPN is required on the phone.

```
Android phone (any network)
        │  HTTPS
        ▼
Cloudflare edge  ──►  Cloudflare Tunnel (outbound from HP server; no open port)
        │
        ▼
Nginx :80  (studyai.binodtiwari.com server block)
        │
        ▼
uvicorn 127.0.0.1:8002  (managed by systemd unit `studyai.service`)
```

### File locations on the HP server

| Artifact | Path | Repo copy |
|---|---|---|
| Nginx site | `/etc/nginx/sites-available/studyai` (symlinked in `sites-enabled`) | `deploy/nginx/studyai` |
| Cloudflared config | `/etc/cloudflared/config.yml` | `deploy/cloudflared/config.yml` |
| systemd unit | `/etc/systemd/system/studyai.service` | `deploy/systemd/studyai.service` |
| Tunnel ID | `d61228a8-8e71-457b-988b-cbaacf646760` | — |

The Nginx site sets `client_max_body_size 50M` so PDF uploads don't 413, disables
buffering on `/chat-stream` so SSE tokens arrive incrementally, and raises
`proxy_read_timeout` to 300s so long `/chat` and `/quiz` generations don't 504.
The FastAPI `/chat-stream` handler also sets `X-Accel-Buffering: no` so streaming
survives even if the Nginx tweak is ever lost.

### Running the backend

The backend is a systemd service. Do **not** launch uvicorn by hand — the service
loads `.env` via `EnvironmentFile=` and restarts on failure.

```bash
sudo systemctl status  studyai   # health check
sudo systemctl restart studyai   # after editing .env or the code
sudo journalctl -u studyai -f    # tail logs
```

### Cloudflare 100s limit

Cloudflare's free plan drops any single origin request that runs longer than
~100 seconds. `/chat-stream` is fine because it emits tokens continuously.
`/quiz` is not streamed — a very large quiz on a slow model can approach the
limit. If quiz size grows past what fits inside 100 s, convert `/quiz` to stream
questions one at a time.

### Fallback: Tailscale

The systemd unit binds uvicorn to `0.0.0.0:8002`, so the Tailscale IP still works
if the tunnel or Cloudflare has an outage:

```
http://100.95.45.33:8002
```

Point the Android build at that URL and re-install. Once the tunnel is healthy,
switch back to `https://studyai.binodtiwari.com`. To later harden the box and
force all public traffic through the tunnel, change `--host 0.0.0.0` to
`--host 127.0.0.1` in `start.sh` (do **not** do this before demo — it kills the
Tailscale fallback).

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

## API — flashcards, quizzes, history

Flashcard sets and quizzes are generated by the LLM from a user-chosen PDF and persisted per user. Relational data (accounts, per-user document index, quiz and flashcard history + full payloads) lives in a single SQLite database at `~/Desktop/Study_AI_users/studyai.db`. Vectors continue to live in per-user ChromaDB stores at `Study_AI_users/{user_id}/chroma_db/` — ChromaDB is already a vector database, so it stays. Chat history is **not** stored on the server — session memory keeps working the same way, but the transcript stays on the phone.

### Flashcards

- `POST /flashcards` — body `{user_id, source_pdf, count}` where `count ∈ {5, 10, 15, 20}`. Cards are drawn from **random** chunks of the chosen PDF, not by similarity. Returns `{id, source_pdf, created_at, cards}`.
- `GET  /flashcards/{user_id}` — index list, newest first.
- `GET  /flashcards/{user_id}/{set_id}` — full set. 404 if the id is not in this user's index.
- `DELETE /flashcards/{user_id}/{set_id}` — remove the file and the index entry.
- `POST /flashcard-reveal` — body `{user_id, set_id, revealed_count}`. The server takes `max(existing, revealed_count)` for `cards_revealed`, so re-taps and retries do not double-count, and bumps `flashcards_revealed_total` by the delta.

### Quizzes

- `POST /quiz` — body `{user_id, num_questions, source_pdf}`. When `user_id` is present the quiz is persisted and the response includes `id`; anonymous callers still get the old `{questions}`-only shape.
- `GET  /quizzes/{user_id}` — index list, newest first. Legacy rows that predate `id`s are still returned so history renders — the client filters on non-null `id` for fetch/delete.
- `GET  /quizzes/{user_id}/{quiz_id}` — full quiz payload.
- `DELETE /quizzes/{user_id}/{quiz_id}` — remove the file and the index entry.
- `POST /quiz-result` — body `{user_id, total_questions, correct, quiz_id?}`. When `quiz_id` is present, the score is attached to that quiz's record. When absent, a bare row is appended (kept so older app builds keep working through the sprint).

### Documents

- `GET  /docs-list?user_id=<id>` — returns `[{filename, timestamp}, ...]` so the picker can show both. `user_id` is required; a missing one returns **400**.
- `DELETE /documents/{user_id}/{filename}` — removes the PDF's chunks from the user's ChromaDB with a `where={"source": filename}` filter, drops the `pdfs_uploaded` entry, and deletes the file on disk. **Quiz and flashcard history generated from that PDF are intentionally kept** — the sets are self-contained JSON payloads that do not need the chunks.

### Progress

- `GET  /progress/{user_id}` — summary + indexes, including `questions_answered_total`, `questions_correct_total`, `flashcards_revealed_total`, `quiz_history`, and `flashcard_sets`. `chroma_db_path` is stripped.

### Ownership and per-user isolation

Every history/retrieve/delete endpoint 404s if the id does not belong to the caller's `user_id`. A client-supplied `source_pdf` is validated against the user's own `pdfs_uploaded` list before any retrieval or generation call runs — user 2 can never point a request at user 1's document by guessing the filename.

### Storage layout

SQLite database: `~/Desktop/Study_AI_users/studyai.db` (WAL mode, `PRAGMA foreign_keys=ON`, `ON DELETE CASCADE` from `users` to `documents`, `quizzes`, and `flashcards`). Four tables:

- `users` — id, name, email, password_hash, created_at.
- `documents` — id, user_id, filename, uploaded_at, `UNIQUE(user_id, filename)`.
- `quizzes` — id (TEXT, e.g. `q_20260706_143022`), user_id, source_pdf, created_at, total_questions, correct, payload (full JSON), is_legacy (1 for pre-`quiz_id` `/quiz-result` rows that only carry `{timestamp, total_questions, correct}`; those render in the old shape).
- `flashcards` — same structural pattern: id, user_id, source_pdf, created_at, card_count, cards_revealed, payload.

Progress totals (`questions_answered_total`, `questions_correct_total`, `flashcards_revealed_total`) are derived from the underlying rows via `SUM()` on read, so they cannot drift.

### One-time JSON → SQLite migration

Run once, manually, per deployment:

```bash
./venv/bin/python -m backend.migrate_to_sqlite
```

The script backs up the entire `Study_AI_users/` tree to a timestamped sibling first, then loads every user's `user_data.json`, `quizzes/*.json`, and `flashcards/*.json` into SQLite. It counts JSON entities independently, compares against `SELECT COUNT(*)` per table, and rolls the whole transaction back on any mismatch. On success it moves each user's original JSON files into `Study_AI_users/{user_id}/_migrated_backup/` — nothing is deleted. Refuses to run if `studyai.db` already has users.

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
├── backend/
│   ├── db.py                        ← SQLite schema + connection helper (WAL)
│   ├── user_store.py                ← account + per-user storage repository (SQLite-backed)
│   └── migrate_to_sqlite.py         ← one-time JSON → SQLite cutover
├── chroma_db/                       ← auto-created by ChromaDB, not committed
└── indexed_files.json               ← auto-created hash cache, not committed

# ~/Desktop/Study_AI_users/            ← runtime data, NOT under this repo
# ├── studyai.db                       ← SQLite (users, documents, quizzes, flashcards)
# └── <user_id>/chroma_db/             ← per-user vector store (unchanged)
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

To use OpenRouter instead of local Ollama inference, set the `OPENROUTER_API_KEY` environment variable and ensure `LLM_PROVIDER = "openrouter"` in `gg.py`. To fall back to fully local inference, set `LLM_PROVIDER = "ollama"`.

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