import fitz  # PyMuPDF
import os
import requests
import json
import base64
import chromadb
import hashlib
from langchain_text_splitters import RecursiveCharacterTextSplitter

# ── CONFIG ──────────────────────────────────────────────
OLLAMA_BASE_URL = "http://localhost:11434"
LLM_MODEL       = "gemma4:26b"
EMBED_MODEL     = "nomic-embed-text"
VISION_MODEL    = "moondream"
TOP_K_CHUNKS    = 5
CHUNK_SIZE      = 500
CHUNK_OVERLAP   = 50
CHROMA_DIR      = "./chroma_db"       # folder created next to your script
HASH_FILE       = "./indexed_files.json"  # tracks which files are already indexed
# ────────────────────────────────────────────────────────


# ── FILE HASHING ─────────────────────────────────────────
def get_file_hash(path):
    """
    Generate an MD5 hash of a PDF file's contents.
    If the file hasn't changed, the hash will be identical.
    This lets us skip re-indexing files that are already in ChromaDB.
    """
    hasher = hashlib.md5()
    with open(path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def load_indexed_hashes():
    """Load the record of already-indexed files from disk."""
    if os.path.exists(HASH_FILE):
        with open(HASH_FILE, "r") as f:
            return json.load(f)
    return {}


def save_indexed_hashes(hashes):
    """Save the updated record of indexed files to disk."""
    with open(HASH_FILE, "w") as f:
        json.dump(hashes, f, indent=2)


# ── TEXT CHUNKING ────────────────────────────────────────
def chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)


# ── PDF EXTRACTION ───────────────────────────────────────
def extract_page_data(pdf_path):
    """Extract text and base64 images from every page of a PDF."""
    doc = fitz.open(pdf_path)
    pages = []
    for i, page in enumerate(doc):
        text = page.get_text().strip()
        images = []
        for img in page.get_images(full=True):
            xref = img[0]
            try:
                base_image = doc.extract_image(xref)
                img_b64 = base64.b64encode(base_image["image"]).decode("utf-8")
                images.append(img_b64)
            except Exception:
                pass
        pages.append({
            "text": text,
            "images": images,
            "source": os.path.basename(pdf_path),
            "page": i + 1
        })
    return pages


# ── OLLAMA CALLS ─────────────────────────────────────────
def get_embedding(text):
    """Get vector embedding from nomic-embed-text on Ollama server."""
    response = requests.post(
        f"{OLLAMA_BASE_URL}/api/embeddings",
        json={"model": EMBED_MODEL, "prompt": text}
    )
    return response.json()["embedding"]


def describe_image(img_b64):
    """Send image to moondream and get a text description back."""
    payload = {
        "model": VISION_MODEL,
        "messages": [{
            "role": "user",
            "content": "Describe everything in this diagram in detail. Include all labels, numbers, names, arrows, colors, and any data shown. Be thorough.",
            "images": [img_b64]
        }],
        "stream": False
    }
    try:
        response = requests.post(
            f"{OLLAMA_BASE_URL}/api/chat",
            json=payload,
            timeout=60
        )
        return response.json()["message"]["content"]
    except Exception as e:
        return f"[Image description unavailable: {e}]"


def stream_llm(prompt):
    """Stream LLM response token by token from llama3.2:1b."""
    payload = {
        "model": LLM_MODEL,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True
    }
    full_response = ""
    with requests.post(
        f"{OLLAMA_BASE_URL}/api/chat",
        json=payload,
        stream=True
    ) as r:
        for line in r.iter_lines():
            if line:
                chunk = json.loads(line)
                content = chunk.get("message", {}).get("content", "")
                print(content, end="", flush=True)
                full_response += content
    return full_response


# ── INDEX A SINGLE PDF ───────────────────────────────────
def index_pdf(pdf_path, collection, chunk_id_start, progress_callback=None):
    """
    Extract, chunk, embed, and store one PDF into ChromaDB.
    Returns the next available chunk_id after indexing.
    progress_callback(pct: int) is called after each page, where pct is 0-100.
    """
    all_texts      = []
    all_embeddings = []
    all_ids        = []
    all_metadata   = []
    chunk_id       = chunk_id_start

    pages = extract_page_data(pdf_path)
    total_pages = len(pages)
    source = os.path.basename(pdf_path)

    for page_index, page_data in enumerate(pages):
        page = page_data["page"]

        # ── TEXT CHUNKS ──
        if page_data["text"]:
            chunks = chunk_text(page_data["text"])
            for chunk in chunks:
                print(f"  [Embedding text] {source} p{page} chunk {chunk_id + 1}...", end="\r")
                embedding = get_embedding(chunk)
                all_texts.append(chunk)
                all_embeddings.append(embedding)
                all_ids.append(f"chunk_{chunk_id}")
                all_metadata.append({
                    "source": source,
                    "page": page,
                    "type": "text"
                })
                chunk_id += 1

        # ── IMAGE CHUNKS ──
        for img_index, img_b64 in enumerate(page_data["images"]):
            print(f"\n  [Describing image] {source} p{page} image {img_index + 1} via moondream...")
            description = describe_image(img_b64)

            if description.strip():
                image_chunk = f"[Diagram on page {page} of {source}]: {description}"
                print(f"  [Embedding image description] chunk {chunk_id + 1}...", end="\r")
                embedding = get_embedding(image_chunk)
                all_texts.append(image_chunk)
                all_embeddings.append(embedding)
                all_ids.append(f"chunk_{chunk_id}")
                all_metadata.append({
                    "source": source,
                    "page": page,
                    "type": "image_description"
                })
                chunk_id += 1
                print(f"  [Image indexed] {source} p{page} image {img_index + 1}   ")

        if progress_callback and total_pages > 0:
            progress_callback(int((page_index + 1) / total_pages * 100))

    if all_texts:
        collection.add(
            documents=all_texts,
            embeddings=all_embeddings,
            ids=all_ids,
            metadatas=all_metadata
        )

    return chunk_id  # return next available ID


# ── BUILD / LOAD VECTORSTORE ─────────────────────────────
def build_vectorstore(pdf_paths):
    """
    Smart vectorstore builder:
    - Creates ChromaDB on disk if it doesn't exist
    - Loads existing ChromaDB if it does exist
    - Only indexes NEW or CHANGED files — skips already-indexed ones
    """
    if isinstance(pdf_paths, str):
        pdf_paths = [pdf_paths]

    # Always use persistent client — saves to CHROMA_DIR folder
    client = chromadb.PersistentClient(path=CHROMA_DIR)

    # Get or create the collection
    try:
        collection = client.get_collection("study_notes")
        print(f"  [Found existing vectorstore at '{CHROMA_DIR}']")
        existing_count = collection.count()
        print(f"  [Currently has {existing_count} chunks stored]")
    except Exception:
        collection = client.create_collection(
            name="study_notes",
            metadata={"hnsw:space": "cosine"}
        )
        print(f"  [Created new vectorstore at '{CHROMA_DIR}']")
        existing_count = 0

    # Load record of which files have been indexed
    indexed_hashes = load_indexed_hashes()

    # Figure out next chunk ID (avoid ID collisions with existing chunks)
    chunk_id_start = existing_count

    files_indexed  = 0
    files_skipped  = 0

    for path in pdf_paths:
        if not os.path.exists(path):
            print(f"  [WARNING] File not found: '{path}' — skipping.")
            continue

        filename = os.path.basename(path)
        current_hash = get_file_hash(path)

        # Check if this exact file (with same content) is already indexed
        if indexed_hashes.get(filename) == current_hash:
            print(f"  [Skipping] '{filename}' — already indexed, unchanged.")
            files_skipped += 1
            continue

        # File is new or has changed — index it
        print(f"\n  [Indexing] '{filename}'...")
        chunk_id_start = index_pdf(path, collection, chunk_id_start)

        # Record that this file is now indexed
        indexed_hashes[filename] = current_hash
        files_indexed += 1

    # Save updated hash record
    save_indexed_hashes(indexed_hashes)

    total = collection.count()
    print(f"\n  [Vectorstore ready]")
    print(f"  Files indexed this run : {files_indexed}")
    print(f"  Files skipped (cached) : {files_skipped}")
    print(f"  Total chunks in DB     : {total}\n")

    return client, collection


# ── RETRIEVE ─────────────────────────────────────────────
def retrieve(collection, question):
    """Embed the question and find the TOP_K most relevant chunks."""
    query_embedding = get_embedding(question)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(TOP_K_CHUNKS, collection.count())
    )

    chunks = results["documents"][0]
    metas  = results["metadatas"][0]

    context_parts = []
    for chunk, meta in zip(chunks, metas):
        label = "📷 Diagram" if meta["type"] == "image_description" else "📄 Text"
        context_parts.append(
            f"[{label} — {meta['source']}, Page {meta['page']}]\n{chunk}"
        )
    return "\n\n---\n\n".join(context_parts)


# ── ANSWER QUESTION ──────────────────────────────────────
def answer_question(collection, question):
    print("\n  [Searching notes and diagrams...] ", end="")
    context = retrieve(collection, question)
    print("done\n")

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

=== RETRIEVED CONTENT ===
{context}

=== STUDENT QUESTION ===
{question}

Format:
[From your notes]: ...
[From general knowledge]: ... (only if needed)"""

    print("[Answer]\n")
    stream_llm(prompt)


# ── GENERATE QUIZ ─────────────────────────────────────────
def generate_quiz(collection):
    print("\n  [Retrieving content for quiz...] ", end="")
    context = retrieve(collection, "key facts important concepts definitions diagrams")
    print("done\n")

    prompt = f"""Using the student notes and diagram descriptions below, create 3 multiple choice questions.
For each question:
- Write a clear question
- Give 4 options (A, B, C, D)
- Mark the correct answer
- Give a one-sentence explanation

=== NOTES AND DIAGRAMS ===
{context}"""

    print("[Quiz]\n")
    stream_llm(prompt)


# ── MAIN ──────────────────────────────────────────────────
def run_pipeline(pdf_paths):
    if isinstance(pdf_paths, str):
        pdf_paths = [pdf_paths]

    print("\n" + "="*55)
    print("  AI STUDY ASSISTANT — RAG + Vision Pipeline")
    print("  Persistent ChromaDB — indexes once, reuses always")
    print("="*55)

    print("\n[Checking vectorstore...]\n")
    client, collection = build_vectorstore(pdf_paths)

    if not collection or collection.count() == 0:
        print("No content indexed. Check your PDF paths.")
        return

    print("="*55)
    print("  READY")
    print(f"  Documents : {len(pdf_paths)}")
    print(f"  Chunks    : {collection.count()}")
    print("  Commands  : type a question, 'quiz', or 'quit'")
    print("="*55)

    while True:
        print("\n")
        user_input = input(">>> ").strip()
        if not user_input:
            continue
        elif user_input.lower() == "quit":
            print("Goodbye!")
            break
        elif user_input.lower() == "quiz":
            generate_quiz(collection)
        else:
            answer_question(collection, user_input)


# ─────────────────────────────────────────────
# run through the folder named demo_pdf and add that to the list of run_pipeline generate it dinamically, we can add more pdfs to that folder
if __name__ == "__main__":
    run_pipeline([os.path.join("demo_pdfs", f) for f in os.listdir("demo_pdfs") if f.endswith(".pdf")])
