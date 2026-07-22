"""Upload visibility: doc appears in /docs-list before indexing finishes, and stays there even if the background indexer crashes.

Indexing runs in a background thread and can take minutes (Ollama +
ChromaDB). Previously the documents row was only inserted at the *end*
of that thread, so an indexing failure — or the user checking their
list before indexing finished — meant the file appeared to have
uploaded successfully (200 OK, bytes on disk) but was invisible in the
app. These tests pin the current contract:

1. `POST /upload` inserts the documents row synchronously, before the
   background thread starts.
2. If the background indexer raises, upload_progress still gets cleared
   (the client's polling terminates) and the documents row stays put
   (the file is on disk; the user can still see and delete it).
"""
import io
import shutil
import sys
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import main_fastapi  # noqa: E402
from backend import db, user_store  # noqa: E402


client = TestClient(main_fastapi.app)


@pytest.fixture
def clean_user(tmp_path, monkeypatch):
    """Fresh SQLite user + isolated upload dir. No live server, no ChromaDB."""
    monkeypatch.setattr(main_fastapi, "UPLOAD_DIR", str(tmp_path))
    user_store.init_user_store()
    email = f"upvis_{uuid.uuid4()}@test.local"
    user = user_store.create_user("UploadVis", email, "password123")
    uid = user["id"]
    yield uid
    with db.connect() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (uid,))
    folder = db._USERS_DIR / str(uid)
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)
    main_fastapi._user_collections.pop(uid, None)
    main_fastapi.upload_progress.pop("visibility_test.pdf", None)


def test_document_row_is_inserted_before_indexing_completes(clean_user, monkeypatch):
    """Even if the background thread does nothing (patched to no-op), the
    doc must appear in /docs-list right after /upload returns 200."""
    monkeypatch.setattr(main_fastapi, "get_collection_for_user", lambda _uid: None)
    # Silence the background thread so the test isn't racing indexing.
    monkeypatch.setattr(main_fastapi, "_run_indexing", lambda *a, **kw: None)

    r = client.post(
        "/upload",
        files={"file": ("visibility_test.pdf", io.BytesIO(b"%PDF-1.4 ..."), "application/pdf")},
        data={"user_id": str(clean_user)},
    )
    assert r.status_code == 200, r.text
    assert r.json()["filename"] == "visibility_test.pdf"

    # The doc is visible immediately — no waiting for background indexing.
    r = client.get(f"/docs-list?user_id={clean_user}")
    assert r.status_code == 200
    docs = r.json()["documents"]
    assert len(docs) == 1
    assert docs[0]["filename"] == "visibility_test.pdf"


def test_indexing_failure_does_not_hide_the_doc_or_leak_progress(clean_user, monkeypatch):
    """Simulate an indexing crash (the real cause of yesterday's blank-list
    bug): ChromaDB `add` raises. The doc must still be in the list, and
    upload_progress must be cleared so the client's polling terminates."""

    class _FakeCollection:
        def count(self):
            return 0
        def add(self, *_a, **_kw):
            raise RuntimeError("simulated readonly database")

    monkeypatch.setattr(main_fastapi, "get_collection_for_user", lambda _uid: _FakeCollection())
    # Real _run_indexing this time — we're testing its error handling.
    # Patch index_pdf so it invokes collection.add (which raises).
    def fake_index_pdf(_filepath, collection, _start, progress_callback=None):
        if progress_callback:
            progress_callback(50)
        collection.add()  # raises
    monkeypatch.setattr(main_fastapi, "index_pdf", fake_index_pdf)

    r = client.post(
        "/upload",
        files={"file": ("visibility_test.pdf", io.BytesIO(b"%PDF-1.4 ..."), "application/pdf")},
        data={"user_id": str(clean_user)},
    )
    assert r.status_code == 200, r.text

    # Give the background thread a chance to run and raise. It's daemon, so
    # we just wait briefly — max 2s, break as soon as progress clears.
    import time
    for _ in range(20):
        if "visibility_test.pdf" not in main_fastapi.upload_progress:
            break
        time.sleep(0.1)

    # Progress cleared → client's /upload-progress polling would return 404
    # ("already complete") and stop.
    assert "visibility_test.pdf" not in main_fastapi.upload_progress

    # Doc row survives the indexing crash — user still sees the file.
    r = client.get(f"/docs-list?user_id={clean_user}")
    assert r.status_code == 200
    docs = r.json()["documents"]
    assert len(docs) == 1
    assert docs[0]["filename"] == "visibility_test.pdf"


def test_reupload_same_filename_does_not_duplicate_the_row(clean_user, monkeypatch):
    """Uploading the same filename twice must not produce two rows —
    documents has a UNIQUE(user_id, filename) constraint, and the
    _register_uploaded_document helper checks before appending."""
    monkeypatch.setattr(main_fastapi, "get_collection_for_user", lambda _uid: None)
    monkeypatch.setattr(main_fastapi, "_run_indexing", lambda *a, **kw: None)

    for _ in range(2):
        r = client.post(
            "/upload",
            files={"file": ("visibility_test.pdf", io.BytesIO(b"data"), "application/pdf")},
            data={"user_id": str(clean_user)},
        )
        assert r.status_code == 200, r.text

    r = client.get(f"/docs-list?user_id={clean_user}")
    docs = r.json()["documents"]
    assert len(docs) == 1
