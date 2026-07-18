"""Wire-contract tests for /docs-list and /flashcards.

These pin the two response/request shapes that shipped mismatched twice
already this sprint (bare-string vs object docs-list; required source_pdf on
flashcards). Runs through Starlette's TestClient so ``response_model`` is
actually exercised on the way out — direct handler calls would bypass it.

The app's lifespan is intentionally NOT triggered (no ``with`` around the
client) so the test doesn't fight the running systemd server for a lock on
``./chroma_db``. None of the endpoints exercised here touch the global
``collection``; per-user collections are opened lazily and use their own
per-user chroma folder.
"""
import json
import shutil
import sys
import uuid
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gg  # noqa: E402
import main_fastapi  # noqa: E402
from backend import user_store  # noqa: E402


client = TestClient(main_fastapi.app)


@pytest.fixture
def clean_user():
    """A fresh user with no uploaded PDFs. Cleans up on teardown."""
    user_store.init_user_store()
    email = f"contract_{uuid.uuid4()}@test.com"
    user = user_store.create_user("Contract Test", email, "password123")
    uid = user["id"]

    yield uid

    folder = user_store._USERS_DIR / str(uid)
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)
    with user_store._db_lock:
        db = user_store._load_db()
        db["users"] = [u for u in db["users"] if u["id"] != uid]
        user_store._save_json_atomic(user_store._USERS_DB, db)
    main_fastapi._user_collections.pop(uid, None)


def _fake_chunks(_collection, _source_pdf, batch_size=1):
    return [{
        "text": "Study material about the solar system.",
        "meta": {"source": "any", "page": 1},
    }]


def _fake_cards_json(count):
    return json.dumps({
        "cards": [
            {
                "question": f"Q{i}?",
                "options": ["a", "b", "c", "d"],
                "correct_index": 0,
                "explanation": "",
            }
            for i in range(count)
        ],
    })


# ── /docs-list contract ────────────────────────────────────

def test_docs_list_empty_state_returns_empty_documents_array(clean_user):
    """Empty state must be {'documents': []} — never null, never a bare
    array, never a bare-string list. response_model enforces this."""
    r = client.get("/docs-list", params={"user_id": clean_user})
    assert r.status_code == 200, r.text
    assert r.json() == {"documents": []}


def test_docs_list_populated_returns_filename_timestamp_objects(clean_user):
    """Populated state must be a list of {filename, timestamp} objects —
    exactly the shape the Android picker parses. If this ever comes back
    as bare strings again the Kotlin JSON reader crashes with
    BEGIN_OBJECT/BEGIN_STRING, same failure as sprint 5."""
    ud = user_store.get_user_data(clean_user)
    ud["pdfs_uploaded"].append({
        "filename": "notes.pdf",
        "timestamp": "2026-07-18T00:00:00Z",
    })
    user_store.save_user_data(clean_user, ud)

    r = client.get("/docs-list", params={"user_id": clean_user})
    assert r.status_code == 200, r.text
    docs = r.json()["documents"]
    assert len(docs) == 1
    assert docs[0] == {"filename": "notes.pdf", "timestamp": "2026-07-18T00:00:00Z"}
    # Sanity: extra fields would be dropped by response_model, so this is
    # the whole shape the client will ever see.
    assert set(docs[0].keys()) == {"filename", "timestamp"}


def test_docs_list_missing_user_id_returns_400():
    r = client.get("/docs-list")
    assert r.status_code == 400
    assert "user_id" in r.json()["detail"].lower()


def test_docs_list_unknown_user_returns_404():
    r = client.get("/docs-list", params={"user_id": 999_999_999})
    assert r.status_code == 404


# ── /flashcards contract ───────────────────────────────────

def test_flashcards_accept_omitted_source_pdf(clean_user, monkeypatch):
    """source_pdf omitted -> all-documents path. Matches /quiz behavior.
    This is the exact request shape that returned 422 in sprint 7."""
    monkeypatch.setattr(gg, "get_llm_response", lambda _p: _fake_cards_json(5))
    monkeypatch.setattr(main_fastapi, "random_chunks_from_source", _fake_chunks)

    r = client.post("/flashcards", json={"user_id": clean_user, "count": 5})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["source_pdf"] is None
    assert len(body["cards"]) == 5


def test_flashcards_accept_explicit_null_source_pdf(clean_user, monkeypatch):
    """Explicit null must match omitted — the Android JSON serializer sends
    both shapes depending on the model, and both must be legal."""
    monkeypatch.setattr(gg, "get_llm_response", lambda _p: _fake_cards_json(5))
    monkeypatch.setattr(main_fastapi, "random_chunks_from_source", _fake_chunks)

    r = client.post("/flashcards", json={"user_id": clean_user, "source_pdf": None, "count": 5})
    assert r.status_code == 200, r.text
    assert r.json()["source_pdf"] is None


def test_flashcards_reject_unowned_source_pdf_with_404(clean_user):
    """A named PDF the user does not own must 404 — even before touching
    the LLM. Prevents cross-user targeting by filename guess."""
    r = client.post(
        "/flashcards",
        json={"user_id": clean_user, "source_pdf": "not-mine.pdf", "count": 5},
    )
    assert r.status_code == 404


def test_flashcards_reject_count_outside_allowed_with_clear_detail(clean_user):
    """Bad count must return a message that names the allowed values —
    not a bare Pydantic 422."""
    r = client.post("/flashcards", json={"user_id": clean_user, "count": 7})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "5" in detail and "20" in detail, detail


def test_flashcards_empty_collection_returns_helpful_400(clean_user):
    """No documents indexed yet + no source_pdf -> 400 with a message that
    tells the user to upload something, not a downstream 500."""
    r = client.post("/flashcards", json={"user_id": clean_user, "count": 5})
    assert r.status_code == 400
    detail = r.json()["detail"].lower()
    assert "upload" in detail or "no indexed" in detail
