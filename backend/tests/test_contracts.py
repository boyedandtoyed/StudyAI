"""Wire-contract tests for the endpoints touched by the JSON→SQLite cutover.

Runs through Starlette's TestClient so ``response_model`` is actually
exercised on the way out — direct handler calls would bypass it. These pin
every response shape the Android app parses:

  /docs-list                       (docs)
  /flashcards                      (POST, generate)
  /progress/{user_id}              (with modern + legacy rows)
  /quizzes/{user_id}               (list index)
  /quizzes/{user_id}/{quiz_id}     (single payload)
  /flashcards/{user_id}            (list index)
  /flashcards/{user_id}/{set_id}   (single payload)

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
from backend import db, user_store  # noqa: E402


client = TestClient(main_fastapi.app)


@pytest.fixture
def clean_user():
    """A fresh user with no uploaded PDFs. Cleans up on teardown via CASCADE."""
    user_store.init_user_store()
    email = f"contract_{uuid.uuid4()}@test.com"
    user = user_store.create_user("Contract Test", email, "password123")
    uid = user["id"]

    yield uid

    with db.connect() as conn:
        conn.execute("DELETE FROM users WHERE id = ?", (uid,))
    folder = db._USERS_DIR / str(uid)
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)
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


# ── /flashcards POST contract ──────────────────────────────

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


# ── /progress contract (SQLite cutover: totals derived on read) ───

def test_progress_response_shape_and_field_types(clean_user):
    """/progress on a fresh account: exact top-level keys, exact types.
    Post-SQLite: totals are derived via SUM(), so they must still surface
    as plain ints (not Decimal, not None) even when there's no history."""
    r = client.get(f"/progress/{clean_user}")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body == {
        "user_id": clean_user,
        "pdfs_uploaded": [],
        "quiz_history": [],
        "flashcard_sets": [],
        "questions_answered_total": 0,
        "questions_correct_total": 0,
        "flashcards_revealed_total": 0,
    }


def test_progress_renders_modern_and_legacy_history_side_by_side(clean_user):
    """The critical SQLite-era contract: a legacy /quiz-result row (no id,
    only {timestamp, total_questions, correct}) must render in the OLD
    shape — no id/source_pdf/created_at keys — while a modern row on the
    same user renders in the NEW shape. Old app builds rely on both."""
    # Modern quiz: full payload + history entry
    user_store.save_quiz_payload(clean_user, "q_new", {
        "id": "q_new", "user_id": clean_user, "source_pdf": "notes.pdf",
        "created_at": "2026-07-10T00:00:00Z",
        "questions": [{"question": "Q", "options": ["a","b","c","d"], "correct_index": 0, "explanation": ""}] * 4,
    })
    ud = user_store.get_user_data(clean_user)
    ud["quiz_history"].append({
        "id": "q_new", "source_pdf": "notes.pdf", "created_at": "2026-07-10T00:00:00Z",
        "total_questions": 4, "correct": 3,
    })
    # Legacy /quiz-result row: shape without id/source_pdf/created_at
    ud["quiz_history"].append({
        "timestamp": "2026-07-11T00:00:00Z", "total_questions": 5, "correct": 2,
    })
    user_store.save_user_data(clean_user, ud)

    r = client.get(f"/progress/{clean_user}")
    assert r.status_code == 200, r.text
    body = r.json()

    modern = [e for e in body["quiz_history"] if e.get("id") == "q_new"]
    assert len(modern) == 1
    m = modern[0]
    assert m["source_pdf"] == "notes.pdf"
    assert m["created_at"] == "2026-07-10T00:00:00Z"
    assert m["total_questions"] == 4
    assert m["correct"] == 3

    legacy = [e for e in body["quiz_history"] if e.get("id") is None]
    assert len(legacy) == 1
    lg = legacy[0]
    # Legacy row must expose timestamp explicitly and NOT carry
    # id/source_pdf/created_at with real values — the Pydantic model
    # tolerates None for the modern-shape fields on legacy rows.
    assert lg["timestamp"] == "2026-07-11T00:00:00Z"
    assert lg["total_questions"] == 5
    assert lg["correct"] == 2

    # Derived totals must sum both rows.
    assert body["questions_answered_total"] == 9
    assert body["questions_correct_total"] == 5

    # chroma_db_path is never leaked to the client.
    assert "chroma_db_path" not in body


# ── /quizzes list + single ────────────────────────────────

def test_list_quizzes_returns_newest_first_by_created_at(clean_user):
    """The list endpoint sorts newest-first by created_at (with timestamp
    fallback for legacy rows). Preserving this ordering is a contract:
    the Android history screen renders in this order."""
    ud = user_store.get_user_data(clean_user)
    # Insert in old-then-new order; list should flip to new-then-old.
    user_store.save_quiz_payload(clean_user, "q_old", {
        "id": "q_old", "user_id": clean_user, "source_pdf": None,
        "created_at": "2026-07-01T00:00:00Z",
        "questions": [{"question": "Q", "options": ["a","b","c","d"], "correct_index": 0, "explanation": ""}],
    })
    user_store.save_quiz_payload(clean_user, "q_new", {
        "id": "q_new", "user_id": clean_user, "source_pdf": None,
        "created_at": "2026-07-15T00:00:00Z",
        "questions": [{"question": "Q", "options": ["a","b","c","d"], "correct_index": 0, "explanation": ""}],
    })
    ud["quiz_history"] += [
        {"id": "q_old", "source_pdf": None, "created_at": "2026-07-01T00:00:00Z", "total_questions": 1, "correct": None},
        {"id": "q_new", "source_pdf": None, "created_at": "2026-07-15T00:00:00Z", "total_questions": 1, "correct": None},
    ]
    user_store.save_user_data(clean_user, ud)

    r = client.get(f"/quizzes/{clean_user}")
    assert r.status_code == 200, r.text
    ids = [q["id"] for q in r.json()["quizzes"]]
    assert ids == ["q_new", "q_old"]


def test_get_quiz_payload_roundtrips_exactly(clean_user):
    """The single-quiz endpoint returns the payload dict byte-for-byte —
    same fields, same values, same nested structure."""
    payload = {
        "id": "q_rt", "user_id": clean_user, "source_pdf": "n.pdf",
        "created_at": "2026-07-06T14:30:00Z",
        "questions": [
            {"question": "Q1", "options": ["a","b","c","d"], "correct_index": 2, "explanation": "e1"},
            {"question": "Q2", "options": ["a","b","c","d"], "correct_index": 0, "explanation": "e2"},
        ],
    }
    user_store.save_quiz_payload(clean_user, "q_rt", payload)
    ud = user_store.get_user_data(clean_user)
    ud["quiz_history"].append({
        "id": "q_rt", "source_pdf": "n.pdf", "created_at": "2026-07-06T14:30:00Z",
        "total_questions": 2, "correct": None,
    })
    user_store.save_user_data(clean_user, ud)

    r = client.get(f"/quizzes/{clean_user}/q_rt")
    assert r.status_code == 200, r.text
    assert r.json() == payload


def test_get_quiz_404_for_other_users_id(clean_user):
    """User B must not be able to fetch user A's quiz by guessing the id,
    even if that id exists in the DB."""
    # Set up a second user, seed a quiz on them, then try to fetch it as clean_user.
    user_store.init_user_store()
    other = user_store.create_user("Other", f"other_{uuid.uuid4()}@t.com", "password123")
    other_uid = other["id"]
    try:
        user_store.save_quiz_payload(other_uid, "q_iso", {
            "id": "q_iso", "user_id": other_uid, "source_pdf": None,
            "created_at": "2026-07-06T14:30:00Z",
            "questions": [{"question": "Q", "options": ["a","b","c","d"], "correct_index": 0, "explanation": ""}],
        })
        ud = user_store.get_user_data(other_uid)
        ud["quiz_history"].append({
            "id": "q_iso", "source_pdf": None, "created_at": "2026-07-06T14:30:00Z",
            "total_questions": 1, "correct": None,
        })
        user_store.save_user_data(other_uid, ud)

        r = client.get(f"/quizzes/{clean_user}/q_iso")
        assert r.status_code == 404
    finally:
        with db.connect() as conn:
            conn.execute("DELETE FROM users WHERE id = ?", (other_uid,))
        folder = db._USERS_DIR / str(other_uid)
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)


# ── /flashcards list + single ─────────────────────────────

def test_list_flashcard_sets_returns_newest_first(clean_user):
    ud = user_store.get_user_data(clean_user)
    for sid, ts in [("f_old", "2026-07-01T00:00:00Z"), ("f_new", "2026-07-15T00:00:00Z")]:
        user_store.save_flashcard_payload(clean_user, sid, {
            "id": sid, "user_id": clean_user, "source_pdf": None, "created_at": ts,
            "cards": [{"question": "Q", "options": ["a","b","c","d"], "correct_index": 0, "explanation": ""}] * 3,
        })
        ud["flashcard_sets"].append({
            "id": sid, "source_pdf": None, "created_at": ts, "card_count": 3, "cards_revealed": 0,
        })
    user_store.save_user_data(clean_user, ud)

    r = client.get(f"/flashcards/{clean_user}")
    assert r.status_code == 200, r.text
    ids = [s["id"] for s in r.json()["flashcard_sets"]]
    assert ids == ["f_new", "f_old"]


def test_get_flashcard_set_payload_roundtrips_exactly(clean_user):
    payload = {
        "id": "f_rt", "user_id": clean_user, "source_pdf": "n.pdf",
        "created_at": "2026-07-06T15:01:12Z",
        "cards": [
            {"question": "C1", "options": ["a","b","c","d"], "correct_index": 1, "explanation": "e1"},
            {"question": "C2", "options": ["a","b","c","d"], "correct_index": 2, "explanation": "e2"},
            {"question": "C3", "options": ["a","b","c","d"], "correct_index": 3, "explanation": "e3"},
        ],
    }
    user_store.save_flashcard_payload(clean_user, "f_rt", payload)
    ud = user_store.get_user_data(clean_user)
    ud["flashcard_sets"].append({
        "id": "f_rt", "source_pdf": "n.pdf", "created_at": "2026-07-06T15:01:12Z",
        "card_count": 3, "cards_revealed": 0,
    })
    user_store.save_user_data(clean_user, ud)

    r = client.get(f"/flashcards/{clean_user}/f_rt")
    assert r.status_code == 200, r.text
    assert r.json() == payload
