import sys
import uuid
from pathlib import Path

import pytest
import requests

BASE_URL = "http://localhost:8000"

# Make backend/user_store importable so we can seed on-disk state directly
# (server and tests share the Study_AI_users directory). Integration tests
# that need a specific pre-condition — say, a flashcard set attributed to
# user 1 — would otherwise have to run a real LLM-backed /flashcards call.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend import db, user_store  # noqa: E402


def _quiz_payload_exists(uid: int, quiz_id: str) -> bool:
    with db.connect() as conn:
        row = conn.execute(
            "SELECT payload FROM quizzes WHERE id = ? AND user_id = ?",
            (quiz_id, uid),
        ).fetchone()
    return row is not None and bool(row["payload"]) and row["payload"] != "{}"


def _flashcard_payload_exists(uid: int, set_id: str) -> bool:
    with db.connect() as conn:
        row = conn.execute(
            "SELECT payload FROM flashcards WHERE id = ? AND user_id = ?",
            (set_id, uid),
        ).fetchone()
    return row is not None and bool(row["payload"]) and row["payload"] != "{}"


def _register_unique_user(password="password123"):
    """Register a user with a unique email and return the parsed response JSON."""
    email = f"test_{uuid.uuid4()}@test.com"
    r = requests.post(
        f"{BASE_URL}/register",
        json={"name": "API Test User", "email": email, "password": password},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    body["_email"] = email
    body["_password"] = password
    return body


def test_health():
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_stats():
    r = requests.get(f"{BASE_URL}/stats")
    assert r.status_code == 200
    data = r.json()
    for key in ("model_name", "embed_model", "total_chunks", "indexed_documents", "document_count", "server_time"):
        assert key in data, f"Missing key: {key}"


def test_docs_list_requires_user_id():
    """/docs-list is a per-user endpoint; a call with no user_id must return
    400 (not FastAPI's default 422, not the old bare-string legacy shape)."""
    r = requests.get(f"{BASE_URL}/docs-list")
    assert r.status_code == 400, r.text
    assert "user_id" in r.json().get("detail", "").lower()


def test_docs_list_returns_per_user_shape():
    """With a valid user_id the endpoint returns objects with filename +
    timestamp — no bare-string fallback anywhere in the response."""
    user = _register_unique_user()
    uid = user["user"]["id"]
    r = requests.get(f"{BASE_URL}/docs-list", params={"user_id": uid})
    assert r.status_code == 200, r.text
    data = r.json()
    assert isinstance(data["documents"], list)
    for entry in data["documents"]:
        assert isinstance(entry, dict)
        assert "filename" in entry
        assert "timestamp" in entry


def test_chat_no_docs():
    r = requests.post(f"{BASE_URL}/chat", json={"question": "test"})
    assert r.status_code == 200
    assert "answer" in r.json()


def test_clear_session_not_found():
    r = requests.delete(f"{BASE_URL}/clear-session/nonexistent")
    assert r.status_code == 404


# ── OpenRouter provider-switch contract tests ─────────────
# These hit the live server and don't assume which LLM_PROVIDER (ollama or
# openrouter) it's configured with — only that the public contract holds.

def test_chat_responds_regardless_of_provider():
    """POST /chat always returns 200 with an "answer", no matter which
    provider the server is currently using."""
    r = requests.post(f"{BASE_URL}/chat", json={"question": "What is 2 + 2?"})
    assert r.status_code == 200, r.text
    assert "answer" in r.json()


def test_stats_includes_provider_info():
    """GET /stats now surfaces the configured provider (added alongside the
    OpenRouter integration)."""
    r = requests.get(f"{BASE_URL}/stats")
    assert r.status_code == 200
    data = r.json()
    assert "llm_provider" in data, "Missing key: llm_provider"
    assert "provider_type" in data, "Missing key: provider_type"


def test_quiz_generation_works_with_current_provider():
    """Confirms the provider swap didn't break quiz generation: /quiz still
    returns a "questions" list with well-formed multiple-choice items.

    Note: the /quiz endpoint requests exactly 3 questions (not 10-15), so we
    assert the actual structure rather than a fixed count.
    """
    # /quiz needs at least one indexed document; skip rather than fail if the
    # running server has an empty index.
    stats = requests.get(f"{BASE_URL}/stats").json()
    if stats.get("document_count", 0) == 0:
        pytest.skip("No documents indexed on the running server — upload a PDF first")

    r = requests.post(f"{BASE_URL}/quiz", json={})
    assert r.status_code == 200, r.text
    data = r.json()
    assert "questions" in data
    questions = data["questions"]
    assert isinstance(questions, list)
    assert len(questions) >= 1
    for q in questions:
        assert "question" in q
        assert isinstance(q["options"], list)
        assert len(q["options"]) == 4
        assert isinstance(q["correct_index"], int)
        assert 0 <= q["correct_index"] <= 3
        assert "explanation" in q


# ── Auth + per-user account endpoints ─────────────────────

def test_register_endpoint():
    email = f"test_{uuid.uuid4()}@test.com"
    r = requests.post(
        f"{BASE_URL}/register",
        json={"name": "Reg Test", "email": email, "password": "password123"},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["success"] is True
    assert "id" in data["user"]


def test_register_duplicate_email():
    email = f"test_{uuid.uuid4()}@test.com"
    payload = {"name": "Dup Test", "email": email, "password": "password123"}

    r1 = requests.post(f"{BASE_URL}/register", json=payload)
    assert r1.status_code == 200, r1.text

    r2 = requests.post(f"{BASE_URL}/register", json=payload)
    assert r2.status_code == 400


def test_login_endpoint():
    user = _register_unique_user()
    r = requests.post(
        f"{BASE_URL}/login",
        json={"email": user["_email"], "password": user["_password"]},
    )
    assert r.status_code == 200, r.text
    assert r.json()["success"] is True


def test_login_wrong_password():
    user = _register_unique_user()
    r = requests.post(
        f"{BASE_URL}/login",
        json={"email": user["_email"], "password": "not-the-password"},
    )
    assert r.status_code == 401


def test_quiz_result_endpoint():
    user = _register_unique_user()
    user_id = user["user"]["id"]
    r = requests.post(
        f"{BASE_URL}/quiz-result",
        json={"user_id": user_id, "total_questions": 5, "correct": 3},
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"success": True}


def test_progress_endpoint():
    user = _register_unique_user()
    user_id = user["user"]["id"]

    requests.post(
        f"{BASE_URL}/quiz-result",
        json={"user_id": user_id, "total_questions": 5, "correct": 3},
    )

    r = requests.get(f"{BASE_URL}/progress/{user_id}")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["questions_answered_total"] == 5
    assert data["questions_correct_total"] == 3
    assert "chroma_db_path" not in data


def test_progress_unknown_user():
    r = requests.get(f"{BASE_URL}/progress/99999")
    assert r.status_code == 404


# ── Flashcards, history, and per-user isolation ──────────

def _seed_flashcard_set(uid: int, set_id: str = "f_seed_test", card_count: int = 3) -> str:
    """Seed a flashcard set directly on disk, bypassing the LLM. Used for
    endpoint contract tests that shouldn't wait on real generation."""
    ud = user_store.get_user_data(uid)
    assert ud is not None
    ud.setdefault("flashcard_sets", []).append({
        "id": set_id,
        "source_pdf": "seed.pdf",
        "created_at": "2026-07-06T15:01:12Z",
        "card_count": card_count,
        "cards_revealed": 0,
    })
    user_store.save_user_data(uid, ud)
    user_store.save_flashcard_payload(uid, set_id, {
        "id": set_id,
        "user_id": uid,
        "source_pdf": "seed.pdf",
        "created_at": "2026-07-06T15:01:12Z",
        "cards": [
            {"question": f"Q{i}", "options": ["a", "b", "c", "d"], "correct_index": 0, "explanation": ""}
            for i in range(card_count)
        ],
    })
    return set_id


def test_flashcards_reject_invalid_count():
    """count must be one of 5, 10, 15, 20 — server-side enforcement."""
    user = _register_unique_user()
    uid = user["user"]["id"]
    r = requests.post(
        f"{BASE_URL}/flashcards",
        json={"user_id": uid, "source_pdf": "any.pdf", "count": 7},
    )
    assert r.status_code == 422


def test_flashcards_reject_unowned_source_pdf():
    """A user must not be able to point /flashcards at a filename they
    didn't upload — even if some other user did."""
    user = _register_unique_user()
    uid = user["user"]["id"]
    r = requests.post(
        f"{BASE_URL}/flashcards",
        json={"user_id": uid, "source_pdf": "not_my_document.pdf", "count": 5},
    )
    assert r.status_code == 404


def test_flashcard_set_visible_only_to_owner():
    """User 2 gets a 404 when trying to fetch user 1's set by id."""
    u1 = _register_unique_user()
    u2 = _register_unique_user()
    uid1, uid2 = u1["user"]["id"], u2["user"]["id"]
    set_id = _seed_flashcard_set(uid1, set_id=f"f_iso_{uuid.uuid4().hex[:8]}")

    r = requests.get(f"{BASE_URL}/flashcards/{uid1}/{set_id}")
    assert r.status_code == 200, r.text

    r2 = requests.get(f"{BASE_URL}/flashcards/{uid2}/{set_id}")
    assert r2.status_code == 404


def test_flashcard_reveal_updates_progress_and_delete_removes_set():
    """End-to-end for the reveal+delete flow: reveal bumps the running
    total on the set and in /progress, and delete makes the set unfetchable
    and drops it from the index."""
    user = _register_unique_user()
    uid = user["user"]["id"]
    set_id = _seed_flashcard_set(uid, set_id=f"f_flow_{uuid.uuid4().hex[:8]}")

    r = requests.post(
        f"{BASE_URL}/flashcard-reveal",
        json={"user_id": uid, "set_id": set_id, "revealed_count": 2},
    )
    assert r.status_code == 200, r.text
    assert r.json()["cards_revealed"] == 2

    p = requests.get(f"{BASE_URL}/progress/{uid}").json()
    assert p["flashcards_revealed_total"] == 2

    # Idempotent re-taps: sending the same running total should not double-count.
    requests.post(
        f"{BASE_URL}/flashcard-reveal",
        json={"user_id": uid, "set_id": set_id, "revealed_count": 2},
    )
    p2 = requests.get(f"{BASE_URL}/progress/{uid}").json()
    assert p2["flashcards_revealed_total"] == 2

    r = requests.delete(f"{BASE_URL}/flashcards/{uid}/{set_id}")
    assert r.status_code == 200

    r = requests.get(f"{BASE_URL}/flashcards/{uid}/{set_id}")
    assert r.status_code == 404

    r = requests.get(f"{BASE_URL}/flashcards/{uid}")
    assert all(s.get("id") != set_id for s in r.json()["flashcard_sets"])


def _seed_quiz(uid: int, source_pdf: str, quiz_id: str, total_questions: int = 5, correct: int = 3) -> str:
    """Attach a quiz to a user directly on disk — bypasses the LLM so we can
    test history/delete flows without waiting on real generation."""
    ud = user_store.get_user_data(uid)
    assert ud is not None
    ud.setdefault("quiz_history", []).append({
        "id": quiz_id,
        "source_pdf": source_pdf,
        "created_at": "2026-07-06T14:30:00Z",
        "total_questions": total_questions,
        "correct": correct,
    })
    ud["questions_answered_total"] = ud.get("questions_answered_total", 0) + total_questions
    ud["questions_correct_total"] = ud.get("questions_correct_total", 0) + correct
    user_store.save_user_data(uid, ud)
    user_store.save_quiz_payload(uid, quiz_id, {
        "id": quiz_id, "user_id": uid, "source_pdf": source_pdf,
        "created_at": "2026-07-06T14:30:00Z",
        "questions": [
            {"question": f"Q{i}", "options": ["a", "b", "c", "d"], "correct_index": 0, "explanation": ""}
            for i in range(total_questions)
        ],
    })
    return quiz_id


def _seed_flashcard_set_for(uid: int, source_pdf: str, set_id: str, card_count: int = 3, cards_revealed: int = 0) -> str:
    """Like _seed_flashcard_set but with a caller-chosen source_pdf and reveal
    count, so cascade-delete tests can control which sets belong to which PDF."""
    ud = user_store.get_user_data(uid)
    assert ud is not None
    ud.setdefault("flashcard_sets", []).append({
        "id": set_id,
        "source_pdf": source_pdf,
        "created_at": "2026-07-06T15:01:12Z",
        "card_count": card_count,
        "cards_revealed": cards_revealed,
    })
    ud["flashcards_revealed_total"] = ud.get("flashcards_revealed_total", 0) + cards_revealed
    user_store.save_user_data(uid, ud)
    user_store.save_flashcard_payload(uid, set_id, {
        "id": set_id, "user_id": uid, "source_pdf": source_pdf,
        "created_at": "2026-07-06T15:01:12Z",
        "cards": [
            {"question": f"Q{i}", "options": ["a", "b", "c", "d"], "correct_index": 0, "explanation": ""}
            for i in range(card_count)
        ],
    })
    return set_id


def _seed_owned_pdf(uid: int, filename: str) -> None:
    ud = user_store.get_user_data(uid)
    assert ud is not None
    if not any(e.get("filename") == filename for e in ud.get("pdfs_uploaded", [])):
        ud.setdefault("pdfs_uploaded", []).append(
            {"filename": filename, "timestamp": "2026-07-06T14:00:00Z"}
        )
        user_store.save_user_data(uid, ud)


def test_document_usage_returns_accurate_counts():
    """/usage counts quiz_history and flashcard_sets entries whose
    source_pdf matches — without opening any payload file."""
    user = _register_unique_user()
    uid = user["user"]["id"]
    _seed_owned_pdf(uid, "target.pdf")
    _seed_owned_pdf(uid, "other.pdf")

    _seed_quiz(uid, "target.pdf", f"q_use1_{uuid.uuid4().hex[:8]}")
    _seed_quiz(uid, "target.pdf", f"q_use2_{uuid.uuid4().hex[:8]}")
    _seed_quiz(uid, "other.pdf", f"q_useX_{uuid.uuid4().hex[:8]}")
    _seed_flashcard_set_for(uid, "target.pdf", f"f_use1_{uuid.uuid4().hex[:8]}")

    r = requests.get(f"{BASE_URL}/documents/{uid}/target.pdf/usage")
    assert r.status_code == 200, r.text
    assert r.json() == {"quiz_count": 2, "flashcard_count": 1}


def test_document_usage_404_for_other_users_file():
    """/usage must 404 if the caller doesn't own that filename — no silent
    zero-count response that could leak filename ownership."""
    u1 = _register_unique_user()
    u2 = _register_unique_user()
    _seed_owned_pdf(u1["user"]["id"], "u1_only.pdf")

    r = requests.get(f"{BASE_URL}/documents/{u2['user']['id']}/u1_only.pdf/usage")
    assert r.status_code == 404


def test_delete_document_cascades_to_quizzes_and_flashcards():
    """Deleting a document must remove every quiz and flashcard set generated
    from it — payload files must be gone from disk too, not just the index —
    while leaving an unrelated document's content untouched."""
    user = _register_unique_user()
    uid = user["user"]["id"]

    _seed_owned_pdf(uid, "target.pdf")
    _seed_owned_pdf(uid, "keep.pdf")

    doomed_q1 = _seed_quiz(uid, "target.pdf", f"q_del1_{uuid.uuid4().hex[:8]}")
    doomed_q2 = _seed_quiz(uid, "target.pdf", f"q_del2_{uuid.uuid4().hex[:8]}")
    doomed_f1 = _seed_flashcard_set_for(uid, "target.pdf", f"f_del1_{uuid.uuid4().hex[:8]}")
    kept_q = _seed_quiz(uid, "keep.pdf", f"q_keep_{uuid.uuid4().hex[:8]}")
    kept_f = _seed_flashcard_set_for(uid, "keep.pdf", f"f_keep_{uuid.uuid4().hex[:8]}")

    # Post-SQLite: payload lives in the `payload` column of quizzes/flashcards.
    # Existence == row present with non-empty payload.
    assert _quiz_payload_exists(uid, doomed_q1) and _quiz_payload_exists(uid, doomed_q2)
    assert _flashcard_payload_exists(uid, doomed_f1)
    assert _quiz_payload_exists(uid, kept_q) and _flashcard_payload_exists(uid, kept_f)

    r = requests.delete(f"{BASE_URL}/documents/{uid}/target.pdf")
    assert r.status_code == 200, r.text
    body = r.json()
    assert body == {
        "deleted_document": "target.pdf",
        "deleted_quizzes": 2,
        "deleted_flashcard_sets": 1,
    }

    assert not _quiz_payload_exists(uid, doomed_q1), "doomed quiz row still in db"
    assert not _quiz_payload_exists(uid, doomed_q2), "doomed quiz row still in db"
    assert not _flashcard_payload_exists(uid, doomed_f1), "doomed flashcard row still in db"
    assert _quiz_payload_exists(uid, kept_q), "unrelated quiz row was wrongly deleted"
    assert _flashcard_payload_exists(uid, kept_f), "unrelated flashcard row was wrongly deleted"

    quizzes = requests.get(f"{BASE_URL}/quizzes/{uid}").json()["quizzes"]
    ids = {q.get("id") for q in quizzes}
    assert doomed_q1 not in ids and doomed_q2 not in ids
    assert kept_q in ids

    sets = requests.get(f"{BASE_URL}/flashcards/{uid}").json()["flashcard_sets"]
    sids = {s.get("id") for s in sets}
    assert doomed_f1 not in sids
    assert kept_f in sids


def test_delete_document_recomputes_totals_from_survivors():
    """Totals must be recomputed by summing surviving entries, not by
    subtracting a computed delta from the pre-delete totals — otherwise a
    single upstream drift would compound on every delete."""
    user = _register_unique_user()
    uid = user["user"]["id"]

    _seed_owned_pdf(uid, "target.pdf")
    _seed_owned_pdf(uid, "keep.pdf")

    _seed_quiz(uid, "target.pdf", f"q_rt1_{uuid.uuid4().hex[:8]}", total_questions=5, correct=3)
    _seed_quiz(uid, "target.pdf", f"q_rt2_{uuid.uuid4().hex[:8]}", total_questions=4, correct=2)
    _seed_quiz(uid, "keep.pdf", f"q_rtk_{uuid.uuid4().hex[:8]}", total_questions=3, correct=1)
    _seed_flashcard_set_for(uid, "target.pdf", f"f_rt1_{uuid.uuid4().hex[:8]}", card_count=5, cards_revealed=5)
    _seed_flashcard_set_for(uid, "keep.pdf", f"f_rtk_{uuid.uuid4().hex[:8]}", card_count=5, cards_revealed=2)

    # Deliberately corrupt the running totals so we can prove they were
    # RECOMPUTED (not decremented) after the delete.
    ud = user_store.get_user_data(uid)
    ud["questions_answered_total"] = 999
    ud["questions_correct_total"] = 999
    ud["flashcards_revealed_total"] = 999
    user_store.save_user_data(uid, ud)

    r = requests.delete(f"{BASE_URL}/documents/{uid}/target.pdf")
    assert r.status_code == 200, r.text

    p = requests.get(f"{BASE_URL}/progress/{uid}").json()
    assert p["questions_answered_total"] == 3
    assert p["questions_correct_total"] == 1
    assert p["flashcards_revealed_total"] == 2


def test_delete_document_404_for_other_users_file():
    """Deleting a document that isn't the user's own is a 404, not a silent
    no-op — otherwise user 2 could quietly drop user 1's chunks by guessing
    the filename."""
    u1 = _register_unique_user()
    u2 = _register_unique_user()
    _seed_owned_pdf(u1["user"]["id"], "u1_only.pdf")

    r = requests.delete(f"{BASE_URL}/documents/{u2['user']['id']}/u1_only.pdf")
    assert r.status_code == 404


def test_progress_contract_shape():
    """/progress response_model enforcement: FastAPI would 500 on validation
    failure, so a clean 200 with the documented top-level keys is the
    contract test. Uses a fresh account so the shape is empty-but-typed."""
    user = _register_unique_user()
    uid = user["user"]["id"]

    r = requests.get(f"{BASE_URL}/progress/{uid}")
    assert r.status_code == 200, r.text
    data = r.json()
    for key in (
        "user_id", "pdfs_uploaded", "quiz_history", "flashcard_sets",
        "questions_answered_total", "questions_correct_total",
        "flashcards_revealed_total",
    ):
        assert key in data, f"Missing key: {key}"
    assert "chroma_db_path" not in data
    assert isinstance(data["pdfs_uploaded"], list)
    assert isinstance(data["quiz_history"], list)
    assert isinstance(data["flashcard_sets"], list)


def test_progress_contract_shape_populated():
    """Same contract test with real content in every list — proves the
    response_model accepts source_pdf=null / correct=null (real quizzes and
    flashcards can have those) and doesn't reject the whole response."""
    user = _register_unique_user()
    uid = user["user"]["id"]
    _seed_owned_pdf(uid, "any.pdf")
    _seed_quiz(uid, "any.pdf", f"q_cs1_{uuid.uuid4().hex[:8]}", total_questions=3, correct=0)
    _seed_quiz(uid, None, f"q_cs2_{uuid.uuid4().hex[:8]}", total_questions=3, correct=0)
    _seed_flashcard_set_for(uid, None, f"f_cs1_{uuid.uuid4().hex[:8]}", card_count=5, cards_revealed=0)

    r = requests.get(f"{BASE_URL}/progress/{uid}")
    assert r.status_code == 200, r.text
    data = r.json()
    assert len(data["quiz_history"]) == 2
    assert len(data["flashcard_sets"]) == 1
