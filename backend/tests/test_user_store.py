"""Unit tests for backend/user_store.py (SQLite-backed).

Each test gets a freshly-created temporary user via the `test_user` fixture,
which registers it under the real ~/Desktop/Study_AI_users database and
cleans it up afterward via ON DELETE CASCADE. Unique per-run emails avoid
collisions with parallel runs.

This file lives outside tests/ on purpose: it's a pure unit test that needs
no running server, so it must not inherit tests/conftest.py's server-skip
fixture.
"""
import shutil
import sys
import uuid
from pathlib import Path

import pytest

# Repo root is two levels up from backend/tests/; put it on sys.path so
# `from backend import user_store` resolves regardless of pytest's rootdir.
REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend import db, user_store


@pytest.fixture
def test_user():
    """Create a temporary user before the test and tear it down afterward."""
    user_store.init_user_store()

    email = f"test_{uuid.uuid4()}@test.com"
    password = "password123"
    name = "Test User"
    user = user_store.create_user(name, email, password)

    ctx = {"user": user, "email": email, "password": password, "name": name,
           "ids": [user["id"]]}
    yield ctx

    # ── teardown: rm the user row (ON DELETE CASCADE removes all derived
    # rows) and the chroma_db folder that create_user provisioned. ──
    with db.connect() as conn:
        for uid in ctx["ids"]:
            conn.execute("DELETE FROM users WHERE id = ?", (uid,))
    for uid in ctx["ids"]:
        folder = db._USERS_DIR / str(uid)
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)


def test_create_user_success(test_user):
    user = test_user["user"]
    assert isinstance(user["id"], int)
    assert user["name"] == test_user["name"]
    assert user["email"] == test_user["email"]
    assert "password_hash" not in user


def test_create_user_duplicate_email(test_user):
    with pytest.raises(ValueError):
        user_store.create_user("Another Name", test_user["email"], "different123")


def test_authenticate_user_correct_password(test_user):
    result = user_store.authenticate_user(test_user["email"], test_user["password"])
    assert result is not None
    assert result["email"] == test_user["email"]
    assert "password_hash" not in result


def test_authenticate_user_wrong_password(test_user):
    assert user_store.authenticate_user(test_user["email"], "wrong-password") is None


def test_user_row_and_chroma_dir_created(test_user):
    """create_user must both insert the users row AND provision the per-user chroma_db folder."""
    uid = test_user["user"]["id"]
    with db.connect() as conn:
        row = conn.execute("SELECT id FROM users WHERE id = ?", (uid,)).fetchone()
    assert row is not None
    assert (db._USERS_DIR / str(uid) / "chroma_db").exists()


def test_get_and_save_user_data(test_user):
    """save_user_data round-trips through the reconciliation path. Totals
    are derived from history, so we exercise them by adding a history row."""
    uid = test_user["user"]["id"]

    data = user_store.get_user_data(uid)
    assert data is not None
    assert data["questions_answered_total"] == 0

    # Simulate the /quiz path: save payload first, then history entry.
    user_store.save_quiz_payload(uid, "q_ut", {
        "id": "q_ut", "user_id": uid, "source_pdf": None,
        "created_at": "2026-07-06T14:30:00Z",
        "questions": [{"question": "Q", "options": ["a","b","c","d"], "correct_index": 0, "explanation": ""}] * 5,
    })
    data["quiz_history"].append({
        "id": "q_ut", "source_pdf": None, "created_at": "2026-07-06T14:30:00Z",
        "total_questions": 5, "correct": 3,
    })
    user_store.save_user_data(uid, data)

    reloaded = user_store.get_user_data(uid)
    assert reloaded["questions_answered_total"] == 5
    assert reloaded["questions_correct_total"] == 3


def test_save_user_data_preserves_legacy_history_row_shape(test_user):
    """A legacy /quiz-result row (no quiz_id, only {timestamp, total, correct})
    must round-trip through save_user_data and come back out in the same
    old shape — the Android app was built against exactly this shape."""
    uid = test_user["user"]["id"]

    data = user_store.get_user_data(uid)
    data["quiz_history"].append({
        "timestamp": "2026-06-03T00:00:00Z",
        "total_questions": 5,
        "correct": 3,
    })
    user_store.save_user_data(uid, data)

    reloaded = user_store.get_user_data(uid)
    legacy = [e for e in reloaded["quiz_history"] if "timestamp" in e and "id" not in e]
    assert legacy == [{"timestamp": "2026-06-03T00:00:00Z", "total_questions": 5, "correct": 3}]
    # Legacy row still counts toward totals derived via SUM.
    assert reloaded["questions_answered_total"] == 5
    assert reloaded["questions_correct_total"] == 3


def test_flashcard_payload_roundtrip(test_user):
    uid = test_user["user"]["id"]
    payload = {"id": "f_test", "cards": [{"question": "Q", "options": ["a", "b", "c", "d"], "correct_index": 0, "explanation": ""}]}
    user_store.save_flashcard_payload(uid, "f_test", payload)

    loaded = user_store.load_flashcard_payload(uid, "f_test")
    assert loaded == payload

    assert user_store.delete_flashcard_payload(uid, "f_test") is True
    assert user_store.load_flashcard_payload(uid, "f_test") is None
    assert user_store.delete_flashcard_payload(uid, "f_test") is False


def test_quiz_payload_roundtrip(test_user):
    uid = test_user["user"]["id"]
    payload = {"id": "q_test", "questions": [{"question": "Q", "options": ["a", "b", "c", "d"], "correct_index": 1, "explanation": ""}]}
    user_store.save_quiz_payload(uid, "q_test", payload)

    loaded = user_store.load_quiz_payload(uid, "q_test")
    assert loaded == payload

    assert user_store.delete_quiz_payload(uid, "q_test") is True
    assert user_store.load_quiz_payload(uid, "q_test") is None
