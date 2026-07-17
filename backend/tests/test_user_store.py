"""Unit tests for backend/user_store.py.

These exercise the real on-disk store under ~/Desktop/Study_AI_users. Each test
gets a freshly-created temporary user via the `test_user` fixture, which cleans
up that user's folder (shutil.rmtree) and its users_db.json record afterward so
the real store isn't polluted. Unique per-run emails avoid collisions.

This file lives outside tests/ on purpose: it's a pure unit test that needs no
running server, so it must not inherit tests/conftest.py's server-skip fixture.
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

from backend import user_store


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

    # ── teardown: remove each created user's folder and db record ──
    for uid in ctx["ids"]:
        folder = user_store._USERS_DIR / str(uid)
        if folder.exists():
            shutil.rmtree(folder, ignore_errors=True)

    with user_store._db_lock:
        db = user_store._load_db()
        db["users"] = [u for u in db["users"] if u["id"] not in ctx["ids"]]
        user_store._save_json_atomic(user_store._USERS_DB, db)


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


def test_user_folder_created(test_user):
    uid = test_user["user"]["id"]
    folder = user_store._USERS_DIR / str(uid)
    assert folder.exists()
    assert (folder / "user_data.json").exists()


def test_get_and_save_user_data(test_user):
    uid = test_user["user"]["id"]

    data = user_store.get_user_data(uid)
    assert data is not None

    data["questions_answered_total"] = 42
    user_store.save_user_data(uid, data)

    reloaded = user_store.get_user_data(uid)
    assert reloaded["questions_answered_total"] == 42


def test_get_user_data_backfills_old_shape(test_user):
    """An old-shape user_data.json (missing flashcard_sets and
    flashcards_revealed_total) heals itself on the next read, without
    losing the fields it already has."""
    uid = test_user["user"]["id"]

    # Simulate an on-disk file written by an older backend version.
    old_shape = {
        "user_id": uid,
        "chroma_db_path": "/tmp/does-not-matter",
        "pdfs_uploaded": [{"filename": "lecture1.pdf", "timestamp": "2026-01-01T00:00:00Z"}],
        "quiz_history": [{"timestamp": "2026-01-02T00:00:00Z", "total_questions": 10, "correct": 8}],
        "questions_answered_total": 10,
        "questions_correct_total": 8,
    }
    user_store.save_user_data(uid, old_shape)

    healed = user_store.get_user_data(uid)
    assert healed is not None
    assert healed["flashcard_sets"] == []
    assert healed["flashcards_revealed_total"] == 0
    # Existing fields are preserved.
    assert healed["questions_answered_total"] == 10
    assert healed["questions_correct_total"] == 8
    assert healed["quiz_history"][0]["correct"] == 8
    assert healed["pdfs_uploaded"][0]["filename"] == "lecture1.pdf"

    # A second read is a no-op — the file already has the new keys.
    healed_again = user_store.get_user_data(uid)
    assert healed_again == healed


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
