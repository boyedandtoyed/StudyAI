"""Migration correctness — JSON fixture tree → SQLite.

Builds a minimal but representative Study_AI_users/ tree in a tmp dir,
points backend.db at it, runs backend.migrate_to_sqlite.main(), then
asserts everything ended up in the DB in the same shape callers already
saw from get_user_data() / load_quiz_payload() / load_flashcard_payload().

The tree covers: multiple users, a document, a modern quiz (with matching
payload file), a legacy /quiz-result row (no id, only timestamp), a
flashcard set, and an entirely empty user. That's the full matrix the
migration script has to handle.
"""
import json
import shutil
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend import db, user_store  # noqa: E402
import backend.migrate_to_sqlite as mig  # noqa: E402


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    """Point db._USERS_DIR + db.DB_PATH at a scratch directory."""
    root = tmp_path / "Study_AI_users"
    root.mkdir()
    monkeypatch.setattr(db, "_USERS_DIR", root)
    monkeypatch.setattr(db, "DB_PATH", root / "studyai.db")
    yield root


def _write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2))


def test_migration_end_to_end(tmp_store):
    root = tmp_store

    _write_json(root / "users_db.json", {
        "next_id": 3,
        "users": [
            {"id": 1, "name": "Alice", "email": "a@x.com", "password_hash": "h1",
             "created_at": "2026-01-01T00:00:00+00:00"},
            {"id": 2, "name": "Bob", "email": "b@x.com", "password_hash": "h2",
             "created_at": "2026-01-02T00:00:00+00:00"},
        ],
    })

    # User 1: one doc, one modern quiz w/ payload, one legacy history row, one flashcard set
    modern_quiz_payload = {
        "id": "q_A", "user_id": 1, "source_pdf": "foo.pdf",
        "created_at": "2026-06-02T00:00:00+00:00",
        "questions": [
            {"question": "Q1", "options": ["a", "b", "c", "d"], "correct_index": 0, "explanation": "e1"},
            {"question": "Q2", "options": ["a", "b", "c", "d"], "correct_index": 1, "explanation": "e2"},
        ],
    }
    modern_fc_payload = {
        "id": "f_A", "user_id": 1, "source_pdf": "foo.pdf",
        "created_at": "2026-06-04T00:00:00+00:00",
        "cards": [
            {"question": f"C{i}", "options": ["a", "b", "c", "d"], "correct_index": i % 4, "explanation": f"e{i}"}
            for i in range(3)
        ],
    }
    _write_json(root / "1" / "user_data.json", {
        "user_id": 1,
        "pdfs_uploaded": [{"filename": "foo.pdf", "timestamp": "2026-06-01T00:00:00+00:00"}],
        "quiz_history": [
            {"id": "q_A", "source_pdf": "foo.pdf", "created_at": "2026-06-02T00:00:00+00:00",
             "total_questions": 2, "correct": 1},
            {"timestamp": "2026-06-03T00:00:00+00:00", "total_questions": 5, "correct": 3},
        ],
        "flashcard_sets": [{
            "id": "f_A", "source_pdf": "foo.pdf",
            "created_at": "2026-06-04T00:00:00+00:00",
            "card_count": 3, "cards_revealed": 2,
        }],
        "questions_answered_total": 7,
        "questions_correct_total": 4,
        "flashcards_revealed_total": 2,
    })
    _write_json(root / "1" / "quizzes" / "q_A.json", modern_quiz_payload)
    _write_json(root / "1" / "flashcards" / "f_A.json", modern_fc_payload)
    (root / "1" / "chroma_db").mkdir()

    # User 2: entirely empty history
    _write_json(root / "2" / "user_data.json", {
        "user_id": 2, "pdfs_uploaded": [], "quiz_history": [], "flashcard_sets": [],
        "questions_answered_total": 0, "questions_correct_total": 0,
        "flashcards_revealed_total": 0,
    })
    (root / "2" / "chroma_db").mkdir()

    mig.main()

    # Users preserved with their explicit ids
    assert user_store.get_user_by_id(1)["email"] == "a@x.com"
    assert user_store.get_user_by_id(2)["email"] == "b@x.com"

    # User 1 progress dict — same shape callers already read
    ud1 = user_store.get_user_data(1)
    assert ud1["pdfs_uploaded"] == [{"filename": "foo.pdf", "timestamp": "2026-06-01T00:00:00+00:00"}]

    # Modern history row round-trips exactly
    modern = [e for e in ud1["quiz_history"] if e.get("id") == "q_A"]
    assert modern == [{
        "id": "q_A", "source_pdf": "foo.pdf", "created_at": "2026-06-02T00:00:00+00:00",
        "total_questions": 2, "correct": 1,
    }]
    # Legacy row round-trips exactly in the OLD shape
    legacy = [e for e in ud1["quiz_history"] if "timestamp" in e and "id" not in e]
    assert legacy == [{
        "timestamp": "2026-06-03T00:00:00+00:00", "total_questions": 5, "correct": 3,
    }]

    # Flashcard set round-trips
    assert ud1["flashcard_sets"] == [{
        "id": "f_A", "source_pdf": "foo.pdf", "created_at": "2026-06-04T00:00:00+00:00",
        "card_count": 3, "cards_revealed": 2,
    }]

    # Totals derived: modern quiz (2/1) + legacy (5/3) = 7/4; reveal = 2
    assert ud1["questions_answered_total"] == 7
    assert ud1["questions_correct_total"] == 4
    assert ud1["flashcards_revealed_total"] == 2

    # Full quiz + flashcard payloads round-trip byte-for-byte
    assert user_store.load_quiz_payload(1, "q_A") == modern_quiz_payload
    assert user_store.load_flashcard_payload(1, "f_A") == modern_fc_payload

    # Empty user round-trips empty
    ud2 = user_store.get_user_data(2)
    assert ud2["quiz_history"] == [] and ud2["flashcard_sets"] == [] and ud2["pdfs_uploaded"] == []

    # New user auto-id continues past max(existing id)=2
    new_user = user_store.create_user("Carla", "c@x.com", "password123")
    assert new_user["id"] == 3

    # Originals stashed, not deleted; ChromaDB folders untouched
    assert (root / "1" / "_migrated_backup" / "user_data.json").exists()
    assert (root / "1" / "_migrated_backup" / "quizzes" / "q_A.json").exists()
    assert (root / "1" / "_migrated_backup" / "flashcards" / "f_A.json").exists()
    assert not (root / "1" / "user_data.json").exists()
    assert not (root / "1" / "quizzes").exists()
    assert (root / "1" / "chroma_db").exists()
    assert (root / "_migrated_backup" / "users_db.json").exists()

    # A whole-tree backup sibling was created
    backups = list(root.parent.glob("Study_AI_users_backup_*"))
    assert len(backups) == 1
    assert (backups[0] / "1" / "user_data.json").exists()


def test_migration_refuses_when_users_db_json_missing(tmp_store):
    """The migration must fail cleanly if there's nothing to migrate — not silently make an empty database."""
    with pytest.raises(SystemExit):
        mig.main()


def test_migration_refuses_second_run_against_populated_db(tmp_store):
    """After a successful run, a second invocation must abort. The archive
    step moves users_db.json into _migrated_backup/, so the second run's
    'no users_db.json' guard fires first — but that same guard prevents
    a partial re-migration that would try to insert duplicate rows."""
    root = tmp_store
    _write_json(root / "users_db.json", {
        "next_id": 2,
        "users": [{"id": 1, "name": "A", "email": "a@x.com",
                   "password_hash": "h", "created_at": "2026-01-01T00:00:00+00:00"}],
    })
    _write_json(root / "1" / "user_data.json", {
        "user_id": 1, "pdfs_uploaded": [], "quiz_history": [], "flashcard_sets": [],
        "questions_answered_total": 0, "questions_correct_total": 0,
        "flashcards_revealed_total": 0,
    })
    mig.main()
    with pytest.raises(SystemExit):
        mig.main()
