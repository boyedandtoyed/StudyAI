"""WAL smoke test: concurrent reads and writes against user_store.

FastAPI serves multiple requests in flight, so the storage layer must
tolerate one request updating a user's data while another reads it. WAL
journal mode lets readers proceed during a write; without WAL, or with a
bad connection pattern, this test would surface `database is locked`.

Uses a tmp-dir database so it doesn't collide with a running server.
"""
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend import db, user_store  # noqa: E402


@pytest.fixture
def tmp_store(tmp_path, monkeypatch):
    root = tmp_path / "Study_AI_users"
    root.mkdir()
    monkeypatch.setattr(db, "_USERS_DIR", root)
    monkeypatch.setattr(db, "DB_PATH", root / "studyai.db")
    user_store.init_user_store()
    yield root


def test_concurrent_reads_and_writes_do_not_lock(tmp_store):
    """Fire N reader + M writer threads at the same user. Every operation
    must complete without `database is locked`. This is the WAL check —
    with the old journal mode, readers block on any live writer."""
    user = user_store.create_user("Conc", "conc@x.com", "password123")
    uid = user["id"]

    NUM_READERS = 16
    NUM_WRITERS = 8
    OPS_PER_THREAD = 20

    errors = []
    lock = threading.Lock()

    def reader(idx):
        try:
            for _ in range(OPS_PER_THREAD):
                ud = user_store.get_user_data(uid)
                assert ud is not None
        except Exception as e:  # noqa: BLE001
            with lock:
                errors.append(("reader", idx, repr(e)))

    def writer(idx):
        try:
            for i in range(OPS_PER_THREAD):
                sid = f"f_{idx}_{i}"
                user_store.save_flashcard_payload(uid, sid, {
                    "id": sid, "user_id": uid, "source_pdf": None,
                    "created_at": "2026-07-06T00:00:00Z",
                    "cards": [{"question": "Q", "options": ["a", "b", "c", "d"],
                               "correct_index": 0, "explanation": ""}] * 3,
                })
        except Exception as e:  # noqa: BLE001
            with lock:
                errors.append(("writer", idx, repr(e)))

    with ThreadPoolExecutor(max_workers=NUM_READERS + NUM_WRITERS) as ex:
        futures = []
        futures += [ex.submit(reader, i) for i in range(NUM_READERS)]
        futures += [ex.submit(writer, i) for i in range(NUM_WRITERS)]
        for f in as_completed(futures):
            f.result()

    assert errors == [], f"Concurrency errors: {errors}"

    # All writes landed (and none clobbered each other).
    with db.connect() as conn:
        n = conn.execute(
            "SELECT COUNT(*) FROM flashcards WHERE user_id = ?", (uid,)
        ).fetchone()[0]
    assert n == NUM_WRITERS * OPS_PER_THREAD
