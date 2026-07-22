"""
SQLite storage backend for StudyAI.

One database file (studyai.db) lives next to the per-user folders under
~/Desktop/Study_AI_users/. ChromaDB is unchanged and still lives at
Study_AI_users/<user_id>/chroma_db/ — this module only owns the relational
data that used to sit in users_db.json and per-user user_data.json /
quizzes/*.json / flashcards/*.json files.

Connection pattern: one sqlite3.Connection per call, opened through the
connect() context manager below. sqlite3 connections aren't safe to share
across threads by default; a fresh connection per call sidesteps that
entirely and stays simple. WAL journal mode lets readers proceed while a
writer holds the DB, so FastAPI's concurrent requests don't queue behind
each other.
"""

import sqlite3
from contextlib import contextmanager
from pathlib import Path

_USERS_DIR = Path.home() / "Desktop" / "Study_AI_users"
DB_PATH = _USERS_DIR / "studyai.db"


@contextmanager
def connect():
    """Yield a sqlite3 Connection with WAL + foreign keys enabled.

    Commits on clean exit, rolls back on exception, always closes.
    Row factory is sqlite3.Row so callers can index columns by name.
    """
    _USERS_DIR.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # journal_mode is persistent per-database but cheap to set; foreign_keys
    # is per-connection and MUST be re-enabled every time.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT NOT NULL,
    email         TEXT NOT NULL UNIQUE,
    password_hash TEXT NOT NULL,
    created_at    TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS documents (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id     INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    filename    TEXT NOT NULL,
    uploaded_at TEXT NOT NULL,
    UNIQUE(user_id, filename)
);

CREATE TABLE IF NOT EXISTS quizzes (
    id              TEXT PRIMARY KEY,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    source_pdf      TEXT,
    created_at      TEXT NOT NULL,
    total_questions INTEGER NOT NULL,
    correct         INTEGER,
    payload         TEXT NOT NULL,
    -- is_legacy=1 marks pre-quiz_id /quiz-result rows that only carry
    -- {timestamp, total_questions, correct}. When rendering history the
    -- caller must emit legacy rows in the old shape (timestamp, no id).
    is_legacy       INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS flashcards (
    id              TEXT PRIMARY KEY,
    user_id         INTEGER NOT NULL REFERENCES users(id) ON DELETE CASCADE,
    source_pdf      TEXT,
    created_at      TEXT NOT NULL,
    card_count      INTEGER NOT NULL,
    cards_revealed  INTEGER NOT NULL DEFAULT 0,
    payload         TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_documents_user  ON documents(user_id);
CREATE INDEX IF NOT EXISTS idx_quizzes_user    ON quizzes(user_id);
CREATE INDEX IF NOT EXISTS idx_flashcards_user ON flashcards(user_id);
"""


def init_schema() -> None:
    """Create all tables if they don't exist. Idempotent — safe on every startup."""
    with connect() as conn:
        conn.executescript(_SCHEMA)


if __name__ == "__main__":
    init_schema()
    print(f"Initialized schema at {DB_PATH}")
