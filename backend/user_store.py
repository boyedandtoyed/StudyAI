"""
Per-user account and storage module — SQLite-backed.

All relational data (accounts, per-user document index, quiz and flashcard
history + payloads) lives in ~/Desktop/Study_AI_users/studyai.db. ChromaDB
is unchanged and still lives at Study_AI_users/<user_id>/chroma_db/.

Public interface preserved from the JSON era so main_fastapi.py and any
existing caller keep working unmodified:
    init_user_store
    create_user / authenticate_user / get_user_by_id
    get_user_data / save_user_data
    save_quiz_payload / load_quiz_payload / delete_quiz_payload
    save_flashcard_payload / load_flashcard_payload / delete_flashcard_payload

get_user_data() returns the same big-dict shape callers already expect,
assembled from queries. save_user_data() does a full reconciliation
against the passed-in dict (upsert what's there, delete what's missing),
in a single transaction. Totals (questions_answered_total etc.) are
derived on read via SUM() so they can't drift.
"""

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from backend import db


def _user_dir(user_id: int) -> Path:
    # Read db._USERS_DIR at call time so tests can monkeypatch it.
    return db._USERS_DIR / str(user_id)


def _chroma_dir(user_id: int) -> Path:
    return _user_dir(user_id) / "chroma_db"


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _hash_password(password: str) -> str:
    return hashlib.sha256(password.encode()).hexdigest()


def _user_row_to_dict(row) -> dict:
    """Users table row -> public dict (no password_hash)."""
    return {
        "id": row["id"],
        "name": row["name"],
        "email": row["email"],
        "created_at": row["created_at"],
    }


def init_user_store() -> None:
    """Create the storage root and the SQLite schema if either is missing. Idempotent — safe to call on every FastAPI startup."""
    db._USERS_DIR.mkdir(parents=True, exist_ok=True)
    db.init_schema()


# ── USERS ────────────────────────────────────────────────

def create_user(name: str, email: str, password: str) -> dict:
    """
    Register a new account and provision its per-user storage folder.

    Raises ValueError if name is blank, email lacks '@', password is shorter
    than 8 characters, or the email is already registered (case-insensitive).
    Returns the new user record without password_hash.
    """
    name = name.strip() if isinstance(name, str) else ""
    email_norm = email.strip().lower() if isinstance(email, str) else ""

    if not name:
        raise ValueError("Name must not be blank")
    if "@" not in email_norm:
        raise ValueError("Email must contain '@'")
    if not isinstance(password, str) or len(password) < 8:
        raise ValueError("Password must be at least 8 characters long")

    password_hash = _hash_password(password)
    created_at = _now_iso()

    with db.connect() as conn:
        existing = conn.execute(
            "SELECT 1 FROM users WHERE LOWER(TRIM(email)) = ?",
            (email_norm,),
        ).fetchone()
        if existing is not None:
            raise ValueError("An account with this email already exists")

        cur = conn.execute(
            "INSERT INTO users (name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
            (name, email_norm, password_hash, created_at),
        )
        user_id = cur.lastrowid

    _chroma_dir(user_id).mkdir(parents=True, exist_ok=True)

    return {
        "id": user_id,
        "name": name,
        "email": email_norm,
        "created_at": created_at,
    }


def authenticate_user(email: str, password: str) -> Optional[dict]:
    """
    Verify credentials. Returns the user record (without password_hash) on
    success, or None if the email is unknown or the password does not match.
    """
    if not isinstance(email, str) or not isinstance(password, str):
        return None

    email_norm = email.strip().lower()
    password_hash = _hash_password(password)

    with db.connect() as conn:
        row = conn.execute(
            "SELECT id, name, email, password_hash, created_at FROM users "
            "WHERE LOWER(TRIM(email)) = ?",
            (email_norm,),
        ).fetchone()

    if row is None:
        return None
    if row["password_hash"] != password_hash:
        return None
    return _user_row_to_dict(row)


def get_user_by_id(user_id: int) -> Optional[dict]:
    """Look up a user by integer id. Returns the record without password_hash, or None if no such user exists."""
    with db.connect() as conn:
        row = conn.execute(
            "SELECT id, name, email, created_at FROM users WHERE id = ?",
            (user_id,),
        ).fetchone()
    return _user_row_to_dict(row) if row else None


# ── USER DATA (BIG DICT COMPAT SURFACE) ──────────────────

def _row_to_history_entry(row) -> dict:
    """Quizzes row -> quiz_history dict shape expected by callers.

    Legacy rows (is_legacy=1, from pre-quiz_id /quiz-result posts) emit the
    old {timestamp, total_questions, correct} shape — no id, no source_pdf,
    no created_at — because the Android app was built against exactly that
    contract for those rows.
    """
    if row["is_legacy"]:
        return {
            "timestamp": row["created_at"],
            "total_questions": row["total_questions"],
            "correct": row["correct"],
        }
    return {
        "id": row["id"],
        "source_pdf": row["source_pdf"],
        "created_at": row["created_at"],
        "total_questions": row["total_questions"],
        "correct": row["correct"],
    }


def _row_to_flashcard_entry(row) -> dict:
    return {
        "id": row["id"],
        "source_pdf": row["source_pdf"],
        "created_at": row["created_at"],
        "card_count": row["card_count"],
        "cards_revealed": row["cards_revealed"],
    }


def get_user_data(user_id: int) -> Optional[dict]:
    """Load the user's per-user data blob. Returns None if no such user exists.

    The dict shape mirrors the original user_data.json contract exactly so
    existing callers (main_fastapi.py's endpoints, tests) keep working:
    user_id, chroma_db_path, pdfs_uploaded, quiz_history, flashcard_sets,
    questions_answered_total, questions_correct_total, flashcards_revealed_total.

    Totals are derived via SUM() so they can never drift from the underlying
    history rows.
    """
    with db.connect() as conn:
        user = conn.execute(
            "SELECT id FROM users WHERE id = ?", (user_id,)
        ).fetchone()
        if user is None:
            return None

        docs = conn.execute(
            "SELECT filename, uploaded_at FROM documents "
            "WHERE user_id = ? ORDER BY id ASC",
            (user_id,),
        ).fetchall()
        quizzes = conn.execute(
            "SELECT id, source_pdf, created_at, total_questions, correct, is_legacy "
            "FROM quizzes WHERE user_id = ? ORDER BY rowid ASC",
            (user_id,),
        ).fetchall()
        sets = conn.execute(
            "SELECT id, source_pdf, created_at, card_count, cards_revealed "
            "FROM flashcards WHERE user_id = ? ORDER BY rowid ASC",
            (user_id,),
        ).fetchall()

        # Totals derived on read — no separate counter columns, no drift.
        qa_total = conn.execute(
            "SELECT COALESCE(SUM(total_questions), 0) FROM quizzes WHERE user_id = ?",
            (user_id,),
        ).fetchone()[0]
        qc_total = conn.execute(
            "SELECT COALESCE(SUM(correct), 0) FROM quizzes WHERE user_id = ?",
            (user_id,),
        ).fetchone()[0]
        fr_total = conn.execute(
            "SELECT COALESCE(SUM(cards_revealed), 0) FROM flashcards WHERE user_id = ?",
            (user_id,),
        ).fetchone()[0]

    return {
        "user_id": user_id,
        "chroma_db_path": str(_chroma_dir(user_id).resolve()),
        "pdfs_uploaded": [
            {"filename": r["filename"], "timestamp": r["uploaded_at"]} for r in docs
        ],
        "quiz_history": [_row_to_history_entry(r) for r in quizzes],
        "flashcard_sets": [_row_to_flashcard_entry(r) for r in sets],
        "questions_answered_total": int(qa_total),
        "questions_correct_total": int(qc_total),
        "flashcards_revealed_total": int(fr_total),
    }


def _legacy_id_for(entry: dict) -> str:
    """Stable synthesized id for legacy /quiz-result rows (no quiz_id).

    Old rows carry only {timestamp, total_questions, correct}. Timestamps
    are ISO-with-microseconds so collisions are extremely unlikely; we still
    fold in the numeric fields so any accidental duplicate timestamp doesn't
    collide.
    """
    ts = str(entry.get("timestamp") or "")
    key = f"{ts}|{entry.get('total_questions')}|{entry.get('correct')}"
    return "legacy_" + hashlib.sha256(key.encode()).hexdigest()[:16]


def save_user_data(user_id: int, data: dict) -> None:
    """Reconcile the SQLite state for this user against the passed-in dict.

    Callers keep handing us the whole user_data blob — same interface as
    the JSON era. Internally we diff:
      - documents: upsert new rows keyed on (user_id, filename); delete any
        row whose filename isn't in the passed list.
      - quiz_history: upsert by quiz id, preserving existing payload / is_legacy;
        legacy entries (no id) get a synthesized id and is_legacy=1;
        delete any quiz for this user not present in the passed list.
      - flashcard_sets: same shape as quizzes.

    Totals in the passed dict are IGNORED — get_user_data() derives them via
    SUM(). Callers that set totals manually (e.g. after cascading delete)
    still work: the deletes happen through this reconciliation, and the next
    read produces the same numbers.

    Creating a user row is NOT this function's job; call create_user first.
    """
    with db.connect() as conn:
        # Documents ────────────────────────────────────────
        docs = data.get("pdfs_uploaded") or []
        wanted_filenames = set()
        for entry in docs:
            fname = entry.get("filename")
            if fname is None:
                continue
            wanted_filenames.add(fname)
            uploaded_at = entry.get("timestamp") or _now_iso()
            # INSERT OR IGNORE preserves the original uploaded_at when a row
            # already exists — matches the JSON code's append-once behavior.
            conn.execute(
                "INSERT OR IGNORE INTO documents (user_id, filename, uploaded_at) "
                "VALUES (?, ?, ?)",
                (user_id, fname, uploaded_at),
            )
        existing_docs = [
            r["filename"] for r in conn.execute(
                "SELECT filename FROM documents WHERE user_id = ?", (user_id,)
            ).fetchall()
        ]
        for fname in existing_docs:
            if fname not in wanted_filenames:
                conn.execute(
                    "DELETE FROM documents WHERE user_id = ? AND filename = ?",
                    (user_id, fname),
                )

        # Quiz history ─────────────────────────────────────
        history = data.get("quiz_history") or []
        wanted_quiz_ids = set()
        for entry in history:
            qid = entry.get("id")
            if qid:
                # Modern row — payload already stored by save_quiz_payload.
                # Update the summary columns only; leave payload / is_legacy alone.
                conn.execute(
                    "UPDATE quizzes SET source_pdf = ?, created_at = ?, "
                    "total_questions = ?, correct = ? "
                    "WHERE id = ? AND user_id = ?",
                    (
                        entry.get("source_pdf"),
                        entry.get("created_at"),
                        int(entry.get("total_questions") or 0),
                        entry.get("correct"),
                        qid,
                        user_id,
                    ),
                )
                wanted_quiz_ids.add(qid)
            else:
                # Legacy /quiz-result row: {timestamp, total_questions, correct}.
                legacy_id = _legacy_id_for(entry)
                conn.execute(
                    "INSERT OR IGNORE INTO quizzes "
                    "(id, user_id, source_pdf, created_at, total_questions, correct, payload, is_legacy) "
                    "VALUES (?, ?, NULL, ?, ?, ?, '{}', 1)",
                    (
                        legacy_id,
                        user_id,
                        entry.get("timestamp") or _now_iso(),
                        int(entry.get("total_questions") or 0),
                        entry.get("correct"),
                    ),
                )
                wanted_quiz_ids.add(legacy_id)
        existing_quiz_ids = [
            r["id"] for r in conn.execute(
                "SELECT id FROM quizzes WHERE user_id = ?", (user_id,)
            ).fetchall()
        ]
        for qid in existing_quiz_ids:
            if qid not in wanted_quiz_ids:
                conn.execute(
                    "DELETE FROM quizzes WHERE id = ? AND user_id = ?",
                    (qid, user_id),
                )

        # Flashcard sets ───────────────────────────────────
        sets = data.get("flashcard_sets") or []
        wanted_set_ids = set()
        for entry in sets:
            sid = entry.get("id")
            if not sid:
                continue
            conn.execute(
                "UPDATE flashcards SET source_pdf = ?, created_at = ?, "
                "card_count = ?, cards_revealed = ? "
                "WHERE id = ? AND user_id = ?",
                (
                    entry.get("source_pdf"),
                    entry.get("created_at"),
                    int(entry.get("card_count") or 0),
                    int(entry.get("cards_revealed") or 0),
                    sid,
                    user_id,
                ),
            )
            wanted_set_ids.add(sid)
        existing_set_ids = [
            r["id"] for r in conn.execute(
                "SELECT id FROM flashcards WHERE user_id = ?", (user_id,)
            ).fetchall()
        ]
        for sid in existing_set_ids:
            if sid not in wanted_set_ids:
                conn.execute(
                    "DELETE FROM flashcards WHERE id = ? AND user_id = ?",
                    (sid, user_id),
                )


# ── QUIZ PAYLOAD I/O ─────────────────────────────────────

def save_quiz_payload(user_id: int, quiz_id: str, payload: dict) -> None:
    """Insert or update the full quiz payload plus its summary columns.

    Called by /quiz when a new quiz is generated — this creates the row
    that save_user_data will later UPDATE with correctness info. If the
    quiz already exists (rare — same id reused), we overwrite payload but
    keep the existing correct value (a /quiz-result may have already
    landed).
    """
    source_pdf = payload.get("source_pdf")
    created_at = payload.get("created_at") or _now_iso()
    questions = payload.get("questions") or []
    total_questions = len(questions)
    payload_json = json.dumps(payload)

    with db.connect() as conn:
        conn.execute(
            "INSERT INTO quizzes "
            "(id, user_id, source_pdf, created_at, total_questions, correct, payload, is_legacy) "
            "VALUES (?, ?, ?, ?, ?, NULL, ?, 0) "
            "ON CONFLICT(id) DO UPDATE SET "
            "source_pdf = excluded.source_pdf, "
            "created_at = excluded.created_at, "
            "total_questions = excluded.total_questions, "
            "payload = excluded.payload",
            (quiz_id, user_id, source_pdf, created_at, total_questions, payload_json),
        )


def load_quiz_payload(user_id: int, quiz_id: str) -> Optional[dict]:
    with db.connect() as conn:
        row = conn.execute(
            "SELECT payload FROM quizzes WHERE id = ? AND user_id = ?",
            (quiz_id, user_id),
        ).fetchone()
    if row is None or not row["payload"]:
        return None
    try:
        payload = json.loads(row["payload"])
    except (ValueError, TypeError):
        return None
    # Empty legacy payloads look like {} — treat as absent so callers 404.
    return payload if payload else None


def delete_quiz_payload(user_id: int, quiz_id: str) -> bool:
    with db.connect() as conn:
        cur = conn.execute(
            "DELETE FROM quizzes WHERE id = ? AND user_id = ?",
            (quiz_id, user_id),
        )
        return cur.rowcount > 0


# ── FLASHCARD PAYLOAD I/O ────────────────────────────────

def save_flashcard_payload(user_id: int, set_id: str, payload: dict) -> None:
    source_pdf = payload.get("source_pdf")
    created_at = payload.get("created_at") or _now_iso()
    cards = payload.get("cards") or []
    card_count = len(cards)
    payload_json = json.dumps(payload)

    with db.connect() as conn:
        conn.execute(
            "INSERT INTO flashcards "
            "(id, user_id, source_pdf, created_at, card_count, cards_revealed, payload) "
            "VALUES (?, ?, ?, ?, ?, 0, ?) "
            "ON CONFLICT(id) DO UPDATE SET "
            "source_pdf = excluded.source_pdf, "
            "created_at = excluded.created_at, "
            "card_count = excluded.card_count, "
            "payload = excluded.payload",
            (set_id, user_id, source_pdf, created_at, card_count, payload_json),
        )


def load_flashcard_payload(user_id: int, set_id: str) -> Optional[dict]:
    with db.connect() as conn:
        row = conn.execute(
            "SELECT payload FROM flashcards WHERE id = ? AND user_id = ?",
            (set_id, user_id),
        ).fetchone()
    if row is None or not row["payload"]:
        return None
    try:
        return json.loads(row["payload"])
    except (ValueError, TypeError):
        return None


def delete_flashcard_payload(user_id: int, set_id: str) -> bool:
    with db.connect() as conn:
        cur = conn.execute(
            "DELETE FROM flashcards WHERE id = ? AND user_id = ?",
            (set_id, user_id),
        )
        return cur.rowcount > 0


# ── PER-OPERATION REPOSITORY API ─────────────────────────
# Cleaner surface for future callers: one function per operation, no big
# dict passed around. main_fastapi.py isn't switched over yet — the compat
# functions above still cover its needs — but new call sites should prefer
# these.

def create_quiz(
    user_id: int,
    quiz_id: str,
    source_pdf: Optional[str],
    created_at: str,
    questions: list,
) -> None:
    """Insert a new quiz. Payload column stores the full JSON exactly as returned to the client."""
    payload = {
        "id": quiz_id,
        "user_id": user_id,
        "source_pdf": source_pdf,
        "created_at": created_at,
        "questions": questions,
    }
    save_quiz_payload(user_id, quiz_id, payload)


def get_quiz(user_id: int, quiz_id: str) -> Optional[dict]:
    return load_quiz_payload(user_id, quiz_id)


def list_quizzes(user_id: int) -> list:
    """Return the quiz history index for a user, insertion order (oldest first).

    Modern rows and legacy rows are rendered in their respective shapes —
    same dispatch as get_user_data() uses.
    """
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT id, source_pdf, created_at, total_questions, correct, is_legacy "
            "FROM quizzes WHERE user_id = ? ORDER BY rowid ASC",
            (user_id,),
        ).fetchall()
    return [_row_to_history_entry(r) for r in rows]


def update_quiz_result(
    user_id: int, quiz_id: str, total_questions: int, correct: int
) -> bool:
    """Set total_questions + correct on an existing quiz. Returns False if the row doesn't exist."""
    with db.connect() as conn:
        cur = conn.execute(
            "UPDATE quizzes SET total_questions = ?, correct = ? "
            "WHERE id = ? AND user_id = ?",
            (int(total_questions), int(correct), quiz_id, user_id),
        )
        return cur.rowcount > 0


def delete_quiz(user_id: int, quiz_id: str) -> bool:
    return delete_quiz_payload(user_id, quiz_id)


def create_flashcard_set(
    user_id: int,
    set_id: str,
    source_pdf: Optional[str],
    created_at: str,
    cards: list,
) -> None:
    payload = {
        "id": set_id,
        "user_id": user_id,
        "source_pdf": source_pdf,
        "created_at": created_at,
        "cards": cards,
    }
    save_flashcard_payload(user_id, set_id, payload)


def get_flashcard_set(user_id: int, set_id: str) -> Optional[dict]:
    return load_flashcard_payload(user_id, set_id)


def list_flashcard_sets(user_id: int) -> list:
    with db.connect() as conn:
        rows = conn.execute(
            "SELECT id, source_pdf, created_at, card_count, cards_revealed "
            "FROM flashcards WHERE user_id = ? ORDER BY rowid ASC",
            (user_id,),
        ).fetchall()
    return [_row_to_flashcard_entry(r) for r in rows]


def update_flashcards_revealed(
    user_id: int, set_id: str, revealed_count: int
) -> Optional[int]:
    """Bump cards_revealed to max(current, revealed_count). Idempotent under retries — same behavior as the /flashcard-reveal endpoint. Returns the new value, or None if the set doesn't exist."""
    with db.connect() as conn:
        row = conn.execute(
            "SELECT cards_revealed FROM flashcards WHERE id = ? AND user_id = ?",
            (set_id, user_id),
        ).fetchone()
        if row is None:
            return None
        new_val = max(int(row["cards_revealed"]), int(revealed_count))
        conn.execute(
            "UPDATE flashcards SET cards_revealed = ? WHERE id = ? AND user_id = ?",
            (new_val, set_id, user_id),
        )
        return new_val


def delete_flashcard_set(user_id: int, set_id: str) -> bool:
    return delete_flashcard_payload(user_id, set_id)
