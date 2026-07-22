"""
One-time cutover from the JSON-file storage layout to SQLite.

Layout being read:
    ~/Desktop/Study_AI_users/
        users_db.json
        <user_id>/
            user_data.json
            quizzes/<quiz_id>.json
            flashcards/<set_id>.json
            chroma_db/               (left untouched)

Behavior:
  1. Copy the entire Study_AI_users/ directory to a timestamped sibling
     backup before touching anything on disk.
  2. Insert users into `users` preserving their existing integer id.
     Bump sqlite_sequence so new registrations continue past MAX(id).
  3. Walk each user's user_data.json, quizzes/*.json, flashcards/*.json;
     insert rows into `documents`, `quizzes`, `flashcards`. Legacy
     /quiz-result rows (no quiz_id, {timestamp, total_questions, correct})
     become is_legacy=1 rows.
  4. Count JSON entities independently and compare against SELECT COUNT(*).
     Any mismatch rolls the whole transaction back and exits non-zero.
  5. On success, move each user's user_data.json + quizzes/ + flashcards/
     into <user_id>/_migrated_backup/. Original JSON is preserved until
     after the demo — nothing is deleted.

Refuses to run if the target DB already has any users (prevents a double
run from silently making a mess). Run manually:

    ./venv/bin/python -m backend.migrate_to_sqlite
"""

import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

from backend import db
from backend.user_store import _legacy_id_for


_USERS_DIR = db._USERS_DIR
_USERS_DB_JSON = _USERS_DIR / "users_db.json"


def _load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


def _summary_from_quiz_payload(payload: dict) -> tuple:
    """Fallback summary derivation when a payload exists but no history row references it."""
    source_pdf = payload.get("source_pdf")
    created_at = payload.get("created_at") or datetime.utcnow().isoformat() + "Z"
    total_questions = len(payload.get("questions") or [])
    return source_pdf, created_at, total_questions


def _summary_from_flashcard_payload(payload: dict) -> tuple:
    source_pdf = payload.get("source_pdf")
    created_at = payload.get("created_at") or datetime.utcnow().isoformat() + "Z"
    card_count = len(payload.get("cards") or [])
    return source_pdf, created_at, card_count


def _preflight(conn) -> None:
    n = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
    if n > 0:
        raise SystemExit(
            f"Refusing to migrate: users table already has {n} row(s). "
            "The target studyai.db is not empty. If you meant to re-migrate, "
            "move studyai.db aside first."
        )


def _backup_tree() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dest = _USERS_DIR.parent / f"Study_AI_users_backup_{stamp}"
    if dest.exists():
        raise SystemExit(f"Backup target already exists: {dest}")
    shutil.copytree(_USERS_DIR, dest, symlinks=False)
    return dest


def _migrate_users(conn, users_json: dict) -> int:
    """Insert users preserving their explicit ids. Returns count inserted."""
    users = users_json.get("users", [])
    for u in users:
        conn.execute(
            "INSERT INTO users (id, name, email, password_hash, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                int(u["id"]),
                u["name"],
                u["email"],
                u["password_hash"],
                u["created_at"],
            ),
        )
    # Bump sqlite_sequence so new user ids continue past the highest existing.
    if users:
        max_id = max(int(u["id"]) for u in users)
        existing = conn.execute(
            "SELECT 1 FROM sqlite_sequence WHERE name = 'users'"
        ).fetchone()
        if existing:
            conn.execute(
                "UPDATE sqlite_sequence SET seq = ? WHERE name = 'users'",
                (max_id,),
            )
        else:
            conn.execute(
                "INSERT INTO sqlite_sequence(name, seq) VALUES ('users', ?)",
                (max_id,),
            )
    return len(users)


def _migrate_user_content(conn, user_id: int) -> tuple:
    """Migrate one user's docs, quizzes, flashcards. Returns (docs, quizzes, flashcards) counts inserted."""
    user_dir = _USERS_DIR / str(user_id)
    user_data_path = user_dir / "user_data.json"
    if not user_data_path.exists():
        return (0, 0, 0)

    user_data = _load_json(user_data_path)

    # Documents ────────────────────────────────────────
    docs_inserted = 0
    for entry in user_data.get("pdfs_uploaded", []):
        fname = entry.get("filename")
        if not fname:
            continue
        uploaded_at = entry.get("timestamp") or ""
        conn.execute(
            "INSERT OR IGNORE INTO documents (user_id, filename, uploaded_at) "
            "VALUES (?, ?, ?)",
            (user_id, fname, uploaded_at),
        )
        docs_inserted += 1

    # Build index of quiz history rows by id (modern) and a list of legacy rows.
    history = user_data.get("quiz_history", []) or []
    modern_history = {h["id"]: h for h in history if h.get("id")}
    legacy_history = [h for h in history if not h.get("id")]

    quiz_dir = user_dir / "quizzes"
    quiz_files = sorted(quiz_dir.glob("*.json")) if quiz_dir.exists() else []

    quizzes_inserted = 0
    seen_quiz_ids = set()

    # 1. Every quiz payload file → modern row. Prefer the history entry's
    #    summary fields when present (they carry the /quiz-result-updated
    #    `correct` value); fall back to the payload's own header.
    for qfile in quiz_files:
        try:
            payload = _load_json(qfile)
        except (OSError, ValueError) as e:
            raise SystemExit(f"Corrupt quiz payload {qfile}: {e}")
        qid = payload.get("id") or qfile.stem
        if qid in seen_quiz_ids:
            continue
        h = modern_history.get(qid)
        if h is not None:
            source_pdf = h.get("source_pdf")
            created_at = h.get("created_at") or ""
            total_questions = int(h.get("total_questions") or 0)
            correct = h.get("correct")
        else:
            source_pdf, created_at, total_questions = _summary_from_quiz_payload(payload)
            correct = None
        conn.execute(
            "INSERT INTO quizzes "
            "(id, user_id, source_pdf, created_at, total_questions, correct, payload, is_legacy) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, 0)",
            (
                qid, user_id, source_pdf, created_at,
                total_questions, correct, json.dumps(payload),
            ),
        )
        seen_quiz_ids.add(qid)
        quizzes_inserted += 1

    # 2. History rows with an id but no matching payload file — rare, but
    #    keep them so the /quizzes list stays the same length. Empty payload.
    for qid, h in modern_history.items():
        if qid in seen_quiz_ids:
            continue
        conn.execute(
            "INSERT INTO quizzes "
            "(id, user_id, source_pdf, created_at, total_questions, correct, payload, is_legacy) "
            "VALUES (?, ?, ?, ?, ?, ?, '{}', 0)",
            (
                qid, user_id, h.get("source_pdf"),
                h.get("created_at") or "",
                int(h.get("total_questions") or 0),
                h.get("correct"),
            ),
        )
        seen_quiz_ids.add(qid)
        quizzes_inserted += 1

    # 3. Legacy /quiz-result rows — no id, no payload file, only
    #    {timestamp, total_questions, correct}.
    for h in legacy_history:
        legacy_id = _legacy_id_for(h)
        conn.execute(
            "INSERT OR IGNORE INTO quizzes "
            "(id, user_id, source_pdf, created_at, total_questions, correct, payload, is_legacy) "
            "VALUES (?, ?, NULL, ?, ?, ?, '{}', 1)",
            (
                legacy_id, user_id,
                h.get("timestamp") or "",
                int(h.get("total_questions") or 0),
                h.get("correct"),
            ),
        )
        quizzes_inserted += 1

    # Flashcards ──────────────────────────────────────
    fc_sets = user_data.get("flashcard_sets", []) or []
    modern_sets = {s["id"]: s for s in fc_sets if s.get("id")}

    fc_dir = user_dir / "flashcards"
    fc_files = sorted(fc_dir.glob("*.json")) if fc_dir.exists() else []

    flashcards_inserted = 0
    seen_set_ids = set()

    for ffile in fc_files:
        try:
            payload = _load_json(ffile)
        except (OSError, ValueError) as e:
            raise SystemExit(f"Corrupt flashcard payload {ffile}: {e}")
        sid = payload.get("id") or ffile.stem
        if sid in seen_set_ids:
            continue
        s = modern_sets.get(sid)
        if s is not None:
            source_pdf = s.get("source_pdf")
            created_at = s.get("created_at") or ""
            card_count = int(s.get("card_count") or 0)
            cards_revealed = int(s.get("cards_revealed") or 0)
        else:
            source_pdf, created_at, card_count = _summary_from_flashcard_payload(payload)
            cards_revealed = 0
        conn.execute(
            "INSERT INTO flashcards "
            "(id, user_id, source_pdf, created_at, card_count, cards_revealed, payload) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                sid, user_id, source_pdf, created_at,
                card_count, cards_revealed, json.dumps(payload),
            ),
        )
        seen_set_ids.add(sid)
        flashcards_inserted += 1

    for sid, s in modern_sets.items():
        if sid in seen_set_ids:
            continue
        conn.execute(
            "INSERT INTO flashcards "
            "(id, user_id, source_pdf, created_at, card_count, cards_revealed, payload) "
            "VALUES (?, ?, ?, ?, ?, ?, '{}')",
            (
                sid, user_id, s.get("source_pdf"),
                s.get("created_at") or "",
                int(s.get("card_count") or 0),
                int(s.get("cards_revealed") or 0),
            ),
        )
        seen_set_ids.add(sid)
        flashcards_inserted += 1

    return (docs_inserted, quizzes_inserted, flashcards_inserted)


def _expected_counts_from_json(user_ids: list) -> tuple:
    """Independently count what should end up in each table, straight from disk."""
    docs = 0
    quizzes = 0
    flashcards = 0
    for uid in user_ids:
        ud_path = _USERS_DIR / str(uid) / "user_data.json"
        if not ud_path.exists():
            continue
        ud = _load_json(ud_path)
        docs += sum(1 for e in ud.get("pdfs_uploaded", []) if e.get("filename"))

        history = ud.get("quiz_history", []) or []
        modern_ids_in_history = {h["id"] for h in history if h.get("id")}
        legacy_count = sum(1 for h in history if not h.get("id"))
        quiz_dir = _USERS_DIR / str(uid) / "quizzes"
        file_ids = (
            {p.stem for p in quiz_dir.glob("*.json")} if quiz_dir.exists() else set()
        )
        quizzes += len(file_ids | modern_ids_in_history) + legacy_count

        set_ids_in_history = {s["id"] for s in ud.get("flashcard_sets", []) if s.get("id")}
        fc_dir = _USERS_DIR / str(uid) / "flashcards"
        set_file_ids = (
            {p.stem for p in fc_dir.glob("*.json")} if fc_dir.exists() else set()
        )
        flashcards += len(set_file_ids | set_ids_in_history)

    return (docs, quizzes, flashcards)


def _archive_json(user_ids: list) -> None:
    """Move the migrated JSON files into <user_id>/_migrated_backup/. Nothing is deleted — kept until after the demo."""
    for uid in user_ids:
        udir = _USERS_DIR / str(uid)
        if not udir.exists():
            continue
        stash = udir / "_migrated_backup"
        stash.mkdir(exist_ok=True)
        user_data_path = udir / "user_data.json"
        if user_data_path.exists():
            shutil.move(str(user_data_path), stash / "user_data.json")
        quiz_dir = udir / "quizzes"
        if quiz_dir.exists():
            shutil.move(str(quiz_dir), stash / "quizzes")
        fc_dir = udir / "flashcards"
        if fc_dir.exists():
            shutil.move(str(fc_dir), stash / "flashcards")
    # Also stash the top-level users_db.json.
    if _USERS_DB_JSON.exists():
        top_stash = _USERS_DIR / "_migrated_backup"
        top_stash.mkdir(exist_ok=True)
        shutil.move(str(_USERS_DB_JSON), top_stash / "users_db.json")


def main() -> None:
    if not _USERS_DB_JSON.exists():
        raise SystemExit(f"No users_db.json at {_USERS_DB_JSON} — nothing to migrate.")

    print(f"[migrate] Source: {_USERS_DIR}")
    backup = _backup_tree()
    print(f"[migrate] Backed up whole tree to: {backup}")

    db.init_schema()

    users_json = _load_json(_USERS_DB_JSON)
    user_ids = [int(u["id"]) for u in users_json.get("users", [])]

    expected_docs, expected_quizzes, expected_flashcards = _expected_counts_from_json(user_ids)
    expected_users = len(user_ids)

    totals = {"docs": 0, "quizzes": 0, "flashcards": 0}

    with db.connect() as conn:
        _preflight(conn)
        users_inserted = _migrate_users(conn, users_json)
        for uid in user_ids:
            d, q, f = _migrate_user_content(conn, uid)
            totals["docs"] += d
            totals["quizzes"] += q
            totals["flashcards"] += f

        # In-transaction verification: mismatch raises → context manager rolls back.
        db_users = conn.execute("SELECT COUNT(*) FROM users").fetchone()[0]
        db_docs = conn.execute("SELECT COUNT(*) FROM documents").fetchone()[0]
        db_quizzes = conn.execute("SELECT COUNT(*) FROM quizzes").fetchone()[0]
        db_fc = conn.execute("SELECT COUNT(*) FROM flashcards").fetchone()[0]

        mismatches = []
        if db_users != expected_users:
            mismatches.append(f"users: expected {expected_users}, got {db_users}")
        if db_docs != expected_docs:
            mismatches.append(f"documents: expected {expected_docs}, got {db_docs}")
        if db_quizzes != expected_quizzes:
            mismatches.append(f"quizzes: expected {expected_quizzes}, got {db_quizzes}")
        if db_fc != expected_flashcards:
            mismatches.append(f"flashcards: expected {expected_flashcards}, got {db_fc}")
        if mismatches:
            raise SystemExit(
                "[migrate] COUNT MISMATCH — rolling back:\n  " + "\n  ".join(mismatches)
            )

    print(
        f"[migrate] Inserted: {users_inserted} users, {totals['docs']} documents, "
        f"{totals['quizzes']} quizzes, {totals['flashcards']} flashcard sets"
    )

    _archive_json(user_ids)
    print("[migrate] JSON originals moved to per-user _migrated_backup/ folders.")
    print("[migrate] Done.")


if __name__ == "__main__":
    main()
