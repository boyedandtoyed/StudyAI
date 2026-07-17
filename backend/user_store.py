"""
Per-user JSON-backed account and storage module.

Layout on disk:
    ~/Desktop/Study_AI_users/
        users_db.json                              -- global account index
        <user_id>/
            chroma_db/                             -- this user's vector store
            user_data.json                         -- per-user summary + indexes
            quizzes/<quiz_id>.json                 -- full quiz payloads
            flashcards/<set_id>.json               -- full flashcard sets

Full quiz and flashcard payloads live in their own files. user_data.json only
carries a lightweight summary/index so it stays cheap to read and rewrite on
every progress update.

Standalone: no imports from gg.py or main_fastapi.py.
"""

import hashlib
import json
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

_USERS_DIR = Path.home() / "Desktop" / "Study_AI_users"
_USERS_DB = _USERS_DIR / "users_db.json"

_QUIZ_SUBDIR = "quizzes"
_FLASHCARD_SUBDIR = "flashcards"

_db_lock = threading.Lock()

# Default shape for user_data.json. The loader fills in any missing keys with
# these defaults so old-shape accounts on disk heal themselves on first read.
_USER_DATA_DEFAULTS: dict = {
    "pdfs_uploaded": [],
    "quiz_history": [],
    "flashcard_sets": [],
    "questions_answered_total": 0,
    "questions_correct_total": 0,
    "flashcards_revealed_total": 0,
}


def _load_db() -> dict:
    """Read users_db.json into a dict. Caller is responsible for locking."""
    with open(_USERS_DB, "r") as f:
        return json.load(f)


def _save_json_atomic(path: Path, data: dict) -> None:
    """Write JSON to a sibling temp file, then rename — avoids torn writes."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w") as f:
        json.dump(data, f, indent=2)
    tmp.replace(path)


def _strip_password(user: dict) -> dict:
    """Return a copy of a user record with the password_hash field removed."""
    return {k: v for k, v in user.items() if k != "password_hash"}


def _user_dir(user_id: int) -> Path:
    return _USERS_DIR / str(user_id)


def _user_data_path(user_id: int) -> Path:
    return _user_dir(user_id) / "user_data.json"


def _quiz_dir(user_id: int) -> Path:
    return _user_dir(user_id) / _QUIZ_SUBDIR


def _flashcard_dir(user_id: int) -> Path:
    return _user_dir(user_id) / _FLASHCARD_SUBDIR


def _payload_path(user_id: int, subdir: str, item_id: str) -> Path:
    # item_id comes from server-generated ids (q_YYYYMMDD_HHMMSS /
    # f_YYYYMMDD_HHMMSS). Strip any path separators as a belt-and-braces
    # guard against a caller ever feeding us an untrusted value.
    safe_id = item_id.replace("/", "_").replace("\\", "_")
    return _user_dir(user_id) / subdir / f"{safe_id}.json"


def init_user_store() -> None:
    """Create the storage root and users_db.json if either is missing. Idempotent — safe to call on every FastAPI startup."""
    with _db_lock:
        _USERS_DIR.mkdir(parents=True, exist_ok=True)
        if not _USERS_DB.exists():
            _save_json_atomic(_USERS_DB, {"users": [], "next_id": 1})


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

    password_hash = hashlib.sha256(password.encode()).hexdigest()
    created_at = datetime.now(timezone.utc).isoformat()

    with _db_lock:
        db = _load_db()

        for existing in db["users"]:
            if existing["email"].strip().lower() == email_norm:
                raise ValueError("An account with this email already exists")

        user_id = db["next_id"]
        db["next_id"] = user_id + 1

        user_record = {
            "id": user_id,
            "name": name,
            "email": email_norm,
            "password_hash": password_hash,
            "created_at": created_at,
        }
        db["users"].append(user_record)
        _save_json_atomic(_USERS_DB, db)

    user_dir = _user_dir(user_id)
    (user_dir / "chroma_db").mkdir(parents=True, exist_ok=True)
    _quiz_dir(user_id).mkdir(parents=True, exist_ok=True)
    _flashcard_dir(user_id).mkdir(parents=True, exist_ok=True)

    user_data = {
        "user_id": user_id,
        "chroma_db_path": str((user_dir / "chroma_db").resolve()),
        **{k: (list(v) if isinstance(v, list) else v) for k, v in _USER_DATA_DEFAULTS.items()},
    }
    save_user_data(user_id, user_data)

    return _strip_password(user_record)


def authenticate_user(email: str, password: str) -> Optional[dict]:
    """
    Verify credentials. Returns the user record (without password_hash) on
    success, or None if the email is unknown or the password does not match.
    """
    if not isinstance(email, str) or not isinstance(password, str):
        return None

    email_norm = email.strip().lower()
    password_hash = hashlib.sha256(password.encode()).hexdigest()

    with _db_lock:
        db = _load_db()

    for user in db["users"]:
        if user["email"].strip().lower() == email_norm:
            if user["password_hash"] == password_hash:
                return _strip_password(user)
            return None
    return None


def get_user_by_id(user_id: int) -> Optional[dict]:
    """Look up a user by integer id. Returns the record without password_hash, or None if no such user exists."""
    with _db_lock:
        db = _load_db()

    for user in db["users"]:
        if user["id"] == user_id:
            return _strip_password(user)
    return None


def _apply_defaults(data: dict) -> bool:
    """Fill in any missing top-level keys in-place using _USER_DATA_DEFAULTS.

    Returns True iff at least one key was added, so the caller knows to
    persist the healed dict.
    """
    changed = False
    for key, default in _USER_DATA_DEFAULTS.items():
        if key not in data:
            data[key] = list(default) if isinstance(default, list) else default
            changed = True
    return changed


def get_user_data(user_id: int) -> Optional[dict]:
    """Load the user's per-user data blob. Returns None if the user folder or user_data.json does not exist.

    Any missing keys (new keys added by later versions of the schema) are
    backfilled with safe defaults from _USER_DATA_DEFAULTS and the healed
    file is written back to disk. This is idempotent — old accounts heal
    themselves on the first read without needing a one-off migration.
    """
    path = _user_data_path(user_id)
    if not path.exists():
        return None
    with open(path, "r") as f:
        data = json.load(f)

    if _apply_defaults(data):
        _save_json_atomic(path, data)

    return data


def save_user_data(user_id: int, data: dict) -> None:
    """Overwrite the user's user_data.json with the given dict, pretty-printed (indent=2). Creates the user folder if missing."""
    user_dir = _user_dir(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    _save_json_atomic(user_dir / "user_data.json", data)


# ── PAYLOAD I/O ──────────────────────────────────────────
# Full quiz and flashcard payloads live in their own files so the summary
# user_data.json stays small — every /flashcard-reveal or /quiz-result would
# otherwise rewrite the entire payload just to bump a counter.

def save_quiz_payload(user_id: int, quiz_id: str, payload: dict) -> None:
    _quiz_dir(user_id).mkdir(parents=True, exist_ok=True)
    _save_json_atomic(_payload_path(user_id, _QUIZ_SUBDIR, quiz_id), payload)


def load_quiz_payload(user_id: int, quiz_id: str) -> Optional[dict]:
    path = _payload_path(user_id, _QUIZ_SUBDIR, quiz_id)
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def delete_quiz_payload(user_id: int, quiz_id: str) -> bool:
    path = _payload_path(user_id, _QUIZ_SUBDIR, quiz_id)
    if not path.exists():
        return False
    path.unlink()
    return True


def save_flashcard_payload(user_id: int, set_id: str, payload: dict) -> None:
    _flashcard_dir(user_id).mkdir(parents=True, exist_ok=True)
    _save_json_atomic(_payload_path(user_id, _FLASHCARD_SUBDIR, set_id), payload)


def load_flashcard_payload(user_id: int, set_id: str) -> Optional[dict]:
    path = _payload_path(user_id, _FLASHCARD_SUBDIR, set_id)
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def delete_flashcard_payload(user_id: int, set_id: str) -> bool:
    path = _payload_path(user_id, _FLASHCARD_SUBDIR, set_id)
    if not path.exists():
        return False
    path.unlink()
    return True
