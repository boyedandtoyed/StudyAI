"""
Per-user JSON-backed account and storage module.

Layout on disk:
    ~/Desktop/Study_AI_users/
        users_db.json                          -- global account index
        <user_id>/
            chroma_db/                         -- this user's vector store
            user_data.json                     -- per-user metadata

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

_db_lock = threading.Lock()


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

    user_dir = _USERS_DIR / str(user_id)
    chroma_dir = user_dir / "chroma_db"
    chroma_dir.mkdir(parents=True, exist_ok=True)

    user_data = {
        "user_id": user_id,
        "chroma_db_path": str(chroma_dir.resolve()),
        "pdfs_uploaded": [],
        "quiz_history": [],
        "questions_answered_total": 0,
        "questions_correct_total": 0,
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


def get_user_data(user_id: int) -> Optional[dict]:
    """Load the user's per-user data blob. Returns None if the user folder or user_data.json does not exist."""
    path = _USERS_DIR / str(user_id) / "user_data.json"
    if not path.exists():
        return None
    with open(path, "r") as f:
        return json.load(f)


def save_user_data(user_id: int, data: dict) -> None:
    """Overwrite the user's user_data.json with the given dict, pretty-printed (indent=2). Creates the user folder if missing."""
    user_dir = _USERS_DIR / str(user_id)
    user_dir.mkdir(parents=True, exist_ok=True)
    _save_json_atomic(user_dir / "user_data.json", data)
