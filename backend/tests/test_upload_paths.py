"""Upload-path helpers: per-user namespacing and filename sanitization.

The Android app sends uploaded PDFs with their original filename. We
preserve that name in the DB and in responses (so the user recognizes
the file in their list), but we place the bytes under a per-user
subfolder so two different users uploading files with the same name
don't clobber each other on disk. And we sanitize the raw filename to
block a client from writing outside the upload directory.
"""
import os
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import main_fastapi  # noqa: E402
from fastapi import HTTPException  # noqa: E402


def test_safe_upload_name_preserves_readable_names():
    """Names with spaces, mixed case, dots, non-ASCII must round-trip
    unchanged — they're what the user sees in the list."""
    assert main_fastapi._safe_upload_name("Lecture 1.pdf") == "Lecture 1.pdf"
    assert main_fastapi._safe_upload_name("Our-Solar-System-Book.pdf") == "Our-Solar-System-Book.pdf"
    assert main_fastapi._safe_upload_name("notes_v2.pdf") == "notes_v2.pdf"


def test_safe_upload_name_strips_directory_portion():
    """A client-supplied path component is stripped down to the basename —
    the primary defense against '../../etc/passwd' style traversal."""
    assert main_fastapi._safe_upload_name("../../etc/passwd") == "passwd"
    assert main_fastapi._safe_upload_name("/absolute/path/to/notes.pdf") == "notes.pdf"
    assert main_fastapi._safe_upload_name("sub/dir/file.pdf") == "file.pdf"


@pytest.mark.parametrize("bad", ["", ".", "..", "\x00.pdf", "notes\x00.pdf"])
def test_safe_upload_name_rejects_empty_dot_and_nul(bad):
    with pytest.raises(HTTPException) as exc:
        main_fastapi._safe_upload_name(bad)
    assert exc.value.status_code == 400


def test_safe_upload_name_rejects_non_string():
    with pytest.raises(HTTPException):
        main_fastapi._safe_upload_name(None)


def test_upload_path_namespaces_authenticated_uploads(tmp_path, monkeypatch):
    """user_id set → demo_pdfs/<user_id>/<filename>. Two users, same name → different paths."""
    monkeypatch.setattr(main_fastapi, "UPLOAD_DIR", str(tmp_path))
    p_a = main_fastapi._upload_path("notes.pdf", 1)
    p_b = main_fastapi._upload_path("notes.pdf", 2)
    assert p_a == str(tmp_path / "1" / "notes.pdf")
    assert p_b == str(tmp_path / "2" / "notes.pdf")
    assert p_a != p_b
    # Parent dirs are created on demand — a fresh install just works.
    assert (tmp_path / "1").is_dir()
    assert (tmp_path / "2").is_dir()


def test_upload_path_anonymous_uploads_stay_flat(tmp_path, monkeypatch):
    """user_id=None → the legacy demo_pdfs/<filename> location, unchanged."""
    monkeypatch.setattr(main_fastapi, "UPLOAD_DIR", str(tmp_path))
    p = main_fastapi._upload_path("notes.pdf", None)
    assert p == str(tmp_path / "notes.pdf")


def test_same_named_uploads_do_not_collide_on_disk(tmp_path, monkeypatch):
    """The whole point: user 1's notes.pdf and user 2's notes.pdf are
    two distinct files with two distinct contents."""
    monkeypatch.setattr(main_fastapi, "UPLOAD_DIR", str(tmp_path))
    p1 = main_fastapi._upload_path("notes.pdf", 1)
    p2 = main_fastapi._upload_path("notes.pdf", 2)
    Path(p1).write_bytes(b"user 1 content")
    Path(p2).write_bytes(b"user 2 content")
    assert Path(p1).read_bytes() == b"user 1 content"
    assert Path(p2).read_bytes() == b"user 2 content"
    assert os.path.basename(p1) == "notes.pdf"
    assert os.path.basename(p2) == "notes.pdf"
