import uuid

import pytest
import requests

BASE_URL = "http://localhost:8000"


def _register_unique_user(password="password123"):
    """Register a user with a unique email and return the parsed response JSON."""
    email = f"test_{uuid.uuid4()}@test.com"
    r = requests.post(
        f"{BASE_URL}/register",
        json={"name": "API Test User", "email": email, "password": password},
    )
    assert r.status_code == 200, r.text
    body = r.json()
    body["_email"] = email
    body["_password"] = password
    return body


def test_health():
    r = requests.get(f"{BASE_URL}/health")
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}


def test_stats():
    r = requests.get(f"{BASE_URL}/stats")
    assert r.status_code == 200
    data = r.json()
    for key in ("model_name", "embed_model", "total_chunks", "indexed_documents", "document_count", "server_time"):
        assert key in data, f"Missing key: {key}"


def test_docs_list():
    r = requests.get(f"{BASE_URL}/docs-list")
    assert r.status_code == 200
    data = r.json()
    assert "documents" in data
    assert isinstance(data["documents"], list)


def test_chat_no_docs():
    r = requests.post(f"{BASE_URL}/chat", json={"question": "test"})
    assert r.status_code == 200
    assert "answer" in r.json()


def test_clear_session_not_found():
    r = requests.delete(f"{BASE_URL}/clear-session/nonexistent")
    assert r.status_code == 404


# ── OpenRouter provider-switch contract tests ─────────────
# These hit the live server and don't assume which LLM_PROVIDER (ollama or
# openrouter) it's configured with — only that the public contract holds.

def test_chat_responds_regardless_of_provider():
    """POST /chat always returns 200 with an "answer", no matter which
    provider the server is currently using."""
    r = requests.post(f"{BASE_URL}/chat", json={"question": "What is 2 + 2?"})
    assert r.status_code == 200, r.text
    assert "answer" in r.json()


def test_stats_includes_provider_info():
    """GET /stats now surfaces the configured provider (added alongside the
    OpenRouter integration)."""
    r = requests.get(f"{BASE_URL}/stats")
    assert r.status_code == 200
    data = r.json()
    assert "llm_provider" in data, "Missing key: llm_provider"
    assert "provider_type" in data, "Missing key: provider_type"


def test_quiz_generation_works_with_current_provider():
    """Confirms the provider swap didn't break quiz generation: /quiz still
    returns a "questions" list with well-formed multiple-choice items.

    Note: the /quiz endpoint requests exactly 3 questions (not 10-15), so we
    assert the actual structure rather than a fixed count.
    """
    # /quiz needs at least one indexed document; skip rather than fail if the
    # running server has an empty index.
    stats = requests.get(f"{BASE_URL}/stats").json()
    if stats.get("document_count", 0) == 0:
        pytest.skip("No documents indexed on the running server — upload a PDF first")

    r = requests.post(f"{BASE_URL}/quiz", json={})
    assert r.status_code == 200, r.text
    data = r.json()
    assert "questions" in data
    questions = data["questions"]
    assert isinstance(questions, list)
    assert len(questions) >= 1
    for q in questions:
        assert "question" in q
        assert isinstance(q["options"], list)
        assert len(q["options"]) == 4
        assert isinstance(q["correct_index"], int)
        assert 0 <= q["correct_index"] <= 3
        assert "explanation" in q


# ── Auth + per-user account endpoints ─────────────────────

def test_register_endpoint():
    email = f"test_{uuid.uuid4()}@test.com"
    r = requests.post(
        f"{BASE_URL}/register",
        json={"name": "Reg Test", "email": email, "password": "password123"},
    )
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["success"] is True
    assert "id" in data["user"]


def test_register_duplicate_email():
    email = f"test_{uuid.uuid4()}@test.com"
    payload = {"name": "Dup Test", "email": email, "password": "password123"}

    r1 = requests.post(f"{BASE_URL}/register", json=payload)
    assert r1.status_code == 200, r1.text

    r2 = requests.post(f"{BASE_URL}/register", json=payload)
    assert r2.status_code == 400


def test_login_endpoint():
    user = _register_unique_user()
    r = requests.post(
        f"{BASE_URL}/login",
        json={"email": user["_email"], "password": user["_password"]},
    )
    assert r.status_code == 200, r.text
    assert r.json()["success"] is True


def test_login_wrong_password():
    user = _register_unique_user()
    r = requests.post(
        f"{BASE_URL}/login",
        json={"email": user["_email"], "password": "not-the-password"},
    )
    assert r.status_code == 401


def test_quiz_result_endpoint():
    user = _register_unique_user()
    user_id = user["user"]["id"]
    r = requests.post(
        f"{BASE_URL}/quiz-result",
        json={"user_id": user_id, "total_questions": 5, "correct": 3},
    )
    assert r.status_code == 200, r.text
    assert r.json() == {"success": True}


def test_progress_endpoint():
    user = _register_unique_user()
    user_id = user["user"]["id"]

    requests.post(
        f"{BASE_URL}/quiz-result",
        json={"user_id": user_id, "total_questions": 5, "correct": 3},
    )

    r = requests.get(f"{BASE_URL}/progress/{user_id}")
    assert r.status_code == 200, r.text
    data = r.json()
    assert data["questions_answered_total"] == 5
    assert data["questions_correct_total"] == 3
    assert "chroma_db_path" not in data


def test_progress_unknown_user():
    r = requests.get(f"{BASE_URL}/progress/99999")
    assert r.status_code == 404
