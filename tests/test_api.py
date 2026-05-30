import requests

BASE_URL = "http://localhost:8000"


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
