import pytest
import requests

BASE_URL = "http://localhost:8000"


@pytest.fixture(scope="session", autouse=True)
def server_running():
    try:
        requests.get(f"{BASE_URL}/health", timeout=2)
    except requests.exceptions.ConnectionError:
        pytest.skip("Server not running at http://localhost:8000 — start the server first")
