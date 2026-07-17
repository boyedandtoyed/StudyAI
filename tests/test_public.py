import pytest
import requests

PUBLIC_URL = "https://studyai.binodtiwari.com"


@pytest.fixture(scope="module")
def public_reachable():
    try:
        requests.get(f"{PUBLIC_URL}/health", timeout=5)
    except requests.exceptions.RequestException:
        pytest.skip(f"{PUBLIC_URL} unreachable — tunnel likely down")


def test_public_health(public_reachable):
    r = requests.get(f"{PUBLIC_URL}/health", timeout=10)
    assert r.status_code == 200
    assert r.json() == {"status": "ok"}
