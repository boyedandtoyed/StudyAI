"""Unit tests for gg.py's LLM-provider layer, exercised in isolation.

These do NOT go through the FastAPI app and do NOT require a running server.
Real network calls are mocked out so the suite is safe to run in CI without an
OPENROUTER_API_KEY. The one test that would make a real call is guarded with
@pytest.mark.skipif so it's skipped for teammates who haven't set up a key.
"""
import os
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# gg.py lives at the repo root; make sure it's importable regardless of where
# pytest's rootdir is resolved.
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gg


def test_openrouter_requires_api_key(monkeypatch):
    """With no API key configured, call_llm_openrouter() raises a clear
    RuntimeError instead of crashing with an unhandled request/KeyError."""
    monkeypatch.setattr(gg, "OPENROUTER_API_KEY", "")

    with pytest.raises(RuntimeError, match="OPENROUTER_API_KEY"):
        gg.call_llm_openrouter("hello")


def test_get_llm_response_respects_provider_switch(monkeypatch):
    """get_llm_response() routes to the correct backend based purely on
    gg.LLM_PROVIDER — verified without any real API calls."""
    # ── provider = "openrouter" → should call call_llm_openrouter, not Ollama ──
    monkeypatch.setattr(gg, "LLM_PROVIDER", "openrouter")
    with patch("gg.call_llm_openrouter", return_value="from-openrouter") as mock_or, \
         patch("gg.requests.post") as mock_post:
        result = gg.get_llm_response("ping")

        assert result == "from-openrouter"
        mock_or.assert_called_once_with("ping")
        mock_post.assert_not_called()  # Ollama path must not be touched

    # ── provider = "ollama" → should hit Ollama, not OpenRouter ──
    monkeypatch.setattr(gg, "LLM_PROVIDER", "ollama")
    fake_response = MagicMock()
    fake_response.json.return_value = {"message": {"content": "from-ollama"}}
    with patch("gg.call_llm_openrouter") as mock_or, \
         patch("gg.requests.post", return_value=fake_response) as mock_post:
        result = gg.get_llm_response("ping")

        assert result == "from-ollama"
        mock_or.assert_not_called()  # OpenRouter path must not be touched
        mock_post.assert_called_once()


@pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="No real OPENROUTER_API_KEY set — skipping live OpenRouter call",
)
def test_openrouter_live_call_smoke():
    """Optional end-to-end smoke test against the real OpenRouter API. Only
    runs when a real key is present, so the suite stays green for everyone else."""
    answer = gg.call_llm_openrouter("Reply with the single word: pong")
    assert isinstance(answer, str)
    assert answer.strip() != ""
