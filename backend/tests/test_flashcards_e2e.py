"""End-to-end tests for the flashcard flow.

These call the route handlers in ``main_fastapi`` directly instead of going
through HTTP — the dev server holds an open ChromaDB against ``./chroma_db``
and running the app's lifespan in-process would deadlock on the same store.
Calling the handlers as plain functions bypasses the lifespan entirely and
still exercises the full server-side logic: validation, ownership check,
chunk retrieval, LLM generation, persistence, progress accounting, and delete.

Two tests:

1. ``test_end_to_end_flashcard_generation_mocked`` — always runs. Patches
   ``gg.get_llm_response`` at the same seam as ``test_openrouter.py`` so no
   API key or network is required.

2. ``test_end_to_end_flashcard_generation_live`` — only runs when
   ``OPENROUTER_API_KEY`` is set (same skip pattern as the live smoke test
   in ``test_openrouter.py``). Hits the real model and confirms the response
   survives our structural validator — this is the test that would have
   caught the 404 we shipped this sprint.
"""
import json
import os
import shutil
import sys
import uuid
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import gg  # noqa: E402
import main_fastapi  # noqa: E402
from backend import user_store  # noqa: E402


@pytest.fixture
def test_user():
    """Create a temp user and seed a fake owned PDF entry. Cleans up on teardown."""
    user_store.init_user_store()

    email = f"test_{uuid.uuid4()}@test.com"
    user = user_store.create_user("E2E Flashcard Test", email, "password123")
    uid = user["id"]

    ud = user_store.get_user_data(uid)
    assert ud is not None
    ud.setdefault("pdfs_uploaded", []).append({
        "filename": "seed.pdf",
        "timestamp": "2026-07-18T00:00:00Z",
    })
    user_store.save_user_data(uid, ud)

    yield {"id": uid, "email": email, "source_pdf": "seed.pdf"}

    # ── teardown ──
    folder = user_store._USERS_DIR / str(uid)
    if folder.exists():
        shutil.rmtree(folder, ignore_errors=True)
    with user_store._db_lock:
        db = user_store._load_db()
        db["users"] = [u for u in db["users"] if u["id"] != uid]
        user_store._save_json_atomic(user_store._USERS_DB, db)

    # Drop the per-user chroma collection cache so a re-used uid doesn't
    # hand back a stale handle to the next test run.
    main_fastapi._user_collections.pop(uid, None)


def _fake_chunks(_collection, _source_pdf, batch_size):
    """Stand-in for gg.random_chunks_from_source so the flashcard path does
    not require real chunks indexed against the user's ChromaDB."""
    return [{
        "text": "Photosynthesis is the process by which plants use sunlight, water, "
                "and carbon dioxide to produce oxygen and glucose.",
        "meta": {"source": "seed.pdf", "page": 1},
    }]


def _make_fake_cards_json(count: int) -> str:
    """Build a valid cards-JSON payload of the shape the LLM is expected to
    return, so the real parse/validate path is still exercised."""
    return json.dumps({
        "cards": [
            {
                "question": f"Mocked question {i}?",
                "options": ["Option A", "Option B", "Option C", "Option D"],
                "correct_index": i % 4,
                "explanation": "Because the test says so.",
            }
            for i in range(count)
        ]
    })


def test_end_to_end_flashcard_generation_mocked(test_user, monkeypatch):
    """Full path: generate -> fetch -> reveal -> progress -> delete -> gone.
    All in-process, with the LLM seam patched to a fixed valid response."""
    uid = test_user["id"]
    source_pdf = test_user["source_pdf"]
    count = 5

    # Same seam test_openrouter.py uses — patch gg.get_llm_response so
    # generate_cards_with_retry (which calls it internally) succeeds without
    # a network round-trip.
    monkeypatch.setattr(gg, "get_llm_response", lambda _prompt: _make_fake_cards_json(count))
    monkeypatch.setattr(main_fastapi, "random_chunks_from_source", _fake_chunks)

    # ── generate ──
    resp = main_fastapi.generate_flashcards(
        main_fastapi.FlashcardRequest(user_id=uid, source_pdf=source_pdf, count=count)
    )
    assert resp["source_pdf"] == source_pdf
    assert len(resp["cards"]) == count
    set_id = resp["id"]
    for card in resp["cards"]:
        assert isinstance(card["question"], str) and card["question"].strip()
        assert len(card["options"]) == 4
        assert 0 <= card["correct_index"] <= 3

    # ── fetch the set back ──
    fetched = main_fastapi.get_flashcard_set(uid, set_id)
    assert fetched["id"] == set_id
    assert len(fetched["cards"]) == count

    # ── reveal, then confirm progress bumps by the reveal count ──
    progress_before = main_fastapi.get_progress(uid)
    revealed_before = int(progress_before.get("flashcards_revealed_total", 0))

    reveal_resp = main_fastapi.record_flashcard_reveal(
        main_fastapi.FlashcardRevealRequest(user_id=uid, set_id=set_id, revealed_count=3)
    )
    assert reveal_resp["cards_revealed"] == 3

    progress_after = main_fastapi.get_progress(uid)
    assert progress_after["flashcards_revealed_total"] == revealed_before + 3

    # ── delete, then confirm gone from both the payload and the index ──
    del_resp = main_fastapi.delete_flashcard_set(uid, set_id)
    assert del_resp["success"] is True

    from fastapi import HTTPException
    with pytest.raises(HTTPException) as exc_info:
        main_fastapi.get_flashcard_set(uid, set_id)
    assert exc_info.value.status_code == 404

    listing = main_fastapi.list_flashcard_sets(uid)
    assert all(entry.get("id") != set_id for entry in listing["flashcard_sets"])


@pytest.mark.skipif(
    not os.environ.get("OPENROUTER_API_KEY"),
    reason="No real OPENROUTER_API_KEY set — skipping live flashcard generation",
)
def test_end_to_end_flashcard_generation_live(test_user, monkeypatch):
    """Same flow as the mocked test but hits the real LLM. Confirms real
    model output survives our structural validator. Chunk retrieval is
    still stubbed since the test user has no indexed PDF — the point of
    this test is the generation + validation path, not the retriever."""
    uid = test_user["id"]
    source_pdf = test_user["source_pdf"]
    count = 5

    monkeypatch.setattr(main_fastapi, "random_chunks_from_source", _fake_chunks)

    resp = main_fastapi.generate_flashcards(
        main_fastapi.FlashcardRequest(user_id=uid, source_pdf=source_pdf, count=count)
    )
    assert len(resp["cards"]) == count
    for card in resp["cards"]:
        assert isinstance(card["question"], str) and card["question"].strip()
        assert len(card["options"]) == 4
        assert 0 <= card["correct_index"] <= 3
        assert isinstance(card["explanation"], str)

    # Clean up the payload we just wrote so the live run leaves no residue
    # (the test_user fixture wipes the user folder anyway, but be tidy).
    main_fastapi.delete_flashcard_set(uid, resp["id"])
