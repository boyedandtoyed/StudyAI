"""Unit tests for gg.parse_and_validate_card_json.

Pure unit tests — no server, no LLM. Same rootdir trick as test_user_store.py.
"""
import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from gg import CardJsonError, parse_and_validate_card_json


def _one_item(**overrides):
    item = {
        "question": "What is 2+2?",
        "options": ["3", "4", "5", "6"],
        "correct_index": 1,
        "explanation": "Basic arithmetic.",
    }
    item.update(overrides)
    return item


def _payload(items, key="questions"):
    return json.dumps({key: items})


def test_accepts_well_formed_questions_payload():
    raw = _payload([_one_item(), _one_item()])
    data = parse_and_validate_card_json(raw, expected_count=2, key="questions")
    assert len(data["questions"]) == 2


def test_accepts_well_formed_cards_payload():
    raw = _payload([_one_item()], key="cards")
    data = parse_and_validate_card_json(raw, expected_count=1, key="cards")
    assert len(data["cards"]) == 1


def test_strips_markdown_code_fences():
    raw = "```json\n" + _payload([_one_item()]) + "\n```"
    data = parse_and_validate_card_json(raw, expected_count=1)
    assert data["questions"][0]["question"] == "What is 2+2?"


def test_rejects_wrong_option_count():
    raw = _payload([_one_item(options=["a", "b", "c"])])
    with pytest.raises(CardJsonError):
        parse_and_validate_card_json(raw, expected_count=1)


def test_rejects_out_of_range_correct_index():
    raw = _payload([_one_item(correct_index=7)])
    with pytest.raises(CardJsonError):
        parse_and_validate_card_json(raw, expected_count=1)


def test_rejects_negative_correct_index():
    raw = _payload([_one_item(correct_index=-1)])
    with pytest.raises(CardJsonError):
        parse_and_validate_card_json(raw, expected_count=1)


def test_rejects_wrong_card_count():
    raw = _payload([_one_item(), _one_item()])
    with pytest.raises(CardJsonError):
        parse_and_validate_card_json(raw, expected_count=5)


def test_rejects_non_json():
    with pytest.raises(CardJsonError):
        parse_and_validate_card_json("sorry, I can't do that", expected_count=1)


def test_rejects_missing_key():
    raw = json.dumps({"answers": [_one_item()]})
    with pytest.raises(CardJsonError):
        parse_and_validate_card_json(raw, expected_count=1, key="questions")


def test_defaults_missing_explanation_to_empty():
    item = _one_item()
    del item["explanation"]
    raw = _payload([item])
    data = parse_and_validate_card_json(raw, expected_count=1)
    assert data["questions"][0]["explanation"] == ""
