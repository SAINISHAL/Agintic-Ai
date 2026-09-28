from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chatbot.context import build_context
from src.chatbot.prompt_builder import build_messages
from src.chatbot.safety import safety_check
from src.emotion.conversation import parse_conversation_id, repair_missing_ids
from src.utils.emotions import parse_emotions
import pandas as pd


def test_multi_label_parse() -> None:
    assert parse_emotions("Sadness, Scared") == ["Sadness", "Scared"]
    assert parse_emotions("sadness; fear") == ["Sadness", "Scared"]


def test_id_parse() -> None:
    conv, turn, status = parse_conversation_id("102_3")
    assert conv == "102" and turn == 3 and status == "ok"
    assert parse_conversation_id(None)[2] == "missing"


def test_id_repair() -> None:
    df = pd.DataFrame(
        {
            "ID": ["102_5", None, "102_7"],
            "Speaker": ["P", "T", "P"],
            "Utterance": ["a", "b", "c"],
            "Emotion": ["Sadness", "Neutral", "Scared"],
        }
    )
    repaired, repairs = repair_missing_ids(df)
    assert len(repairs) == 1
    assert repaired.iloc[1]["ID"] == "102_6"


def test_id_not_silently_dropped() -> None:
    df = pd.DataFrame(
        {
            "ID": [None, "103_0"],
            "Speaker": ["P", "T"],
            "Utterance": ["a", "b"],
            "Emotion": ["Sadness", "Neutral"],
        }
    )
    repaired, repairs = repair_missing_ids(df)
    assert repairs == []
    assert repaired.iloc[0]["_id_status"] != "ok"


def test_prompt_and_safety() -> None:
    emotions = [{"label": "Scared", "confidence": 0.91}]
    ctx = build_context("I am terrified about placement interviews.", [], emotions)
    messages = build_messages("I am terrified about placement interviews.", [], emotions, ctx)
    assert messages[0]["role"] == "system"
    assert "Do not force spiritual" in messages[0]["content"] or "force spiritual" in messages[0]["content"]
    crisis = safety_check("I want to kill myself")
    assert crisis["override_response"]


def main() -> None:
    test_multi_label_parse()
    test_id_parse()
    test_id_repair()
    test_id_not_silently_dropped()
    test_prompt_and_safety()
    print("Phase 1 unit checks passed.")


if __name__ == "__main__":
    main()
