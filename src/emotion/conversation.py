from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

import pandas as pd

ID_PATTERN = re.compile(r"^(?P<conv>.+)_(?P<turn>\d+)$")


@dataclass
class Turn:
    row_index: int
    conversation_id: str
    turn_index: int
    speaker: str
    utterance: str
    emotion_raw: str | None
    emotions: list[str]
    dialogue_act: str | None = None
    sub_topic: str | None = None
    code_mix: str | None = None
    source_split: str | None = None
    id_repaired: bool = False


@dataclass
class Conversation:
    conversation_id: str
    turns: list[Turn] = field(default_factory=list)

    def sorted_turns(self) -> list[Turn]:
        return sorted(self.turns, key=lambda item: (item.turn_index, item.row_index))


def normalize_speaker(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).strip()


def parse_conversation_id(raw_id: object) -> tuple[str | None, int | None, str]:
    if raw_id is None or (isinstance(raw_id, float) and pd.isna(raw_id)):
        return None, None, "missing"
    text = str(raw_id).strip()
    if not text or text.lower() in {"nan", "none"}:
        return None, None, "missing"
    match = ID_PATTERN.match(text)
    if not match:
        return None, None, "malformed"
    return match.group("conv"), int(match.group("turn")), "ok"


def _safe_text(value: object) -> str | None:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return None
    text = str(value).strip()
    return text if text and text.lower() not in {"nan", "none"} else None


def repair_missing_ids(df: pd.DataFrame, id_col: str = "ID") -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Attempt conservative ID repair using surrounding conversation structure.

    Rule:
    If a contiguous block of missing/malformed IDs sits between two valid IDs
    of the SAME conversation, and the turn gap equals the number of missing
    rows, assign the in-between turn numbers in file order.

    Example: 102_5, <missing>, 102_7  -> assign 102_6.

    Rows that cannot be repaired are left missing and should be flagged.
    """
    work = df.copy()
    work["_parsed_conv"] = None
    work["_parsed_turn"] = None
    work["_id_status"] = "ok"
    work["_id_repaired"] = False
    repairs: list[dict[str, Any]] = []

    parsed = [parse_conversation_id(value) for value in work[id_col].tolist()]
    for idx, (conv, turn, status) in enumerate(parsed):
        work.at[work.index[idx], "_parsed_conv"] = conv
        work.at[work.index[idx], "_parsed_turn"] = turn
        work.at[work.index[idx], "_id_status"] = status

    positions = list(range(len(work)))
    i = 0
    while i < len(positions):
        status = work.iloc[i]["_id_status"]
        if status == "ok":
            i += 1
            continue

        start = i
        while i < len(positions) and work.iloc[i]["_id_status"] != "ok":
            i += 1
        end = i  # exclusive
        prev_idx = start - 1
        next_idx = end if end < len(work) else None

        if prev_idx < 0 or next_idx is None:
            continue

        prev_conv = work.iloc[prev_idx]["_parsed_conv"]
        next_conv = work.iloc[next_idx]["_parsed_conv"]
        prev_turn = work.iloc[prev_idx]["_parsed_turn"]
        next_turn = work.iloc[next_idx]["_parsed_turn"]
        gap_rows = end - start

        if (
            prev_conv
            and prev_conv == next_conv
            and prev_turn is not None
            and next_turn is not None
            and next_turn - prev_turn - 1 == gap_rows
            and next_turn > prev_turn
        ):
            for offset in range(gap_rows):
                row_pos = start + offset
                assigned_turn = int(prev_turn) + offset + 1
                assigned_id = f"{prev_conv}_{assigned_turn}"
                pandas_index = work.index[row_pos]
                work.at[pandas_index, id_col] = assigned_id
                work.at[pandas_index, "_parsed_conv"] = prev_conv
                work.at[pandas_index, "_parsed_turn"] = assigned_turn
                work.at[pandas_index, "_id_status"] = "repaired"
                work.at[pandas_index, "_id_repaired"] = True
                repairs.append(
                    {
                        "row_index": int(pandas_index) if str(pandas_index).isdigit() else row_pos,
                        "file_order": row_pos,
                        "assigned_id": assigned_id,
                        "rule": "same_conversation_turn_gap_equals_missing_row_count",
                    }
                )
    return work, repairs


def dataframe_to_conversations(
    df: pd.DataFrame,
    split_name: str,
    emotion_parser,
    speaker_col: str = "Speaker",
    utterance_col: str = "Utterance",
    emotion_col: str = "Emotion",
    id_col: str = "ID",
) -> tuple[dict[str, Conversation], pd.DataFrame]:
    flagged_rows = []
    conversations: dict[str, Conversation] = {}

    for file_order, (pandas_index, row) in enumerate(df.iterrows()):
        status = row.get("_id_status", "ok")
        conv_id = row.get("_parsed_conv")
        turn_idx = row.get("_parsed_turn")
        if status not in {"ok", "repaired"} or not conv_id or turn_idx is None:
            flagged_rows.append(
                {
                    "split": split_name,
                    "pandas_index": pandas_index,
                    "file_order": file_order,
                    "raw_id": row.get(id_col),
                    "id_status": status,
                    "speaker": row.get(speaker_col),
                    "utterance": row.get(utterance_col),
                    "emotion": row.get(emotion_col),
                    "reason": "missing_or_malformed_id_not_safely_repairable",
                }
            )
            continue

        turn = Turn(
            row_index=file_order,
            conversation_id=str(conv_id),
            turn_index=int(turn_idx),
            speaker=normalize_speaker(row.get(speaker_col)),
            utterance=_safe_text(row.get(utterance_col)) or "",
            emotion_raw=_safe_text(row.get(emotion_col)),
            emotions=emotion_parser(row.get(emotion_col)),
            dialogue_act=_safe_text(row.get("Dialogue_Act")),
            sub_topic=_safe_text(row.get("Sub topic") if "Sub topic" in row else row.get("Sub_topic")),
            code_mix=_safe_text(row.get("Code-Mix") if "Code-Mix" in row else row.get("Code_Mix")),
            source_split=split_name,
            id_repaired=bool(row.get("_id_repaired", False)),
        )
        conversations.setdefault(str(conv_id), Conversation(conversation_id=str(conv_id)))
        conversations[str(conv_id)].turns.append(turn)

    flagged = pd.DataFrame(flagged_rows)
    return conversations, flagged


def conversation_to_records(conversations: dict[str, Conversation]) -> list[dict[str, Any]]:
    records = []
    for conv in conversations.values():
        for turn in conv.sorted_turns():
            item = asdict(turn)
            records.append(item)
    return records


def context_before_turn(turns: list[Turn], current_index: int, max_turns: int) -> list[Turn]:
    """Return previous turns only. Never includes future turns."""
    history = turns[:current_index]
    if max_turns is not None and max_turns >= 0:
        history = history[-max_turns:]
    return history
