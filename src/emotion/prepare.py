from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from src.emotion.conversation import (
    conversation_to_records,
    dataframe_to_conversations,
    repair_missing_ids,
)
from src.emotion.dataset import build_emotion_examples
from src.emotion.quality import (
    analyze_split,
    conversation_overlap_report,
    load_counselling_csv,
    write_quality_report,
)
from src.utils.config import load_config, project_root, resolve_path
from src.utils.emotions import parse_emotions
from src.utils.io import ensure_dir, save_json, save_jsonl


def _conversation_ids(conversations: dict) -> set[str]:
    return set(conversations.keys())


def prepare_emotion_data(config_path: str | Path) -> dict[str, Any]:
    root = project_root()
    config = load_config(config_path)
    paths = config["paths"]
    processed_dir = ensure_dir(resolve_path(paths["processed_dir"], root))
    reports_dir = ensure_dir(resolve_path(paths["reports_dir"], root))

    split_frames = {}
    split_reports = []
    conversations_by_split = {}
    flagged_all = []

    for split, key in (("train", "train_csv"), ("val", "val_csv"), ("test", "test_csv")):
        csv_path = resolve_path(paths[key], root)
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Missing {split} counselling CSV: {csv_path}. Place the file as described in data/README.md."
            )
        df = load_counselling_csv(csv_path)
        analysis = analyze_split(df, split)
        repaired = analysis.pop("_repaired_frame")
        split_reports.append({k: v for k, v in analysis.items() if k != "_repaired_frame"})
        split_frames[split] = repaired
        conversations, flagged = dataframe_to_conversations(
            repaired,
            split_name=split,
            emotion_parser=parse_emotions,
        )
        conversations_by_split[split] = conversations
        if not flagged.empty:
            flagged_all.append(flagged)

        records = conversation_to_records(conversations)
        save_jsonl(records, processed_dir / f"{split}_turns.jsonl")
        examples = build_emotion_examples(
            conversations,
            emotion_list=config["emotions"],
            num_context_turns=int(config.get("num_context_turns", 4)),
            require_labels=True,
        )
        save_jsonl(examples, processed_dir / f"{split}_examples.jsonl")
        save_json(
            {
                "n_conversations": len(conversations),
                "n_turns": len(records),
                "n_labeled_examples": len(examples),
            },
            processed_dir / f"{split}_summary.json",
        )

    if flagged_all:
        flagged_df = pd.concat(flagged_all, ignore_index=True)
        flagged_path = reports_dir / "flagged_missing_ids.csv"
        flagged_df.to_csv(flagged_path, index=False)
    else:
        flagged_path = reports_dir / "flagged_missing_ids.csv"
        pd.DataFrame(columns=["split", "reason"]).to_csv(flagged_path, index=False)

    overlap = conversation_overlap_report(
        _conversation_ids(conversations_by_split["train"]),
        _conversation_ids(conversations_by_split["val"]),
        _conversation_ids(conversations_by_split["test"]),
    )
    quality = {
        "splits": split_reports,
        "overlap": overlap,
        "id_repair_rule": (
            "Fill missing IDs only when a contiguous missing block sits between two valid IDs "
            "of the same conversation and the turn gap equals the missing row count. "
            "Unrepairable rows are flagged and excluded from reconstruction, not silently deleted."
        ),
        "flagged_file": str(flagged_path),
        "no_future_turn_leakage": True,
    }
    write_quality_report(quality, reports_dir)

    lengths = []
    for split, convs in conversations_by_split.items():
        for conv in convs.values():
            lengths.append({"split": split, "conversation_id": conv.conversation_id, "n_turns": len(conv.turns)})
    pd.DataFrame(lengths).to_csv(reports_dir / "conversation_lengths.csv", index=False)

    return {
        "processed_dir": str(processed_dir),
        "reports_dir": str(reports_dir),
        "overlap": overlap,
        "flagged_rows": int(sum(len(df) for df in flagged_all) if flagged_all else 0),
    }
