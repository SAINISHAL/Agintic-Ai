from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from src.emotion.conversation import parse_conversation_id, repair_missing_ids
from src.utils.emotions import parse_emotions
from src.utils.io import ensure_dir, save_json


EXPECTED_COUNSELLING_COLUMNS = [
    "ID",
    "Speaker",
    "Utterance",
    "Code-Mix",
    "Emotion",
    "Dialogue_Act",
    "Sub topic",
]


def _column_map(df: pd.DataFrame) -> dict[str, str]:
    aliases = {
        "id": "ID",
        "speaker": "Speaker",
        "utterance": "Utterance",
        "code-mix": "Code-Mix",
        "code_mix": "Code-Mix",
        "codemix": "Code-Mix",
        "emotion": "Emotion",
        "dialogue_act": "Dialogue_Act",
        "dialogue act": "Dialogue_Act",
        "sub topic": "Sub topic",
        "sub_topic": "Sub topic",
        "subtopic": "Sub topic",
    }
    mapping = {}
    for col in df.columns:
        key = str(col).strip().lower()
        mapping[col] = aliases.get(key, col)
    return mapping


def load_counselling_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = _column_map(df)
    df = df.rename(columns=rename)
    return df


def _missing_count(series: pd.Series) -> int:
    return int(series.isna().sum() + (series.astype(str).str.strip().isin(["", "nan", "None"])).sum())


def analyze_split(df: pd.DataFrame, split_name: str) -> dict[str, Any]:
    report: dict[str, Any] = {"split": split_name, "n_rows": int(len(df))}
    present = [col for col in EXPECTED_COUNSELLING_COLUMNS if col in df.columns]
    missing_cols = [col for col in EXPECTED_COUNSELLING_COLUMNS if col not in df.columns]
    report["present_columns"] = present
    report["missing_columns"] = missing_cols

    for col in present:
        report[f"missing_{col}"] = int(df[col].isna().sum())

    if "ID" in df.columns:
        parsed = [parse_conversation_id(v) for v in df["ID"]]
        report["missing_ids"] = sum(status == "missing" for _, _, status in parsed)
        report["malformed_ids"] = sum(status == "malformed" for _, _, status in parsed)
        conv_ids = [conv for conv, _, status in parsed if status == "ok" and conv]
        report["n_conversations_raw"] = len(set(conv_ids))

    if "Speaker" in df.columns:
        speakers = df["Speaker"].fillna("").astype(str).str.strip()
        report["speaker_distribution"] = speakers.value_counts(dropna=False).to_dict()
        report["invalid_speakers"] = int((~speakers.str.upper().isin({"P", "T", "PATIENT", "THERAPIST", "USER", "COUNSELLOR", ""})).sum())
        report["missing_speakers"] = int((speakers == "").sum() + df["Speaker"].isna().sum())

    if "Utterance" in df.columns:
        utterances = df["Utterance"].fillna("").astype(str).str.strip()
        report["missing_utterances"] = int((utterances == "").sum() + df["Utterance"].isna().sum())
        report["duplicate_utterance_count"] = int(utterances.duplicated().sum())

    report["duplicate_rows"] = int(df.duplicated().sum())

    if "Emotion" in df.columns:
        report["missing_emotion_labels"] = int(df["Emotion"].isna().sum())
        raw_counts = df["Emotion"].fillna("<MISSING>").astype(str).value_counts().head(50).to_dict()
        report["raw_emotion_value_counts_top50"] = raw_counts
        parsed_labels = [parse_emotions(v) for v in df["Emotion"]]
        flat = [lab for labs in parsed_labels if labs for lab in labs]
        report["parsed_emotion_distribution"] = dict(Counter(flat))
        report["empty_parsed_emotions"] = sum(1 for labs in parsed_labels if not labs)
        unknown_tokens = []
        for raw in df["Emotion"].dropna().astype(str):
            for token in [p.strip() for p in raw.replace(";", ",").split(",")]:
                if token and not parse_emotions(token):
                    unknown_tokens.append(token)
        report["unknown_emotion_tokens"] = dict(Counter(unknown_tokens))

    repaired_df, repairs = repair_missing_ids(df)
    report["id_repairs_applied"] = len(repairs)
    report["id_repair_examples"] = repairs[:20]
    report["_repaired_frame"] = repaired_df
    return report


def conversation_overlap_report(train_ids: set[str], val_ids: set[str], test_ids: set[str]) -> dict[str, Any]:
    return {
        "train_conversations": len(train_ids),
        "validation_conversations": len(val_ids),
        "test_conversations": len(test_ids),
        "intersection_train_val": sorted(train_ids & val_ids),
        "intersection_train_test": sorted(train_ids & test_ids),
        "intersection_val_test": sorted(val_ids & test_ids),
        "n_intersection_train_val": len(train_ids & val_ids),
        "n_intersection_train_test": len(train_ids & test_ids),
        "n_intersection_val_test": len(val_ids & test_ids),
        "leakage_detected": bool(train_ids & val_ids or train_ids & test_ids or val_ids & test_ids),
    }


def write_quality_report(report: dict[str, Any], output_dir: Path) -> Path:
    ensure_dir(output_dir)
    json_path = output_dir / "data_quality_report.json"
    markdown_path = output_dir / "data_quality_report.md"
    serializable = {k: v for k, v in report.items() if not k.startswith("_")}
    save_json(serializable, json_path)

    lines = ["# Data quality report", ""]
    overlap = report.get("overlap", {})
    lines.append("## Split overlap")
    lines.append("")
    lines.append(f"- Train conversations: {overlap.get('train_conversations')}")
    lines.append(f"- Validation conversations: {overlap.get('validation_conversations')}")
    lines.append(f"- Test conversations: {overlap.get('test_conversations')}")
    lines.append(f"- intersection(train, val): {overlap.get('n_intersection_train_val')}")
    lines.append(f"- intersection(train, test): {overlap.get('n_intersection_train_test')}")
    lines.append(f"- intersection(val, test): {overlap.get('n_intersection_val_test')}")
    lines.append(f"- Leakage detected: {overlap.get('leakage_detected')}")
    lines.append("")
    lines.append("Missing IDs are **reported and never silently dropped**.")
    lines.append("Repair rule: fill missing IDs only when they sit between two valid IDs of the same conversation and the turn gap equals the number of missing rows.")
    lines.append("Unrepairable rows are written to `flagged_missing_ids.csv` and excluded from conversation reconstruction.")
    lines.append("")
    for split in report.get("splits", []):
        lines.append(f"## {split['split']}")
        lines.append("")
        lines.append(f"- Rows: {split['n_rows']}")
        lines.append(f"- Missing IDs: {split.get('missing_ids')}")
        lines.append(f"- Malformed IDs: {split.get('malformed_ids')}")
        lines.append(f"- ID repairs applied: {split.get('id_repairs_applied')}")
        lines.append(f"- Missing utterances: {split.get('missing_utterances')}")
        lines.append(f"- Missing speakers: {split.get('missing_speakers')}")
        lines.append(f"- Missing emotion labels: {split.get('missing_emotion_labels')}")
        lines.append(f"- Duplicate rows: {split.get('duplicate_rows')}")
        lines.append("")
    markdown_path.write_text("\n".join(lines), encoding="utf-8")
    return markdown_path
