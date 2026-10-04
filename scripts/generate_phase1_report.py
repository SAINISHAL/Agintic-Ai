from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.utils.config import load_config, project_root
from src.utils.io import ensure_dir, load_json


def _read(path: Path):
    if path.exists():
        return load_json(path)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Compile Phase 1 markdown evaluation report")
    args = parser.parse_args()
    del args
    root = project_root()
    out = ensure_dir(root / "outputs")
    quality = _read(root / "outputs" / "emotion" / "data_quality_report.json")
    emotion_eval = _read(root / "outputs" / "emotion" / "evaluation_summary.json")
    chatbot_eval = _read(root / "outputs" / "chatbot" / "automatic_eval_summary.json")
    qwen_model_name = load_config(root / "configs" / "qwen.yaml").get("model_name", "Qwen")

    lines = [
        "# Phase 1 evaluation report",
        "",
        "This report aggregates artifacts produced by the Phase 1 scripts.",
        f"Phase 1 scope: emotion detection (fine-tuned RoBERTa) + chatbot replies from base {qwen_model_name} via prompting only.",
        "No Qwen fine-tuning or Gita SFT is part of this phase.",
        "Automatic chatbot scores are heuristics and are labelled as such.",
        "",
        "## Data quality and leakage",
        "",
    ]
    if quality:
        overlap = quality.get("overlap", {})
        lines += [
            f"- Train conversations: {overlap.get('train_conversations')}",
            f"- Val conversations: {overlap.get('validation_conversations')}",
            f"- Test conversations: {overlap.get('test_conversations')}",
            f"- intersection(train, val): {overlap.get('n_intersection_train_val')}",
            f"- intersection(train, test): {overlap.get('n_intersection_train_test')}",
            f"- intersection(val, test): {overlap.get('n_intersection_val_test')}",
            f"- Leakage detected: {overlap.get('leakage_detected')}",
            "",
            quality.get("id_repair_rule", ""),
            "",
        ]
    else:
        lines.append("Run `python scripts/prepare_emotion_data.py` after placing CSVs in `data/`.")
        lines.append("")

    lines += ["## Emotion classifier", ""]
    if emotion_eval:
        ft = emotion_eval.get("fine_tuned", {})
        lines += [
            f"- Threshold used (tuned on validation, not assumed 0.5): {emotion_eval.get('threshold_used')}",
            f"- Micro F1: {ft.get('micro_f1')}",
            f"- Macro F1: {ft.get('macro_f1')}",
            f"- Weighted F1: {ft.get('weighted_f1')}",
            f"- Precision (micro): {ft.get('precision_micro')}",
            f"- Recall (micro): {ft.get('recall_micro')}",
            f"- Exact match ratio: {ft.get('exact_match_ratio')}",
            "",
            "### Emotion baselines",
            "",
            "```json",
            json.dumps(emotion_eval.get("baselines"), indent=2),
            "```",
            "",
        ]
    else:
        lines.append("Run `python scripts/train_emotion.py` then `python scripts/evaluate_emotion.py`.")
        lines.append("")

    lines += ["## Chatbot (automatic evaluation)", ""]
    if chatbot_eval:
        lines.append("```json")
        lines.append(json.dumps(chatbot_eval, indent=2))
        lines.append("```")
        lines.append("")
        lines.append(
            "Inspectable file columns: conversation_id, turn_index, user_message, true_emotion, "
            "predicted_emotion, model_response, grounding_used, automatic_scores."
        )
        lines.append("")
    else:
        lines.append("Run `python scripts/evaluate_chatbot.py` after the emotion checkpoint is in `models/emotion/best`.")
        lines.append("")

    lines += [
        "## Baseline vs Phase 1",
        "",
        "- Emotion: keyword / majority baselines vs fine-tuned RoBERTa (see emotion evaluation).",
        f"- Response model: base {qwen_model_name} via prompting only. The prompt carries the detected emotion, "
        "inferred need, and short session history; no adapter is loaded.",
        "",
        "## Out of scope (later phases)",
        "",
        "Qwen QLoRA fine-tuning and Gita Q&A SFT (`scripts/train_qwen.py`, `scripts/prepare_qwen_data.py`, "
        "`scripts/evaluate_qwen.py` are kept in the repo but are not part of this phase).",
        "RAG, vector databases, agentic workflows, long-term memory, reranking, and the full safety agent were not implemented.",
        "",
    ]
    path = out / "phase1_evaluation.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    print("Wrote", path)


if __name__ == "__main__":
    main()
