from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chatbot.chatbot import CounsellingChatbot
from src.qwen.evaluate import automatic_response_scores
from src.utils.config import load_config, project_root, resolve_path
from src.utils.io import ensure_dir, load_jsonl, save_json, save_jsonl


def run_eval(generator_path: Path | None, label: str, rows: list[dict], limit: int, chatbot: CounsellingChatbot) -> list[dict]:
    out = []
    for item in rows[:limit]:
        user_message = item.get("user_message") or ""
        true_emotion = item.get("true_emotion") or []
        result = chatbot.chat(user_message, conversation_history=[], persist=False)
        pred = [e["label"] for e in result.get("emotions", [])]
        scores = automatic_response_scores(user_message, result["response"], pred, true_emotion)
        out.append(
            {
                "split_label": label,
                "conversation_id": item.get("conversation_id"),
                "user_message": user_message,
                "true_emotion": true_emotion,
                "predicted_emotion": pred,
                "model_response": result["response"],
                "grounding_used": result.get("grounding_used"),
                "automatic_scores": scores,
            }
        )
        chatbot.clear_history()
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Automatic + inspectable Qwen / chatbot evaluation")
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--finetuned-only", action="store_true")
    args = parser.parse_args()
    root = project_root()
    qwen_cfg = load_config(root / "configs" / "qwen.yaml")
    processed = resolve_path(qwen_cfg["paths"]["processed_dir"], root)
    heldout = processed / "heldout_counselling.jsonl"
    if not heldout.exists():
        raise FileNotFoundError("Run python scripts/prepare_qwen_data.py first.")
    rows = load_jsonl(heldout)
    reports = ensure_dir(resolve_path(qwen_cfg["paths"]["reports_dir"], root))

    chatbot = CounsellingChatbot()
    chatbot.initialize()
    inspectable = run_eval(None, "current_pipeline", rows, args.limit, chatbot)
    save_jsonl(inspectable, reports / "inspectable_eval.jsonl")

    def avg(key: str) -> float:
        vals = [row["automatic_scores"][key] for row in inspectable]
        return round(sum(vals) / max(1, len(vals)), 3)

    summary = {
        "evaluation_type": "automatic_heuristic_plus_inspectable_file",
        "n": len(inspectable),
        "mean_emotional_alignment": avg("emotional_alignment"),
        "mean_context_relevance": avg("context_relevance"),
        "mean_empathy": avg("empathy"),
        "mean_coherence": avg("coherence"),
        "mean_cultural_relevance": avg("cultural_relevance"),
        "share_spirituality_forced": round(
            sum(1 for row in inspectable if row["automatic_scores"]["spirituality_unnecessarily_forced"]) / max(1, len(inspectable)),
            3,
        ),
        "mean_response_appropriateness": avg("response_appropriateness"),
        "inspectable_file": str(reports / "inspectable_eval.jsonl"),
        "note": (
            "These scores are automatic heuristics for Phase 1, not human ratings. "
            "Compare base Qwen (fallback) vs models/qwen/best after fine-tuning by re-running this script."
        ),
    }
    save_json(summary, reports / "automatic_eval_summary.json")
    print("Wrote", reports / "inspectable_eval.jsonl")
    print(summary)


if __name__ == "__main__":
    main()
