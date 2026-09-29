from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chatbot.chatbot import CounsellingChatbot
from src.chatbot.evaluate import automatic_response_scores
from src.utils.config import load_config, project_root, resolve_path
from src.utils.io import ensure_dir, load_jsonl, save_json, save_jsonl


def select_patient_turns(examples: list[dict], patient_speakers: set[str]) -> list[dict]:
    rows = []
    for item in examples:
        speaker = str(item.get("speaker", "")).strip()
        utterance = (item.get("utterance") or "").strip()
        if speaker in patient_speakers and utterance:
            rows.append(item)
    return rows


def run_eval(rows: list[dict], limit: int, chatbot: CounsellingChatbot) -> list[dict]:
    out = []
    for item in rows[:limit]:
        user_message = item["utterance"]
        true_emotion = item.get("emotions") or []
        result = chatbot.chat(user_message, conversation_history=[], persist=False)
        pred = [e["label"] for e in result.get("emotions", [])]
        scores = automatic_response_scores(user_message, result["response"], pred, true_emotion)
        out.append(
            {
                "split_label": "emotion_test_patient_turns",
                "conversation_id": item.get("conversation_id"),
                "turn_index": item.get("turn_index"),
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
    parser = argparse.ArgumentParser(
        description="Phase 1 chatbot evaluation: emotion classifier + prompting-only Qwen on emotion test patient turns"
    )
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--emotion-config", default=str(ROOT / "configs" / "emotion.yaml"))
    parser.add_argument("--chatbot-config", default=str(ROOT / "configs" / "chatbot.yaml"))
    args = parser.parse_args()
    root = project_root()

    emotion_cfg = load_config(args.emotion_config)
    chatbot_cfg = load_config(args.chatbot_config)
    processed = resolve_path(emotion_cfg["paths"]["processed_dir"], root)
    test_file = processed / "test_examples.jsonl"
    if not test_file.exists():
        raise FileNotFoundError(f"Missing {test_file}. Run python scripts/prepare_emotion_data.py first.")

    patient_speakers = {str(s) for s in emotion_cfg.get("patient_speakers", ["P", "Patient", "USER", "User"])}
    rows = select_patient_turns(load_jsonl(test_file), patient_speakers)
    if not rows:
        raise RuntimeError(f"No patient turns found in {test_file} for speakers {sorted(patient_speakers)}.")

    reports = ensure_dir(resolve_path(chatbot_cfg.get("paths", {}).get("reports_dir", "outputs/chatbot"), root))

    chatbot = CounsellingChatbot(chatbot_config_path=args.chatbot_config)
    chatbot.initialize()
    inspectable = run_eval(rows, args.limit, chatbot)
    save_jsonl(inspectable, reports / "inspectable_eval.jsonl")

    def avg(key: str) -> float:
        vals = [row["automatic_scores"][key] for row in inspectable]
        return round(sum(vals) / max(1, len(vals)), 3)

    summary = {
        "evaluation_type": "automatic_heuristic_plus_inspectable_file",
        "response_model": "prompting_only_base_qwen" if not chatbot_cfg["qwen"].get("use_finetuned", False) else "finetuned_qwen",
        "emotion_backend": "roberta_checkpoint" if chatbot.emotion_model is not None else "keyword_baseline",
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
            "Inputs are patient turns from the official emotion test split; no Gita or SFT data is required."
        ),
    }
    save_json(summary, reports / "automatic_eval_summary.json")
    print("Wrote", reports / "inspectable_eval.jsonl")
    print(summary)


if __name__ == "__main__":
    main()
