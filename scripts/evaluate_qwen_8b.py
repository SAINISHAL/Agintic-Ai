"""Evaluation and comparison script for Qwen3-8B with Emotion Detection.

Tests both base Qwen3-8B and fine-tuned Qwen3-8B (LoRA) across benchmark
scenarios including moral dilemmas, grief, anxiety, duty, and Gita wisdom.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.chatbot.chatbot import CounsellingChatbot
from src.emotion.inference import EmotionClassifier
from src.qwen.inference import QwenGenerator
from src.utils.config import load_config, project_root, resolve_path
from src.utils.io import ensure_dir, save_json

TEST_PROMPTS = [
    {
        "category": "Duty and Conflict (Karma Yoga)",
        "prompt": "I have to make a tough business decision that will hurt some employees' livelihoods, but if I don't, the entire company will fail. I feel paralyzed by guilt. What does the Gita say about doing one's duty when all choices feel painful?",
    },
    {
        "category": "Grief and Loss (Sorrow / Sadness)",
        "prompt": "I recently lost my father, who was my anchor. I feel an overwhelming void and sadness every single day. How can I cope with this grief?",
    },
    {
        "category": "Anxiety and Fear of Failure (Future Anxiety)",
        "prompt": "I have a major competitive exam coming up next week. I haven't slept well because I keep picturing myself failing and humiliating my family. How do I stop this racing mind?",
    },
    {
        "category": "Anger and Betrayal",
        "prompt": "A close friend stole credit for my project and got promoted over me. I am consumed by rage and thoughts of revenge. How do I regain my peace?",
    },
    {
        "category": "Search for Purpose / Detachment",
        "prompt": "I work 60 hours a week, earn good money, but feel completely empty inside. What is the meaning of all this effort if none of it brings true happiness?",
    },
]


def run_evaluation(config_path: Path, use_finetuned: bool = False) -> list[dict[str, Any]]:
    print(f"Initializing CounsellingChatbot (use_finetuned={use_finetuned})...")
    chatbot = CounsellingChatbot(chatbot_config_path=config_path)
    # Temporarily override finetuned flag if requested
    chatbot.config["qwen"]["use_finetuned"] = use_finetuned
    chatbot.initialize()

    results = []
    print("\n" + "=" * 60)
    print(f" EVALUATION RUN: {'Fine-Tuned Qwen3-8B' if use_finetuned else 'Base Qwen3-8B'}")
    print("=" * 60 + "\n")

    for item in TEST_PROMPTS:
        category = item["category"]
        prompt = item["prompt"]
        print(f"[{category}]")
        print(f"User: {prompt}\n")

        # Step 1: Detect Emotion
        emotion_out = chatbot.detect_emotion(prompt)
        detected_emotions = emotion_out.get("predicted_emotions", [])
        print(f"Detected Emotions: {', '.join(detected_emotions) if detected_emotions else 'Neutral'}")

        # Step 2: Generate response
        response = chatbot.respond(prompt)
        print(f"Assistant: {response}\n")
        print("-" * 60)

        results.append({
            "category": category,
            "prompt": prompt,
            "detected_emotions": detected_emotions,
            "emotion_scores": emotion_out.get("probabilities", {}),
            "response": response,
        })
        chatbot.reset()

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate Qwen3-8B with Emotion Detection")
    parser.add_argument("--config", default=str(ROOT / "configs" / "chatbot.yaml"))
    parser.add_argument("--finetuned", action="store_true", help="Evaluate fine-tuned adapter instead of base model")
    parser.add_argument("--output", default=str(ROOT / "outputs" / "qwen_8b_eval.json"))
    args = parser.parse_args()

    results = run_evaluation(Path(args.config), use_finetuned=args.finetuned)
    output_path = Path(args.output)
    ensure_dir(output_path.parent)
    save_json(results, output_path)
    print(f"\nEvaluation complete. Results saved to: {output_path}")


if __name__ == "__main__":
    main()
