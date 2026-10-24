"""Model comparison script — run the same prompt through Qwen and MentalHealth LLM.

Usage:
    .venv\\Scripts\\python.exe scripts/compare_models.py

Runs inference **sequentially** and unloads the first model before loading the
second to stay within 16 GB RAM.  Results are printed as a table and saved to
``outputs/chatbot/model_comparison.json``.
"""

from __future__ import annotations

import gc
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch

from src.chatbot.prompt_builder import build_messages, SYSTEM_PROMPT
from src.utils.config import load_config, project_root


# ── Test prompts ──────────────────────────────────────────────────────────────
TEST_PROMPTS = [
    {
        "category": "Everyday stress",
        "message": "I've been so stressed with work deadlines, I can't sleep at night.",
    },
    {
        "category": "Academic pressure",
        "message": "My placements are coming up and I have zero preparation. I feel like a failure.",
    },
    {
        "category": "Grief",
        "message": "My grandmother passed away last month and I feel completely empty inside.",
    },
    {
        "category": "Relationship difficulty",
        "message": "My parents keep comparing me to my sibling and it makes me feel worthless.",
    },
    {
        "category": "Confusion",
        "message": "I don't know what to do with my life after college. Everything feels uncertain.",
    },
    {
        "category": "Spiritual perspective",
        "message": "Can the Bhagavad Gita help me deal with my fear of failure?",
    },
    {
        "category": "Non-spiritual request",
        "message": "What are some practical ways to manage my anxiety before an interview?",
    },
]


def _build_simple_messages(prompt: str) -> list[dict[str, str]]:
    """Build a minimal message list for direct model testing."""
    context = {
        "history_text": "(none)",
        "need": "emotional support",
        "allow_cultural_grounding": False,
        "emotion_labels": [],
    }
    return build_messages(prompt, [], [], context)


def _run_with_model(generator, prompt: str) -> dict:
    """Run a single prompt and collect timing."""
    messages = _build_simple_messages(prompt)
    t0 = time.time()
    try:
        response = generator.generate(messages)
        latency = time.time() - t0
        return {"response": response, "latency_s": round(latency, 2), "success": True, "error": None}
    except Exception as exc:
        latency = time.time() - t0
        return {"response": "", "latency_s": round(latency, 2), "success": False, "error": str(exc)}


def run_comparison():
    root = project_root()
    cfg = load_config(root / "configs" / "chatbot.yaml")
    results = []

    # ── 1. Qwen ───────────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("Loading Qwen model…")
    print("=" * 60)
    from src.qwen.inference import QwenGenerator

    qwen_cfg = load_config(root / cfg["qwen"]["config_path"])
    gen_cfg = qwen_cfg.get("generation", {})
    base_name = cfg["qwen"].get("base_model_name", "Qwen/Qwen3-0.6B")

    qwen_gen = QwenGenerator(
        model_path=base_name,
        base_model_name=base_name,
        load_in_4bit=False,
        generation_config=gen_cfg,
        inference_backend="local",
    )

    for item in TEST_PROMPTS:
        out = _run_with_model(qwen_gen, item["message"])
        results.append({
            "category": item["category"],
            "prompt": item["message"],
            "model": f"Qwen ({base_name})",
            **out,
        })
        print(f"  [{item['category']}] Qwen: {out['latency_s']}s — {out['response'][:80]}…")

    # Unload Qwen to free RAM
    del qwen_gen
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print("\nQwen unloaded.\n")

    # ── 2. Mental Health LLM ──────────────────────────────────────────────
    print("=" * 60)
    print("Loading Mental Health LLM…")
    print("=" * 60)
    from src.mental_llm.inference import MentalHealthGenerator

    ml_cfg = cfg.get("mental_llm", {})
    model_id = ml_cfg.get("model_id", "ShivomH/Falcon3-1B-MentalHealth")
    ml_gen = MentalHealthGenerator(
        model_id=model_id,
        device=ml_cfg.get("device", "auto"),
        torch_dtype=ml_cfg.get("torch_dtype", "auto"),
        generation_config={
            "max_new_tokens": ml_cfg.get("max_new_tokens", 256),
            "temperature": ml_cfg.get("temperature", 0.7),
            "top_p": ml_cfg.get("top_p", 0.9),
            "repetition_penalty": ml_cfg.get("repetition_penalty", 1.05),
        },
    )

    for item in TEST_PROMPTS:
        out = _run_with_model(ml_gen, item["message"])
        results.append({
            "category": item["category"],
            "prompt": item["message"],
            "model": f"MentalHealth ({model_id})",
            **out,
        })
        print(f"  [{item['category']}] MentalLLM: {out['latency_s']}s — {out['response'][:80]}…")

    del ml_gen
    gc.collect()

    # ── Save results ──────────────────────────────────────────────────────
    out_dir = root / "outputs" / "chatbot"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Results saved to {out_path}")

    # ── Print summary table ───────────────────────────────────────────────
    print("\n" + "=" * 80)
    print(f"{'Category':<25} {'Model':<35} {'Latency':>8} {'OK?':>4}")
    print("-" * 80)
    for r in results:
        print(f"{r['category']:<25} {r['model']:<35} {r['latency_s']:>7.1f}s {'✓' if r['success'] else '✗':>4}")
    print("=" * 80)


if __name__ == "__main__":
    run_comparison()
