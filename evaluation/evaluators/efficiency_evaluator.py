"""Efficiency and latency benchmarking for Emotion-Aware Culturally Grounded Counselling Chatbot."""

import json
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.abspath("."))

from src.chatbot.chatbot import CounsellingChatbot


def benchmark_efficiency() -> dict[str, Any]:
    print("Initializing CounsellingChatbot for efficiency benchmark...")
    bot = CounsellingChatbot()
    bot.initialize()

    test_queries = [
        {"id": "eff_01", "query": "I feel anxious and overwhelmed by decisions in my career.", "category": "career_anxiety"},
        {"id": "eff_02", "query": "How can I stay calm when faced with sudden failures?", "category": "failure_coping"},
        {"id": "eff_03", "query": "What is the true nature of selfless duty in the Gita?", "category": "spiritual_duty"}
    ]

    query_results = []
    ttfts = []
    total_times = []
    tps_list = []

    for item in test_queries:
        q = item["query"]
        t0 = time.time()
        t_first = None
        chunks = []

        for chunk in bot.chat_stream(q, persist=False):
            if t_first is None:
                t_first = time.time() - t0
            chunks.append(chunk)

        total_t = time.time() - t0
        full_text = "".join(chunks)
        words = len(full_text.split())
        est_tokens = int(words * 1.3)
        gen_time = max(0.001, total_t - (t_first or 0.0))
        tps = est_tokens / gen_time

        ttfts.append(t_first or 0.0)
        total_times.append(total_t)
        tps_list.append(tps)

        query_results.append({
            "id": item["id"],
            "category": item["category"],
            "query": q,
            "ttft_s": round(t_first or 0.0, 3),
            "total_time_s": round(total_t, 3),
            "tokens_generated": est_tokens,
            "tokens_per_sec": round(tps, 2),
            "response_preview": full_text[:120] + "..."
        })
        print(f"Query {item['id']} done: TTFT={t_first:.2f}s, Total={total_t:.2f}s, Speed={tps:.1f} tok/s")

    summary = {
        "device": "CPU (Intel i5-12th Gen)",
        "generator": "Qwen/Qwen3-0.6B",
        "precision": "FP32",
        "mean_ttft_s": round(sum(ttfts) / len(ttfts), 3),
        "mean_total_time_s": round(sum(total_times) / len(total_times), 3),
        "mean_tokens_per_sec": round(sum(tps_list) / len(tps_list), 2),
        "min_tokens_per_sec": round(min(tps_list), 2),
        "max_tokens_per_sec": round(max(tps_list), 2),
        "benchmarks": query_results
    }

    os.makedirs("outputs/evaluation", exist_ok=True)
    with open("outputs/evaluation/efficiency_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Efficiency benchmark complete: Mean TTFT = {summary['mean_ttft_s']}s, Mean Speed = {summary['mean_tokens_per_sec']} tok/s")
    return summary


if __name__ == "__main__":
    benchmark_efficiency()
