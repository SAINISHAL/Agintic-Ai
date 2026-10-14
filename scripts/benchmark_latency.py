import sys
import time
from pathlib import Path

ROOT = Path(r"D:\Agentic AI")
sys.path.insert(0, str(ROOT))

from src.chatbot.chatbot import CounsellingChatbot

print("Initializing CounsellingChatbot...")
bot = CounsellingChatbot()
bot.initialize()

test_queries = [
    ("I feel anxious and overwhelmed by decisions in my career.", "Anxiety"),
    ("How can I stay calm when faced with sudden failures?", "Fear"),
    ("What is the true nature of selfless duty in the Gita?", "Neutral"),
]

print("\n" + "=" * 65)
print("END-TO-END PIPELINE LATENCY BENCHMARK (Streaming & Warm A100)")
print("=" * 65)

for i, (q, expected_emo) in enumerate(test_queries, 1):
    print(f"\n--- Query {i}: \"{q}\" ---")
    t0 = time.time()
    t_first = None
    chunks = []
    
    for chunk in bot.chat_stream(q, persist=True):
        if t_first is None:
            t_first = time.time() - t0
        chunks.append(chunk)
    
    total_t = time.time() - t0
    full_text = "".join(chunks)
    words = len(full_text.split())
    est_tokens = int(words * 1.3)
    tps = est_tokens / (total_t - t_first) if (total_t - t_first) > 0 else 0
    
    print(f"Time to First Token (TTFT): {t_first:.2f}s")
    print(f"Total Generation Time:       {total_t:.2f}s")
    print(f"Estimated Tokens Generated:  ~{est_tokens} tokens ({words} words)")
    print(f"Effective Generation Speed:  {tps:.1f} tokens/sec")
    print(f"Response Excerpt:            {full_text[:130]}...")

print("\n" + "=" * 65)
print("ALL BENCHMARKS COMPLETED SUCCESSFULLY")
print("=" * 65)
