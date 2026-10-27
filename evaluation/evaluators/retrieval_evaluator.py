"""RAG retrieval evaluator over ChromaDB Master Dataset index."""

import json
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.abspath("."))

from src.rag.indexer import VectorStore
from src.rag.embeddings import EmbeddingModel


def evaluate_retrieval(
    suite_path: str = "evaluation/data/retrieval_test_suite.json",
    index_dir: str = "data/rag_index",
    k_values: list[int] | None = None
) -> dict[str, Any]:
    if k_values is None:
        k_values = [1, 3, 5, 10]

    with open(suite_path, "r", encoding="utf-8") as f:
        queries = json.load(f)

    store = VectorStore(index_dir=index_dir)
    embed_model = EmbeddingModel("sentence-transformers/all-MiniLM-L6-v2", device="cpu")

    results_by_query = []
    latencies = []

    k_hits = {k: 0 for k in k_values}
    mrr_sum = 0.0

    # Key terms associated with themes for semantic checking
    theme_keywords = {
        "grief": ["grief", "sorrow", "mourning", "loss", "tears", "separation", "distress"],
        "anger": ["anger", "rage", "forgiveness", "furious", "wrath", "calm"],
        "duty": ["duty", "dharma", "responsibility", "obligation", "arjuna", "battle"],
        "courage": ["courage", "strength", "fear", "brave", "power", "confidence"],
        "detachment": ["detachment", "equanimity", "action", "outcome", "surrender", "balance"],
        "support": ["support", "loyalty", "brother", "lakshmana", "devotion", "companion"],
        "decision": ["decision", "regret", "stress", "choice", "impulsive", "mind"],
        "surrender": ["surrender", "peace", "devotion", "faith", "trust"],
        "isolation": ["isolation", "lonely", "forest", "dandaka", "exile", "alone"],
        "meditation": ["mind", "meditation", "control", "focus", "restless", "calm"],
        "contentment": ["jealousy", "envy", "contentment", "peace", "desire"],
        "impermanence": ["mortality", "impermanence", "change", "transition", "death"],
        "ethical": ["dilemma", "righteousness", "duty", "conflict", "truth"],
        "hope": ["hope", "perseverance", "patience", "despair", "determination"],
        "empathy": ["listening", "empathy", "compassion", "understanding", "validation"],
        "ego": ["ego", "arrogance", "pride", "downfall", "ravana"],
        "righteousness": ["vibhishana", "righteousness", "dharma", "loyalty", "truth"],
        "humility": ["hanuman", "humility", "service", "devotion", "strength"],
        "selflessness": ["bharata", "selfless", "paduka", "throne", "brotherhood"],
        "bereavement": ["tara", "vali", "lamentation", "mourning", "widow", "death"]
    }

    for item in queries:
        qid = item["id"]
        q_text = item["query"]
        theme = item["expected_theme"].lower()

        # Find matching keywords for this theme
        relevant_words = []
        for key, words in theme_keywords.items():
            if key in theme:
                relevant_words.extend(words)
        if not relevant_words:
            relevant_words = theme.split()

        start_time = time.perf_counter()
        query_vec = embed_model.encode([q_text])[0]
        max_k = max(k_values)
        retrieved_raw = store.query(query_vec, top_k=max_k)
        elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        latencies.append(elapsed_ms)

        retrieved_docs = [r.get("document", "") for r in retrieved_raw]
        retrieved_distances = [r.get("distance", 1.0) for r in retrieved_raw]

        # Check hits at each k
        first_relevant_rank = None
        for rank, doc in enumerate(retrieved_docs, 1):
            doc_lower = doc.lower()
            if any(w in doc_lower for w in relevant_words):
                if first_relevant_rank is None:
                    first_relevant_rank = rank

        for k in k_values:
            if first_relevant_rank is not None and first_relevant_rank <= k:
                k_hits[k] += 1

        if first_relevant_rank is not None:
            mrr_sum += 1.0 / first_relevant_rank

        results_by_query.append({
            "id": qid,
            "query": q_text,
            "theme": theme,
            "first_relevant_rank": first_relevant_rank,
            "top_1_snippet": retrieved_docs[0][:150] if retrieved_docs else "",
            "top_1_distance": retrieved_distances[0] if retrieved_distances else 1.0,
            "latency_ms": elapsed_ms
        })

    n = len(queries)
    recall_at_k = {f"Recall@{k}": k_hits[k] / n for k in k_values}
    mean_mrr = mrr_sum / n
    avg_latency = sum(latencies) / len(latencies)

    summary = {
        "num_queries": n,
        "k_values": k_values,
        "metrics": {
            **recall_at_k,
            "MRR": mean_mrr,
            "mean_latency_ms": avg_latency,
            "min_latency_ms": min(latencies),
            "max_latency_ms": max(latencies)
        },
        "query_results": results_by_query
    }

    os.makedirs("outputs/evaluation", exist_ok=True)
    with open("outputs/evaluation/retrieval_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Retrieval evaluation complete: Recall@1={recall_at_k['Recall@1']:.2%}, Recall@5={recall_at_k['Recall@5']:.2%}, MRR={mean_mrr:.4f}, Latency={avg_latency:.1f}ms")
    return summary


if __name__ == "__main__":
    evaluate_retrieval()
