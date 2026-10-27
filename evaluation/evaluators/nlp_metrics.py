"""Pure Python, zero-dependency implementations of ROUGE-1, ROUGE-2, ROUGE-L, and BLEU-2."""

import re
from collections import Counter
from typing import Any


def tokenize(text: str) -> list[str]:
    """Tokenize lowercase words."""
    return re.findall(r"\b\w+\b", text.lower())


def get_ngrams(tokens: list[str], n: int) -> list[tuple[str, ...]]:
    return [tuple(tokens[i : i + n]) for i in range(len(tokens) - n + 1)]


def compute_rouge_n(hypothesis_tokens: list[str], reference_tokens: list[str], n: int = 1) -> dict[str, float]:
    hyp_ngrams = get_ngrams(hypothesis_tokens, n)
    ref_ngrams = get_ngrams(reference_tokens, n)

    if not hyp_ngrams or not ref_ngrams:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    hyp_counts = Counter(hyp_ngrams)
    ref_counts = Counter(ref_ngrams)

    overlap = 0
    for ng, count in hyp_counts.items():
        overlap += min(count, ref_counts.get(ng, 0))

    precision = overlap / len(hyp_ngrams) if hyp_ngrams else 0.0
    recall = overlap / len(ref_ngrams) if ref_ngrams else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def lcs_length(x: list[str], y: list[str]) -> int:
    m = len(x)
    n = len(y)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if x[i - 1] == y[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    return dp[m][n]


def compute_rouge_l(hypothesis_tokens: list[str], reference_tokens: list[str]) -> dict[str, float]:
    if not hypothesis_tokens or not reference_tokens:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0}

    lcs = lcs_length(hypothesis_tokens, reference_tokens)
    precision = lcs / len(hypothesis_tokens)
    recall = lcs / len(reference_tokens)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1}


def compute_bleu_2(hypothesis_tokens: list[str], reference_tokens: list[str]) -> float:
    if not hypothesis_tokens or not reference_tokens:
        return 0.0

    import math

    p1 = compute_rouge_n(hypothesis_tokens, reference_tokens, n=1)["precision"]
    p2 = compute_rouge_n(hypothesis_tokens, reference_tokens, n=2)["precision"]

    if p1 <= 0:
        return 0.0

    # Brevity penalty
    c = len(hypothesis_tokens)
    r = len(reference_tokens)
    bp = 1.0 if c > r else math.exp(1 - r / c) if c > 0 else 0.0

    p2_smooth = p2 if p2 > 0 else (0.5 / len(hypothesis_tokens))
    geo_mean = math.exp(0.5 * math.log(p1) + 0.5 * math.log(p2_smooth))
    return bp * geo_mean


def compute_generation_metrics(prediction: str, reference: str) -> dict[str, float]:
    hyp_tokens = tokenize(prediction)
    ref_tokens = tokenize(reference)

    r1 = compute_rouge_n(hyp_tokens, ref_tokens, n=1)
    r2 = compute_rouge_n(hyp_tokens, ref_tokens, n=2)
    rl = compute_rouge_l(hyp_tokens, ref_tokens)
    bleu2 = compute_bleu_2(hyp_tokens, ref_tokens)

    return {
        "rouge1_f1": r1["f1"],
        "rouge1_recall": r1["recall"],
        "rouge1_precision": r1["precision"],
        "rouge2_f1": r2["f1"],
        "rouge2_recall": r2["recall"],
        "rouge2_precision": r2["precision"],
        "rougel_f1": rl["f1"],
        "rougel_recall": rl["recall"],
        "rougel_precision": rl["precision"],
        "bleu2": bleu2
    }
