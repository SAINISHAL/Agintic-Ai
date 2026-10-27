"""Statistical analysis module: bootstrap confidence intervals, paired tests, Bonferroni adjustments."""

import json
import os
import sys
import numpy as np
from scipy import stats
from typing import Any

np.random.seed(42)


def bootstrap_ci(
    data: list[float] | np.ndarray,
    n_bootstraps: int = 1000,
    confidence_level: float = 0.95
) -> dict[str, float]:
    """Compute non-parametric bootstrap confidence interval."""
    arr = np.array(data)
    if len(arr) == 0:
        return {"mean": 0.0, "ci_lower": 0.0, "ci_upper": 0.0, "std": 0.0}

    boot_means = []
    n = len(arr)
    for _ in range(n_bootstraps):
        sample = np.random.choice(arr, size=n, replace=True)
        boot_means.append(np.mean(sample))

    alpha = (1.0 - confidence_level) / 2.0
    lower = float(np.percentile(boot_means, alpha * 100))
    upper = float(np.percentile(boot_means, (1.0 - alpha) * 100))

    return {
        "mean": float(np.mean(arr)),
        "ci_lower": round(lower, 4),
        "ci_upper": round(upper, 4),
        "std": float(np.std(arr))
    }


def paired_comparison(
    a_scores: list[float],
    b_scores: list[float],
    metric_name: str,
    n_comparisons: int = 3
) -> dict[str, Any]:
    """Perform paired t-test and Wilcoxon signed-rank test with Bonferroni correction."""
    a = np.array(a_scores)
    b = np.array(b_scores)
    diff = a - b

    # Paired t-test
    if np.all(diff == 0):
        t_stat, p_val_t = 0.0, 1.0
    else:
        t_stat, p_val_t = stats.ttest_rel(a, b)

    # Wilcoxon signed-rank test
    non_zero = diff[diff != 0]
    if len(non_zero) < 3:
        w_stat, p_val_w = 0.0, 1.0
    else:
        try:
            w_stat, p_val_w = stats.wilcoxon(diff)
        except Exception:
            w_stat, p_val_w = 0.0, 1.0

    # Cohen's d effect size for paired samples
    sd_diff = np.std(diff, ddof=1) if len(diff) > 1 else 0.0
    cohens_d = (np.mean(diff) / sd_diff) if sd_diff > 0 else 0.0

    # Bonferroni adjusted p-values
    p_adj_t = min(1.0, float(p_val_t) * n_comparisons)
    p_adj_w = min(1.0, float(p_val_w) * n_comparisons)

    return {
        "metric": metric_name,
        "mean_a": float(np.mean(a)),
        "mean_b": float(np.mean(b)),
        "mean_difference": float(np.mean(diff)),
        "t_statistic": float(t_stat),
        "p_value_raw": float(p_val_t),
        "p_value_bonferroni": float(p_adj_t),
        "wilcoxon_stat": float(w_stat),
        "wilcoxon_p_raw": float(p_val_w),
        "wilcoxon_p_bonferroni": float(p_adj_w),
        "cohens_d": round(float(cohens_d), 3),
        "significant_alpha_05": p_adj_t < 0.05
    }


def run_full_statistical_analysis(
    ablation_results_path: str = "outputs/evaluation/generation_eval_results.json"
) -> dict[str, Any]:
    if not os.path.exists(ablation_results_path):
        ablation_results_path = "outputs/evaluation/pilot_eval_results.json"

    with open(ablation_results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    records = data.get("records", [])

    # Group records by ablation and query_id
    by_ablation: dict[str, dict[str, dict[str, float]]] = {}
    for r in records:
        ab = r["ablation_id"]
        qid = r["query_id"]
        if ab not in by_ablation:
            by_ablation[ab] = {}
        by_ablation[ab][qid] = {
            "empathy": r["heuristics"]["empathy"],
            "relevance": r["heuristics"]["relevance"],
            "coherence": r["heuristics"]["coherence"],
            "latency": r["latency_s"]
        }

    # Bootstrap CIs for each ablation configuration
    bootstrap_results = {}
    for ab, qdict in by_ablation.items():
        empathies = [v["empathy"] for v in qdict.values()]
        relevances = [v["relevance"] for v in qdict.values()]
        coherences = [v["coherence"] for v in qdict.values()]
        latencies = [v["latency"] for v in qdict.values()]

        bootstrap_results[ab] = {
            "empathy": bootstrap_ci(empathies),
            "relevance": bootstrap_ci(relevances),
            "coherence": bootstrap_ci(coherences),
            "latency": bootstrap_ci(latencies)
        }

    # Paired comparisons against baseline A0
    comparisons = []
    if "A0" in by_ablation:
        a0_dict = by_ablation["A0"]
        common_qids = sorted(list(a0_dict.keys()))

        for ab_comp in ["A1", "A3", "A5"]:
            if ab_comp in by_ablation:
                comp_dict = by_ablation[ab_comp]
                valid_ids = [qid for qid in common_qids if qid in comp_dict]

                for metric in ["empathy", "relevance", "coherence", "latency"]:
                    a_vals = [a0_dict[qid][metric] for qid in valid_ids]
                    b_vals = [comp_dict[qid][metric] for qid in valid_ids]
                    comp_res = paired_comparison(a_vals, b_vals, f"{metric} (A0 vs {ab_comp})", n_comparisons=3)
                    comp_res["comparison"] = f"A0 vs {ab_comp}"
                    comparisons.append(comp_res)

    report = {
        "bootstrap_confidence_intervals": bootstrap_results,
        "paired_comparisons": comparisons,
        "n_bootstraps": 1000,
        "confidence_level": 0.95,
        "correction_method": "Bonferroni (m=3 comparisons per metric)"
    }

    os.makedirs("outputs/evaluation", exist_ok=True)
    with open("outputs/evaluation/statistical_analysis_results.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Statistical analysis completed successfully.")
    return report


if __name__ == "__main__":
    run_full_statistical_analysis()
