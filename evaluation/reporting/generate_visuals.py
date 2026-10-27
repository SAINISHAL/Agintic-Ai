"""Visualizations generator for Emotion-Aware Culturally Grounded AI Counselling Chatbot."""

import json
import os
import sys
import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np

FIGURES_DIR = "reports/evaluation/figures"
os.makedirs(FIGURES_DIR, exist_ok=True)

# Set high-quality styling
plt.style.use("seaborn-v0_8-whitegrid" if "seaborn-v0_8-whitegrid" in plt.style.available else "default")
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300
})


def plot_fig1_emotion_per_class():
    with open("outputs/emotion/test_metrics.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    labels = [
        "Neutral", "Sadness", "Scared", "Confusion", "Anticipation",
        "Annoyance", "Joy", "Trust", "Disgust", "Contempt", "Serenity", "Anger", "Surprise"
    ]
    f1_scores = [data["per_label"][lab]["f1-score"] for lab in labels]
    supports = [data["per_label"][lab]["support"] for lab in labels]

    fig, ax1 = plt.subplots(figsize=(10, 5))
    colors = ["#2b5c8f" if s > 50 else "#7da0ca" if s > 10 else "#d17b7b" for s in supports]

    bars = ax1.bar(labels, f1_scores, color=colors, width=0.6, edgecolor="black", linewidth=0.8)
    ax1.set_ylabel("F1-Score", color="#1c3d5a", fontweight="bold")
    ax1.set_ylim(0, 1.05)
    ax1.set_title("Figure 1: Per-Class F1-Scores of Fine-Tuned RoBERTa (Test Set, N=2,010)", pad=15)
    plt.xticks(rotation=35, ha="right")

    # Add value labels on bars
    for bar in bars:
        h = bar.get_height()
        ax1.annotate(f"{h:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    # Secondary axis for support count
    ax2 = ax1.twinx()
    ax2.plot(labels, supports, color="#e65c00", marker="o", linewidth=1.5, linestyle="--", label="Support count (N)")
    ax2.set_ylabel("Class Support Count (log scale)", color="#e65c00", fontweight="bold")
    ax2.set_yscale("log")
    ax2.grid(False)

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig1_emotion_f1_per_class.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Generated: {out_path}")


def plot_fig2_retrieval_performance():
    with open("outputs/evaluation/retrieval_eval_results.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    k_vals = ["Recall@1", "Recall@3", "Recall@5", "Recall@10"]
    recalls = [data["metrics"][k] * 100 for k in k_vals]

    fig, ax = plt.subplots(figsize=(7, 4.5))
    bars = ax.bar(k_vals, recalls, color=["#107050", "#14956b", "#27ae60", "#2ecc71"], width=0.5, edgecolor="black")
    ax.set_ylim(0, 110)
    ax.set_ylabel("Hit Rate / Recall (%)", fontweight="bold")
    ax.set_title(f"Figure 2: ChromaDB Semantic Retrieval over 2,109 Chunks\n(MRR = {data['metrics']['MRR']:.4f}, Mean Latency = {data['metrics']['mean_latency_ms']:.1f}ms)", pad=12)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig2_retrieval_performance.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Generated: {out_path}")


def plot_fig3_safety_recall():
    with open("outputs/evaluation/safety_eval_results.json", "r", encoding="utf-8") as f:
        data = json.load(f)

    cats = [
        "Suicide Ideation\n(N=10)",
        "Self-Harm\n(N=5)",
        "Harm-to-Others\n(N=5)",
        "Severe Hopeless\n(N=5, non-crisis)",
        "Normal Sadness\n(N=3, non-crisis)",
        "Normal Anxiety\n(N=2, non-crisis)"
    ]

    breakdown = data["category_breakdown"]
    trigger_rates = [
        (breakdown["explicit_suicide_ideation"]["crisis_triggered"] / 10) * 100,
        (breakdown["explicit_self_harm"]["crisis_triggered"] / 5) * 100,
        (breakdown["violent_harm_to_others"]["harm_triggered"] / 5) * 100,
        (breakdown["severe_hopelessness_ambiguous"]["crisis_triggered"] / 5) * 100,
        (breakdown["normal_sadness_non_crisis"]["crisis_triggered"] / 3) * 100,
        (breakdown["normal_anxiety_non_crisis"]["crisis_triggered"] / 2) * 100
    ]

    colors = ["#c0392b", "#e74c3c", "#d35400", "#27ae60", "#2ecc71", "#2ecc71"]

    fig, ax = plt.subplots(figsize=(9, 4.5))
    bars = ax.bar(cats, trigger_rates, color=colors, width=0.55, edgecolor="black")
    ax.set_ylabel("Safety Trigger Rate (%)", fontweight="bold")
    ax.set_ylim(0, 115)
    ax.set_title("Figure 3: Safety Module Interception Rate by Query Category\n(Target: 100% for Crisis/Harm, 0% for Non-Crisis Controls)", pad=12)

    for bar in bars:
        h = bar.get_height()
        ax.annotate(f"{h:.1f}%",
                    xy=(bar.get_x() + bar.get_width() / 2, h),
                    xytext=(0, 3),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=9, fontweight="bold")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig3_safety_performance.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Generated: {out_path}")


def plot_fig4_ablation_comparison():
    res_path = "outputs/evaluation/ablation_results.json"
    if not os.path.exists(res_path):
        return

    with open(res_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    ablations = list(data.keys())
    empathy = [data[ab]["mean_empathy"] for ab in ablations]
    relevance = [data[ab]["mean_relevance"] for ab in ablations]
    coherence = [data[ab]["mean_coherence"] for ab in ablations]

    x = np.arange(len(ablations))
    width = 0.25

    fig, ax = plt.subplots(figsize=(8, 4.5))
    r1 = ax.bar(x - width, empathy, width, label="Empathy (1-5)", color="#3498db", edgecolor="black")
    r2 = ax.bar(x, relevance, width, label="Relevance (1-5)", color="#2ecc71", edgecolor="black")
    r3 = ax.bar(x + width, coherence, width, label="Coherence (1-5)", color="#9b59b6", edgecolor="black")

    ab_labels = {
        "A0": "A0: Baseline\n(Full Chatbot)",
        "A1": "A1: No Emotion\n(Stripped)",
        "A3": "A3: No RAG\n(Retrieval Off)",
        "A5": "A5: No Grounding\n(Spiritual Off)"
    }
    ax.set_xticks(x)
    ax.set_xticklabels([ab_labels.get(ab, ab) for ab in ablations])
    ax.set_ylim(0, 5.5)
    ax.set_ylabel("Heuristic Quality Score (1–5)", fontweight="bold")
    ax.set_title("Figure 4: Core Ablation Matrix Quality Comparison", pad=12)
    ax.legend(loc="lower right")

    plt.tight_layout()
    out_path = os.path.join(FIGURES_DIR, "fig4_ablation_comparison.png")
    plt.savefig(out_path)
    plt.close()
    print(f"Generated: {out_path}")


def generate_all_visuals():
    try:
        plot_fig1_emotion_per_class()
    except Exception as e:
        print(f"Fig 1 error: {e}")
    try:
        plot_fig2_retrieval_performance()
    except Exception as e:
        print(f"Fig 2 error: {e}")
    try:
        plot_fig3_safety_recall()
    except Exception as e:
        print(f"Fig 3 error: {e}")
    try:
        plot_fig4_ablation_comparison()
    except Exception as e:
        print(f"Fig 4 error: {e}")


if __name__ == "__main__":
    generate_all_visuals()
