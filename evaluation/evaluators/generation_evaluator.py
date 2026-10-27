"""Generation and ablation evaluator for Emotion-Aware Culturally Grounded Counselling Chatbot."""

import json
import os
import sys
import time
from typing import Any

sys.path.insert(0, os.path.abspath("."))

from src.chatbot.chatbot import CounsellingChatbot
from src.chatbot.safety import grounding_used
from evaluation.evaluators.nlp_metrics import compute_generation_metrics


def score_heuristic_quality(user_msg: str, response: str, emotions: list[str]) -> dict[str, float]:
    """Compute reproducible heuristic quality scores on a 1-5 scale."""
    resp_lower = response.lower()
    user_lower = user_msg.lower()

    # 1. Empathy Score (Validation of user emotion and supportive listening cues)
    empathy_cues = [
        "understand", "sounds", "must be", "valid", "hear you", "take your time",
        "normal to feel", "overwhelmed", "gently", "sorry", "support", "care", "listen"
    ]
    empathy_hits = sum(1 for cue in empathy_cues if cue in resp_lower)
    # Check if detected emotion is acknowledged
    emotion_acknowledged = any(e.lower() in resp_lower for e in emotions)
    empathy_score = min(5.0, 1.0 + (empathy_hits * 0.6) + (1.0 if emotion_acknowledged else 0.0))

    # 2. Context Relevance Score (Direct addressing of user's key problem)
    user_words = set(re_word for re_word in user_lower.split() if len(re_word) > 3)
    overlap = sum(1 for w in user_words if w in resp_lower)
    overlap_ratio = overlap / len(user_words) if user_words else 0.5
    relevance_score = min(5.0, max(1.0, 1.0 + overlap_ratio * 5.0))

    # 3. Coherence Score (Logical length, punctuation, avoidance of repetitive loops)
    words = response.split()
    word_count = len(words)
    length_penalty = 0.0
    if word_count < 15:
        length_penalty = 2.0
    elif word_count > 250:
        length_penalty = 1.0
    
    # Repetition check (unique words / total words)
    unique_ratio = len(set(words)) / word_count if word_count > 0 else 1.0
    rep_penalty = 2.0 if unique_ratio < 0.5 else (1.0 if unique_ratio < 0.65 else 0.0)
    coherence_score = max(1.0, 5.0 - length_penalty - rep_penalty)

    # 4. Grounding Fidelity (If cultural grounding is present, does it frame gracefully?)
    is_grounded = grounding_used(response)
    grounding_score = 4.0 if is_grounded else 2.5

    return {
        "empathy": round(empathy_score, 2),
        "relevance": round(relevance_score, 2),
        "coherence": round(coherence_score, 2),
        "grounding_score": round(grounding_score, 2),
        "is_grounded": is_grounded,
        "word_count": word_count
    }


def run_ablation_sample(
    chatbot: CounsellingChatbot,
    user_message: str,
    ablation_id: str,
    reference: str = ""
) -> dict[str, Any]:
    """Execute a single query under a specific ablation configuration."""
    start_time = time.perf_counter()

    # Determine condition parameters
    if ablation_id == "A0":  # Baseline: Emotion + RAG + Cultural Grounding
        em_out = chatbot.detect_emotion(user_message)
        emotions = em_out.get("emotions", [])
        context = chatbot.build_context(user_message, emotions, [])
        retrieved = chatbot.retrieve(user_message, emotions, context)
        resp = chatbot.generate_response(user_message, emotions, context, [], safety_note=None, retrieved=retrieved)

    elif ablation_id == "A1":  # Emotion awareness DISABLED
        emotions = []  # Stripped
        context = chatbot.build_context(user_message, [], [])
        retrieved = chatbot.retrieve(user_message, [], context)
        resp = chatbot.generate_response(user_message, [], context, [], safety_note=None, retrieved=retrieved)

    elif ablation_id == "A3":  # RAG retrieval DISABLED
        em_out = chatbot.detect_emotion(user_message)
        emotions = em_out.get("emotions", [])
        context = chatbot.build_context(user_message, emotions, [])
        retrieved = []  # No RAG
        resp = chatbot.generate_response(user_message, emotions, context, [], safety_note=None, retrieved=[])

    elif ablation_id == "A5":  # Cultural grounding DISABLED
        em_out = chatbot.detect_emotion(user_message)
        emotions = em_out.get("emotions", [])
        context = chatbot.build_context(user_message, emotions, [])
        context["allow_cultural_grounding"] = False  # Denied
        retrieved = []
        resp = chatbot.generate_response(user_message, emotions, context, [], safety_note=None, retrieved=[])

    else:
        raise ValueError(f"Unknown ablation ID: {ablation_id}")

    elapsed_s = time.perf_counter() - start_time
    words = len(resp.split())
    speed_wps = words / elapsed_s if elapsed_s > 0 else 0.0

    emotion_labels = [e["label"] for e in emotions] if emotions else []
    heuristics = score_heuristic_quality(user_message, resp, emotion_labels)

    nlp_scores = {}
    if reference:
        nlp_scores = compute_generation_metrics(resp, reference)

    return {
        "ablation_id": ablation_id,
        "response": resp,
        "emotions_used": emotion_labels,
        "rag_chunks_used": len(retrieved) if 'retrieved' in locals() and retrieved else 0,
        "latency_s": round(elapsed_s, 2),
        "words_per_sec": round(speed_wps, 2),
        "heuristics": heuristics,
        "nlp_metrics": nlp_scores
    }


def evaluate_generation(
    pilot: bool = False,
    n_patient: int = 15,
    n_gita: int = 10,
    ablations: list[str] | None = None
) -> dict[str, Any]:
    if ablations is None:
        ablations = ["A0", "A1", "A3", "A5"]

    if pilot:
        n_patient = 3
        n_gita = 2
        print(f"--- Running Stage 4 PILOT Evaluation (n_patient={n_patient}, n_gita={n_gita}) ---")
    else:
        print(f"--- Running Full Generation & Ablation Evaluation (n_patient={n_patient}, n_gita={n_gita}) ---")

    with open("evaluation/data/generation_test_suite.json", "r", encoding="utf-8") as f:
        patient_cases = json.load(f)[:n_patient]

    with open("evaluation/data/gita_test_suite.json", "r", encoding="utf-8") as f:
        gita_cases = json.load(f)[:n_gita]

    print("Initializing CounsellingChatbot...")
    chatbot = CounsellingChatbot()
    chatbot.initialize()

    all_records = []
    ablation_stats = {
        ab: {
            "count": 0,
            "empathy_sum": 0.0,
            "relevance_sum": 0.0,
            "coherence_sum": 0.0,
            "grounding_count": 0,
            "latency_sum": 0.0,
            "rouge1_f1_sum": 0.0,
            "rougel_f1_sum": 0.0,
            "bleu2_sum": 0.0,
            "nlp_evaluated_count": 0
        }
        for ab in ablations
    }

    # 1. Patient counselling queries across ablations
    for idx, case in enumerate(patient_cases, 1):
        uid = case.get("id", f"pat_{idx}")
        msg = case["utterance"]
        print(f"Evaluating Patient Turn {idx}/{len(patient_cases)}: '{msg[:40]}...'")

        for ab in ablations:
            result = run_ablation_sample(chatbot, msg, ab, reference="")
            result["query_id"] = uid
            result["query_type"] = "patient_counselling"
            result["query_text"] = msg
            all_records.append(result)

            h = result["heuristics"]
            ablation_stats[ab]["count"] += 1
            ablation_stats[ab]["empathy_sum"] += h["empathy"]
            ablation_stats[ab]["relevance_sum"] += h["relevance"]
            ablation_stats[ab]["coherence_sum"] += h["coherence"]
            if h["is_grounded"]:
                ablation_stats[ab]["grounding_count"] += 1
            ablation_stats[ab]["latency_sum"] += result["latency_s"]

    # 2. Gita queries across ablations (evaluates ROUGE & grounding)
    for idx, case in enumerate(gita_cases, 1):
        q = case["question"]
        ref = case["reference_answer"]
        print(f"Evaluating Gita Query {idx}/{len(gita_cases)}: '{q[:40]}...'")

        for ab in ablations:
            result = run_ablation_sample(chatbot, q, ab, reference=ref)
            result["query_id"] = f"gita_{case['chapter']}_{idx}"
            result["query_type"] = "gita_grounding"
            result["query_text"] = q
            result["reference"] = ref
            all_records.append(result)

            h = result["heuristics"]
            m = result["nlp_metrics"]
            ablation_stats[ab]["count"] += 1
            ablation_stats[ab]["empathy_sum"] += h["empathy"]
            ablation_stats[ab]["relevance_sum"] += h["relevance"]
            ablation_stats[ab]["coherence_sum"] += h["coherence"]
            if h["is_grounded"]:
                ablation_stats[ab]["grounding_count"] += 1
            ablation_stats[ab]["latency_sum"] += result["latency_s"]

            if m:
                ablation_stats[ab]["rouge1_f1_sum"] += m.get("rouge1_f1", 0.0)
                ablation_stats[ab]["rougel_f1_sum"] += m.get("rougel_f1", 0.0)
                ablation_stats[ab]["bleu2_sum"] += m.get("bleu2", 0.0)
                ablation_stats[ab]["nlp_evaluated_count"] += 1

    # Summarize ablation comparison
    summary_table = {}
    for ab, st in ablation_stats.items():
        cnt = st["count"]
        nlp_cnt = st["nlp_evaluated_count"]
        summary_table[ab] = {
            "n_total": cnt,
            "mean_empathy": round(st["empathy_sum"] / cnt, 2) if cnt > 0 else 0.0,
            "mean_relevance": round(st["relevance_sum"] / cnt, 2) if cnt > 0 else 0.0,
            "mean_coherence": round(st["coherence_sum"] / cnt, 2) if cnt > 0 else 0.0,
            "cultural_grounding_rate": round(st["grounding_count"] / cnt, 2) if cnt > 0 else 0.0,
            "mean_latency_s": round(st["latency_sum"] / cnt, 2) if cnt > 0 else 0.0,
            "mean_rouge1_f1": round(st["rouge1_f1_sum"] / nlp_cnt, 4) if nlp_cnt > 0 else 0.0,
            "mean_rougel_f1": round(st["rougel_f1_sum"] / nlp_cnt, 4) if nlp_cnt > 0 else 0.0,
            "mean_bleu2": round(st["bleu2_sum"] / nlp_cnt, 4) if nlp_cnt > 0 else 0.0,
        }

    output_file = "outputs/evaluation/pilot_eval_results.json" if pilot else "outputs/evaluation/generation_eval_results.json"
    os.makedirs("outputs/evaluation", exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump({"summary": summary_table, "records": all_records}, f, indent=2)

    # Also save ablation_results.json
    with open("outputs/evaluation/ablation_results.json", "w", encoding="utf-8") as f:
        json.dump(summary_table, f, indent=2)

    print(f"\nGeneration evaluation finished. Saved to {output_file}.")
    print("Ablation Summary Table:")
    for ab, vals in summary_table.items():
        print(f"  {ab}: Empathy={vals['mean_empathy']}, Rel={vals['mean_relevance']}, Coh={vals['mean_coherence']}, Grounding={vals['cultural_grounding_rate']}, Latency={vals['mean_latency_s']}s, ROUGE-1={vals['mean_rouge1_f1']}")

    return {"summary": summary_table, "records": all_records}


if __name__ == "__main__":
    is_pilot = "--pilot" in sys.argv
    evaluate_generation(pilot=is_pilot)
