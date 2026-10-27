"""Safety and crisis evaluator for Emotion-Aware Culturally Grounded AI Counselling Chatbot."""

import json
import os
import sys
from typing import Any

# Ensure workspace root is in sys.path
sys.path.insert(0, os.path.abspath("."))

from src.chatbot.safety import safety_check, CRISIS_RESPONSE


def evaluate_safety(suite_path: str = "evaluation/data/safety_test_suite.json") -> dict[str, Any]:
    with open(suite_path, "r", encoding="utf-8") as f:
        cases = json.load(f)

    results = []
    tp_crisis = 0
    fn_crisis = 0
    fp_crisis = 0
    tn_crisis = 0

    tp_harm = 0
    fn_harm = 0
    fp_harm = 0
    tn_harm = 0

    category_stats: dict[str, dict[str, int]] = {}

    for case in cases:
        cid = case["id"]
        cat = case["category"]
        text = case["text"]
        expected_crisis = case.get("is_crisis", False)
        expected_harm = case.get("is_harm_to_others", False)

        check_res = safety_check(text)
        detected_crisis = check_res["crisis"]
        detected_harm = check_res["harm_to_others"]
        override = check_res["override_response"]

        if cat not in category_stats:
            category_stats[cat] = {"total": 0, "crisis_triggered": 0, "harm_triggered": 0}
        category_stats[cat]["total"] += 1
        if detected_crisis:
            category_stats[cat]["crisis_triggered"] += 1
        if detected_harm:
            category_stats[cat]["harm_triggered"] += 1

        # Crisis stats
        if expected_crisis and not expected_harm:
            if detected_crisis:
                tp_crisis += 1
            else:
                fn_crisis += 1
        elif not expected_crisis:
            if detected_crisis:
                fp_crisis += 1
            else:
                tn_crisis += 1

        # Harm stats
        if expected_harm:
            if detected_harm:
                tp_harm += 1
            else:
                fn_harm += 1
        else:
            if detected_harm:
                fp_harm += 1
            else:
                tn_harm += 1

        # Validate helpline integrity if crisis triggered
        helpline_valid = True
        if detected_crisis and override:
            helpline_valid = ("1800-599-0019" in override) or ("9152987821" in override)

        results.append({
            "id": cid,
            "category": cat,
            "text": text,
            "expected_crisis": expected_crisis,
            "detected_crisis": detected_crisis,
            "expected_harm": expected_harm,
            "detected_harm": detected_harm,
            "override_present": bool(override),
            "helpline_valid": helpline_valid
        })

    crisis_recall = tp_crisis / (tp_crisis + fn_crisis) if (tp_crisis + fn_crisis) > 0 else 1.0
    crisis_fp_rate = fp_crisis / (fp_crisis + tn_crisis) if (fp_crisis + tn_crisis) > 0 else 0.0

    harm_recall = tp_harm / (tp_harm + fn_harm) if (tp_harm + fn_harm) > 0 else 1.0
    harm_fp_rate = fp_harm / (fp_harm + tn_harm) if (fp_harm + tn_harm) > 0 else 0.0

    summary = {
        "total_test_cases": len(cases),
        "crisis_cases": tp_crisis + fn_crisis,
        "crisis_recall": crisis_recall,
        "crisis_false_negatives": fn_crisis,
        "crisis_false_positives": fp_crisis,
        "crisis_fp_rate": crisis_fp_rate,
        "harm_cases": tp_harm + fn_harm,
        "harm_recall": harm_recall,
        "harm_false_negatives": fn_harm,
        "harm_false_positives": fp_harm,
        "harm_fp_rate": harm_fp_rate,
        "category_breakdown": category_stats,
        "results": results
    }

    os.makedirs("outputs/evaluation", exist_ok=True)
    with open("outputs/evaluation/safety_eval_results.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Safety evaluation complete: Crisis Recall = {crisis_recall:.2%}, Harm Recall = {harm_recall:.2%}, Crisis FP Rate = {crisis_fp_rate:.2%}")
    return summary


if __name__ == "__main__":
    evaluate_safety()
