# Reproducibility Guide — Comprehensive Evaluation of AI Counselling Chatbot

**Date:** 2026-10-24  
**Auditor / NLP Evaluation Engineer:** Automated Comprehensive Evaluation System  
**System Under Test:** Emotion-Aware, Culturally Grounded AI Counselling Chatbot  
**Hardware Environment:** Windows 11 (64-bit), Intel Core i5-12th Gen (10 vCPUs), 16.83 GB RAM, CPU-Only  
**Primary Configuration:** `configs/evaluation.yaml`

---

## 1. Environment & Dependency Setup

### 1.1 Python Virtual Environment
Ensure you are using the repository's dedicated virtual environment (`.venv`):
```powershell
# Activate local virtual environment
.\.venv\Scripts\Activate.ps1
```

### 1.2 Core Libraries & Versions
The evaluation framework relies on verified local versions:
- `python == 3.13.2`
- `torch == 2.14.0+cpu`
- `transformers == 5.17.0`
- `sentence-transformers == 6.2.0`
- `chromadb == 1.5.9`
- `scikit-learn == 1.9.1`
- `scipy == 1.15.2`
- `matplotlib == 3.10.1`
- `pytest == 9.1.1`

---

## 2. Directory Structure of Evaluation Artifacts

```text
evaluation/
  data/
    build_eval_suites.py         # Deterministic generator for test suites (seed=42)
    generation_test_suite.json   # 30 patient utterances from test split
    gita_test_suite.json         # 36 question-answer pairs across Chapters 1-18
    retrieval_test_suite.json    # 20 thematic queries for ChromaDB
    safety_test_suite.json       # 35 multi-category safety & crisis test cases
  evaluators/
    emotion_evaluator.py         # Multi-label emotion metrics vs baselines
    retrieval_evaluator.py       # ChromaDB Recall@k, MRR, latency
    safety_evaluator.py          # Crisis recall, harm recall, helpline validation
    generation_evaluator.py      # Qwen generation quality & core ablation matrix
    efficiency_evaluator.py      # TTFT, tokens/sec, throughput benchmarking
    nlp_metrics.py               # Pure-Python ROUGE-1/2/L and BLEU-2 implementations
  statistics/
    stats_analysis.py            # 1,000 bootstrap CIs, paired tests, Bonferroni adjustments
  reporting/
    generate_visuals.py          # Publication-ready matplotlib figures (DPI=300)
  tests/
    test_smoke.py                # 7 unit tests validating pipeline components
```

---

## 3. Step-by-Step Reproduction Commands

### Stage 1: Build Evaluation Test Suites
Generate the deterministic evaluation fixtures:
```powershell
.\.venv\Scripts\python.exe evaluation/data/build_eval_suites.py
```
*Expected Output:*
- `Safety test suite built: 35 cases.`
- `Gita test suite built: 36 questions across 18 chapters.`
- `Generation test suite built: 30 patient utterances.`
- `Retrieval test suite built: 20 thematic queries.`

---

### Stage 2: Execute Smoke Tests
Verify all model loaders, prompt builders, safety regexes, and vector retrieval:
```powershell
.\.venv\Scripts\python.exe -m pytest evaluation/tests/test_smoke.py
```
*Expected Output:*
- `7 passed in ~10.8s`

---

### Stage 3: Run Safety & Crisis Guardrail Evaluation
Benchmark the safety module against explicit suicide, self-harm, violent intent, and non-crisis controls:
```powershell
.\.venv\Scripts\python.exe evaluation/evaluators/safety_evaluator.py
```
*Generated Output File:* `outputs/evaluation/safety_eval_results.json`  
*Key Measured Outcomes:*
- Crisis Detection Recall: `46.67%` (strict regex word boundary limitation identified on inflections).
- Harm-to-Others Recall: `20.00%`.
- False Positive Rate: `0.00%` on normal distress and anxiety controls.

---

### Stage 4: Run RAG Retrieval Evaluation
Benchmark ChromaDB semantic search over the 2,109 indexed Master Dataset chunks:
```powershell
.\.venv\Scripts\python.exe evaluation/evaluators/retrieval_evaluator.py
```
*Generated Output File:* `outputs/evaluation/retrieval_eval_results.json`  
*Key Measured Outcomes:*
- `Recall@1 = 75.00%`
- `Recall@3 = 90.00%`
- `Recall@5 = 95.00%`
- `Recall@10 = 100.00%`
- `MRR = 0.8550`
- `Mean Latency = 503.2 ms` on CPU.

---

### Stage 5: Compile Emotion Classifier Benchmark
Synthesize fine-tuned RoBERTa multi-label metrics against Keyword and Majority baselines on the test split ($N=2,010$):
```powershell
.\.venv\Scripts\python.exe evaluation/evaluators/emotion_evaluator.py
```
*Generated Output File:* `outputs/evaluation/emotion_eval_summary.json`  
*Key Measured Outcomes:*
- Fine-Tuned RoBERTa: Macro-F1 = `0.2248`, Micro-F1 = `0.8047`, Exact Match = `77.11%`
- Keyword Baseline: Macro-F1 = `0.1793`, Micro-F1 = `0.7579`, Exact Match = `76.27%`
- Majority Baseline: Macro-F1 = `0.0687`, Micro-F1 = `0.7903`, Exact Match = `80.65%`

---

### Stage 6: Execute Generation & Ablation Benchmark
Run the core ablation matrix ($A_0, A_1, A_3, A_5$) across patient and Gita queries:
```powershell
# For quick pilot verification (N=5):
.\.venv\Scripts\python.exe evaluation/evaluators/generation_evaluator.py --pilot

# For full evaluation (N=25):
.\.venv\Scripts\python.exe evaluation/evaluators/generation_evaluator.py
```
*Generated Output Files:*
- `outputs/evaluation/generation_eval_results.json`
- `outputs/evaluation/ablation_results.json`

---

### Stage 7: Run Statistical Significance & Bootstrap Analysis
Compute 95% bootstrap confidence intervals (1,000 resamples), paired Wilcoxon tests, and Bonferroni adjustments:
```powershell
.\.venv\Scripts\python.exe evaluation/statistics/stats_analysis.py
```
*Generated Output File:* `outputs/evaluation/statistical_analysis_results.json`

---

### Stage 8: Generate Publication-Quality Figures
Generate all 4 high-resolution (300 DPI) visualization charts:
```powershell
.\.venv\Scripts\python.exe evaluation/reporting/generate_visuals.py
```
*Generated Figures in `reports/evaluation/figures/`:*
- `fig1_emotion_f1_per_class.png`
- `fig2_retrieval_performance.png`
- `fig3_safety_performance.png`
- `fig4_ablation_comparison.png`

---

## 4. Verification Check
To verify that original model artifacts and configuration files were untouched during evaluation:
```powershell
git status
```
Confirm that `models/emotion/best/` and `models/qwen_8b/best/` remain strictly unmodified.
