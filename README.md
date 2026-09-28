# Phase 1 — Emotion-Aware, Culturally Grounded Counselling Chatbot

Phase 1 only: conversation history → emotion detection → context → prompt → Qwen3-4B.

No RAG, vector database, agents, long-term memory, reranking, or full safety agent.

## Place datasets here first

Copy files into these folders (names must match):

```
data/counselling/my_dataset_train.csv
data/counselling/my_dataset_val.csv
data/counselling/my_dataset_test.csv

data/gita/Chapter_1_QA.csv
...
data/gita/Chapter_18_QA.csv
```

See `data/README.md` for columns.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

CUDA is detected automatically. On CPU, Qwen 4-bit quantization is skipped. On some Windows setups `bitsandbytes` fails; training then falls back to LoRA without 4-bit.

## Run Phase 1

```bash
python scripts/test_e2e.py
python scripts/prepare_emotion_data.py
python scripts/train_emotion.py
python scripts/evaluate_emotion.py
python scripts/prepare_qwen_data.py
python scripts/train_qwen.py
python scripts/evaluate_qwen.py
python scripts/generate_phase1_report.py
streamlit run app.py
```

Training Qwen3-4B is optional before the UI. If `models/qwen/best` is missing, the app loads base `Qwen/Qwen3-4B` as the **prompting-only baseline**. If `models/emotion/best` is missing, emotion uses a keyword baseline.

## Architecture

```
USER
  → session history (this chat only)
  → RoBERTa multi-label emotion (sigmoid + tuned threshold)
  → context / need (rule-based, Phase 1)
  → prompt_builder
  → Qwen3-4B (QLoRA adapter if trained)
  → response
```

Gita grounding is **optional**. The prompt tells the model not to start from “According to the Bhagavad Gita…” unless it actually helps.

## Emotion model

- Multi-label (not softmax). `"Sadness, Scared"` → `["Sadness", "Scared"]`.
- Input: current utterance + **previous** turns only (no future leak).
- Loss: `BCEWithLogitsLoss` / sigmoid.
- Threshold is **tuned on validation** (not assumed 0.5) and saved in `models/emotion/best/label_mapping.json`.
- Official train/val/test CSVs are the split. Overlap is reported; missing IDs are **not silently deleted**.

Missing-ID repair rule: fill only when a missing block sits between two valid IDs of the **same** conversation and the turn gap equals the number of missing rows. Otherwise rows go to `outputs/emotion/flagged_missing_ids.csv` and are left out of conversation reconstruction.

## Qwen SFT mix

Structured `messages` jsonl (not raw CSV concat):

1. Counselling: history + emotion + patient utterance → **actual therapist** reply.
2. Gita Q&A: original **answer preserved**.
3. Emotion-aware cultural: counselling-style wrap only when the Gita question looks counselling-relevant.

Counselling targets stay therapist text so the model also sees replies **without** scripture.

## Config

Edit `configs/emotion.yaml`, `configs/qwen.yaml`, `configs/chatbot.yaml`. Do not hardcode batch sizes in code.

## Outputs

| Path | What |
|------|------|
| `outputs/emotion/data_quality_report.md` | EDA, missing IDs, split overlap |
| `outputs/emotion/evaluation_summary.json` | F1 / exact match vs baselines |
| `outputs/qwen/inspectable_eval.jsonl` | user message, true/pred emotion, response |
| `outputs/phase1_evaluation.md` | Combined Phase 1 report |

Automatic chatbot scores are **heuristics**, labelled as such.

## Later phases (do not exist yet)

RAG / FAISS / Chroma, psychology or cultural retrieval, agents, user profiles, persistent memory, Qwen embeddings, web search, full safety agent.
