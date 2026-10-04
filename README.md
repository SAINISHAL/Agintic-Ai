# Phase 1 — Emotion-Aware, Culturally Grounded Counselling Chatbot

Repo: [SAINISHAL/Agintic-Ai](https://github.com/SAINISHAL/Agintic-Ai)

Phase 1 only: conversation history → emotion detection (fine-tuned RoBERTa) → context → prompt → **base Qwen3-4B, prompting only** → response.

Phase 1 does **not** fine-tune Qwen and does **not** use the Gita dataset for training. The Qwen QLoRA / Gita SFT code is kept in the repo for a later phase but is not part of the Phase 1 run flow.

No RAG, vector database, agents, long-term memory, reranking, or full safety agent.

## Datasets

The trained emotion checkpoint ships in `models/emotion/best`, so no dataset is needed just to run the chatbot.

Counselling CSVs are needed only if you want to **retrain** the emotion model or run `evaluate_chatbot.py` (it reads patient turns from the processed emotion test split):

```
data/counselling/my_dataset_train.csv
data/counselling/my_dataset_val.csv
data/counselling/my_dataset_test.csv
```

Gita `Chapter_*_QA.csv` files are **not required** in Phase 1 (only for the later Qwen SFT phase).

See `data/README.md` for columns.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

The emotion weights are stored with Git LFS. After cloning, run `git lfs install` then `git lfs pull` so `models/emotion/best/encoder/model.safetensors` is the real ~500 MB file, not a small pointer.

CUDA is detected automatically. Qwen3-4B loads in 4-bit on a CUDA GPU; on CPU, 4-bit is skipped and generation is slow.

On Google Colab, a GPU runtime often ships an old `torchao` (for example 0.10.0) and an old `bitsandbytes`. After `pip install -r requirements.txt`, also run:

```bash
pip install -U "bitsandbytes>=0.46.1" "torchao>=0.16.0"
```

Then restart the runtime and install again if needed.

## Run Phase 1

The trained emotion checkpoint is already in `models/emotion/best` (plus reports in `outputs/emotion`). You do **not** need to train anything to run the chatbot.

```bash
python scripts/test_e2e.py
python scripts/evaluate_chatbot.py --limit 20   # optional; needs processed emotion test split
python scripts/generate_phase1_report.py
streamlit run app.py
```

`evaluate_chatbot.py` reads patient turns from `data/processed/emotion/test_examples.jsonl` and writes to `outputs/chatbot/`. If that file is missing, run `python scripts/prepare_emotion_data.py` first (needs the counselling CSVs).

Optional: retrain the emotion classifier.

```bash
python scripts/prepare_emotion_data.py
python scripts/train_emotion.py
python scripts/evaluate_emotion.py
```

Response generation always uses base `Qwen/Qwen3-4B` via prompting (`qwen.use_finetuned: false` in `configs/chatbot.yaml`). If `models/emotion/best` is missing, emotion falls back to a keyword baseline.

## Architecture

```
USER
  → session history (this chat only)
  → RoBERTa multi-label emotion (sigmoid + tuned threshold)
  → context / need (rule-based, Phase 1)
  → prompt_builder (detected emotion + need + short history)
  → Qwen3-4B (base model, prompting only)
  → response
```

Gita grounding is **optional prompt text**, not training. The prompt tells the model not to start from “According to the Bhagavad Gita…” unless it actually helps.

## Emotion model

- Multi-label (not softmax). `"Sadness, Scared"` → `["Sadness", "Scared"]`.
- Runtime emotion input: current user utterance only; prior turns remain response context so earlier assistant text cannot override the user's current emotion.
- Loss: `BCEWithLogitsLoss` / sigmoid.
- Threshold is **tuned on validation** (not assumed 0.5) and saved in `models/emotion/best/label_mapping.json`.
- Official train/val/test CSVs are the split. Overlap is reported; missing IDs are **not silently deleted**.

Missing-ID repair rule: fill only when a missing block sits between two valid IDs of the **same** conversation and the turn gap equals the number of missing rows. Otherwise rows go to `outputs/emotion/flagged_missing_ids.csv` and are left out of conversation reconstruction.

## Config

Edit `configs/emotion.yaml` and `configs/chatbot.yaml`. `configs/qwen.yaml` is read only for `model_name` and `generation` settings in Phase 1. Do not hardcode batch sizes in code.

## Outputs

| Path | What |
|------|------|
| `outputs/emotion/data_quality_report.md` | EDA, missing IDs, split overlap |
| `outputs/emotion/evaluation_summary.json` | F1 / exact match vs baselines |
| `outputs/chatbot/inspectable_eval.jsonl` | user message, true/pred emotion, response |
| `outputs/chatbot/automatic_eval_summary.json` | heuristic chatbot scores |
| `outputs/phase1_evaluation.md` | Combined Phase 1 report |

Automatic chatbot scores are **heuristics**, labelled as such.

## Later phases (not part of Phase 1)

### Qwen QLoRA fine-tuning with Gita SFT (code kept, not run)

`scripts/prepare_qwen_data.py`, `scripts/train_qwen.py`, `scripts/evaluate_qwen.py`, `src/qwen/train.py`, `src/qwen/prepare_data.py` build a structured `messages` jsonl (counselling therapist replies + Gita Q&A + emotion-aware cultural wraps) and fine-tune Qwen3-4B with QLoRA. To enable it later, place the Gita CSVs in `data/gita/`, run those scripts, and set `qwen.use_finetuned: true` in `configs/chatbot.yaml`.

### Other

RAG / FAISS / Chroma, psychology or cultural retrieval, agents, user profiles, persistent memory, Qwen embeddings, web search, full safety agent.
