# Project Inventory — Emotion-Aware, Culturally Grounded AI Counselling Chatbot

**Audit Date:** 2026-10-24  
**Auditor:** Automated inspection (Stage 1)  
**Environment:** Windows 11, Intel i5-12th Gen (10 cores), 16.83 GB RAM, CPU-only (no CUDA)

---

## 1. Component Inventory Table

| Component | Actual Path | Purpose | Inputs | Outputs | Current Status | Evaluation Feasibility |
|---|---|---|---|---|---|---|
| Streamlit UI | `app.py` | Main web application entry point | Browser HTTP requests | Chat UI responses | ✅ Running on port 8501 | Via browser; not automated |
| Launch script | `run_local.ps1` | PowerShell local launcher for CPU mode | None | Starts Streamlit | ✅ Working | Invoke from CLI |
| Chatbot orchestration | `src/chatbot/chatbot.py` | Coordinates emotion → context → RAG → generate → safety pipeline | user text + history | response dict with emotions, rag_sources, safety | ✅ Fully operational | Directly callable |
| Safety module | `src/chatbot/safety.py` | Crisis detection, harm-to-others detection, helpline override | user text | override_response, crisis bool, harm_to_others bool | ✅ Operational, uses regex patterns | Fully testable with fixture messages |
| Context builder | `src/chatbot/context.py` | Infer user need, allow/deny cultural grounding | user text + emotion labels | need string, allow_cultural_grounding bool | ✅ Operational | Testable with fixed inputs |
| Prompt builder | `src/chatbot/prompt_builder.py` | Construct LLM messages with emotion, context, RAG evidence | user msg, history, emotions, context, RAG results | list[dict] messages | ✅ Operational | Testable without LLM |
| Chatbot heuristic evaluator | `src/chatbot/evaluate.py` | Automatic response quality scoring | user msg, response, emotion labels | empathy, coherence, relevance heuristic scores | ✅ Operational; acknowledged as heuristic only | Can run on any response corpus |
| Emotion classifier (fine-tuned) | `src/emotion/inference.py` + `models/emotion/best/` | Multi-label emotion prediction from text | text string | emotions list, probabilities, threshold | ✅ Checkpoint loaded and verified | Fully evaluable against test split |
| Emotion keyword baseline | `src/emotion/baseline.py` | Rule-based emotion prediction fallback | text string | emotion labels | ✅ Operational | Already evaluated |
| Emotion model module | `src/emotion/model.py` | RobertaMultiLabelClassifier definition | tokenized input | logits | ✅ Used in inference | Inspected, not standalone |
| Emotion training | `src/emotion/train.py` | Training loop for RoBERTa classifier | DataLoader + config | checkpoint | ✅ Script exists; training complete | Not re-run |
| Emotion evaluation | `src/emotion/evaluate.py` | Evaluate checkpoint on test set | model dir + test examples | metrics JSON | ✅ Existing results in outputs/ | Already run; results exist |
| Emotion metrics | `src/emotion/metrics.py` | Compute multi-label metrics | y_true, y_prob, labels, threshold | F1, precision, recall, Hamming, etc. | ✅ Used by evaluate.py | Ready for reuse |
| Qwen generator | `src/qwen/inference.py` (QwenGenerator) | Local Qwen3 causal LM inference | messages list[dict] | generated string | ✅ Operational; currently runs Qwen/Qwen3-0.6B (base, no LoRA, CPU) | Evaluable |
| Qwen LoRA adapter | `models/qwen_8b/best/` | PEFT LoRA adapter for Qwen3-8B base | Qwen3-8B base | conditioned generation | ⚠️ Adapter present (349 MB safetensors), but base Qwen3-8B (~16GB) **not locally available** — requires download, infeasible in 16GB RAM with other models loaded | NOT LOADABLE on current hardware |
| Mental Health LLM | `src/mental_llm/inference.py` (MentalHealthGenerator) | Inference adapter for HuggingFace causal LMs | messages list[dict] | generated string | ⚠️ Configured for ShivomH/Falcon3-1B-MentalHealth; would require ~6.7GB RAM FP32. With only 1.78GB RAM available, NOT loadable while Qwen is running | Feasible after Qwen unloaded; requires model download |
| RAG pipeline | `src/rag/rag_pipeline.py` (RAGPipeline) | Vector index + semantic retrieval facade | user query, emotion labels, need | list[RetrievalResult] | ✅ Operational; 2,109 chunks indexed from Master Dataset | Fully evaluable |
| RAG embeddings | `src/rag/embeddings.py` | SentenceTransformers embedding wrapper | text | dense vector | ✅ Uses all-MiniLM-L6-v2 | Evaluable |
| RAG indexer | `src/rag/indexer.py` (VectorStore) | ChromaDB vector store management | TextChunks + embedding model | Chroma collection | ✅ Index exists at data/rag_index/ | Evaluable |
| RAG retriever | `src/rag/retriever.py` | Semantic similarity retrieval | query + embedding model | RetrievalResult list | ✅ Operational | Evaluable |
| RAG loader | `src/rag/loader.py` | Load JSON/CSV datasets into record format | file paths | list[Record] | ✅ Supports Master Dataset JSON, Gita CSV, Counselling CSV | Inspected |
| RAG text processor | `src/rag/text_processor.py` | Split records into overlapping text chunks | records, chunk_size, chunk_overlap | list[TextChunk] | ✅ Operational | Inspected |
| Chatbot eval script | `scripts/evaluate_chatbot.py` | End-to-end chatbot eval on patient turns | test_examples.jsonl | automatic scores + inspectable JSONL | ✅ Run in Phase 1; results in outputs/chatbot/ | Reusable |
| Latency benchmark | `scripts/benchmark_latency.py` | End-to-end latency measurement | 3 hardcoded queries | TTFT, total time, tokens/sec | ✅ Script exists; runnable | Run independently |
| Mental LLM comparison | `scripts/compare_models.py` | Sequential Qwen vs MentalLLM comparison | TEST_PROMPTS list | model_comparison.json | ✅ Script ready; blocked by RAM (needs sequential loading) | Runnable after freeing memory |
| Mental LLM evaluation suite | `scripts/evaluate_mental_llm.py` | 10-scenario evaluation with quality checks | provider arg | evaluation JSON | ✅ Script ready | Runnable |
| Gita data preparation | `scripts/prepare_gita_only.py` | Prepare Gita QA for SFT | Chapter CSVs | processed JSONL | ✅ Script exists | Not used yet in eval |
| Train emotion | `scripts/train_emotion.py` | Train RoBERTa emotion classifier | emotion config | checkpoint | ✅ Training complete; do not re-run | Read-only |
| Phase 1 report generator | `scripts/generate_phase1_report.py` | Generate phase1_evaluation.md | evaluation outputs | report MD | ✅ Already run | Reusable |
| Unit tests — mental LLM | `tests/test_mental_llm.py` | Mock-based tests for MentalHealthGenerator | None (mocked) | 12 test results | ✅ 12/12 PASSED | Runnable any time |
| Unit tests — comparison | `tests/test_model_comparison.py` | Mock-based tests for compare/eval scripts | None (mocked) | 8 test results | ✅ 8/8 PASSED (20/20 total) | Runnable any time |

---

## 2. Model Inventory Summary

| Model | ID / Checkpoint | Architecture | Parameters | Fine-tuning | Status | CPU Feasibility |
|---|---|---|---|---|---|---|
| Emotion classifier | `models/emotion/best/` (roberta-base fine-tuned) | RoBERTa-base + linear head (13 outputs) | 125M (encoder) + tiny classifier | Multi-label SFT on counselling dataset | ✅ Loaded | ✅ Yes (~500 MB) |
| Qwen3-0.6B (active) | `Qwen/Qwen3-0.6B` (HuggingFace hub) | Qwen3 causal LM | ~600M | None (prompting only) | ✅ Loaded via HF cache | ✅ Yes (~2.4GB FP32) |
| Qwen3-8B + LoRA | `Qwen/Qwen3-8B` + `models/qwen_8b/best/` | Qwen3 causal LM + PEFT LoRA (r=32, α=64) | ~8B base + 349MB adapter | Counselling SFT (QLoRA) | ⚠️ Adapter present; base NOT locally downloaded | ❌ No (base needs ~32GB FP32, 16GB FP16 — exceeds available RAM) |
| Falcon3-1B-MentalHealth | `ShivomH/Falcon3-1B-MentalHealth` (HuggingFace hub) | LlamaForCausalLM (Falcon3-1B-Instruct base) | 1.67B | Mental health counselling fine-tune | ⚠️ Configured; not downloaded yet | ⚠️ Marginal (6.7GB FP32; only 1.78GB RAM free currently) |
| MentaLLaMA-7B | `klyang/MentaLLaMA-chat-7B` (HuggingFace hub) | LlamaForCausalLM (LLaMA-2-chat-7B) | ~6.7B | Mental health instruction fine-tune | ❌ Not configured as active; 14GB FP16 | ❌ No (exceeds 16GB RAM) |
| all-MiniLM-L6-v2 | `sentence-transformers/all-MiniLM-L6-v2` (HF hub) | DistilBERT-based bi-encoder | 22M | SentenceTransformers pre-trained | ✅ Loaded for RAG | ✅ Yes (~90 MB) |
| ChromaDB vector store | `data/rag_index/` | ChromaDB collection | 2,109 vectors (384-dim) | None (index only) | ✅ Indexed and ready | ✅ Yes |

---

## 3. Key Findings from Inspection

1. **Active generator is Qwen3-0.6B in prompting-only mode.** The LoRA adapter targets Qwen3-8B which cannot run on this 16GB CPU machine. The fallback to the 0.6B base model is intentional behavior.
2. **Emotion classifier is multi-label** (13 classes, sigmoid activation, threshold=0.45) — not single-label.
3. **Test metrics already exist** from a prior evaluation run: micro-F1=0.8047, macro-F1=0.2248, exact-match=0.7711.
4. **Safety module uses regex only** — 3 crisis patterns, 2 harm-to-others patterns. No ML safety model.
5. **RAG indexes only the Master Dataset** (2,109 chunks from Ramayana JSON files). Gita Chapter CSVs and counselling CSVs are **NOT currently indexed** — they require separate loaders to be added to `rag.dataset_paths`.
6. **12,902 Gita QA records** across 18 chapters (schema: chapter, verse_source, question, answer). 3 duplicate questions in Chapter 1 only.
7. **Counselling dataset**: 8,322 train / 1,191 val / 2,010 test rows, multi-speaker (P=patient, T=therapist), 174 duplicate utterances in test, 66 missing IDs in train.
8. **No uploaded research papers were found in the repository.** They must be sourced externally.

---

## 4. Hardware & Environment

- **OS:** Windows 11
- **CPU:** Intel i5-1235U equivalent (10 cores)
- **RAM:** 16.83 GB total; **~1.78 GB currently available** (Streamlit + Qwen loaded)
- **GPU:** None (PyTorch CPU build — `torch.cuda.is_available()` = False)
- **Python:** 3.13.2
- **Key packages:** torch 2.14.0+cpu, transformers 5.17.0, sentence-transformers 6.2.0, chromadb 1.5.9, peft 0.21.0, pytest 9.1.1

---

## 5. Existing Evaluation Results (Pre-existing, Not Fabricated)

Verified from `outputs/emotion/test_metrics.json` and `outputs/emotion/evaluation_summary.json`:

| Metric | Fine-tuned RoBERTa | Majority Baseline | Keyword Baseline |
|---|---|---|---|
| Micro F1 | 0.8047 | 0.7903 | 0.7579 |
| Macro F1 | 0.2248 | 0.0687 | 0.1793 |
| Weighted F1 | 0.8034 | 0.6918 | 0.7175 |
| Micro Precision | 0.8172 | 0.8065 | 0.7655 |
| Micro Recall | 0.7925 | 0.7749 | 0.7505 |
| Exact Match Ratio | 0.7711 | 0.8065 | 0.7627 |

**Per-class noteworthy findings (test set, n=2,010):**
- **Neutral** (n=1,621): F1=0.9149 — dominant class, heavily drives micro metrics
- **Sadness** (n=213): F1=0.6024 — best non-neutral class
- **Scared** (n=76): F1=0.4183
- **Anger** (n=4), **Contempt** (n=10), **Disgust** (n=10), **Serenity** (n=8), **Surprise** (n=3), **Trust** (n=17): **F1=0.000** — zero-shot rare classes
- **Macro F1 = 0.2248** due to class imbalance (Neutral dominates with 1,621 / 2,010 = 80.6%)
