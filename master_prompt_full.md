<USER_REQUEST>
# MASTER CODING-AGENT PROMPT: Comprehensive Evaluation of an Emotion-Aware, Culturally Grounded AI Counselling Chatbot

## 1. Role and objective

Act as a senior AI/ML research engineer, NLP evaluation specialist, RAG researcher, and reproducibility-focused experimental scientist.

I have an existing **Emotion-Aware, Culturally Grounded AI Counselling Chatbot**. Your task is to conduct a rigorous, end-to-end evaluation of the actual system, its individual models, its retrieval pipeline, its cultural and spiritual grounding, and its safety mechanisms.

The primary research references are all three uploaded papers:

1. **Spiritual-LLM: Gita Inspired Mental Health Therapy in the Era of LLMs** — `2506.19185v1 gitas.pdf`
2. **SpiritRAG: A Q&A System for Religion and Spirituality in the United Nations Archive** — `SpiritRAG_A_QA_System_for_Religion_and_Spiritualit.pdf`
3. **Development of a Ramayana-based Counseling Module for Persons with Common Mental Health Disorders** — `kurhade-et-al-2024-development-of-a-ramayana-based-counseling-module-for-persons-with-common-mental-health-disorders.pdf`

These papers investigate different research questions. Do not assume that they use the same datasets, models, metrics, evaluation protocols, or definitions of success.

Your goal is to determine:

- How well each existing component performs independently.
- How much each component contributes to the complete chatbot.
- Whether emotion awareness improves response quality.
- Whether RAG improves factual accuracy and source grounding.
- Whether spiritual or culturally grounded responses are accurate, relevant, respectful, and appropriate.
- How the existing Qwen+LoRA generator compares with a candidate mental-health-oriented LLM.
- Whether the complete chatbot is safe, useful, reliable, and computationally practical.
- How its measured results compare with relevant results actually reported in the three research papers.

**Do not begin by modifying code or training models. First inspect the actual project, uploaded papers, datasets, checkpoints, configuration, and evaluation scripts. Then design and implement the evaluation.**

---

## 2. Critical rules and constraints

Follow these rules throughout the project.

### 2.1 Inspect before implementing

Before making changes, inspect:

- The entire project directory structure.
- All existing source files related to inference, training, orchestration, safety, RAG, and data processing.
- All model checkpoints and adapter configurations.
- Existing evaluation scripts, notebooks, test cases, and experiment results.
- Dataset formats, sample counts, column names, metadata, duplicate records, and data quality.
- The three complete research papers, including methodology, experimental setup, tables, figures, limitations, and appendices where available.
- Dependency versions, hardware availability, memory requirements, and model loading configurations.

Create an initial inspection report before implementing anything.

Do not assume a file, function, model, dataset, metric, or capability exists merely because it is mentioned in this prompt. Verify it from the actual repository.

### 2.2 Preserve the existing system

Treat the current project as a research system whose original results and behavior must remain reproducible.

Known architecture elements to verify during inspection include:

- Streamlit interface: `app.py`.
- Launch script: `run_local.ps1`.
- Emotion classifier: fine-tuned RoBERTa-base, potentially multi-label.
- Emotion inference: `src/emotion/inference.py`.
- Emotion fallback: `src/emotion/baseline.py`.
- Existing emotion checkpoint: `models/emotion/best/`.
- Primary generation model: Qwen3-8B or a configured fallback such as Qwen3-0.6B.
- Existing Qwen LoRA adapter: `models/qwen_8b/best/`.
- Qwen inference: `src/qwen/inference.py`.
- Chat orchestration: `src/chatbot/chatbot.py`.
- Safety logic: `safety.py`.
- Context management: `context.py`.
- Prompt construction: `prompt_builder.py`.
- Counselling and Gita-related data: `data/counselling/` and `data/gita/`.
- Potentially available master dataset containing approximately 735 JSON files.

These are starting points, not guaranteed current paths. Confirm the real repository structure and adapt accordingly.

**Never overwrite, delete, merge into, or silently modify existing trained checkpoints or LoRA adapters.** Do not retrain existing models unless explicitly authorized. Keep original inference behavior available as a baseline.

Implement evaluation code separately wherever possible. If integration changes are necessary, keep them minimal, backward-compatible, and reversible.

### 2.3 Research integrity

- Never fabricate experimental scores, benchmark results, sample counts, citations, or statistical significance.
- Never claim that a paper evaluated a metric unless its methodology or results explicitly support that claim.
- Clearly distinguish published results, locally measured results, manually assessed results, and proposed experiments.
- Do not compare scores obtained using incompatible datasets, label definitions, test splits, prompts, metrics, or evaluation protocols as though they were directly comparable.
- Do not treat a high BLEU, ROUGE, BERTScore, or similar text-overlap score as proof of counselling quality, empathy, cultural appropriateness, or clinical safety.
- Report failed experiments, skipped tests, unavailable models, out-of-memory errors, and incomplete evaluations honestly.
- Do not describe the chatbot as clinically validated unless the available evidence supports that conclusion.

---

## 3. Phase A — Complete project and artifact audit

Before implementing evaluation code, create a structured inventory.

### 3.1 Project inventory

Identify:

1. Main application entry points.
2. Model inference modules.
3. Model training and fine-tuning scripts.
4. Model checkpoints and adapters.
5. Tokenizers and model configurations.
6. RAG indexing, embedding, retrieval, reranking, and generation modules.
7. Emotion classification modules.
8. Safety and crisis-handling modules.
9. Prompt templates.
10. Conversation memory and context handling.
11. Dataset loaders and preprocessing scripts.
12. Existing unit tests and integration tests.
13. Existing evaluation code and saved results.
14. Dependencies and environment configuration.
15. Available CPU, GPU, RAM, and storage resources.

Produce a table with these columns:

| Component | Actual path | Purpose | Inputs | Outputs | Current status | Evaluation feasibility |
|---|---|---|---|---|---|---|

Inspect code before filling in the table.

### 3.2 Model inventory

Identify every model actually used or configured in the project.

At minimum, investigate:

- The existing emotion classifier.
- The existing Qwen generator with its LoRA adapter.
- The base Qwen model, if accessible.
- The candidate mental-health-oriented LLM.
- The embedding model used by RAG.
- Any reranker.
- Any secondary classifier or auxiliary model.
- Any fallback models.

For each model, record:

- Exact repository or checkpoint identifier.
- Model architecture and parameter count, when verifiable.
- Fine-tuning method.
- Adapter dependencies.
- Tokenizer.
- Model revision or commit, where available.
- License and usage restrictions, where verifiable.
- Device, precision, quantization, and loading configuration.
- Actual inference entry point.
- Memory requirements observed during testing.
- Whether it is runnable in the current environment.
- Whether its output is deterministic under the selected settings.

Do not load every large model into memory simultaneously. Evaluate sequentially and release resources between runs.

If the candidate mental-health LLM is not configured or accessible, inspect the actual project and existing configuration to identify it. If its identity remains ambiguous, document that limitation rather than silently substituting another model.

### 3.3 Dataset inventory

Inspect every relevant dataset, including all uploaded chapter-wise QA CSV files:

- `Chapter_1_QA.csv` through `Chapter_18_QA.csv`.
- All counselling datasets.
- All Gita or spiritual knowledge datasets.
- The master JSON dataset, if present.
- Any benchmark, test, or validation datasets.
- Existing evaluation fixtures.

For each dataset, report:

- File path.
- Format and schema.
- Number of records.
- Missing values.
- Duplicate records.
- Duplicate questions or semantically similar questions.
- Language distribution.
- Relevant metadata fields.
- Whether answers are human-authored, model-generated, or unknown.
- Whether labels are available.
- Whether records contain personal or sensitive information.
- Whether the dataset is appropriate for each planned evaluation.

Preserve the original datasets.

Check whether chapter-wise files contain overlapping or duplicate content. Do not assume that every record is independent or that every answer is a verified gold-standard answer.

Create a dataset manifest and a data-quality report.

### 3.4 Research-paper audit

Read all three uploaded papers in full, not just their abstracts.

For each paper, extract:

- Research objective.
- Research questions.
- Model architecture.
- Dataset names and sizes.
- Data preprocessing.
- Train, validation, and test split methodology.
- Baselines.
- Experimental conditions.
- Evaluation metrics and their definitions.
- Published numerical results.
- Human evaluation procedures.
- Statistical methods.
- Reported limitations.
- Reproducibility information.
- Any available code or dataset references.
- Relevant tables and figures, with page or section references.

Create a **paper evidence matrix**:

| Paper | Research objective | Method | Dataset | Baselines | Metric | Published value | Experimental conditions | Page/table/section |
|---|---|---|---|---|---|---|---|---|

For each metric, preserve the exact meaning, units, evaluation split, and experimental conditions reported by the paper.

If a metric or numerical value is not reported, write `Not reported`. If the paper's text cannot be extracted reliably, inspect the relevant pages visually or use another appropriate local extraction method.

Do not invent results to fill gaps.

### 3.5 Initial deliverables

Before implementing evaluation code, save:

- `reports/evaluation/project_inventory.md`
- `reports/evaluation/model_inventory.csv`
- `reports/evaluation/dataset_inventory.csv`
- `reports/evaluation/paper_evidence_matrix.csv`
- `reports/evaluation/initial_audit.md`

Use actual discovered paths and verified facts.

Then proceed to the evaluation design.

---

## 4. Phase B — Design the experimental protocol

Design the experiments around the actual system and the evidence found in the three papers.

Create an evaluation plan before running large experiments.

### 4.1 Research questions

Evaluate at least the following research questions:

**RQ1 — Emotion classification:** How accurately does the existing emotion classifier identify the emotions expressed in user messages?

**RQ2 — Generation quality:** How does the existing Qwen+LoRA generator perform on counselling-oriented prompts?

**RQ3 — Mental-health LLM comparison:** How does the candidate mental-health-oriented LLM compare with the existing generator under controlled conditions?

**RQ4 — Retrieval quality:** How accurately and effectively does the RAG subsystem retrieve relevant and trustworthy passages?

**RQ5 — Grounded generation:** Does RAG improve answer relevance, faithfulness, and source traceability?

**RQ6 — Emotion awareness:** Does conditioning generation on emotion predictions improve empathy, emotional appropriateness, and response usefulness?

**RQ7 — Cultural and spiritual grounding:** Does the chatbot accurately and respectfully use relevant Gita or other culturally grounded material when appropriate?

**RQ8 — End-to-end quality:** How does the integrated chatbot perform across helpfulness, empathy, relevance, grounding, coherence, and safety?

**RQ9 — Safety:** Does the system handle distress, self-harm disclosures, crisis situations, and other sensitive requests appropriately?

**RQ10 — Efficiency:** What are the latency, throughput, and memory costs of each model and the integrated pipeline?

**RQ11 — Research comparison:** Which of the published findings in the three reference papers can be reproduced, compared, or only discussed qualitatively given the available artifacts?

Do not force a question to be answered if the necessary labels, reference answers, models, or data are unavailable. Record the limitation and propose the smallest additional data collection needed.

### 4.2 Evaluation splits and leakage prevention

Create reproducible evaluation splits with fixed random seeds.

Prefer existing official test splits if they are valid and available. Otherwise, construct documented splits appropriate to the task.

Prevent leakage by:

- Removing exact duplicates across splits.
- Checking near-duplicate questions and answers.
- Grouping related conversations, source documents, or records before splitting when needed.
- Preventing passages from the same source or conversation from unintentionally appearing in both development and test sets when that would inflate performance.
- Keeping the final test set separate from prompt tuning and model selection.
- Avoiding test-set-driven changes to prompts, thresholds, and retrieval parameters.

For chapter-wise Gita QA, retain chapter metadata and investigate whether evaluation should include both within-chapter and cross-chapter generalization.

For all splits, save record IDs or stable hashes, split assignments, seeds, and dataset versions. Do not expose unnecessary sensitive text in logs.

### 4.3 Reproducibility

Record the following for every run:

- Experiment ID.
- Timestamp.
- Git commit or source snapshot, if available.
- Dataset version and split identifier.
- Model and adapter identifiers.
- Prompt-template version.
- Random seeds.
- Decoding parameters.
- Device and precision.
- Software and library versions.
- RAG configuration.
- Metrics and evaluation configuration.
- Runtime and memory measurements.
- Warnings, failures, and skipped cases.

Use configuration files instead of hardcoding experimental settings throughout the codebase.

---

## 5. Phase C — Evaluate the emotion classifier

Evaluate the existing emotion model independently before evaluating its influence on generation.

### 5.1 Verify the label schema

Inspect the actual model configuration, training scripts, and dataset labels.

The current project may use these labels:

- Anger
- Annoyance
- Anticipation
- Confusion
- Contempt
- Disgust
- Joy
- Neutral
- Sadness
- Scared
- Serenity
- Surprise
- Trust

Verify whether the classifier is genuinely multi-label or single-label. Do not assume these labels or their ordering without inspecting the model artifacts.

Confirm the exact label-to-index mapping and output activation.

### 5.2 Metrics

For a verified multi-label task, compute:

- Micro precision, recall, and F1.
- Macro precision, recall, and F1.
- Weighted F1, when meaningful.
- Per-class precision, recall, F1, support, and confusion patterns.
- Hamming loss.
- Exact-match subset accuracy.
- Micro and macro average precision, if the ground truth and score outputs support them.
- Ranking metrics such as label-ranking average precision, if appropriate.
- Calibration metrics, such as Brier score or expected calibration error, where meaningful.
- Performance at the existing decision threshold and at a threshold selected using validation data only.

For a verified single-label task, additionally or alternatively compute:

- Accuracy.
- Macro F1.
- Weighted F1.
- Multiclass confusion matrix.
- Per-class metrics.
- Appropriate probability-based metrics where valid.

Do not report multi-label metrics as though the task were single-label, or vice versa.

### 5.3 Error analysis

Investigate:

- Frequent confusion between related emotions.
- Rare-class performance.
- Neutral versus non-neutral classification.
- Ambiguous and mixed-emotion messages.
- Short, indirect, sarcastic, or context-dependent messages.
- Domain shift between the classifier's original training data and counselling messages.
- Confidence calibration and uncertainty.
- Failures caused by missing conversation context.

Provide representative anonymized examples where permitted.

### 5.4 Baselines

Compare against the existing baseline classifier if one exists.

If useful and inexpensive, add simple baselines such as majority-class prediction or a transparent rule-based baseline. Label new baselines clearly and do not substitute them for existing ones.

Do not retrain the existing classifier or modify its checkpoint as part of the evaluation.

---

## 6. Phase D — Evaluate the existing Qwen+LoRA generator

Evaluate the actual configured Qwen base model with the existing LoRA adapter.

Verify that the correct adapter is loaded and that the intended inference path is being used.

### 6.1 Controlled generation

Use a fixed test set and record:

- Exact model and adapter identifiers.
- System and user prompts.
- Whether emotion information is supplied.
- Whether retrieved context is supplied.
- Whether spiritual grounding is enabled.
- Sampling settings and maximum output length.
- Random seed where supported.
- Any output truncation or generation failures.

Use deterministic decoding for the primary comparison where appropriate, then optionally evaluate multiple sampling seeds for robustness.

### 6.2 Generation metrics

Where suitable reference responses exist, calculate metrics such as:

- ROUGE-L.
- BERTScore.
- Other reference-based metrics only when justified by the task.

Use these as supporting indicators, not definitive measures of counselling quality.

Evaluate semantic relevance, answer completeness, and consistency with the question.

For open-ended counselling responses, prioritize a well-defined human or validated rubric-based evaluation over reference-text overlap alone.

### 6.3 Base model versus LoRA

If the original base model is accessible and compatible, compare:

1. Base Qwen without the LoRA adapter.
2. Qwen with the existing LoRA adapter.
3. Any additional generator included in the project.

Use the same prompts, test examples, context, decoding settings, and evaluation criteria wherever possible.

If the base model cannot be loaded, mark the comparison as unavailable and explain why.

Do not merge, overwrite, or permanently modify the adapter.

---

## 7. Phase E — Evaluate the mental-health-oriented LLM

Inspect the configured candidate model and determine whether it can run in the available environment.

Evaluate it as a separate generator before integrating it into the complete chatbot.

### 7.1 Fair comparison

Use the same test examples and equivalent prompting conditions for the mental-health LLM and Qwen+LoRA.

Run at least these conditions where technically meaningful:

- Candidate mental-health LLM alone.
- Existing Qwen+LoRA alone.
- Base Qwen without LoRA, if accessible.
- Candidate model with RAG, if compatible.
- Existing Qwen+LoRA with RAG.

Keep system instructions, output constraints, available evidence, and evaluation rubrics as consistent as possible.

If a model requires a different prompt format, document the difference and avoid claiming a perfectly controlled comparison.

### 7.2 Evaluate

Measure:

- Relevance.
- Helpfulness.
- Empathy.
- Emotional appropriateness.
- Clarity and coherence.
- Actionability.
- Hallucination risk.
- Faithfulness to supplied evidence.
- Cultural sensitivity.
- Appropriate use of spiritual references.
- Safety-policy compliance.
- Refusal appropriateness when relevant.
- Latency and memory use.

Document model limitations, license constraints, access problems, and hardware requirements.

Do not assume that a model described as mental-health-oriented is clinically safe, superior, or suitable for deployment.

Do not replace the existing generator in the main application as a side effect of this experiment. Use an explicit model-selection configuration or an isolated evaluation runner.

---

## 8. Phase F — Evaluate RAG independently

Inspect the actual retrieval pipeline before selecting metrics.

Identify:

- Document sources.
- Chunking strategy.
- Chunk size and overlap.
- Embedding model.
- Vector database or index.
- Retrieval algorithm.
- Top-k setting.
- Similarity function.
- Metadata filtering.
- Reranker, if any.
- Context assembly.
- Source attribution and citation mechanism.
- Index creation and update procedures.

Determine whether the RAG system retrieves Gita-related knowledge, counselling material, the chapter-wise QA CSV content, or other sources. Do not assume that all uploaded files are currently indexed.

### 8.1 Retrieval evaluation

Build a representative query set and establish relevance judgments.

Use verified relevant passages or manually reviewed judgments whenever possible.

Calculate, as appropriate:

- Recall@k.
- Precision@k.
- Hit rate@k.
- Mean reciprocal rank (MRR).
- Mean average precision (MAP).
- nDCG@k.
- Retrieval latency.
- Context size and token usage.

State the relevance definition and the value of k used for each metric.

Do not compute retrieval metrics against guessed relevance labels. If relevance judgments are missing, create a documented annotation workflow or report that the metric cannot yet be measured.

### 8.2 Grounded-answer evaluation

For questions with reliable reference answers or source passages, evaluate:

- Faithfulness to retrieved evidence.
- Answer relevance.
- Correctness of factual claims.
- Citation or source attribution correctness.
- Completeness of relevant information.
- Unsupported-claim rate.
- Contradictions between the answer and its sources.
- Appropriate handling of missing evidence.

If using a framework such as RAGAS or another automated evaluator, verify that it supports the actual data schema and version. Document its metrics, judge model, dependencies, and limitations.

Automated LLM judges must not be treated as ground truth.

### 8.3 Retrieval ablations

Compare, where supported:

1. No retrieval.
2. Standard retrieval.
3. Retrieval with reranking, if a reranker exists.
4. Alternative top-k settings.
5. Alternative chunking configurations, if feasible.
6. Metadata-filtered versus unfiltered retrieval, if meaningful.

Keep the underlying question set fixed.

Do not change multiple retrieval variables at once in a comparison intended to identify the effect of a single variable.

---

## 9. Phase G — Conduct ablation studies

Ablation studies must estimate the contribution of individual components.

Inspect the actual implementation and define a clear baseline before building ablations.

### 9.1 Core ablation matrix

Evaluate the following configurations wherever the architecture supports them:

| Experiment | Emotion awareness | RAG | Spiritual grounding | Generator |
|---|---|---|---|---|
| A0: Existing baseline | Existing default | Existing default | Existing default | Existing default |
| A1: Emotion disabled | Off | Same as A0 | Same as A0 | Same as A0 |
| A2: Emotion enabled | On | Same as A0 | Same as A0 | Same as A0 |
| A3: Retrieval disabled | Same as A0 | Off | Same as A0 | Same as A0 |
| A4: Retrieval enabled | Same as A0 | On | Same as A0 | Same as A0 |
| A5: Spiritual grounding disabled | Same as A0 | Same as A0 | Off | Same as A0 |
| A6: Spiritual grounding enabled | Same as A0 | Same as A0 | On | Same as A0 |
| A7: Qwen+LoRA | Controlled | Controlled | Controlled | Qwen+LoRA |
| A8: Mental-health LLM | Controlled | Controlled | Controlled | Candidate LLM |
| A9: Alternative generator + RAG | Controlled | On | Controlled | Compatible alternative |

The table describes experimental intentions, not verified system capabilities. Resolve ambiguous configurations from the actual code.

Do not silently equate disabling spiritual grounding with removing all relevant cultural context. Define the operational meaning of every switch.

For fair component comparisons, hold all other variables constant.

### 9.2 Interaction effects

If resources permit, conduct a factorial experiment examining interactions such as:

- Emotion awareness × RAG.
- Emotion awareness × generator.
- RAG × generator.
- Spiritual grounding × RAG.

Use a full factorial design only if the number of configurations is practical. Otherwise, use a documented reduced design.

Estimate main effects and interaction effects where the design supports them.

Do not interpret differences between experiments as causal if the conditions changed in uncontrolled ways.

### 9.3 Report incremental value

For each ablation, report:

- Metric values.
- Absolute difference from the relevant baseline.
- Relative difference when meaningful and numerically valid.
- Confidence intervals.
- Statistical test and adjusted p-value, if applicable.
- Sample size.
- Runtime and memory changes.
- Representative qualitative examples.
- Failure cases and limitations.

A component should not be described as beneficial merely because one metric improved. Consider safety, grounding, response quality, and efficiency together.

---

## 10. Phase H — Evaluate end-to-end chatbot performance

Evaluate the actual application pipeline from user input to final response.

Use the real orchestration code and configured models, not an approximate implementation that bypasses key modules.

Test both isolated single-turn questions and multi-turn conversations.

### 10.1 Evaluation dimensions

Assess the following dimensions using a documented scoring rubric.

**Response quality**
- Relevance to the user's stated concern.
- Helpfulness and practical usefulness.
- Clarity and coherence.
- Completeness without unnecessary verbosity.
- Consistency across turns.
- Appropriate follow-up questions.

**Emotional intelligence**
- Correct interpretation of the user's emotional state.
- Empathy and validation.
- Appropriate tone.
- Avoidance of emotional overconfidence.
- Responsiveness to changes in emotional context.

**Cultural and spiritual grounding**
- Accuracy of spiritual or textual references.
- Relevance to the question.
- Correct representation of the source's meaning.
- Respectful and non-coercive presentation.
- Appropriate use of secular alternatives.
- Avoidance of fabricated quotations or unsupported scriptural claims.
- Respect for users who do not want spiritual guidance.

**RAG and factual reliability**
- Evidence support for factual claims.
- Correct source attribution.
- Appropriate uncertainty.
- Avoidance of fabricated citations.
- Appropriate admission when the knowledge base does not contain an answer.

**Conversation quality**
- Context retention.
- Avoidance of repetitive advice.
- Consistency with previous turns.
- Responsiveness to user corrections.
- Correct use of available conversation history.

**Safety**
- Recognition of crisis signals.
- Appropriate handling of self-harm disclosures.
- Avoidance of harmful or coercive advice.
- Appropriate boundaries around diagnosis and treatment.
- Suitable recommendations to seek qualified human support.
- Safe handling of sensitive personal information.

**Operational performance**
- End-to-end latency.
- Component-level latency.
- Peak memory usage, where measurable.
- Failed requests.
- Model loading time.
- Throughput, if supported.
- Stability over repeated calls.

### 10.2 Scoring rubric

Create a versioned rubric, preferably using a 1–5 scale for qualitative dimensions, with explicit behavioral anchors.

For example:

- **1 — Poor:** materially incorrect, irrelevant, insensitive, or unsupported.
- **2 — Weak:** partially relevant but with important shortcomings.
- **3 — Adequate:** generally acceptable, with identifiable limitations.
- **4 — Good:** relevant, clear, empathetic, and well-supported.
- **5 — Excellent:** highly relevant, nuanced, appropriate, and well-supported.

Refine the anchors for each dimension. A score of 5 for empathy must mean something different from a score of 5 for factual faithfulness.

Safety must not be reduced to a single average score. Define critical failure categories separately.

### 10.3 Human evaluation

Design a blinded human evaluation protocol where feasible.

Include:

- Anonymized or appropriately de-identified test cases.
- Randomized model response ordering.
- Multiple independent raters when possible.
- A written scoring rubric.
- Inter-rater agreement, such as Cohen's kappa for suitable two-rater categorical ratings or Krippendorff's alpha for suitable multi-rater settings.
- Adjudication procedures for disagreements.
- Rater instructions and limitations.
- An annotation file that can be reused.

Do not claim human evaluation was performed until actual human ratings exist.

If only one researcher is available, label the evaluation as single-rater exploratory assessment.

Use qualified clinical reviewers for clinical judgments where feasible. Do not treat a general-purpose LLM judge as a substitute for clinical expertise.

---

## 11. Phase I — Safety and crisis evaluation

Safety is a critical requirement, not an optional performance metric.

Inspect the existing `safety.py`, crisis detection, Indian helpline responses, and any routing or refusal logic before designing safety tests.

**Never remove, weaken, bypass, or silently change existing safety mechanisms to improve another metric.**

### 11.1 Safety test categories

Create a structured test suite covering:

- Ordinary emotional distress.
- Sadness, grief, loneliness, and anxiety.
- Ambiguous expressions of hopelessness.
- Explicit self-harm disclosures.
- Suicidal ideation and crisis-related messages.
- Requests for dangerous self-harm instructions.
- Users describing immediate danger.
- Repeated crisis disclosures across turns.
- Crisis language expressed indirectly or in varied wording.
- Requests for diagnosis or medication instructions.
- Requests beyond the chatbot's intended scope.
- Spiritual interpretations of distress.
- Users who decline spiritual guidance.
- Attempts to override safety instructions.
- Hallucinated or invalid helpline details.
- Contradictions between safety rules and generated content.

Use safe, controlled test fixtures. Do not use real vulnerable people as experimental subjects without appropriate approvals and safeguards.

### 11.2 Safety metrics

Depending on available annotations and test design, calculate:

- Crisis detection recall.
- Crisis detection precision.
- False-negative rate.
- False-positive rate.
- Correct escalation or response rate.
- Unsafe response rate.
- Critical safety failure count.
- Appropriate refusal rate.
- Safe completion rate for benign requests.
- Helpline correctness rate, where independently verified.
- Safety regression count compared with the original system.

Report each critical category separately.

A high average safety score must not conceal even a small number of serious failures.

Use a predefined acceptance threshold for critical safety tests. If any critical regression occurs, flag it clearly and do not recommend deployment until it is reviewed.

### 11.3 Safety preservation and regression

Run the original safety tests before changes and after any evaluation-related integration change.

If the project has no tests, create a baseline test suite around the current behavior before changing the implementation.

Do not modify safety logic merely to make the test suite pass. Report actual failures and recommendations separately.

Do not provide real-time crisis intervention claims or imply that this research evaluation establishes clinical effectiveness.

---

## 12. Phase J — Compare results with the three research papers

This section is mandatory.

Create a paper-by-paper comparison based on the actual uploaded documents.

### 12.1 Spiritual-LLM paper

Inspect the paper's actual methodology and results.

Determine which experiments concern:

- Gita-inspired mental-health responses.
- Model architecture and adaptation.
- Datasets and task definitions.
- Baseline model comparisons.
- Response quality or other reported evaluation dimensions.
- Human or automated evaluation.
- Reported limitations.

Extract the exact metrics and values reported in the paper.

Determine which of these can be reproduced with the current system and available data.

Do not assume that the current chatbot implements the same architecture or training procedure.

### 12.2 SpiritRAG paper

Inspect the paper's actual RAG design and experimental methodology.

Extract only the retrieval, question-answering, grounding, or other metrics actually reported.

Identify differences between the paper's information domain and the chatbot's own knowledge sources.

Determine whether its retrieval evaluation protocol can be adapted to the current system.

If the original dataset, relevance judgments, or implementation are unavailable, distinguish an adapted local experiment from a faithful reproduction of the paper.

Do not claim that the chatbot reproduces SpiritRAG's results simply because both systems use retrieval and religion-related content.

### 12.3 Ramayana-based counselling module paper

Inspect the paper's actual counseling-module development and validation methods.

Extract its reported validation process, evaluator roles, criteria, sample sizes, and numerical results where available.

Determine whether its methods can inform evaluation of cultural relevance, appropriateness, content validation, or counseling-related quality in the chatbot.

Do not equate validation of a Ramayana-based counselling module with validation of an LLM or chatbot.

Do not transfer clinical efficacy claims from the paper to this system.

If the paper reports content-validation findings rather than model benchmark scores, compare the evaluation methodology and relevant criteria—not unrelated numerical metrics.

### 12.4 Evidence-based comparison table

Create a table with these columns:

| Reference paper | Published metric or finding | Published value | Original conditions | Local equivalent metric | Local result | Direct comparability | Explanation |
|---|---|---|---|---|---|---|---|

Use these rules:

- Use the exact published value only after verifying it in the paper.
- Record the correct dataset, sample size, baseline, and evaluation conditions.
- Mark unreported values as `Not reported`.
- Mark unavailable local measurements as `Not evaluated`.
- Mark incompatible comparisons as `Not directly comparable`.
- Distinguish `Faithful reproduction`, `Adapted reproduction`, `Related local experiment`, and `Qualitative comparison`.
- Include page, section, table, or figure references for published evidence.
- Do not manufacture a common score by averaging unrelated metrics.
- Do not compare a paper's training result with the chatbot's test result without explicitly explaining the difference.
- Do not claim superiority based on incomparable experiments.

If web access is useful for locating official code, datasets, or paper versions, clearly separate externally obtained evidence from the uploaded papers. Do not replace the uploaded papers with different documents without explanation.

---

## 13. Phase K — Statistical analysis

Use statistical methods appropriate to the metric, test set, and experimental design.

### 13.1 Confidence intervals

Where feasible, calculate confidence intervals for:

- Classification metrics.
- Retrieval metrics.
- Generation evaluation metrics.
- Human evaluation scores.
- Safety detection rates.
- End-to-end response-quality scores.

Use bootstrap confidence intervals where appropriate. For classification, use a suitable stratified or paired resampling method when needed.

Document the method and number of resamples.

### 13.2 Paired model comparisons

Because the models will often answer the same test questions, prefer paired comparisons.

Depending on the metric and design, consider:

- Paired bootstrap tests.
- McNemar's test for paired classification outcomes.
- Wilcoxon signed-rank tests for suitable paired score distributions.
- Paired permutation tests.
- Appropriate tests for proportions.
- Mixed-effects models if the number of raters, prompts, or repeated observations justifies them.

Correct for multiple comparisons when conducting many hypothesis tests, using a suitable method such as Holm's procedure.

Report:

- Effect size.
- Confidence interval.
- Test statistic, when relevant.
- p-value.
- Adjusted p-value, when applicable.
- Sample size.
- Assumptions and limitations.

Do not run a statistical test that does not match the data or experimental design.

Do not describe a result as statistically significant without the corresponding valid analysis.

If sample sizes are too small, report descriptive results and explain the limitation.

### 13.3 Practical significance

Distinguish statistical significance from practical importance.

Discuss whether improvements justify additional latency, memory usage, complexity, or safety risk.

Do not rank models using a single arbitrary weighted average unless the weighting scheme is justified, defined in advance, and reported transparently.

---

## 14. Phase L — Efficiency and resource evaluation

The development environment may include a Windows 11 laptop with an Intel i5-1235U CPU, 16 GB RAM, and integrated Intel Iris Xe graphics. GPU evaluation may require a separate Colab or Kaggle runtime.

Verify the current environment rather than assuming the hardware is unchanged.

### 14.1 Measure

Where technically feasible, record:

- Model initialization time.
- Warm-up time.
- Mean and median inference latency.
- p90 and p95 latency, if the number of observations supports them.
- Throughput.
- Input and output token counts.
- Peak CPU RAM.
- Peak GPU memory.
- RAG retrieval latency.
- Time spent in emotion classification.
- Time spent in generation.
- Total end-to-end latency.
- Failed and timed-out requests.

State the hardware, precision, batch size, concurrency, warm-up procedure, and number of measured runs.

Separate cold-start measurements from steady-state measurements.

Do not compare latency values obtained under substantially different hardware conditions without clearly labeling them.

### 14.2 Resource constraints

- Do not assume that every 7B or 8B model can run locally.
- Do not load all candidate models into GPU memory simultaneously.
- Avoid unnecessary full-dataset generation runs before a small pilot.
- Use configurable batch sizes and resumable evaluation.
- Release model resources between experiments.
- Support CPU execution where feasible.
- Detect CUDA availability rather than hardcoding it.
- Handle out-of-memory errors gracefully.
- Do not silently switch models after a failure.
- Save completed results incrementally so interrupted experiments can resume.

Use staged execution: a small smoke test, a pilot run, then the full evaluation.

---

## 15. Phase M — Automated evaluation implementation

After completing the initial audit and designing the protocol, implement a modular evaluation framework.

Adapt the directory structure to the existing repository. Do not duplicate functionality unnecessarily.

A possible structure is:

```text
evaluation/
    README.md
    configs/
        evaluation.yaml
        model_configs.yaml
        dataset_configs.yaml
    audit/
        inspect_project.py
        inspect_datasets.py
        inspect_papers.py
    data/
        build_splits.py
        validate_data.py
        leakage_checks.py
    evaluators/
        emotion_evaluator.py
        generation_evaluator.py
        mental_health_llm_evaluator.py
        retrieval_evaluator.py
        grounded_answer_evaluator.py
        end_to_end_evaluator.py
        safety_evaluator.py
        efficiency_evaluator.py
    ablations/
        run_ablations.py
    statistics/
        confidence_intervals.py
        paired_tests.py
        multiple_comparisons.py
    reporting/
        generate_tables.py
        generate_plots.py
        generate_report.py
    tests/
        test_data_integrity.py
        test_model_loading.py
        test_metrics.py
        test_safety_regression.py
        test_reproducibility.py
    results/
        .gitkeep
```

This is a suggested layout, not a mandate. Reuse existing project conventions when practical.

### 15.1 Requirements

The framework must:

- Reuse the actual production inference functions where appropriate.
- Support individual component evaluation.
- Support the complete chatbot pipeline.
- Support configurable ablations.
- Use stable record IDs.
- Save raw predictions and intermediate outputs when permitted.
- Save metric summaries in CSV or JSON.
- Support interrupted-run recovery.
- Separate data preparation, inference, scoring, statistics, and reporting.
- Include input validation and meaningful error messages.
- Avoid hardcoded absolute paths.
- Avoid committing large generated files or private user data to version control.
- Make it easy to rerun one experiment without rerunning every model.
- Track skipped and failed examples.
- Keep original model artifacts read-only.
- Provide a dry-run or smoke-test mode.

### 15.2 Reuse existing evaluation scripts

If evaluation scripts already exist:

1. Inspect them.
2. Determine whether they are correct.
3. Reuse them where appropriate.
4. Avoid creating redundant implementations.
5. Document any required fixes.
6. Preserve existing results and provide a clear migration path if necessary.

Do not replace working code simply to conform to the suggested directory structure.

---

## 16. Phase N — Tests and validation

Implement tests for the evaluation framework itself.

Include:

- Dataset schema validation.
- Duplicate detection.
- Split reproducibility.
- Metric calculations against small, manually verifiable examples.
- Correct handling of missing labels.
- Correct handling of empty predictions.
- Correct multi-label versus single-label classification behavior.
- Model-loading failure handling.
- RAG evaluation with known relevant and irrelevant passages.
- Citation validation.
- Ablation switch correctness.
- Paired comparison alignment by record ID.
- Statistical analysis on synthetic test fixtures.
- Safety regression checks.
- Incremental result saving and recovery.
- Configuration validation.

Use mock models or small synthetic fixtures for unit tests when the actual model is too expensive to load.

Keep mock results clearly separate from real experimental results.

Run the tests and record the actual outcomes.

Do not report a test as passing unless it was executed successfully.

---

## 17. Phase O — Tables, graphs, and visual analysis

Generate publication-quality tables and graphs from the actual result files.

Every graph must be generated from saved measurements, not manually typed values.

### 17.1 Required tables

Produce, where applicable:

1. Project and model inventory.
2. Dataset quality and split summary.
3. Research-paper evidence matrix.
4. Emotion-classifier metrics.
5. Generator comparison.
6. Mental-health LLM comparison.
7. Retrieval metrics.
8. Grounded-answer metrics.
9. End-to-end quality scores.
10. Safety evaluation.
11. Ablation results.
12. Statistical comparison results.
13. Efficiency and resource measurements.
14. Paper-versus-local comparison.
15. Failure and limitation summary.

### 17.2 Required graphs

Produce relevant graphs such as:

- Per-class emotion F1 scores.
- Emotion confusion matrix.
- Precision-recall curves when suitable.
- Retrieval Recall@k and nDCG@k.
- Model comparison across evaluation dimensions.
- Ablation improvements over baseline.
- Confidence intervals for key comparisons.
- Latency comparison.
- Memory usage comparison.
- Safety performance by category.
- Error-category distribution.

Avoid plots that obscure uncertainty or make incompatible values look directly comparable.

Include titles, axis labels, metric definitions, units, sample sizes, and legends.

When a metric has no valid observations, omit the corresponding graph or explicitly indicate that the result is unavailable.

### 17.3 Visual quality

Use consistent formatting and readable labels. Save plots as high-resolution PNG and, where practical, vector PDF or SVG.

Do not use fabricated placeholders as though they were results.

---

## 18. Phase P — Research-style report

Generate a research-style report in Markdown and, if practical, PDF.

Suggested output:

`reports/evaluation/final_evaluation_report.md`

Optionally generate:

`reports/evaluation/final_evaluation_report.pdf`

The report must contain:

### Title
Comprehensive Evaluation of an Emotion-Aware, Culturally Grounded AI Counselling Chatbot

### Abstract
Summarize the objective, actual evaluation design, key measured findings, limitations, and conclusions. Write this only after results exist.

### 1. Introduction
Describe the system and research objectives based on the actual implementation.

### 2. System architecture
Document the real emotion classifier, generator, LoRA adapter, RAG pipeline, orchestration, and safety components.

### 3. Related work
Summarize the three uploaded papers accurately. Cite the actual paper sections, tables, and pages.

### 4. Research questions
List the research questions addressed by the experiments.

### 5. Datasets
Document sources, schemas, sizes, splits, label definitions, quality checks, and leakage prevention.

### 6. Experimental methodology
Explain model configurations, prompts, metrics, baselines, ablations, human evaluation, statistical methods, and hardware.

### 7. Results
Report measured results with tables, graphs, uncertainty estimates, and sample sizes.

### 8. Ablation studies
Analyze the contribution of emotion awareness, RAG, spiritual grounding, and generator choice.

### 9. Safety evaluation
Describe critical safety tests, outcomes, failure cases, and unresolved risks.

### 10. Comparison with prior work
Compare only genuinely comparable metrics and findings from the three papers.

### 11. Error analysis
Discuss recurring errors, representative examples, and likely causes.

### 12. Efficiency
Report runtime, latency, memory usage, and deployment constraints.

### 13. Discussion
Interpret the findings without overclaiming.

### 14. Limitations
Document missing data, unavailable models, small sample sizes, annotation limitations, judge bias, compute constraints, and non-comparable published results.

### 15. Conclusion
Summarize what the experiments establish and what remains unknown.

### 16. Future work
Recommend targeted improvements based on measured evidence.

### References
Include the three uploaded papers and any additional sources actually used.

### Appendices
Include the metric definitions, evaluation rubric, configuration details, additional tables, safety test categories, and reproduction instructions.

Use appropriate academic citations. Do not invent publication metadata, DOI values, authors, or references.

---

## 19. Phase Q — Reproducibility package

Create a complete reproduction guide:

`reports/evaluation/REPRODUCE.md`

It must specify:

- Required Python version.
- Dependency installation instructions.
- Environment variables and configuration.
- Model and adapter paths.
- Dataset paths.
- How to prepare the evaluation splits.
- How to run smoke tests.
- How to evaluate each individual model.
- How to evaluate RAG.
- How to run ablations.
- How to run end-to-end evaluation.
- How to run safety tests.
- How to generate statistical comparisons.
- How to generate tables and graphs.
- How to generate the final report.
- Expected output files.
- Approximate resource requirements based on observed measurements.
- How to resume an interrupted experiment.
- Known limitations and troubleshooting steps.

Use commands appropriate to the actual Windows environment and any supported Linux or Colab workflow.

Do not assume that a particular command works until it has been verified.

Create a single top-level command or script that can execute the complete evaluation in stages, while still allowing individual experiments to run independently.

---

## 20. Phase R — Execution order

Follow this sequence strictly.

### Stage 1: Audit

- Inspect project files.
- Read the three papers.
- Inspect all relevant datasets.
- Inspect current evaluation scripts.
- Record available hardware.
- Create the initial audit documents.

**Do not modify production code or model artifacts during this stage.**

### Stage 2: Experimental design

- Define research questions.
- Map paper metrics to local evaluation possibilities.
- Select valid datasets and splits.
- Define baselines and ablations.
- Define evaluation rubrics and safety criteria.
- Specify statistical methods.
- Create the evaluation configuration.

### Stage 3: Smoke tests

- Validate dataset loaders.
- Load each feasible model individually.
- Test one or a few examples per pipeline.
- Confirm that raw predictions and metric outputs are saved correctly.
- Check that safety mechanisms remain active.
- Estimate memory and runtime.

### Stage 4: Pilot evaluation

Run a small, representative pilot for each model and subsystem.

Check for:

- Broken inference paths.
- Incorrect label mappings.
- Data leakage.
- Invalid references.
- Memory failures.
- Incompatible output formats.
- Misconfigured ablations.
- Metric implementation errors.

Fix evaluation-code problems before the full run.

### Stage 5: Full evaluation

Run the complete feasible experiment matrix with reproducible configurations.

Save outputs incrementally. Do not silently skip failed experiments.

### Stage 6: Statistical analysis

Compute confidence intervals, paired comparisons, multiple-comparison adjustments, and effect sizes where justified.

### Stage 7: Reporting

Generate tables, graphs, the research report, the comparison with the three papers, and the reproduction guide.

### Stage 8: Final verification

- Rerun the automated tests.
- Verify all reported values against raw result files.
- Verify paper claims against the actual papers.
- Check that no scores were invented.
- Check that original model artifacts remain unchanged.
- Check that existing safety mechanisms have not regressed.
- Verify all reproduction commands and output paths.
- Produce a final implementation and experiment summary.

---

## 21. Acceptance criteria

The task is complete only when the following requirements have been addressed or explicitly marked as blocked.

- [ ] All three uploaded research papers have been inspected.
- [ ] The actual project architecture has been documented.
- [ ] All relevant datasets and chapter-wise QA files have been inspected.
- [ ] Existing evaluation scripts have been reviewed.
- [ ] The emotion classifier has been evaluated using metrics appropriate to its actual label schema.
- [ ] The Qwen+LoRA generator has been evaluated.
- [ ] The candidate mental-health LLM has been evaluated if accessible.
- [ ] The base Qwen model has been evaluated if accessible and relevant.
- [ ] RAG retrieval quality has been evaluated where suitable relevance judgments exist.
- [ ] Grounded generation has been evaluated where suitable evidence is available.
- [ ] Core ablation studies have been conducted where the system supports them.
- [ ] End-to-end chatbot responses have been evaluated.
- [ ] Safety and crisis handling have been tested without weakening safeguards.
- [ ] Efficiency measurements have been collected where feasible.
- [ ] Statistical comparisons have been performed where justified.
- [ ] Tables and graphs have been generated from actual results.
- [ ] A research-style report has been generated.
- [ ] Published-paper comparisons include exact evidence and comparability labels.
- [ ] A reproduction guide and configuration files have been created.
- [ ] Automated tests have been executed and their actual outcomes recorded.
- [ ] Original trained model and adapter artifacts remain intact.
- [ ] All incomplete experiments and limitations are documented.

If a requirement cannot be completed because of missing labels, unavailable models, inaccessible datasets, hardware constraints, or insufficient annotations, do not fabricate a result. Record the blocker, explain its effect, and provide a concrete next step.

---

## 22. Final response required from the coding agent

After completing the work, provide a concise but detailed implementation summary containing:

1. What you inspected.
2. Which three papers were analyzed.
3. The actual models and datasets evaluated.
4. Which experiments completed successfully.
5. Which experiments failed or remain incomplete.
6. The most important measured results.
7. The key ablation findings.
8. Any critical safety issues.
9. Which published results were directly comparable, adapted, or not comparable.
10. The location of the generated tables, graphs, raw results, and report.
11. The commands needed to reproduce the experiments.
12. The tests executed and their actual pass/fail outcomes.
13. Confirmation of whether original model artifacts and safety mechanisms were preserved.
14. The most important limitations and recommended next experiments.

Separate facts, measured results, interpretations, and recommendations.

**Final instruction:** Begin with Stage 1 only. Inspect the real project, the actual uploaded papers, the datasets, and existing evaluation code before making implementation changes. After the audit, proceed through the stages in order. Do not invent evidence, silently change the research question, overwrite model artifacts, weaken safety mechanisms, or claim that an experiment succeeded unless it actually ran and its results were verified.

</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-10-24T16:17:06+05:30.

The user's current state is as follows:
Active Document: d:\Agentic AI\configs\chatbot.yaml (LANGUAGE_YAML)
Cursor is on line: 1
Other open documents:
- d:\Agentic AI\configs\chatbot.yaml (LANGUAGE_YAML)
- d:\Agentic AI\src\qwen\inference.py (LANGUAGE_PYTHON)
- d:\Agentic AI\run_local.ps1 (LANGUAGE_POWERSHELL)
Browser State:
  Page 2D5CA485EA59D0B8D5FF3E0E9E21C978 (Culturally Grounded Counselling Chatbot) - http://localhost:8501/ [ACTIVE]
    Viewport: 1280x585, Page Height: 585
  Page 6B4CC7E4799A104650409750CCF172A9 (IND 40-0 (4.3) (Sanju Samson 14(16), Abhishek Sharma 22(13)) vs New Zealand 180-5 ((20.0)) 2nd-T20 | India tour of New Zealand 2026 - CREX) - https://crex.com/cricket-live-score/ind-vs-nz-2nd-t20i-india-tour-of-new-zealand...
    Viewport: 1280x585, Page Height: 2778
</ADDITIONAL_METADATA>
<USER_SETTINGS_CHANGE>
The user changed setting `Model Selection` from Gemini 3.8 Flash (Medium) to Claude Sonnet 4.6 (Thinking). No need to comment on this change if the user doesn't ask about it. If reporting what model you are, please use a human readable name instead of the exact string.
</USER_SETTINGS_CHANGE>