# AI Semester Project — Experiment Log

**Project:** Evaluating Prompting Strategies for SMS Spam Detection  
**Course:** CSI-4130/5130 — Artificial Intelligence  
**Last updated:** 2026-11-03 (E01, E02, E02-R and E03 measured; E04 synthetic case study reviewed)  
**Purpose:** Preserve every experiment as a separate iteration for the final report, GitHub README, and presentation. Entries flagged **pending** are not measured results.

## Shared experimental setup

- **Dataset:** UCI SMS Spam Collection, exact duplicate text messages removed.
- **Split:** 3,870 training messages; 1,290 held-out testing messages; 240 messages sampled for evaluation, seed 42.
- **Evaluation class distribution:** 210 ham and 30 spam.
- **Obfuscation:** 48 of 240 evaluation messages actually modified (23 ham, 25 spam); other messages remained unchanged. Replacement patterns include `free → fr33`, `win → w1n`, `claim → cl@im`.
- **Traditional model:** TF-IDF word unigrams and bigrams + Logistic Regression with balanced class weights.
- **Language model:** `google/flan-t5-base`, greedy generation (zero-shot and few-shot), evaluated on exactly the same 240 messages as the baseline.
- **Reported metrics:** Accuracy, spam precision, spam recall, spam F1, invalid output count, inference time.
- **Colab environment as supplied:** Python 3.13.16; NVIDIA Tesla T4 (15 GiB); PyTorch 2.5.1+cu121; Transformers 5.20.2. These are user-reported installed versions, not independently reverified for each experiment.

## E01 — Initial prompting experiment (completed)

**Prompt configuration:** Original classification instruction; four few-shot examples sampled from the training split (two ham, two spam). Zero-shot had no examples. No fine-tuning.

### Primary comparison: full evaluation set (n=240)

| Model | Scenario | Accuracy | Spam precision | Spam recall | Spam F1 | Inference (s) |
|---|---|---:|---:|---:|---:|---:|
| Few-shot | Obfuscated | 16.2% | 13.0% | 100.0% | 23.0% | 209.17 |
| Few-shot | Original | 16.2% | 13.0% | 100.0% | 23.0% | 199.59 |
| TF-IDF + LR | Obfuscated | 97.5% | 92.9% | 86.7% | 89.7% | 0.01 |
| TF-IDF + LR | Original | 97.9% | 93.1% | 90.0% | 91.5% | 0.01 |
| Zero-shot | Obfuscated | 15.0% | 12.8% | 100.0% | 22.7% | 102.19 |
| Zero-shot | Original | 15.0% | 12.8% | 100.0% | 22.7% | 112.67 |

### Modified-message subset (n=48)

| Model | Scenario | Accuracy | Spam precision | Spam recall | Spam F1 | Inference (s) |
|---|---|---:|---:|---:|---:|---:|
| TF-IDF + LR | Original | 95.8% | 100.0% | 92.0% | 95.8% | 0.01 |
| Zero-shot | Original | 52.1% | 52.1% | 100.0% | 68.5% | 112.67 |
| Few-shot | Original | 52.1% | 52.1% | 100.0% | 68.5% | 199.59 |
| TF-IDF + LR | Obfuscated | 93.8% | 100.0% | 88.0% | 93.6% | 0.01 |
| Zero-shot | Obfuscated | 52.1% | 52.1% | 100.0% | 68.5% | 102.19 |
| Few-shot | Obfuscated | 52.1% | 52.1% | 100.0% | 68.5% | 209.17 |

**Output behavior on original messages:**

| Method | Predicted spam | Predicted ham |
|---|---:|---:|
| TF_IDF_+_LR_Original | 29 | 211 |
| Zero_shot_Original | 234 | 6 |
| Few_shot_Original | 231 | 9 |


**Observations:**

1. TF-IDF + Logistic Regression achieved **91.5%** spam F1 on original messages and **89.7%** on obfuscated messages (a decrease of approximately **1.9 percentage points**).
2. Zero-shot FLAN-T5 produced **22.7%** spam F1 on both conditions; few-shot produced **23.0%** on both. Both methods had 100% spam recall but very low spam precision because they classified nearly every message as spam.
3. Original zero-shot marked 234/240 messages spam; few-shot marked 231/240 spam. The predicted-spam rate greatly exceeded the true spam prevalence (30/240 = 12.5%).
4. An unchanged F1 after the keyword substitutions **does not establish robustness**: these prompting configurations already labeled almost everything as spam.
5. Of the 240 messages, only 48 were modified by the dictionary-based obfuscation. Report the changed-only subset separately to avoid dilution by 192 unmodified messages.
6. These observations describe the actual measured outputs; they do **not** establish the root cause of the LLM's spam-heavy predictions.

**Files received for E01:** `metrics.csv`, `metrics_changed_only.csv`, `message_predictions.csv`, `result_summary.txt`, `spam_f1_comparison.png` (preserve these as raw evidence).

## E02 — Revised prompting experiment (completed)

**Prompt changes from E01:** Zero-shot and few-shot share a rewritten classification instruction that explicitly contrasts scams/unsolicited promotions with routine conversations and appointments. Few-shot now uses four short handcrafted examples (two ham, two spam) rather than training-sampled examples. No model fine-tuning was performed. **Causal limitation:** Few-shot changed both instruction wording and example content; separate attribution to either change is not possible.

**Comparability:** Verified the E01 and E02 message-prediction files have identical 240 row-ordered SMS texts, labels, obfuscated texts, and `was_changed` indicators. The TF-IDF + LR predictions are also identical across experiments. Model and seed are unchanged according to uploaded summaries.

### E02 primary results (full evaluation set, n=240)

| Model | Scenario | Accuracy | Spam precision | Spam recall | Spam F1 | Time (s) |
|---|---|---:|---:|---:|---:|---:|
| TF-IDF + LR | Original | 97.9% | 93.1% | 90.0% | 91.5% | 0.01 |
| TF-IDF + LR | Obfuscated | 97.5% | 92.9% | 86.7% | 89.7% | 0.01 |
| Zero-shot | Original | 82.1% | 37.7% | 66.7% | 48.2% | 125.13 |
| Zero-shot | Obfuscated | 80.8% | 34.0% | 56.7% | 42.5% | 101.07 |
| Few-shot | Original | 46.7% | 19.0% | 100.0% | 31.9% | 150.98 |
| Few-shot | Obfuscated | 46.2% | 18.9% | 100.0% | 31.7% | 150.43 |

### Direct E01 vs E02 comparison (spam F1)

| Method | E01 original F1 | E02 original F1 | Change | E02 obfuscated F1 |
|---|---:|---:|---:|---:|
| TF-IDF + LR | 91.5% | 91.5% | +0.0 pp | 89.7% |
| Zero-shot | 22.7% | 48.2% | +25.5 pp | 42.5% |
| Few-shot | 23.0% | 31.9% | +8.9 pp | 31.7% |

### E02 prediction distributions on original messages

Ground truth: **30 spam / 210 ham** (12.5% spam). An always-ham classifier would have 87.5% overall accuracy, but zero spam recall/F1.

| Method | E01 predicted spam | E02 predicted spam | E02 TP | E02 FP | E02 TN | E02 FN |
|---|---:|---:|---:|---:|---:|---:|
| TF-IDF + LR | 29 | 29 | 27 | 2 | 208 | 3 |
| Zero-shot | 234 | 53 | 20 | 33 | 177 | 10 |
| Few-shot | 231 | 158 | 30 | 128 | 82 | 0 |

### Keyword obfuscation sensitivity

48/240 messages were modified (25 spam, 23 ham). Of the remaining 192, their original and obfuscated texts are identical. On all 240, zero-shot F1 dropped from **48.2%** (original) to **42.5%** (obfuscated); few-shot from **31.9%** to **31.7%**; and TF-IDF+LR from **91.5%** to **89.7%**.

**Only the 48 modified messages:**

| Method | E02 original F1 (changed only) | E02 obfuscated F1 (changed only) | Change |
|---|---:|---:|---:|
| TF-IDF + LR | 95.8% | 93.6% | -2.2 pp |
| Zero-shot | 80.9% | 72.7% | -8.1 pp |
| Few-shot | 74.6% | 73.5% | -1.1 pp |

**How many predictions actually changed between the E02 original and obfuscated text:**

| Method | Prediction changes under obfuscation | Newly missed spam |
|---|---:|---:|
| TF-IDF + LR | 1 | 1 |
| Zero-shot | 7 | 4 |
| Few-shot | 1 | 0 |

**Analysis and key findings:**

1. **Zero-shot benefited most from the prompt revision.** F1 increased by **25.5 percentage points**. Its predicted-spam count dropped from **234/240** to **53/240**. False positives fell from **204** to **33**, while true positives dropped from **30** to **20**, revealing a precision-versus-recall trade-off.
2. **Few-shot improved but remained spam-heavy.** F1 increased by **8.9 percentage points**, and it now marks **158/240** messages as spam rather than **231/240**. It detects all 30 spam messages but incorrectly flags 128/210 legitimate messages.
3. **Zero-shot exceeded few-shot on the E02 sample** (48.2% vs 31.9% original F1). Four handcrafted examples are not inherently better than no examples for this model/prompt; results may change with example choice or prompt design.
4. **The traditional model continues to dominate:** TF-IDF+LR achieved 91.5% original F1 with an inference time recorded as 0.01 s, whereas LLM calls took about 101–151 s for each 240-message scenario. Time measurements reflect different computational approaches and execution context; do not treat as a controlled hardware benchmark.
5. **Obfuscation now degrades zero-shot F1:** -5.7 percentage points overall, and -8.2 percentage points on the 48 actually modified messages. This is a more informative robustness finding than E01's unchanged F1 (which resulted from excessive spam predictions).
6. The evidence supports **improvement associated with revised prompts**, but not a universal claim that prompt wording alone caused all changes or that the evaluated model is generally weak on spam detection. The sample is small and has only 30 spam texts.

**Received artifacts (E02):** `metrics(1).csv`, `metrics_changed_only(1).csv`, `message_predictions(1).csv`, `result_summary(1).txt`, `spam_f1_comparison(1).png`. The files with `(1)` are E02; preserve E01's unnumbered exports separately.

## E02-R — Reproduction of improved-prompt configuration (completed, November 3, 2026)

Two uploaded ZIP exports contained **exactly matching** `metrics(2).csv` / `metrics(3).csv`, `message_predictions(2).csv` / `message_predictions(3).csv`, and their original/obfuscated plots. They represent duplicate downloads from one Colab session, not two different experiments. Their metrics and 240 row-level predictions are identical. Compared with E02's earlier run, the predictions and evaluation metrics remain unchanged; inference times vary.

| Method | Scenario | Spam F1 | Accuracy | Inference (s) |
|---|---|---:|---:|---:|
| Few-shot | Obfuscated | 0.3175 | 0.4625 | 194.17 |
| Few-shot | Original | 0.3191 | 0.4667 | 192.02 |
| TF-IDF + LR | Obfuscated | 0.8966 | 0.9750 | 0.02 |
| TF-IDF + LR | Original | 0.9153 | 0.9792 | 0.02 |
| Zero-shot | Obfuscated | 0.4250 | 0.8083 | 126.14 |
| Zero-shot | Original | 0.4819 | 0.8208 | 126.88 |

**Interpretation:** E02 reproduction is stable at the prediction level. No improvement over E02 is established from these uploads alone. The unmodified comparison chart contains E02's three models, **not** E03's added methods.

## E03 — Alternative classification techniques (completed; November 3, 2026)

**Methods:** (a) FLAN-T5 conditional candidate-label log-likelihood scoring over `ham`/`spam` using the E02 zero-shot prompt; (b) character 3–5 gram TF-IDF + class-balanced Logistic Regression. No LLM fine-tuning. Both were evaluated on the same 240 held-out messages as prior experiments. Uploaded source files: `metrics_e03.csv`, `metrics_e03_changed_only.csv`, `predictions_e03.csv`, `metrics_all_methods.csv`.

### Full 240-message evaluation

| Method | Scenario | Accuracy | Spam precision | Spam recall | Spam F1 |
|---|---|---:|---:|---:|---:|
| FLAN-T5 Label Scoring | Original | 94.2% | 86.4% | 63.3% | 73.1% |
| Character TF-IDF + LR | Original | 98.8% | 100.0% | 90.0% | 94.7% |
| FLAN-T5 Label Scoring | Obfuscated | 93.3% | 85.0% | 56.7% | 68.0% |
| Character TF-IDF + LR | Obfuscated | 98.8% | 100.0% | 90.0% | 94.7% |

### Changed-only 48-message evaluation

| Method | Scenario | Accuracy | Spam F1 |
|---|---|---:|---:|
| FLAN-T5 Label Scoring | Original | 83.3% | 81.0% |
| Character TF-IDF + LR | Original | 97.9% | 98.0% |
| FLAN-T5 Label Scoring | Obfuscated | 79.2% | 75.0% |
| Character TF-IDF + LR | Obfuscated | 97.9% | 98.0% |

### Row-level error analysis and controlled comparisons

- **Character TF-IDF + LR:** 27 true-positive spam detections, 3 missed spam, zero false-positive spam flags, 210 true-negative ham messages on both original and obfuscated texts. **94.7% spam F1** in both conditions; **97.96% F1** on only the 48 altered messages. Its **240 predictions are identical** between original and obfuscated variants, including the 48 modified messages. This supports stability against this limited substitution set, *not* general adversarial robustness.
- **Word TF-IDF + LR (existing baseline):** 91.5% original F1, 89.7% obfuscated F1. Character features improved original F1 by **3.2 percentage points** and obfuscated F1 by **5.1 percentage points** on this sample.
- **FLAN-T5 Label Scoring:** Original: TP 19, FN 11, FP 3, TN 207; original spam F1 **73.1%**, accuracy **94.2%**. Obfuscated: TP 17, FN 13, FP 3, TN 207; F1 **68.0%**, accuracy **93.3%**. Both altered predictions were spam messages correctly flagged before substitution but labeled ham afterward. These two substitutions include `offer → 0ffer` and `claim → cl@im` / `call → c@ll` in individual messages.
- **Compared with E02 zero-shot text generation**, candidate-label scoring increased original F1 from 48.2% to 73.1%, **+24.9 percentage points**; however original spam recall was slightly lower (66.7% → 63.3%). This isolates *classification procedure* more clearly than E01→E02, though procedures' timing/thresholds are not matched.
- Inference time for label scoring was **not retained** by E03's export function (null/NaN); **do not claim a speed comparison**. Char-model reported inference (0.08 s) also excludes training time and is not a controlled runtime benchmark.
- **Important limitation:** E02/E03 configuration choices were assessed on a reused 240-message development/test sample. Report these as exploratory/development comparisons. A final, untouched held-out set should assess generalization before claiming a final best model. Sampling variability is meaningful with only 30 positive cases.

## E04 — Modern SMS challenge set (completed, qualitative only)

**Data:** 12 synthetic texts written for the project, organized into 2 personal, 4 political, 3 phishing, and 3 commercial examples. The `topic` field is a topical category, **not a ground-truth ham/spam label**. Predictions are recorded in `modern_sms_case_study(1).csv`.

| Topic (count) | Word TF-IDF flagged spam | Zero-shot flagged spam | Few-shot flagged spam | Candidate scoring flagged spam |
|---|---:|---:|---:|---:|
| personal (2) | 0 | 0 | 0 | 0 |
| political (4) | 1 | 3 | 4 | 1 |
| phishing (3) | 2 | 3 | 3 | 1 |
| commercial (3) | 2 | 3 | 3 | 2 |

- Political SMS are not intrinsically spam or fraudulent: a legitimate subscribed reminder, unsolicited campaign solicitation, and an impersonation scam may warrant different treatment. Evaluate consent and fraud signals separately where ground truth can be assigned.
- The few-shot model flagged all four synthetic political messages as spam, including an opt-in reminder; candidate scoring flagged one of four.
- Candidate scoring flagged only one of three simulated phishing messages as spam; the few-shot and zero-shot generation models flagged all three. **No accuracy, recall, or F1 should be calculated** on this case study without a separately documented annotation rubric and labels.
- Future work: a larger, privacy-preserving, independently labeled and untouched modern-message evaluation set. Distinguish **topic** (`political`, `personal`, `commercial`, `phishing`) from **spam status** (`solicited`, `unsolicited`, `fraudulent`) as distinct questions.

## Report/presentation narrative (measured vs qualitative)

1. E01 established a severe spam-prediction bias in FLAN-T5's initial prompts (22.7–23.0% F1), with word TF-IDF achieving 91.5%.
2. E02 revised instructions/examples, raising zero-shot F1 to 48.2% and few-shot to 31.9%; interpretation requires noting both instructions and example choice changed.
3. E03 tested new decision methods; candidate scoring reached 73.1% F1, character TF-IDF reached 94.7% F1 with no false-positive cases on the 240-message evaluation sample.
4. Obfuscation experiments modified only 48 messages. Compare both full-sample and changed-only metrics. Character TF-IDF was stable for these substitutions; label scoring lost two spam detections.
5. E04 provides twelve **illustrative synthetic examples**, not a quantitative performance benchmark. Political outreach should not be equated with fraud.
6. Final step: frozen method selection and a fresh untouched test split, ideally with F1 uncertainty estimates, before writing final claims of superiority.
