# SMS Spam Detection — Experiment Results & Tracking

**Snapshot:** November 9, 2026. Source: user-uploaded Colab CSV outputs from the cleaned notebook. All metrics shown below are measured values, not estimates.

## Project experiment definitions

- **E01:** Original word-level TF-IDF + Logistic Regression baseline and original zero-shot/few-shot FLAN-T5 prompts (240 development SMS).
- **E02:** Revised prompts and four handcrafted few-shot examples, same checkpoint and development messages.
- **E03:** FLAN-T5 candidate-label log-probability scoring and character 3–5-gram TF-IDF + Logistic Regression.
- **E04:** Qualitative comparison on 12 *synthetic*, topic-labeled SMS examples, including political outreach; no verified binary spam labels.
- **E05:** Frozen E02/E03 methods evaluated on 300 *different* holdout SMS messages (263 ham, 37 spam). Not used for prompt development.

## E05 primary outcome (300 unseen messages)

| Method | Original F1 | Obfuscated F1 | Original accuracy | Original precision | Original recall | TP | FP | FN | TN |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Word TF-IDF + LR | 92.3% | 94.6% | 98.0% | 87.8% | 97.3% | 36 | 5 | 1 | 258 |
| E02 Zero-shot | 62.4% | 55.6% | 88.3% | 51.8% | 78.4% | 29 | 27 | 8 | 236 |
| E02 Few-shot | 34.9% | 34.7% | 54.0% | 21.1% | 100.0% | 37 | 138 | 0 | 125 |
| E03 FLAN-T5 Label Scoring | 78.1% | 70.0% | 95.3% | 92.6% | 67.6% | 25 | 2 | 12 | 261 |
| E03 Character TF-IDF + LR | 93.3% | 93.0% | 98.3% | 92.1% | 94.6% | 35 | 3 | 2 | 260 |

**Interpretation:** Character TF-IDF has the highest original-SMS F1 (93.3%), about 1.0 pp above word TF-IDF (92.3%). Difference should not be described as statistically significant without a paired uncertainty analysis. Word TF-IDF catches 36/37 spam but gives 5 false positives; character TF-IDF catches 35/37 with 3 false positives. Label scoring has only 2 false positives but misses 12 spam. E02 few-shot catches all spam but falsely flags 138/263 ham.

## Development vs unseen (original SMS spam F1)

| Model | Development E01–E03 (n=240) | E05 unseen (n=300) | Change (percentage points) |
|---|---:|---:|---:|
| Word TF-IDF + LR | 91.5% | 92.3% | +0.8 pp |
| E02 Zero-shot | 48.2% | 62.4% | +14.2 pp |
| E02 Few-shot | 31.9% | 34.9% | +3.0 pp |
| E03 FLAN-T5 Label Scoring | 73.1% | 78.1% | +5.0 pp |
| E03 Character TF-IDF + LR | 94.7% | 93.3% | -1.4 pp |

## E05 obfuscation outcomes

- **38/300** E05 messages were modified by the predefined substitution dictionary; the other 262 were unchanged.
- Word TF-IDF F1 **rose** from 92.3% to 94.6% on the full E05 sample. This is sample-specific, not evidence that obfuscation generally improves performance.
- Character TF-IDF F1 **fell slightly** from 93.3% to 93.0% on full E05, unlike its identical 94.7% scores on the development set.
- Candidate scoring F1 **fell** from 78.1% to 70.0%; zero-shot generation **fell** from 62.4% to 55.6%.
- On the *38 modified messages only*, the subset is not representative of the full test distribution: 22 spam / 16 ham. Report its metrics separately, not as main accuracy.
- Inference-time comparison caveat: the exported obfuscated runtime is **incremental runtime on only changed messages**, whereas original runtime is on all messages; these times cannot be compared directly.

## E04 modern SMS qualitative case study

12 synthetic messages: 2 personal, 4 political, 3 phishing, 3 commercial. Topic is not equivalent to a binary spam/ham ground-truth label; do not compute accuracy or F1 on these data.

| Category | N | Word TF-IDF says spam | E02 zero-shot says spam | E02 few-shot says spam | FLAN label scoring says spam | Character TF-IDF says spam |
|---|---:|---:|---:|---:|---:|---:|
| Personal | 2 | 0 | 0 | 0 | 0 | 0 |
| Political | 4 | 1 | 3 | 4 | 1 | 0 |
| Phishing | 3 | 2 | 3 | 3 | 1 | 1 |
| Commercial | 3 | 2 | 3 | 3 | 2 | 2 |

**Qualitative finding:** All four political-themed messages were called spam by E02 few-shot, but only one was called spam by candidate scoring. Because these examples include potentially solicited campaign reminders, category and consent context matter.

## Historical E01–E03 trace

Original model-development measurements are retained as separate historical outputs and as newly rerun per-experiment CSVs. The cleaned notebook preserves two named prompt functions so earlier prompt definitions are not overwritten. Keep E05 frozen; do not tune models on its outcomes.

## Suggested final-report storyline

1. Historical UCI SMS classification benchmark and class imbalance.
2. E01 initial prompts overpredicted spam; simple word-level Logistic Regression established a strong baseline.
3. E02 prompt refinements reduced false alarms for zero-shot prediction but few-shot remained biased toward spam.
4. E03 candidate scoring improved FLAN-T5 F1, and character-level features improved classical ML on the development set.
5. E04 demonstrated ambiguity of modern political/transactional SMS under a binary spam/ham definition (qualitative only).
6. E05 provided a separate held-out estimate and identified a precision–recall tradeoff across methods.

## Methodological qualifications to include

- E05 results reflect one fixed 300-message holdout sample with only 37 spam; point estimates have uncertainty.
- Models were chosen after inspecting the 240 development messages. E05 should remain untouched for any further tuning.
- Obfuscation uses a limited predetermined dictionary and only affects a subset of messages.
- UCI messages are historical; E04 is synthetic, small, and not representative of current campaign texts.
- Public pretrained FLAN-T5 may have seen parts of widely available public corpora during pretraining, which was not audited.
- Original vs obfuscated text are paired observations, not independent datasets.

## Next deliverables

- Final report: use E05 table and confusion-matrix counts as primary quantitative findings, E01–E03 as evolution, E04 as qualitative case study.
- Presentation: emphasize the 93.3% character TF-IDF F1 and the surprising few-shot false-positive tradeoff; do not claim a statistically proven edge over word TF-IDF.
- Keep single versioned ZIP export and one citations section at the end of Colab.
