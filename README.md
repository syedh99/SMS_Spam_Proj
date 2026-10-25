# Evaluating LLM Prompting Strategies for SMS Spam Detection

**CSI-4130/5130 Artificial Intelligence Course Project**

## Project overview / problem statement
SMS spam is an everyday classification problem. This project evaluates whether a small instruction-tuned language model can distinguish `spam` from `ham` using zero-shot and few-shot prompts. It compares both strategies with a traditional TF-IDF + logistic regression classifier, then measures their robustness to visually disguised spam keywords. This is a **model evaluation project**, not an LLM fine-tuning project.

## Research questions
1. How does FLAN-T5-base with zero-shot prompting compare against a conventional supervised classifier?
2. Does few-shot prompting change spam precision, recall, or F1?
3. How do all three methods respond when common SMS keywords are visually obfuscated?

## Data
- **UCI SMS Spam Collection** (public historical research dataset): https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- Labels: `ham`, `spam`.
- Preprocessing: remove exact duplicate messages; stratified 75% training / 25% held-out test split; sample a reproducible subset of the test partition for the LLM comparison.
- The same test messages are evaluated for all methods. Few-shot examples are drawn exclusively from the training partition.

## Models and approaches
- **Baseline:** TF-IDF with unigram and bigram features + scikit-learn logistic regression with balanced class weights.
- **LLM:** Google's open `google/flan-t5-base`, using deterministic generation (`do_sample=False`).
- **Prompting:** zero-shot and four-example few-shot prompts.
- **Robustness:** substitute tokens such as `free` -> `fr33`; evaluate original and altered messages as paired cases. Also analyze only the changed examples to avoid diluting effects.

## How to run (Google Colab)
1. Open [Google Colab](https://colab.research.google.com/).
2. Upload `SMS_Spam_Prompting_Project.ipynb` using **File > Upload notebook**.
3. Choose **Runtime > Change runtime type > T4 GPU**.
4. Click **Runtime > Run all**. The notebook installs `sentencepiece` and scikit-learn if needed, automatically downloads the UCI data and open model weights, and runs all comparisons.
5. At the end, download `project_results.zip`, which contains a summary, metric tables, message-level predictions, and a chart.
6. Change `DEMO_MESSAGE` in Section 12 to demonstrate the three methods with a new SMS.

**Hardware:** tested by design for Colab T4 (16 GB VRAM), Python 3.13, PyTorch 2.5.1, Transformers 5.20.2. It has **not been executed end-to-end in the author's environment**, so record and investigate any issues observed in your actual Colab run. No paid API key or fine-tuning is required.

**Speed/scale:** `EVAL_SAMPLES=240` is the default illustrative run. Increase to 400–800 for more reliable estimates if your runtime budget permits. Results depend on the actual run, model downloads, model version, and sampled test data.

## Expected outputs
The notebook creates a `project_results/` folder containing:
- `metrics.csv`: precision, recall, F1, accuracy, invalid outputs, execution time for every comparison.
- `metrics_changed_only.csv`: metrics on obfuscation-affected SMS only.
- `message_predictions.csv`: held-out SMS, labels, modifications, and method predictions.
- `spam_f1_comparison.png`: grouped F1 chart.
- `result_summary.txt`: compact experiment summary.

**No quantitative results are reported in this README until the notebook has actually been run.** Add measured findings after completing experiments.

## Limitations
Historical and predominantly English-language SMS; small evaluation subset; potential memorization of public benchmark data by pretrained models; deterministic and narrow obfuscation dictionary; one LLM checkpoint; uncertain generalization and no production-grade security claim.

## Repository progress plan
Commit the project incrementally rather than uploading everything at once:
1. Project proposal and this README.
2. Dataset loading, cleaning and train/test split.
3. TF-IDF baseline and FLAN-T5 prompt variants.
4. Obfuscation experiment, quantitative evaluation, and error analysis.
5. Results, final report, charts, and presentation link.

## Project report & presentation
See `REPORT_TEMPLATE.md` and `PRESENTATION_OUTLINE.md`. Insert *measured* results and link to your recorded presentation when complete.

## Citations and acknowledgments
- Almeida, T. A., Hidalgo, J. M. G., and Yamakami, A. (2011). *Contributions to the Study of SMS Spam Filtering: New Collection and Results*. ACM DocEng. https://archive.ics.uci.edu/dataset/228/sms+spam+collection
- Chung, H. W., et al. (2022). *Scaling Instruction-Finetuned Language Models*. https://arxiv.org/abs/2210.11416
- Google FLAN-T5-base model card and license (Apache-2.0): https://huggingface.co/google/flan-t5-base
- Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR 12, 2825–2830. https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html

**Acknowledgment:** Credit any outside technical or writing assistance you actually use, including AI-assisted drafting or debugging, in accordance with your instructor's academic-integrity policy. Independently validate and explain your submitted code, experiments, and conclusions.
