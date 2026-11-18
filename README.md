# Hallucination Early-Warning for RAG Answers

A sentence-level hallucination detector for answers that an AI writes **from supplied documents** (RAG).
It colours every sentence **green / amber / red** and shows the signals behind the colour, using three kinds of warning signs:

| Sign | What it measures | Model used (pre-trained, not fine-tuned) |
|---|---|---|
| **A. Word-level doubt** | How unsure the model is on each word (entropy of the next-word probabilities) | Qwen2.5-3B-Instruct |
| **B. Meaning-level doubt** (semantic entropy) | The model re-writes a sentence 5 times; do the versions mean the same? | Qwen2.5-3B-Instruct + DeBERTa-large-MNLI |
| **C. Document check** | Does the document support the sentence? (supported / not mentioned / contradicted) | DeBERTa-large-MNLI |

A small logistic regression (the only trained part) mixes the signs into one risk score per sentence.
Everything is evaluated against **human-marked hallucinations** in the RAGTruth dataset.

> **Honest scope.** This is a student project: a prototype and an evaluation built on published ideas
> (the MASE paper, semantic entropy, SelfCheckGPT-style checking). It is **not** a new detector, and all findings are
> **preliminary** (one small model, one dataset, 200 answers).
> The one question it adds: *should the document check, instead of word-level doubt, decide which sentences get the
> expensive re-writing?* It also reports calibration (ECE, Brier) and cost.

---

## Folder structure

```
hallucination-early-warning-rag/
├── README.md                          <- this file
├── requirements.txt                   <- Python libraries
├── .gitignore
├── notebooks/
│   └── hallucination_prototype_v3.ipynb   <- MAIN FILE (already executed; outputs are visible)
├── data/                              <- RAGTruth is downloaded here automatically (not committed)
├── results/
│   └── RESULTS_SUMMARY.md             <- all result tables (the CSV files go here after a run)
└── docs/                              <- project report, slides, demo screenshots (see docs/README.md)
```

**Main file to run:** `notebooks/hallucination_prototype_v3.ipynb`

---

## Requirements

- **Python:** 3.10 or newer. Tested on **Python 3.13** in Google Colab.
- **GPU:** an NVIDIA GPU with CUDA is required. Tested on a 15 GB Tesla T4 (free Colab). 12 GB or more is recommended.
- **Internet** for the first run: downloads RAGTruth and two models (about 8 GB in total).
- **Time:** the 200-answer run plus the two baselines takes a few hours on a T4; it saves progress every 5 answers and resumes.
- **Libraries** (see `requirements.txt`): torch, transformers, accelerate, scikit-learn, pandas, numpy, gradio, jupyterlab.
  The file lists **minimum** versions. To record the exact versions you used, run
  `pip freeze | grep -i -E "^(torch|transformers|accelerate|scikit-learn|pandas|numpy|gradio)=="`.

## Installation

### Option A: Google Colab (the tested way, easiest)

1. Upload `notebooks/hallucination_prototype_v3.ipynb` to Colab (or open it from GitHub with *File > Open notebook > GitHub*).
2. `Runtime > Change runtime type > T4 GPU`.
3. `Runtime > Run all`. The first cell installs the libraries.

Progress is saved to Google Drive (`MyDrive/hallucination_project`) because `USE_DRIVE = True`.
If Colab disconnects, run all cells again and it resumes. Use a different Google account and the files start from zero
unless you copy the saved files (`features_v2.csv`, `spans_v2.json`, `baseline_v2.csv`, `selfcheck_v2.csv`) into the same folder name.

### Option B: your own machine (Linux / Windows / macOS with an NVIDIA GPU)

```bash
git clone https://github.com/tejaswaroop282/hallucination-early-warning-rag.git
cd hallucination-early-warning-rag

python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate

# 1) install torch with CUDA from https://pytorch.org/get-started/locally/  (pick your CUDA version)
# 2) then the rest:
pip install -r requirements.txt

jupyter lab notebooks/hallucination_prototype_v3.ipynb
```

In the **settings cell** set `USE_DRIVE = False`. Optionally choose folders:

```bash
export HALLU_WORK=./work     # where progress and result files are saved (default ./work)
export HALLU_DATA=./data     # where RAGTruth is downloaded (default ./data)
```

> Only the Colab route has been tested end to end. The local route uses the same notebook (paths and the Google Drive step are
> switched automatically when the notebook is not running in Colab).

## Dataset

**RAGTruth** (Niu et al., ACL 2024): RAG answers from several AI models with human-marked hallucinated spans.
Source: <https://github.com/ParticleMedia/RAGTruth> (folder `dataset`).

- **Automatic (default):** the notebook downloads `response.jsonl` and `source_info.jsonl` into `data/` the first time.
- **Manual:** download those two files from the link above and put them in `data/` (or in the folder set by `HALLU_DATA`).

The notebook uses the **test split**, keeps answers with quality = "good", and picks answers alternately from the
hallucinated and the clean group (pool of 1,886 answers). With `N_RESP = 200` this gives 1,425 sentences.

The two models download automatically from Hugging Face on first use: `Qwen/Qwen2.5-3B-Instruct` and
`microsoft/deberta-large-mnli`. No account is needed (an optional `HF_TOKEN` only speeds up downloads).

---

## Train / test / run

There is **no large neural-network training** in this project: Qwen and DeBERTa are used as they are.
The only trained part is the small logistic-regression mixer. Everything runs from the one notebook.

| Step | Where in the notebook | What it does |
|---|---|---|
| **Smoke test** (about 10 min) | Settings cell: `N_RESP = 10`, `BASE_N = 5`, then Run all | Checks that everything works |
| **Compute signals** | Sections 1 to 5b | Scores every sentence (A, B, C) and runs the two baselines. Resumes if interrupted |
| **Train** | Section 6 | Fits the logistic-regression mixer for each method variant |
| **Test / evaluate** | Sections 6 to 11 | 5-fold cross-validation, split by answer (no answer is in both train and test): AUROC, AUPRC, F1, ECE, Brier, paired tests, gating, early warning, baselines |
| **Run the demo** | Section 13 | Starts a Gradio web app (a temporary public link is printed). Paste a document and question to see coloured sentences |
| **Save** | Section 14 | Writes all result tables to the work folder and downloads the main ones |

**Full run (default settings):** `N_RESP = 200`, `BASE_N = 100`, then `Runtime > Run all`.

Optional headless run on your own machine (runs every cell, including the demo cell; not tested outside Colab):

```bash
jupyter nbconvert --to notebook --execute notebooks/hallucination_prototype_v3.ipynb \
  --output executed.ipynb --output-dir results --ExecutePreprocessor.timeout=-1
```

## Output files (written to the work folder; copy them to `results/`)

| File | Content |
|---|---|
| `results_table.csv` | Main comparison of all signal combinations (AUROC, AUPRC, F1, ECE, Brier) |
| `per_task.csv` | Results per task type (QA, Summary, Data-to-text) |
| `paired_tests.csv`, `threshold_comparison.csv` | Does the expensive step help? Global vs per-task threshold |
| `four_boxes.csv`, `three_way.csv`, `three_way_categories.csv` | Confident/unsure vs document support; 3-way document check |
| `gating_results.csv`, `gating_difference.csv` | Which check should decide who gets the expensive re-writing |
| `early_warning.csv` | How often the alarm fires in the first words of a wrong span |
| `answer_level_vs_baseline.csv`, `cost_vs_baseline.csv`, `selfcheck_sentence_level.csv`, `selfcheck_answer_level.csv` | Comparison with the two re-implemented baselines, plus cost |
| `safety_rule.csv` | Effect of the "never show green if the document clearly does not back it" rule |
| `sentences_scored.csv`, `features_v2.csv`, `spans_v2.json`, `baseline_v2.csv`, `selfcheck_v2.csv` | Raw per-sentence and per-answer data |

---

## Results (200 answers, 1,425 sentences, 155 hallucinated = 10.9%)

Sentence level, 5-fold cross-validation grouped by answer. 95% ranges come from resampling answers.
All numbers are **preliminary**. Full tables are in [`results/RESULTS_SUMMARY.md`](results/RESULTS_SUMMARY.md).

| Method | AUROC (95% range) |
|---|---|
| A. word-level doubt only | 0.707 (0.66 to 0.76) |
| B. semantic entropy only | 0.591 (0.55 to 0.63) |
| D. re-write disagreement only | 0.665 (0.62 to 0.71) |
| C. document check only | 0.811 (0.78 to 0.84) |
| A + C (cheap, no re-writing) | 0.834 (0.81 to 0.86) |
| **Full prototype (A + B + C + D)** | **0.843 (0.82 to 0.87)**, F1 0.458, ECE 0.014, Brier 0.080 |

Main findings so far:

- The **direct document check is the strongest single signal**; it beats word-level doubt by +0.103 AUROC (range +0.043 to +0.165).
- The expensive sampling adds only a **small** gain over the cheap model A + C: +0.010 AUROC (range -0.004 to +0.023, not clearly different) and +0.029 AUPRC (range +0.003 to +0.059, clearly different but borderline).
- If only 40% of sentences get the expensive re-writing, **document-first** sends 86% of the hallucinated sentences to it
  (word-doubt-first: 66%, random: 41%). The final AUROC is nearly the same for both orders (difference under 0.01; clearly different only at the 10% setting).
- About 23% of the hallucinations are sentences where the model is **confident but the document does not support them**.
- **One threshold per task type** improves F1 from 0.432 to 0.458 (+0.025, range +0.010 to +0.044).
- A three-way document check (supported / contradicted / not mentioned) did **not** improve on the plain one.
- The early-warning idea holds only weakly: the alarm fires in the first word of a wrong span for 36% of spans (25% within five words) versus 10% for normal text. The paper's 73% was not reproduced.
- Re-implemented baselines (same 100 answers): the paper's whole-answer semantic entropy scores at chance (answer-level AUROC 0.472);
  the SelfCheck-style check scores 0.564 (answer level) and 0.666 (sentence level), against 0.826 and 0.853 for the full prototype.
  Re-writing every sentence costs about 1.6 times as much as the baselines; document-first gating at 40% costs about 0.6 times.
- The safety rule (never green when "not backed" > 0.9) raises recall but lowers precision and F1: it is a trade-off, not an improvement.

## Limitations

- One small model (Qwen2.5-3B) and one dataset (RAGTruth), balanced on purpose (more hallucinations than real use).
- RAGTruth marks claims as hallucinated when the document does not support them, even if they are true, so a document check
  measures something very close to the label definition. This favours the document check.
- The answers were written by other AI models; the doubt is measured by Qwen, not by the writer.
- Data-to-text makes up about half of the sentences; Summary has only 19 hallucinated sentences, so its ranges are wide.
- The ablation without any document information (A + B + D) was not run.
- Cost is counted as generated token slots, not seconds.

## Troubleshooting

- **"A CUDA GPU is required"**: switch on a GPU (Colab: `Runtime > Change runtime type`).
- **Out of memory**: lower `MAX_TOK` (for example 1000) or `N_SAMPLES` (for example 3) in the settings cell.
- **Colab disconnected**: run all cells again; saved answers are not repeated.
- **Download fails**: put `response.jsonl` and `source_info.jsonl` in `data/` by hand (see Dataset).
- **"resumed with ... answers" does not appear**: the saved files are in the wrong folder or have different names.

## References and credits

- C. Jiang, H. Zhang, R. Ye. *Multi-scale adaptive semantic entropy framework: hierarchical detection and early warning mechanism for large language model hallucinations.* Neurocomputing 695 (2026) 134000.
- C. Niu et al. *RAGTruth: A hallucination corpus for developing trustworthy retrieval-augmented language models.* ACL 2024, pp. 10862-10878.
- S. Farquhar et al. *Detecting hallucinations in large language models using semantic entropy.* Nature 630 (2024) 625-630.
- P. Manakul et al. *SelfCheckGPT: zero-resource black-box hallucination detection.* EMNLP 2023.
- Models: Qwen2.5-3B-Instruct (Qwen Team), DeBERTa-large-MNLI (Microsoft).
