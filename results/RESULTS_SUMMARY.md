# Results summary

Transcribed from the outputs of the executed notebook `notebooks/hallucination_prototype_v3.ipynb`
(200 answers, 1,425 sentences, 155 hallucinated = 10.9%; 5-fold cross-validation grouped by answer; 95% ranges from resampling answers).
The raw CSV files are written to the work folder when the notebook runs; copy them into this folder.
All findings are preliminary (one small model, one dataset).

## 1. Signal comparison (ablation study)

| Method | AUROC (95% range) | AUPRC | Precision | Recall | F1 | ECE | Brier |
|---|---|---|---|---|---|---|---|
| A word-level doubt | 0.707 (0.66-0.76) | 0.234 | 0.204 | 0.555 | 0.299 | 0.011 | 0.092 |
| B semantic entropy | 0.591 (0.55-0.63) | 0.134 | 0.139 | 0.645 | 0.228 | 0.005 | 0.096 |
| D re-write disagreement | 0.665 (0.62-0.71) | 0.164 | 0.164 | 0.652 | 0.262 | 0.002 | 0.094 |
| C document check | 0.811 (0.78-0.84) | 0.313 | 0.254 | 0.690 | 0.372 | 0.005 | 0.084 |
| C3 three-way document check | 0.807 (0.78-0.84) | 0.291 | 0.254 | 0.735 | 0.378 | 0.011 | 0.085 |
| A + B | 0.706 (0.66-0.76) | 0.230 | 0.197 | 0.381 | 0.260 | 0.010 | 0.092 |
| A + C | 0.834 (0.81-0.86) | 0.335 | 0.316 | 0.671 | 0.430 | 0.023 | 0.083 |
| A + C3 | 0.831 (0.80-0.86) | 0.328 | 0.307 | 0.619 | 0.410 | 0.018 | 0.083 |
| B + C | 0.822 (0.80-0.85) | 0.356 | 0.290 | 0.594 | 0.390 | 0.016 | 0.082 |
| A + B + C | 0.835 (0.81-0.86) | 0.348 | 0.333 | 0.639 | 0.438 | 0.010 | 0.082 |
| A + B + C3 | 0.833 (0.81-0.86) | 0.345 | 0.340 | 0.626 | 0.441 | 0.016 | 0.082 |
| **Full: A + B + C + D** | **0.843 (0.82-0.87)** | 0.362 | 0.341 | 0.697 | 0.458 | 0.014 | 0.080 |
| Full with three-way check | 0.841 (0.81-0.87) | 0.358 | 0.332 | 0.710 | 0.453 | 0.015 | 0.081 |

Random guessing: AUROC 0.50, AUPRC about 0.109.

## 2. By task type (AUROC)

| Task | Sentences | Hallucinated | A | B | C | Full |
|---|---|---|---|---|---|---|
| Data-to-text | 743 | 99 | 0.712 | 0.580 | 0.772 | 0.814 |
| QA | 383 | 37 | 0.722 | 0.611 | 0.856 | 0.881 |
| Summary | 299 | 19 | 0.766 | 0.625 | 0.763 | 0.839 |

## 3. Paired tests (what each component adds)

| Question | AUROC difference (95% range) | Clear? | AUPRC difference (95% range) | Clear? |
|---|---|---|---|---|
| Full vs A+C (does the expensive sampling help?) | +0.010 (-0.004 to +0.023) | no | +0.029 (+0.003 to +0.059) | yes |
| A+B+C vs A+C (does semantic entropy add?) | +0.001 (-0.009 to +0.011) | no | +0.015 (-0.004 to +0.035) | no |
| A+C vs C (does word doubt add?) | +0.023 (+0.007 to +0.041) | yes | +0.022 (-0.031 to +0.073) | no |
| C vs A (document check vs word doubt) | +0.103 (+0.043 to +0.165) | yes | +0.077 (+0.015 to +0.151) | yes |
| Three-way vs plain document check (full model) | -0.002 (-0.006 to +0.001) | no | -0.007 (-0.028 to +0.012) | no |

## 4. Four boxes (confidence vs document support)

| Box | Sentences | Hallucinated | Rate | Share of all hallucinations |
|---|---|---|---|---|
| confident, document supports | 521 | 7 | 1.3% | 4.5% |
| confident, document does not support | 192 | 36 | 18.8% | 23.2% |
| unsure, document supports | 409 | 24 | 5.9% | 15.5% |
| unsure, document does not support | 303 | 88 | 29.0% | 56.8% |

## 5. Who decides the expensive re-writing? (gating)

Reach = share of hallucinated sentences that reach the expensive step.

| Share re-written | Reach doc-first | Reach doubt-first | Reach random | AUROC doc-first | AUROC doubt-first | AUROC random |
|---|---|---|---|---|---|---|
| 10% | 35.5% | 24.5% | 9.7% | 0.845 | 0.834 | 0.835 |
| 20% | 55.5% | 36.1% | 20.3% | 0.842 | 0.837 | 0.836 |
| 30% | 72.3% | 50.3% | 30.2% | 0.843 | 0.840 | 0.837 |
| 40% | 85.8% | 65.8% | 40.7% | 0.845 | 0.837 | 0.837 |
| 50% | 92.3% | 72.3% | 50.7% | 0.845 | 0.841 | 0.838 |
| 60% | 95.5% | 81.9% | 61.1% | 0.843 | 0.839 | 0.839 |
| 80% | 98.1% | 96.1% | 80.0% | 0.843 | 0.840 | 0.842 |
| 100% | 100% | 100% | 100% | 0.843 | 0.843 | 0.843 |

AUROC difference, document-first minus doubt-first: +0.0105 (+0.0036 to +0.0194) at 10% (clearly different); not clearly different at 20% to 80%.

## 6. Early warning (154 human-marked wrong spans)

| Words after span start | Alarm fired | 95% range | Normal text (by design) |
|---|---|---|---|
| 1 | 35.7% | 29-44% | 10% |
| 2 | 26.0% | 20-33% | 10% |
| 3 | 28.6% | 22-36% | 10% |
| 4 | 26.6% | 20-34% | 10% |
| 5 | 25.3% | 19-33% | 10% |
| 6 | 26.0% | 20-33% | 10% |
| 7 | 24.7% | 19-32% | 10% |
| 8 | 22.1% | 16-29% | 10% |
| 9 | 22.1% | 16-29% | 10% |
| 10 | 20.8% | 15-28% | 10% |

The alarm fires somewhere inside 64% of the spans (chance level for independent words about 52%); median delay 0 words.
Average word-level doubt: 0.967 inside wrong spans, 0.869 elsewhere.

## 7. Global vs per-task thresholds (full model)

| Threshold | Precision | Recall | F1 | F1 Data-to-text | F1 QA | F1 Summary |
|---|---|---|---|---|---|---|
| one global threshold | 0.319 | 0.671 | 0.432 | 0.426 | 0.483 | 0.400 |
| one per task type | 0.341 | 0.697 | 0.458 | 0.424 | 0.575 | 0.473 |

F1 difference +0.025 (95% range +0.010 to +0.044). Task thresholds: Data-to-text 0.222, QA 0.142, Summary 0.314 (fallback 0.223).

## 8. Baselines (first 100 answers: 713 sentences, 72 hallucinated; 51 of the 100 answers contain a hallucination)

Sentence level:

| Method | AUROC (95% range) | AUPRC |
|---|---|---|
| SelfCheck-style baseline | 0.666 (0.61-0.71) | 0.152 |
| A word-level doubt | 0.691 (0.61-0.75) | 0.202 |
| B semantic entropy | 0.621 (0.57-0.67) | 0.133 |
| C document check | 0.812 (0.77-0.86) | 0.266 |
| A + C | 0.840 (0.81-0.88) | 0.297 |
| **Full prototype** | **0.853 (0.82-0.89)** | 0.325 |

Answer level (each answer scored by its highest-risk sentence):

| Method | AUROC (95% range) | AUPRC |
|---|---|---|
| Whole-answer semantic entropy (the paper's baseline) | 0.472 (0.37-0.58) | 0.485 |
| SelfCheck-style, worst sentence | 0.564 (0.45-0.66) | 0.582 |
| SelfCheck-style, average sentence | 0.479 (0.37-0.59) | 0.489 |
| A word-level doubt, worst sentence | 0.786 (0.70-0.86) | 0.760 |
| C document check, worst sentence | 0.740 (0.63-0.84) | 0.666 |
| A + C, worst sentence | 0.798 (0.70-0.88) | 0.725 |
| **Full prototype, worst sentence** | **0.826 (0.74-0.90)** | 0.751 |

Cost per answer (generated token slots): whole-answer baseline 808; SelfCheck-style 775;
ours, every sentence re-written 1,322 (1.64x); ours, document-first gating at 40% 495 (0.61x).

## 9. Safety rule

| Rule | Precision | Recall | F1 | Flagged sentences |
|---|---|---|---|---|
| none (model only) | 0.341 | 0.697 | 0.458 | 317 |
| also flag if not backed > 0.80 | 0.297 | 0.748 | 0.425 | 391 |
| also flag if not backed > 0.90 | 0.309 | 0.729 | 0.434 | 366 |
| also flag if not backed > 0.95 | 0.315 | 0.716 | 0.438 | 352 |
| also flag if not backed > 0.99 | 0.325 | 0.697 | 0.444 | 332 |
