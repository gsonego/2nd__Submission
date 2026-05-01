# Tuning Review: IMDB Sentiment Classification Project

## Context

The core project (EDA, model design, baseline evaluation, methodology, and original analysis) was completed prior to this tuning phase. Hyperparameter tuning was subsequently added per lecturer request, resulting in 3 additional tuning runs per model (4 total runs per model: 1 baseline + 3 tuning). This document summarises what tuning revealed and identifies every notebook cell that requires wording or content updates to reflect the new results.

- **Models**: Logistic Regression (TF-IDF), LSTM (Keras/TensorFlow), BERT (bert-base-uncased, HuggingFace)
- **Data split**: 20,000 train / 5,000 validation / 25,000 test
- **Total runs**: 12 (3 models × 4 runs each)

---

## Results Summary

### Logistic Regression — 4 Runs

| Run                | Val Acc    | Test Acc    | Precision | Recall  | F1          | Key Hyperparams                       |
| ------------------ | ---------- | ----------- | --------- | ------- | ----------- | ------------------------------------- |
| LR_BASELINE        | 0.8928     | 0.87936     | 0.8754    | 0.88464 | 0.87999     | C=1.0, max_iter=1000 (default solver) |
| LR_TUNING_1        | 0.8908     | 0.87968     | 0.8787    | 0.88104 | 0.87984     | C=2.0, liblinear                      |
| LR_TUNING_2        | 0.8868     | 0.87540     | 0.8689    | 0.88416 | 0.87648     | C=0.5, liblinear (worst)              |
| **LR_TUNING_3** ✅ | **0.8928** | **0.87988** | 0.8765    | 0.88432 | **0.88041** | C=1.25, lbfgs                         |

**Selected best: LR_TUNING_3** — marginally highest F1 (0.88041) and tied highest val accuracy. Already documented in the notebook at line 2883: _"LR_TUNING_3 is selected as the final Logistic Regression model"_.

---

### LSTM — 4 Runs

| Run                  | Val Acc    | Test Acc    | Precision | Recall  | F1           | Key Hyperparams                                                   |
| -------------------- | ---------- | ----------- | --------- | ------- | ------------ | ----------------------------------------------------------------- |
| LSTM_BASELINE        | 0.5646     | 0.55028     | 0.5979    | 0.3071  | 0.4058       | emb=64, units=64, no dropout, 3 epochs                            |
| **LSTM_TUNING_1** ✅ | **0.8724** | **0.85308** | 0.838018  | 0.87536 | **0.856282** | emb=64, units=64, dropout=0.2, rec_dropout=0.2, 5 epochs, Adam    |
| LSTM_TUNING_2        | 0.8592     | 0.84252     | 0.822572  | 0.87344 | 0.847243     | emb=128, units=128, dropout=0.3, rec_dropout=0.2, 6 epochs, Adam  |
| LSTM_TUNING_3        | 0.6524     | 0.64640     | 0.5979    | 0.89448 | 0.7167       | emb=64, units=64, dropout=0.2, rec_dropout=0.1, lr=5e-4, 8 epochs |

**Selected best: LSTM_TUNING_1** — highest test accuracy (0.85308) and highest F1 (0.856282). The notebook code comment at line 4417 (`# Confusion Matrix for best LSTM run: Tuning 1`) is **correct** and requires no change.

> **Notable finding**: LSTM tuning produced the most dramatic improvement of all three models. The baseline was severely overfitting — validation accuracy _declined_ across every training epoch (epoch 1: 0.5758 → epoch 2: 0.5712 → epoch 3: 0.5646), indicating the model was memorising training data from the very first epoch. Adding dropout (TUNING_1: dropout=0.2, recurrent_dropout=0.2) completely fixed this, enabling the model to converge properly and achieve test accuracy 0.85308 — a **55% relative improvement**. After tuning, LSTM is nearly on par with Logistic Regression (0.85 vs 0.88).

---

### BERT — 4 Runs

| Run                  | Val Acc    | Test Acc    | Precision  | Recall     | F1         | Key Hyperparams                          |
| -------------------- | ---------- | ----------- | ---------- | ---------- | ---------- | ---------------------------------------- |
| **BERT_BASELINE** ✅ | 0.9190     | **0.92252** | **0.9232** | 0.9218     | 0.9225     | lr=2e-5, epochs=2, batch=16, warmup=0.0  |
| BERT_TUNING_1        | 0.9218     | 0.92164     | 0.9183     | **0.9256** | 0.9219     | lr=3e-5, epochs=2, batch=16, warmup=0.0  |
| BERT_TUNING_2        | 0.9206     | 0.92180     | 0.9153     | 0.9296     | 0.9224     | lr=2e-5, epochs=2, batch=32, warmup=0.0  |
| BERT_TUNING_3        | **0.9236** | 0.92204     | 0.9159     | 0.9294     | **0.9226** | lr=2e-5, epochs=2, batch=16, warmup=0.10 |

**Selected best: BERT_BASELINE** — highest test accuracy (0.92252). All four configurations performed within 0.001 of each other, demonstrating BERT's robustness to moderate hyperparameter variation. Note: BERT_TUNING_3 achieves the highest validation accuracy (0.9236) and highest F1 (0.9226), but test accuracy is the primary selection criterion.

---

## Best Run Analysis

### Logistic Regression: Why LR_TUNING_3?

Tuning had minimal impact across all LR runs — the range of test accuracy is only 0.00448 (0.87540 to 0.87988). The baseline C=1.0 configuration was already near-optimal for TF-IDF features. TUNING_3 (C=1.25, lbfgs solver) produced a marginal improvement in F1 (+0.04%). The convergence of results across all four runs suggests LR's performance ceiling is determined more by the TF-IDF feature representation than by regularisation strength.

### LSTM: Why LSTM_TUNING_1?

The baseline LSTM demonstrated severe overfitting from the first epoch — validation accuracy was declining while training accuracy improved. Without regularisation, the model had no mechanism to generalise. TUNING_1 introduced both standard dropout (0.2) and recurrent dropout (0.2), which prevented the model from over-relying on specific activation patterns. This single change produced a test accuracy improvement of over 0.30 points (0.55 → 0.85).

TUNING_2 used a larger architecture (emb=128, units=128) with more dropout (0.3), yet underperformed TUNING_1, suggesting that the original smaller architecture was sufficient once regularised — adding capacity without controlling the regularisation budget marginally hurt performance.

TUNING_3 used a lower Adam learning rate (lr=5e-4 versus the default 0.001). This caused slow convergence; the model peaked at validation accuracy 0.6524 at epoch 3, then began overfitting, triggering early stopping before reaching competitive performance. Lower learning rates require more epochs to converge, and the interaction with early stopping led to premature termination.

### BERT: Why BERT_BASELINE?

BERT's fine-tuned representations are robust by nature. All four configurations — varying learning rate (2e-5 vs 3e-5), batch size (16 vs 32), and warmup schedule (0% vs 10%) — produce virtually identical results. The test accuracy spread across all four runs is only 0.00088 (0.92164 to 0.92252). This confirms that BERT's performance is primarily determined by the quality of its pre-trained representations, not fine-tuning hyperparameters within this narrow range. The baseline configuration represents the simplest and most interpretable choice.

---

## Tuning Observations (Key Learnings for Step 8)

### Logistic Regression

Tuning confirmed that the baseline configuration was near-optimal. Increasing regularisation (lower C values) slightly hurt performance; increasing it above baseline gave negligible improvement. The model's performance is largely a function of TF-IDF feature quality. This is consistent with the theoretical expectation: linear models trained on bag-of-words features are relatively insensitive to regularisation strength once data is sufficient.

### LSTM

The most impactful finding of the tuning phase. The baseline suffered from a fundamental architectural problem (no regularisation), not a tuning deficiency. Dropout is not a tuning parameter in the traditional sense — it is a structural decision that enables the model to function at all on this task. Once this was addressed (TUNING_1), the model achieved performance nearly on par with Logistic Regression (0.85 vs 0.88). The architecture size (64 units) proved sufficient; the larger 128-unit model in TUNING_2 did not improve results. The learning rate sensitivity in TUNING_3 highlights that LSTM networks require careful scheduler choices, particularly when combined with early stopping.

### BERT

Tuning confirmed BERT's robustness. Within the commonly recommended fine-tuning ranges (lr 2e-5 to 3e-5, batch 16 to 32, warmup 0 to 10%), performance is virtually identical. This is consistent with published BERT fine-tuning guidance (Devlin et al., 2019), which notes that performance is relatively stable for downstream tasks within these parameter ranges. The BERT baseline is a strong choice for deployment precisely because it does not rely on tuned hyperparameters.

---

## Post-Tuning Model Comparison

| Stage                      | LSTM       | LR     | BERT   |
| -------------------------- | ---------- | ------ | ------ |
| **Pre-tuning (baseline)**  | 0.550      | 0.879  | 0.923  |
| **Post-tuning (best run)** | **0.853**  | 0.880  | 0.923  |
| **Change**                 | **+0.303** | +0.001 | ~0.000 |

Pre-tuning: BERT (0.92) >> LR (0.88) >> LSTM (0.55) — large gap between LSTM and the others.

Post-tuning: BERT (0.92) > LR (0.88) ≈ LSTM (0.85) — LSTM is now nearly on par with LR. The tuning phase fundamentally changed the relative standing of the three models.

---

## Notebook Updates Required

The following cells require wording or content updates to reflect the tuning results. All changes are limited to **markdown narrative cells** — no code cells require modification.

---

### Change 1 — Line 3453: LSTM Baseline Accuracy Claim (Factually Wrong)

**Location**: Markdown cell in the LSTM Baseline Evaluation section.

**Current text**:

> "The LSTM model achieved a test accuracy of approximately 0.66, with an F1-score of 0.58."

**Problem**: The actual LSTM_BASELINE results are test accuracy = 0.55028 and F1 = 0.4058. Neither figure matches the cell. No LSTM run achieved 0.66/0.58 (TUNING_3 achieved 0.64640, which is closest but still different).

**Suggested replacement**:

> "The LSTM baseline model achieved a test accuracy of approximately 0.55, with an F1-score of 0.41. These results indicate severe overfitting — the model failed to generalise to unseen data. Validation accuracy declined across all three training epochs (0.5758 → 0.5712 → 0.5646), a clear sign that the absence of regularisation caused the model to memorise training samples rather than learn general patterns. Hyperparameter tuning, documented below, addressed this by introducing dropout regularisation, which dramatically improved test accuracy to 0.85."

---

### Change 2 — Line 3457: Stale "Absence of Hyperparameter Tuning" Phrase

**Location**: Same markdown cell as Change 1, or nearby.

**Current text** (approximate):

> "...the absence of hyperparameter tuning."

**Problem**: Tuning has since been performed. This phrase is factually outdated.

**Suggested replacement**: Remove or rephrase to:

> "...the absence of dropout regularisation in the baseline configuration."

---

### Change 3 — Missing LSTM Final Evaluation Section (Add after ~line 4360)

**Location**: After the LSTM tuning results comparison table (after line 4360).

**Problem**: The Logistic Regression section has a `### Final Evaluation of Logistic Regression Baseline and Tuning Runs` markdown cell (lines 2874–2883) with bullet points per run and a model selection statement. BERT has a similar section at line 5787. LSTM has no equivalent — the tuning comparison table appears but is not followed by any narrative summary or selection statement.

**Content to add** (new markdown cell):

```markdown
### Final Evaluation of LSTM Baseline and Tuning Runs

Based on the comparison table above, the following observations are made:

- **LSTM_BASELINE**: Validation accuracy = 0.5646, Test accuracy = 0.55028, F1 = 0.4058. The model severely overfitted — validation accuracy declined across all three epochs. The absence of dropout regularisation prevented effective generalisation.
- **LSTM_TUNING_1**: Validation accuracy = 0.8724, Test accuracy = **0.85308**, F1 = **0.856282**. Adding dropout (0.2) and recurrent dropout (0.2) completely resolved the overfitting issue. The model converged smoothly across 5 epochs and achieved the best results across all LSTM configurations. ← **Best**
- **LSTM_TUNING_2**: Validation accuracy = 0.8592, Test accuracy = 0.84252, F1 = 0.847243. Larger architecture (emb=128, units=128) with higher dropout (0.3). Despite increased capacity, this run underperformed TUNING_1 marginally, suggesting the smaller architecture is sufficient once regularised.
- **LSTM_TUNING_3**: Validation accuracy = 0.6524, Test accuracy = 0.64640, F1 = 0.7167. Using a lower Adam learning rate (lr=5e-4, half the default 0.001) caused slow convergence. The model peaked at epoch 3 and began overfitting, leading to early stopping before reaching competitive performance.

Based on this comparison, **LSTM_TUNING_1 is selected as the final LSTM model**, having achieved the highest test accuracy (0.85308) and F1-score (0.8563) among all LSTM configurations.
```

---

### Change 4 — Missing BERT Final Evaluation Narrative (Add after ~line 5807)

**Location**: After the BERT tuning results code cell output (after line 5807, before the `## Step 6` markdown cell at line 5835).

**Problem**: The BERT Final Evaluation section (line 5787) has a header and description cell, and a code cell that outputs `Best BERT run: BERT_BASELINE`. However, there is no narrative markdown cell with per-run bullet points and a selection statement, unlike the LR section at lines 2878–2883.

**Content to add** (new markdown cell):

```markdown
### Final Evaluation of BERT Baseline and Tuning Runs

Based on the comparison table above, the following observations are made:

- **BERT_BASELINE**: Validation accuracy = 0.9190, Test accuracy = **0.92252**, Precision = **0.9232**, Recall = 0.9218, F1 = 0.9225. The standard fine-tuning configuration (lr=2e-5, batch=16, no warmup) achieved the highest test accuracy of all BERT runs. ← **Best**
- **BERT_TUNING_1**: Validation accuracy = 0.9218, Test accuracy = 0.92164, Precision = 0.9183, Recall = **0.9256**, F1 = 0.9219. Increasing the learning rate to 3e-5 slightly improved validation accuracy and recall, but marginally reduced test accuracy.
- **BERT_TUNING_2**: Validation accuracy = 0.9206, Test accuracy = 0.92180, Precision = 0.9153, Recall = 0.9296, F1 = 0.9224. Doubling the batch size to 32 had minimal impact. Results are nearly identical to the baseline, consistent with published fine-tuning guidance.
- **BERT_TUNING_3**: Validation accuracy = **0.9236**, Test accuracy = 0.92204, Precision = 0.9159, Recall = 0.9294, F1 = **0.9226**. Adding a 10% warmup schedule achieved the highest validation accuracy and F1, but marginally lower test accuracy than the baseline.

Based on this comparison, **BERT_BASELINE is selected as the final BERT model**, having achieved the highest test accuracy (0.92252). All four configurations performed within 0.001 of each other, demonstrating BERT's robustness to moderate hyperparameter variation within the standard fine-tuning range.
```

---

### Change 5 — Line 5929: Step 7 LSTM Accuracy Claim (Critical — Very Wrong)

**Location**: Step 7 model comparison section.

**Current text** (approximate):

> "The LSTM model achieved a lower test accuracy of approximately 0.66."

**Problem**: After TUNING_1, the best LSTM run achieved test accuracy 0.85308 — not 0.66. The figure 0.66 is not close to any LSTM run's results (TUNING_3 achieved 0.646, which is the nearest). Reporting 0.66 in the final comparison is a significant factual error.

**Suggested replacement**:

> "The LSTM model, following hyperparameter tuning, achieved a test accuracy of approximately 0.85 (best run: LSTM_TUNING_1, test accuracy = 0.85308). The baseline LSTM achieved only 0.55 due to severe overfitting; the addition of dropout regularisation in the tuning phase produced a 30-point improvement, narrowing the gap considerably between LSTM and Logistic Regression."

---

### Change 6 — Lines 5977–5984: Step 8 Findings — No Tuning Context (Missing Content)

**Location**: Step 8 Discussion / Findings section.

**Problem**: The Findings section describes "three models were implemented" and their results, but contains no mention of the hyperparameter tuning phase, its outcomes, or the key learnings from it. This is a significant omission given the dramatic LSTM improvement and the contrasting tuning sensitivity observed across models.

**Content to add** (new paragraph within Step 8):

> **Hyperparameter Tuning Outcomes**: A tuning phase was conducted for each of the three models, comprising three additional runs per model beyond the baseline. The outcomes differed markedly across model families:
>
> - **Logistic Regression**: Tuning produced negligible gains (test accuracy range: 0.875–0.880). The baseline C=1.0 configuration was already near-optimal, confirming that LR performance on this task is constrained by the TF-IDF feature representation rather than regularisation strength.
> - **LSTM**: Tuning produced the most dramatic improvement of the project. The baseline LSTM severely overfitted (test accuracy 0.55), with validation accuracy declining across every training epoch. Introducing dropout regularisation (LSTM_TUNING_1: dropout=0.2, recurrent_dropout=0.2) resolved the overfitting and improved test accuracy to 0.85 — a 55% relative improvement. This result substantially closes the performance gap between LSTM and Logistic Regression and highlights that regularisation is a structural requirement for LSTM models on this task, not an optional tuning parameter.
> - **BERT**: Tuning confirmed BERT's robustness. All four BERT configurations performed within 0.001 of each other, regardless of learning rate (2e-5 vs 3e-5), batch size (16 vs 32), or warmup schedule (0% vs 10%). BERT's performance is primarily determined by its pre-trained representations, not fine-tuning hyperparameters within standard ranges.

---

### Change 7 (Minor) — Line 5044: BERT Baseline Recall Discrepancy

**Location**: BERT baseline narrative markdown cell.

**Current text**:

> "with precision of 0.921, recall of 0.924"

**Actual values**: precision = 0.9232, recall = 0.9218

**Suggested replacement**:

> "with precision of 0.923, recall of 0.922"

(Low priority — minor rounding discrepancy, no impact on conclusions.)

---

## What NOT to Change

The following sections are correct and should remain unchanged:

- **Core project structure** — section headers, EDA analysis cells, preprocessing pipeline
- **Model architecture rationale** — the reasoning for choosing each model family
- **Step 6 overall structure** — only update the LSTM accuracy figure within it (Change 5)
- **LR Final Evaluation section** (lines 2874–2883) — already complete and correct
- **Line 4417 code comment** — `# Confusion Matrix for best LSTM run: Tuning 1` is **correct**; LSTM_TUNING_1 is indeed the best run
- **All code cells** — no code cells require modification; only markdown narrative cells
- **BERT code cells and training output cells** — outputs reflect actual executed results
- **Original model selection rationale** — BERT > LR > LSTM ranking by test accuracy is still valid post-tuning (0.923 > 0.880 > 0.853); tuning narrowed the LSTM gap but did not change the ranking
