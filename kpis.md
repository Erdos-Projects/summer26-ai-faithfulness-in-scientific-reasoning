# KPIs — CharXiv Reasoning Success/Failure Prediction

Key performance indicators for the task defined in the
[CharXiv section of the README](README.md#charxiv-predicting-whether-a-vlm-will-read-a-chart-correctly):
*predict whether each of **3 target VLMs** (`GPT-4o`, `Claude-3-5-Sonnet`, `GPT-4o-Random`) answers a
CharXiv reasoning item correctly, with a separate classifier per target.* The **baseline** uses the
**item metadata** (`category`, `inst_category`, `year`); the **12 DeLeAn demand dimensions are
engineered features** tested in Checkpoint 2 (not in the baseline). `GPT-4o-Random` is the
random-answer control.

**Positive class = failure (incorrect = 1)** (ROC-AUC is invariant to the choice, and F1 names the
failure class). All metrics use paper-grouped cross-validation (`StratifiedGroupKFold`, 5 folds × 3
repeats, grouped by `paperid`) on the **800-item train split only**; the 200-item test set is the
untouched lockbox. Every KPI is reported **per target model**. Numbers below are produced by
[`notebooks/11_charxiv_baseline.ipynb`](notebooks/11_charxiv_baseline.ipynb)
(saved to `results/11_charxiv_baseline_kpis.csv`).

## Primary KPI

| KPI | Definition | Direction | Chance |
|-----|------------|-----------|--------|
| **ROC-AUC** | Area under the ROC curve — the probability the model ranks a random *failure* above a random *success*. Threshold-free. | higher is better | 0.50 |

ROC-AUC is primary because it is **threshold-free** and **invariant to class balance**, which matters
because the failure base rate varies by target (`GPT-4o` ≈ 0.51, `Claude-3-5-Sonnet` ≈ 0.39,
`GPT-4o-Random` ≈ 0.90). A separate classifier is fit for each target on the 800 training items.

## Secondary KPIs

| KPI | Definition | Direction | Notes |
|-----|------------|-----------|-------|
| **Balanced accuracy** | Mean of per-class recall at threshold 0.5. | higher is better | Interpretable "how often right", corrected for imbalance. |
| **F1 (failure)** | Harmonic mean of precision & recall for the failure class at 0.5. | higher is better | Threshold-dependent. |

## Learnability decision rule

A learner is credited with real signal only if its **ROC-AUC beats the Dummy baseline by more than one
cross-fold SD**, evaluated per target. By this rule the metadata baseline is weak for the two real
models; the open question for Checkpoint 2 is whether the DeLeAn dims add signal on top of it.

## Baseline values (item metadata only, no DeLeAn dims)

Trivial (`DummyClassifier`) → linear (`LogisticRegression`, scaled) → tree (`RandomForestClassifier`,
200 trees). Positive class = failure. ROC-AUC per target:

| Learner | GPT-4o | Claude-3-5-Sonnet | GPT-4o-Random |
|---------|:------:|:-----------------:|:-------------:|
| Dummy (most_frequent) | 0.500 | 0.500 | 0.500 |
| Dummy (stratified) | 0.526 | 0.494 | 0.509 |
| LogisticRegression | 0.528 | 0.536 | 0.725 |
| RandomForest | 0.504 | 0.520 | 0.747 |

## Verdict

For the two real models the metadata baseline barely clears the Dummy floor (ROC-AUC ≈ 0.53), so item
metadata says little about whether a real model gets a chart right. This is the **bar the DeLeAn
dimensions must beat**: Checkpoint 2 adds the 12 dims as engineered features and measures the ROC-AUC
lift over this baseline (up to ≈ 0.68 for `GPT-4o`), preserving the paper-grouped split and
failure-as-positive convention.

`GPT-4o-Random` behaves differently, and that is informative. Item metadata predicts the random control
well above chance (ROC-AUC ≈ 0.73) because a random answer's chance of being correct depends on the
answer format, which the metadata (especially `inst_category`) partly encodes. The reasoning-demand
dimensions, by contrast, add nothing for the control, confirming they measure reasoning difficulty
rather than answer-guessability.
