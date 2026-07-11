# Evaluation Plan — CharXiv Reasoning Failure Prediction

Checkpoint 3 deliverable. Defines how models are evaluated *before* serious modeling: the unit of
analysis, the split strategy that mirrors deployment, the metrics, and the stress/leakage tests.
Companion files: [kpis.md](kpis.md) (metrics), [README.md](README.md) (problem statement),
[notebooks/15_charxiv_metric_evaluation.ipynb](notebooks/15_charxiv_metric_evaluation.ipynb)
(the framework run on the baselines), `src/splits/` (splitter), `src/eval/` (CV core + stress tests).

## 1. Unit of analysis

- One row per **item** (800 train items), with a separate classifier fit for each of the **3 target
  models** (`GPT-4o`, `Claude-3-5-Sonnet`, `GPT-4o-Random`). Each classifier shares the item feature
  block and predicts that target's failure label; every KPI is reported per target.
- **Target:** `y = 1` = **failure** (incorrect), the positive class.
- **Features:** the Checkpoint-2 default pipeline — the **12 DeLeAn demand dims** (`CL` dropped). Item
  metadata (`category`, `inst_category`, `year`) is the cheap baseline the demand dims must beat and is
  off in the final feature set (Ckpt-2 ablation). No model's correctness (`answer`, any VLM's `score`,
  or `Human`) ever enters the features.
- **Control:** `GPT-4o-Random` is a random-answer model. The demand dimensions cannot predict a random
  answer, so its classifier is expected near chance; it is never tuned.

## 2. Split strategy — the `item_holdout` regime

We evaluate a **single deployment regime**: predict **failure on a new chart** (an unseen paper) for a
given target model. Splitter: `ItemHoldoutSplit` = `StratifiedGroupKFold` grouped by `paperid`
(5 folds × 3 repeats), so **no `paperid` (hence no item) straddles a fold** while the failure rate
stays balanced across folds. This mirrors the intended use — scoring newly-published charts. The same
folds are reused for each target's classifier.

Geographic and temporal leakage are **N/A** — the data has no spatial or time dimension.

### The 800/200 holdout and its caveat
The fixed 800/200 split (seed `20260618`, `charxiv_analysis/train_test_split.json`) is **item-level
random, not paper-grouped**, so a few papers have figures in both train and test. Decision: **keep**
the split for continuity; the CV above is fully paper-grouped (the unbiased estimator), so the
straddle affects only the final-holdout boundary. **Mitigation:** at the single final evaluation
(Ckpt 5) we additionally report metrics with straddling test items excluded. The 200-item test set is
**untouched** here and is scored **once**.

## 3. Leakage taxonomy applied to this data

- **Group-level** → paper-grouped `ItemHoldoutSplit` (no `paperid` straddles a fold).
- **Feature leakage** → outcome fields never used; no model's correctness is a feature, including
  `Human` — the features are item-intrinsic (the chart's demand profile and its metadata).
- **Post-outcome leakage** → demand is scored **verdict-blind** (only the question + figure reach the
  annotator), so it is available before the outcome.
- **Preprocessing contamination** → every transformer is fit **per fold** inside an sklearn
  `Pipeline` (`src/eval/core.py`), never on the test rows.
- **Hyperparameter-tuning leakage** → tuning is **not** done this checkpoint (baselines only). For
  Checkpoint 4, tuning uses **nested CV** with the same grouped splitter in the inner loop, for the
  two real targets only.
- **Overfitting the split** → grouped CV with repeats; a sealed 200-item holdout scored once.
- **Geographic / temporal** → not applicable.

## 4. Metrics and objectives

Primary **ROC-AUC** (threshold-free, robust to class imbalance); secondaries balanced accuracy and F1
— full definitions in [kpis.md](kpis.md). Beyond point metrics:
- **Calibration:** reliability diagrams per target; `CalibratedClassifierCV` (isotonic/Platt) is the
  planned fix if calibrated probabilities matter for the decision.
- **Cost-sensitive decision:** deployment scenario = predicted failures are flagged for human review;
  a false negative (missed failure) and a false positive (wasted review) carry costs. We report the
  expected-cost-minimizing threshold via an OOF sweep (and a `make_scorer` cost objective) on the
  primary real target. **Stated assumption:** symmetric cost by default; a 3:1 (miss-failure :
  false-alarm) scenario shifts the threshold down. Adjust the ratio to the real review economics.

## 5. Robustness & stress tests (`src/eval/stress_tests.py` → `results/stress_tests/`)

Run on the primary real target (`GPT-4o`) under `item_holdout` (dated `2026-07-03`):
- **Adversarial shuffled-target** — permute `y`, refit; real AUC ≈ **0.68** vs shuffled ≈ **0.51–0.54**
  → the signal is real, not leakage/overfit.
- **Dropped-feature** — where the signal lives: **metadata only ≈ 0.53 → demand dims ≈ 0.68 →
  demand + metadata ≈ 0.67.** The **DeLeAn dims are the signal**; metadata adds nothing on top —
  the point of the DeLeAn annotation.
- **Cohort / distribution shift** — per-cohort AUC by arXiv `category` and question type
  (`inst_category`); metadata used for *slicing only*, never as a feature. No fragile subgroup.
- **Outlier sensitivity** — `StandardScaler` vs `RobustScaler` and excluding IsolationForest
  demand-outlier items: AUC ≈ 0.68 in all three → performance does not hinge on outliers or scaler.

## 6. Audit trail — why this reflects deployment, and what might break

- **`item_holdout`** mirrors the real use: score *new charts* for a given target model. Paper-grouping
  is what makes the estimate honest (no chart/paper leaks across the fold boundary).
- **Assumptions that may break:** demand scores come from a single Sonnet annotation pass (annotator
  drift would shift features); the target roster and the CharXiv val distribution are fixed (results
  may not transfer to other models or very different chart styles); the few-paper train/test straddle
  is a small known leak at the final-holdout boundary, mitigated at final eval.

## 7. Scope of this checkpoint

Only the **baseline** models (Dummy, LogisticRegression, RandomForest) have been run through the
framework — see [notebooks/15_charxiv_metric_evaluation.ipynb](notebooks/15_charxiv_metric_evaluation.ipynb)
and `results/cv_metrics__item_holdout__2026-07-03.csv` (per target: `GPT-4o` LogReg ≈ 0.68,
`Claude-3-5-Sonnet` ≈ 0.63, `GPT-4o-Random` near chance). Tuned modeling and the single final-holdout
evaluation are Checkpoint 4/5.
