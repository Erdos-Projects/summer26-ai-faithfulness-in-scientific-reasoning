# CharXiv Failure Prediction: Executive Summary

## Problem and data

This project asks whether we can predict when a vision-language model will fail a chart-reasoning
question. The material is the CharXiv benchmark, one thousand validation charts drawn from arXiv papers,
each with a single reasoning question and a pre-graded record of whether a given model answered it
correctly. Every chart also carries a model-agnostic DeLeAn annotation that scores twelve cognitive-demand
dimensions from zero to five, produced by a verdict-blind process that never sees the correct answer. We
predict failure for three target models, GPT-4o, Claude-3-5-Sonnet, and a GPT-4o-Random control, with a
separate classifier for each. The positive class is failure, so a label of one means the model got the
item wrong. GPT-4o-Random answers at random and fails about ninety percent of items, which makes it a
negative control, because the cognitive demand of an item cannot predict a random answer.

## Feature engineering and the central question

The engineered features are the twelve demand dimensions, of which eleven are used because one dimension
is near-constant and carries almost no information. The item metadata is kept only as a cheap baseline
that the demand dimensions must beat. The central question is whether the demand profile adds predictive
value over that baseline. The ablation below, measured with paper-grouped cross-validation, shows that it
does for the two real models and, as designed, does not for the random control.

| Target | Metadata baseline ROC-AUC | Demand dimensions ROC-AUC |
| --- | --- | --- |
| GPT-4o | 0.528 | 0.676 |
| Claude-3-5-Sonnet | 0.536 | 0.628 |
| GPT-4o-Random | 0.727 | 0.580 |

## Modeling journey and final choice

Model selection compared four richer families against a plain logistic regression under nested,
paper-grouped cross-validation, so that no paper straddled a fold and no hyperparameter was tuned on the
data it was scored against. A regularized logistic regression, a random forest, a histogram gradient
boosting model, and XGBoost all landed at the same level as the plain logistic regression, near 0.68 for
GPT-4o and 0.63 for Claude-3-5-Sonnet. The tree models drifted toward very shallow depth during tuning,
which points to a ceiling in the available signal rather than a limitation of the model family, and a
bootstrap stability check and a TabPFN flexibility check left that reading in place. We therefore chose
one logistic regression per target, the simplest strong model and the most interpretable one.

## Results and interpretation

The table below reports the cross-validated key performance indicators on the training items and the
single-scoring result on the sealed two-hundred-item test set, restricted to the one hundred ninety-four
items whose papers do not straddle the split.

| Target | CV ROC-AUC | CV balanced accuracy | CV F1 | Test ROC-AUC |
| --- | --- | --- | --- | --- |
| GPT-4o | 0.676 | 0.631 | 0.650 | 0.553 |
| Claude-3-5-Sonnet | 0.628 | 0.567 | 0.379 | 0.694 |
| GPT-4o-Random | 0.580 | 0.500 | 0.947 | 0.588 |

Because the final models are linear, their coefficients read directly as the contribution of each demand
dimension to the odds of failure. One dimension dominates for both real models, the demand to calibrate
what is known against what is unknown, which raises failure odds, while quantitative-reasoning and
gestalt-shape demands lower them. The signal is concentrated in a few dimensions rather than spread across
all of them.

The single test result was surprising, because GPT-4o scored below its cross-validation number and
Claude-3-5-Sonnet scored above. A balance audit found no significant difference between the training and
test items after a Holm adjustment, and a split-sensitivity study that resampled the split one hundred
times placed the GPT-4o test draw below the second percentile and the Claude draw near the ninetieth. The
gap is sampling variation in one small test set, and the cross-validation estimates, averaged over many
folds, are the more reliable figures.

## Limitations and next steps

The test set is small, so the final numbers are reported with bootstrap confidence intervals rather than
single values. Performance plateaus near 0.68 and 0.63 across every model family, which suggests that the
demand dimensions, while a clear improvement over the metadata baseline, carry a modest ceiling of signal.
The natural next steps are to enrich the feature set with chart-structural and question-level information,
to extend the target set beyond three models, and to give the flexible families a full
hardware-accelerated treatment before concluding that added complexity cannot help. The intended use is
benchmark diagnosis, meaning triage of the items a model is likely to fail together with an explanation of
which cognitive demands drive that failure, and not a deployment gate.
