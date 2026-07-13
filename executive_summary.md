**AI Faithfulness in Scientific Reasoning: Executive Summary**

Team: Awndre Gamache and Hayden Everett

**Overview**

This project asks whether we can predict when a vision language model (VLM) will fail a scientific chart-reasoning task. We use the CharXiv benchmark dataset, which consists of 1,000 charts drawn from arXiv papers, each with a single question and a pre-graded record of whether a given VLM could answer the question. Every question-chart pair receives a model-agnostic annotation that contains scores for twelve cognitive-demand dimensions. The demand scores range from 0 to 5 and are assigned via a verdict-blind process that never sees the correct answer. We predict failure for two real VLMs (GPT-4o and Sonnet 3.5) as well as a control that never sees the chart associated with each question (GPT-4o random). We measure performance using ROC-AUC as our primary KPI along with balanced accuracy and F1 score as secondary KPIs. 

**Central Question**

Can a classifier trained on an item's 12-dimension demand profile alone predict failure for a given target model before the model ever sees the item?

Hypothesis: The cognitive demand of claim-evidence pairs drives VLM reasoning failure

1\. Demand \= difficulty: Each rubric dimension captures a real cognitive cost an item imposes — how much visual search the claim/question forces (VL), how many values must be read and combined (MA), how subtle a shape judgment is required (GS), how much attention, quantitative reasoning, or background knowledge is needed. An item's demand profile is a decomposition of its intrinsic difficulty into interpretable parts.

2\. Direct relationship with correctness: For every dimension, higher demand → lower probability of a correct answer. Failed items should score higher than passed items, and the effects should be roughly additive.  An item that is demanding on several dimensions at once should be hardest of all.

3\. Item-intrinsic: Because demand is a property of the claim-evidence pair and not of any model, the same demand profile should predict failure for every model. 

**Sonnet annotation of CharXiv** 

The demand scores themselves were produced by Claude Sonnet acting as an automated annotator over the 1,000 reasoning items of the CharXiv validation split. Each (item × dimension) cell—12,000 in total—was scored by an isolated Sonnet subagent that saw only the chart image, the reasoning question, and a single rubric at a time, so every dimension was judged independently and without contamination from other rubrics. The pipeline was deliberately verdict-blind: the ground-truth answer never appeared in any prompt, so the demand scores describe the item, not the answer. Annotation ran as a resumable prepare → dispatch → collect pipeline that fanned out one subagent per cell in sharded waves, scraped each transcript for the 0–5 score, and recorded per-call token and cost provenance in a SQLite database (annotations\_charxiv.db); all 12,000 annotations parsed successfully. Notably, no model evaluation run was needed for the correctness side of the analysis—CharXiv ships pre-graded 0/1 scores for 39 models, which were loaded into the same database and joined to the demand profiles.

**Feature engineering**  

The feature discovery process is an inductive, LLM-driven pass to build a codebook of demand dimensions–properties that describe what verifying a scientific claim against an evidence figure requires of a verifier (visual decoding, reasoning, knowledge). These demand dimensions extend the ADeLe/DeLeAn demand-annotation methodology from text-only to figure-grounded tasks. A 300-item discovery sample of 200 SciVer single-figure items and 100 SciClaimEval figure-evidence items was annotated in 30 batches of 10 by Claude Opus 4.8 subagents that were kept blind to verdict labels; each batch scored every item on the current codebook (seeded with 23 dimensions from ADeLe and the Tolan et al. visual-processing taxonomy) and proposed new dimensions the codebook failed to capture, with new proposals fed forward into later batches so counts measured marginal novelty. The pass produced 89 emergent dimensions (112 total) without reaching the pre-registered saturation criterion, validated verdict-blindness via twin-pair invariance checks, and flagged leakage-suspect and deep-reasoning dimensions for review. Subsequent stages grouped the sub-features into 9 parent dimensions, ran a per-item spread test to identify which parents best separate easy from hard items, and drafted three of them \- Visual Localization/Grounding, Multi-Element Aggregation, and Gestalt/Shape Judgment \- into full ADeLe-style 0–5 rubrics that entered a final 12-dimension battery. More detail on this process can be found in the feature discovery folder.

The engineered features are the twelve demand dimensions, of which eleven are used because one dimension is near-constant and carries almost no information. The item metadata is kept only as a cheap baseline that the demand dimensions must beat. The central question is whether the demand profile adds predictive value over that baseline. The ablation below, measured with paper-grouped cross-validation, shows that it does for the two real models but not for the random control.

| Target | Metadata baseline ROC-AUC | Demand dimensions ROC-AUC |
| :---- | :---- | :---- |
| GPT-4o | 0.528 | 0.676 |
| Sonnet 3.5 | 0.536 | 0.628 |
| GPT-4o blind | 0.727 | 0.580 |

**Modeling journey and final choice**

Model selection compared four tree-based classifiers against plain logistic regression under nested, paper-grouped cross-validation, so that no question-evidence pairs from the same paper appeared in both the training and validation sets. We repeated this process on 30 bootstrap resamples of our training data since the original dataset is small. Logistic regression, random forest, histogram gradient boosting, and XGBoost classifiers all achieved comparable performance on our KPIs. Even TabPFN–a pretrained, transformer-based classifier designed specifically for tabular data–could not deliver substantially improved performance. We therefore chose one logistic regression as our final model due to its relative simplicity and greater interpretability.

**Results and post hoc audit**

The table below reports the cross-validated key performance indicators on the training items and the single-scoring result on the test set.

| Target | ROC-AUC (CV) | Balanced accuracy (CV) | F1 (CV) | ROC-AUC (Test) |
| :---- | :---- | :---- | :---- | :---- |
| GPT-4o | 0.676 | 0.631 | 0.650 | 0.553 |
| Sonnet 3.5 | 0.628 | 0.567 | 0.379 | 0.694 |
| GPT-4o blind | 0.580 | 0.500 | 0.947 | 0.588 |

The single test result was surprising, because GPT-4o scored below its cross-validation number and Sonnet 3.5 scored above. A balance audit found no significant difference between the training and test items after a Holm adjustment, and a split-sensitivity study that resampled the split one hundred times placed the GPT-4o test draw below the second percentile and the Sonnet 3.5 draw near the ninetieth. The gap is sampling variation in one small test set, and the cross-validation estimates, averaged over many folds, are the more reliable figures.

**Limitations and conclusions**

The main limitation to keep in mind is that we are working with a small dataset, and next steps would involve obtaining more examples with richer features. To answer our opening question, we find that demand scores, including new emergent demand dimensions, are modestly successful in predicting when and why a VLM will fail on a scientific reasoning task. For researchers and businesses, demand scores can assist in routing tasks to expensive frontier models, and since the rubrics transfer across VLMs and benchmarks, the annotations outlive any single model release.