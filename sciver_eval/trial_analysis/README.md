# trial_analysis/

Canonical home for derived figures/tables over the runs in `../predictions.db`.
Everything here is an **output**, regenerated from the DB — not source data.
Analysis scripts resolve this folder via `sciver_eval.db.analysis_dir()`, so new
trial analyses should write here too.

Runs in the DB: 1 = Haiku native (5 trials), 2 = Haiku entailed (5t),
3 = Haiku refuted (5t), 4 = Sonnet entailed (1t), 5 = Sonnet refuted (1t).

## Venn artifacts — `viz_venn.py`

Paired correctness on the 817 chart items: an entailed run (truth = "yes") vs a
refuted run (truth = "no"). Defaults to **majority vote across all trials** of each
run; pass `--trial N` for a single trial. Figure titles auto-label the model and
aggregation. By-claim-type output filename derives from `--out`'s stem.

    # Haiku, maj. vote of 5 trials (default runs 2 vs 3)
    python -m sciver_eval.viz_venn --by-claim-type
    # Sonnet, single trial
    python -m sciver_eval.viz_venn --entailed-run 4 --refuted-run 5 --by-claim-type \
        --out trial_analysis/venn_entailed_refuted_sonnet.png

| region | meaning |
|---|---|
| `correct_either_way` | "yes" to entailed AND "no" to refuted — discriminates |
| `both_entailed` | "yes" in both conditions (always-yes) |
| `both_refuted` | "no" in both conditions (always-no / skeptic) |
| `wrong_either_way` | "no" to entailed AND "yes" to refuted — inverted |

Files: `venn_entailed_refuted{,_by_claim_type}.{png,csv}` (Haiku),
`venn_entailed_refuted_sonnet{,_by_claim_type}.{png,csv}` (Sonnet).

## `agreement_entailed_refuted.png` / `.csv` — `viz_agreement.py`
2x2 of whether the entailed and refuted conditions AGREE on per-item correctness
(diagonal = agree: both right / both wrong; off-diagonal = disagree). Reports
observed agreement + Cohen's kappa. Same majority-vote default + `--trial N`.

    python -m sciver_eval.viz_agreement [--entailed-run 2 --refuted-run 3]
    # threshold sweep: item "correct" iff correct on >=K of 5 trials
    python -m sciver_eval.viz_agreement --min-correct K --out .../agreement_entailed_refuted_Kof5.png

Haiku maj.-of-5 (3/5): κ = -0.113 (anti-agreement) — "refuted-only" cell dominates
(47.1%), i.e. item difficulty is condition/bias-driven, not intrinsic. Threshold
sweep `agreement_entailed_refuted_{3,4,5}of5`: tightening to unanimous (5/5) exposes
a hard core (both-wrong 14->156) and shrinks robust-correct (357->148); κ -> -0.018.

## `rubric_demand_by_correct.png` / `.csv` — `viz_rubric_demand.py`
Joins predictions with DeLeAn rubric demand from `rubric_scoring/annotations_prod.db`
(ALL 12 fully-scored dims: pass-1 AS/AT/MCr/QLl/QLq/VO + pass-3 CL/GS/KNf/MA/MCu/VL;
partial dims like SNs@200 excluded). Emergent (authored figure-grounding) dims
GS/MA/VL are shaded purple + ◆. Compares mean demand per dim between items the model
is **confidently CORRECT** (5/5 on both conditions, n=148) vs **confidently INCORRECT**
(unanimous per condition but not aligned with both, n=183; strict both-wrong is n=1
due to skeptic bias). SEM bars + Mann-Whitney U per dim.

    python -m sciver_eval.viz_rubric_demand [--entailed-run 2 --refuted-run 3]

Result: confident failures demand significantly MORE on standard DeLeAn reasoning dims
— Calibrating Knowns/Unknowns MCu (Δ+0.44, p<.001), Identifying-Relevant-Info MCr
(+0.33, p<.001), Attention-and-Scan AS (+0.24, p<.001), Volume VO (+0.17, p<.01).
The 3 EMERGENT figure-grounding dims (GS/MA/VL) do NOT separate (all ns). ⇒ failure
is driven by metacognition/relevant-info selection, not visual figure-grounding.

## `run1_pcorrect_hist.png` / `.csv` — `viz_difficulty.py`
Run-1 per-item difficulty: p(correct) over the 5 trials, bucketed (0.0 = always
wrong, 1.0 = always right, between = stochastic).

    python -m sciver_eval.viz_difficulty [--run 1]

## Headline (Haiku vs Sonnet, entailed/refuted accuracy)
Haiku: entailed 0.503, refuted 0.882 (balanced 0.693).
Sonnet: entailed 0.432, refuted 0.956 (balanced 0.694).
Sonnet is a *stronger skeptic*, not a better reader — same balanced accuracy, more "no".

> Caveat: Haiku Venn uses maj.-vote-of-5 (denoised); Sonnet is single-trial (noisier).
> For a strict comparison use `--trial 1` on the Haiku runs.
