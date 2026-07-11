# CharXiv analysis — session notes

What we built in this session, and *why*. The focus is the analysis layer: how each
figure is constructed, the decisions behind it, and what it shows. Everything lives in
`charxiv_analysis/` and reads from `charxiv_scoring/annotations_charxiv.db`.

Context: CharXiv rubric scoring was already complete — **12,000 annotations = 1,000
items × 12 DeLeAn demand dimensions**, all `parse_ok`. This session is the first
analysis pass, deliberately **replicating the SciVer rubric→correctness analyses**
(`sciver_eval/viz_rubric_*.py`) on CharXiv so the two benchmarks are methodologically
parallel and directly comparable.

---

## 1. Data model and the core join

Three things have to be joined, and getting the join right is the foundation of every
figure below:

- **Demand scores** (`annotation` table): one 0–5 score per (item, dim). These are
  **model-agnostic** — they describe how much each cognitive dimension the *item*
  demands, not any model's behaviour. 1,000 items × 12 dims.
- **Correctness** (`model_score` table): per (model, item) outcome. We use
  **`task='reasoning'`** only. Why: reasoning has exactly **1 sub-question per figure**,
  so it maps **1:1** to the rubric items; the `descriptive` task has 4 sub-questions per
  figure and has no clean single-label mapping to an item's demand profile. Score is
  `1`=correct, `0`=wrong, `-1`=dropped (only 28 rows, excluded).
- **Leakage key** (`item.paperid`): a few papers contribute 2–3 figures (979 distinct
  papers over 1,000 items), used for grouped cross-validation.

**39 models** are in the leaderboard, including the `Human` ceiling and the
`GPT-4o-Random` baseline. We keep all 39 by default; `GPT-4o-Random` doubles as a
negative control (it should show no real structure) and `Human` as a ceiling.

## 2. Train/test split

`train_test_split.json` — a fixed **800 train / 200 test** split, seed `20260618`,
created once and cached. **The 200 test items are never read by any analysis here.**
All modeling and figures use train only. (Exception by design: the item-difficulty
histogram in §3e is a pure correctness-count distribution and uses all 1,000 — it
touches no rubric features and no train/test modeling.)

Caveat we logged: the split is **item-level random, not paper-grouped**, so a handful
of multi-figure papers can have one figure in train and another in test. Within CV we
group by `paperid` so no paper straddles folds, but the train/test boundary itself
doesn't — a small, known leakage path if we later score the held-out test set.

## 3. The analyses

Shared conventions across all figures:
- **Global dim order** (by pooled Mann-Whitney significance, same on every chart):
  `VL, AS, MCr, MCu, MA, VO, AT, GS, QLl, KNf, QLq, CL`.
- **Emergent figure-grounding dims** `{VL, GS, MA}` are highlighted purple/◆ (authored
  in-house, vs the standard DeLeAn dims).
- **Models ordered by reasoning accuracy**, descending (Human first → GPT-4o-Random
  last), so per-model pages are comparable.
- **Pooled vs per-model**: "pooled / all-models" = every (model, item) observation
  thrown into one set (an item recurs once per model, weighted by how many models saw
  it). "per-model" = that model's items only. We use pooled for aggregate views and
  per-model where the per-subject unit matters.

### 3a. Mann-Whitney: demand by correctness — `mann_whitney_analysis.py`
**What:** per model, split items into CORRECT vs INCORRECT (reasoning), and for each
dim compare the demand-score distributions with a two-sided Mann-Whitney U test.
**Why:** the first, most direct question — *do harder-on-dimension-X items get failed
more?* Replicates SciVer's `viz_rubric_demand.py`.
**Form:** started as grouped bar charts (mean ± SEM + significance stars); **changed to
split violins + interspersed jittered points + gold mean diamonds** so the full
distribution is visible, not just the mean. Strip points are subsampled (≤600/dim/group)
so the 31k-observation pooled chart stays legible.
**Output:** 39 per-model + 1 pooled chart, `charxiv_mann_whitney_all.pdf`, CSVs.
**Finding:** failed items demand **more** on most dims (Δ = incorrect − correct positive,
highly significant for VL/AS/MCr/MCu/MA/VO/AT) — same direction as SciVer. **GS is the
exception** (correct items demand *more* Gestalt/shape). CL/QLq/KNf are n.s. pooled.

### 3b. Per-dimension line graphs — `dim_lines.py`
**What:** 12 figures (one per dim); x-axis = 39 models (45° labels, accuracy order),
two lines (correct/incorrect), marker area ∝ #items behind each mean.
**Why:** shows whether the correct-vs-incorrect demand gap is consistent *across models*
for a given dimension, rather than collapsing everything into one chart.
**Error bars:** initially SD, **switched to SEM** at request — with SD the bars (~1.2–1.4
on a 0–5 scale) swamp the signal; SEM makes the per-model gap legible. (SD reflects score
spread; SEM reflects precision of the mean — the latter is what shows separation.)
**Output:** 12 PNGs, PDF, `dim_lines_long.csv`.

### 3c. Kendall tau-b inter-dimension dependency — `kendall_tau.py`
**What:** 12×12 Kendall tau-b correlation of demand dims, computed *within* each group,
as a 3-panel heatmap **CORRECT | INCORRECT | Δ**. 39 per-model + 1 pooled.
**Why:** beyond per-dim means (§3a), does the *coupling between dimensions* change
between solved and failed items? tau-b (not Spearman) because demand is a coarse ordinal
0–5 scale with many ties, which tau-b handles honestly. The Δ panel is auto-scaled to
surface small dependency shifts.
**Scope decision:** we **omitted** SciVer's per-pair permutation + BH-FDR significance
test — at 40 models × 10k permutations × 66 pairs it's intractable. The raw Δτ tables
(`kendall_tau_diff.csv`) are kept so significance on a specific pair can be computed
later.

### 3d. Random forest: rubric demand → failure — `random_forest.py`
**What:** predict `y=1`=INCORRECT (failure as the positive class) from the 12 demand
dims. One RF per model + a `COMBINATION_ALL` pooled model (all 39×800 rows).
**Why:** the multivariate question — taken together, how much do the demand dims predict
*which items a model fails*, and which dims carry it? Replicates `viz_rubric_clf.py`.
**Method:** paper-grouped `StratifiedGroupKFold` (no `paperid` straddles a fold),
**5 folds × 3 repeats = 15 folds**; report mean ± SD across folds. **Permutation
importance** (held-out folds, scored by ROC-AUC).
**Tractability decisions** (vs SciVer's config): RF only (dropped HistGB), 200 trees,
5 permutation repeats, and **drop-column importance omitted** — necessary to run 40
models. Importance is therefore permutation-only.
**Findings:**
- Signal is **real but modest**: per-model AUC mean **0.594** (0.52–0.66); **35/39 beat
  chance by >1 fold-SD**. Pooled AUC **0.573 ± 0.013** (tightest, n=31k → most reliable).
- Top failure-predictors: **GS (Gestalt/Shape) and MA (Multi-Element Aggregation)**,
  then MCu, VL, AS. GS is top-4 in 27/39 models, MA in 23/39; the pooled model agrees.
- **This inverts SciVer.** On SciVer the emergent figure-grounding dims were near-zero
  and failure tracked metacognition/relevant-info; on CharXiv (pure chart-reading) the
  figure-grounding dims **dominate**. A clean, interpretable benchmark difference.
**Reading caveats (also on the PDF cover page):** weak predictor → use for explanation,
not deployment. Importance shows **magnitude not direction** — a dim can be a strong RF
predictor while, in the Mann-Whitney view, higher demand went with *correct* answers
(GS); the RF flags it as informative either way. Error bars on importance and the AUC ±
are **SD across the 15 folds** (population, ddof=0) — not SEM, and not the within-shuffle
spread (that inner variability is averaged away per fold first).
**Cover page:** page 1 of `charxiv_random_forest_all.pdf` is a reader's guide explaining
the panels, axes, and conclusions (added via `cover_page()` in the script; merged onto
the existing PDF with pypdf to avoid a full rerun).

### 3e. Item difficulty histogram — `item_difficulty.py`
**What:** for each of the 1,000 items, count how many of the 39 models solve its
reasoning question; histogram those counts (x = #models correct 0–39, y = #items).
**Why:** a per-item "solvability" distribution — orthogonal to demand, it characterises
the benchmark's difficulty spread directly.
**Finding:** strongly right-skewed. Mean ≈ 11, median 9; **16 items solved by no model**,
**none solved by all 39** (ceiling 36). CharXiv reasoning is hard across the board.

### 3f. Radial / ADeLe-style figure — `radial_diagrams.py`
**What:** styled after the General Scales / ADeLe Fig. 1. **Panel a** = ability radar;
**panel b** = CharXiv demand-frequency polar stacked histogram (12 dims, levels 0–5).
Plus per-model ability radars spanning the accuracy range.
**The key methodological decision (ability definition):** the naive "ability = demand
level where P(correct) crosses 0.5" **collapses** here — pooled accuracy is ~28%, so the
characteristic curves sit below 0.5 and the crossing is undefined for almost every dim.
Two separate fixes, per reviewer feedback:
1. **Definition:** use the **area under the fitted logistic characteristic curve** over
   [0,5] (the paper's preferred AUC definition), not the crossing. It degrades
   gracefully (stays a small positive number on the demand scale when the left plateau
   is low), stays non-negative, and keeps dimensions comparable. The logistic's level-0
   value is allowed to float to the model's *base reliability* rather than being forced
   to 1.0.
2. **Unit:** ability is **per-model**, not a property of a pooled mixture. A single
   curve over all 39 models is the success rate of a 39-model mixture (~28%), which
   reads as near-zero ability everywhere and is *not* an ability profile. So panel A
   shows the **mean of the 39 per-model abilities** (a mean of per-subject abilities),
   and we add per-model radars for models spanning the range.
**Result:** mean AUC-abilities span ~0.95 (MCu) to ~1.57 (GS/AS) — non-degenerate.
Per-model radars show Human's profile lifted off the floor and ChartGemma/GPT-4o-Random
genuinely collapsed toward the center — a real result, not an artifact.
**Caveat & extra:** AUC-as-is **blends base competence with demand tolerance** when the
plateau is low. To isolate demand tolerance we also compute a **floor/ceiling-normalized
ability** ("the demand at which the model loses half of whatever competence it has") and
store it in `ability_long.csv`; the figure uses AUC-as-is and says so. Radial axis is
**scaled to the data**, not a fixed 0–8, so small profiles aren't dots.

## 4. SciVer reference & what didn't port

The SciVer rubric→correctness sequence was: **MWU bar → Spearman → Kendall tau-b →
RF (+HistGB) classifier → frozen-RF-on-4/5-items**. On CharXiv we did MWU, Kendall,
and RF (we went straight to Kendall, skipping a standalone Spearman, since tau-b is the
preferred ordinal measure). The **frozen-RF-on-4/5** step did **not** port — it exploits
SciVer's paired entailed×refuted twin design, which CharXiv has no analog for.

Label-structure difference worth remembering: SciVer had *one* confident label per item
(5/5 unanimous across paired runs); CharXiv has a label per **(model, item)**. That's
why every CharXiv analysis chooses explicitly between **pooled** and **per-model**.

## 5. Open decisions / possible next steps
- Apply the RF to the held-out 200-item test set (requires refitting on full train; CV
  already estimates this — test would be the lockbox confirmation). Note the
  item-level-split caveat in §2.
- Boosting comparison: `HistGradientBoostingClassifier` (no new dep, matches SciVer)
  rather than XGBoost — the signal, not the learner, is the ceiling (~0.59 AUC).
- SHAP on the pooled RF to recover direction/interactions (resolves "magnitude not
  direction").
- Multi-dataset panel B (SciVer / SciClaimEval demand profiles) for a paper-style
  comparison; per-model normalized demand-tolerance radars.

## 6. Regenerate
```bash
.venv/bin/python charxiv_analysis/mann_whitney_analysis.py
.venv/bin/python charxiv_analysis/dim_lines.py
.venv/bin/python charxiv_analysis/kendall_tau.py
.venv/bin/python charxiv_analysis/random_forest.py     # ~10 min (40 models); cover page included
.venv/bin/python charxiv_analysis/item_difficulty.py
.venv/bin/python charxiv_analysis/radial_diagrams.py
```
All scripts reuse the loaders and cached split from `mann_whitney_analysis.py`
(`load_rubric`, `make_or_load_split`, `load_correctness`), so the split and dim
conventions stay consistent. (numpy 2.x note: use `np.trapezoid`, not `np.trapz`.)
