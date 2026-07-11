"""CharXiv DeLeAn rubric demand vs per-model reasoning correctness (Mann-Whitney).

Replicates the SciVer `sciver_eval/viz_rubric_demand.py` analysis on CharXiv.

For every model in the CharXiv leaderboard we split its reasoning-question answers
into CORRECT vs INCORRECT, then for each of the 12 model-agnostic DeLeAn demand
dimensions compare the demand-score distributions of the two groups with a
two-sided Mann-Whitney U test. One bar chart per model (12 dims, correct vs
incorrect, SEM error bars, significance stars), plus one pooled "all-models
average" chart. Dimensions are shown in a SINGLE global order (by the pooled
chart's MWU significance, p ascending) so every per-model plot is comparable.

Data join (CharXiv `charxiv_scoring/annotations_charxiv.db`):
  * annotation  -> {item_id: {dim_code: demand 0-5}}  (1000 items x 12 dims)
  * model_score -> reasoning task (1 sub_q per figure), score 1=correct / 0=wrong
                   / -1=dropped; item_id maps 1:1 to the rubric items.

Train/test: a fixed 800/200 split is created once and cached in
`charxiv_analysis/train_test_split.json`. ALL analysis here uses TRAIN ONLY;
the 200 test items are never read.

    python charxiv_analysis/mann_whitney_analysis.py
"""
import csv
import json
import sqlite3
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from scipy.stats import mannwhitneyu
import pandas as pd
import seaborn as sns

ROOT = Path(__file__).resolve().parent
DB = ROOT.parent / "charxiv_scoring" / "annotations_charxiv.db"
OUTDIR = ROOT / "mann_whitney"
SPLIT_PATH = ROOT / "train_test_split.json"

N_TRAIN = 800
SEED = 20260618

# Authored figure-grounding dims (emergent), highlighted; the rest are standard DeLeAn.
EMERGENT = {"VL", "GS", "MA"}
EMERGENT_COLOR = "#6a3d9a"
CORRECT_C = "#2ca25f"
INCORRECT_C = "#de2d26"
STRIP_MAX = 600        # cap jittered points per group (pooled groups are huge)


def load_rubric(con):
    """{item_id: {dim_code: float demand}}, {dim_code: dim_name}."""
    rubric = {}
    names = {}
    for item_id, code, name, score in con.execute(
        "SELECT item_id, dim_code, dim_name, score FROM annotation WHERE parse_ok=1"
    ):
        rubric.setdefault(item_id, {})[code] = float(score)
        names[code] = name
    return rubric, names


def make_or_load_split(item_ids):
    """Deterministic 800/200 split, cached to disk. TEST IDS ARE NEVER USED HERE."""
    if SPLIT_PATH.exists():
        d = json.loads(SPLIT_PATH.read_text())
        return set(d["train_ids"]), set(d["test_ids"])
    ids = sorted(item_ids)
    rng = np.random.default_rng(SEED)
    perm = rng.permutation(len(ids))
    train = sorted(ids[i] for i in perm[:N_TRAIN])
    test = sorted(ids[i] for i in perm[N_TRAIN:])
    SPLIT_PATH.write_text(json.dumps(
        {"seed": SEED, "n_train": len(train), "n_test": len(test),
         "train_ids": train, "test_ids": test}, indent=2))
    print(f"wrote split -> {SPLIT_PATH} ({len(train)} train / {len(test)} test)")
    return set(train), set(test)


def load_correctness(con, train_ids):
    """{model: {item_id: 0/1}} for the reasoning task, train items only, -1 dropped."""
    out = {}
    for model, item_id, score in con.execute(
        "SELECT model, item_id, score FROM model_score WHERE task='reasoning'"
    ):
        if item_id in train_ids and score in (0, 1):
            out.setdefault(model, {})[item_id] = score
    return out


def per_dim_stats(rubric, dims, correct_ids, incorrect_ids):
    """{dim: {correct:(mean,sem), incorrect:(mean,sem), delta, p}}."""
    sem = lambda a: a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
    stats = {}
    for d in dims:
        vc = np.array([rubric[i][d] for i in correct_ids if d in rubric[i]])
        vi = np.array([rubric[i][d] for i in incorrect_ids if d in rubric[i]])
        try:
            p = mannwhitneyu(vc, vi, alternative="two-sided").pvalue
        except ValueError:
            p = 1.0
        mc = vc.mean() if len(vc) else 0.0
        mi = vi.mean() if len(vi) else 0.0
        stats[d] = {"correct": (mc, sem(vc)), "incorrect": (mi, sem(vi)),
                    "delta": mi - mc, "p": p}
    return stats


def _tidy(rubric, dims, correct_ids, incorrect_ids):
    """Long DataFrame [dim, score, group] over the two groups' item-level demand."""
    rows = []
    for grp, ids in (("CORRECT", correct_ids), ("INCORRECT", incorrect_ids)):
        for i in ids:
            for d in dims:
                if d in rubric[i]:
                    rows.append((d, rubric[i][d], grp))
    return pd.DataFrame(rows, columns=["dim", "score", "group"])


CAPTION = ("Split violins = demand distribution (KDE); jittered points = items "
           f"(≤{STRIP_MAX}/group shown); ◆ diamond = mean.  "
           "Δ = incorrect − correct (positive ⇒ failed items higher-demand).  "
           "Mann-Whitney U: *** p<.001, ** p<.01, * p<.05, ns.  "
           "Dims in fixed global order; ◆/purple = emergent dim.")

PALETTE = {"CORRECT": CORRECT_C, "INCORRECT": INCORRECT_C}
HUE_ORDER = ["CORRECT", "INCORRECT"]


def make_figure(dims, names, stats, rubric, correct_ids, incorrect_ids, title):
    df = _tidy(rubric, dims, correct_ids, incorrect_ids)
    n_corr, n_inc = len(correct_ids), len(incorrect_ids)
    fig, ax = plt.subplots(figsize=(16, 7))

    for i, d in enumerate(dims):                      # shade emergent dims
        if d in EMERGENT:
            ax.axvspan(i - 0.5, i + 0.5, color=EMERGENT_COLOR, alpha=0.08, zorder=0)

    sns.violinplot(data=df, x="dim", y="score", hue="group", order=dims,
                   hue_order=HUE_ORDER, split=True, gap=0.12, cut=0, inner=None,
                   density_norm="width", palette=PALETTE, linewidth=0.8, alpha=0.55, ax=ax)
    # jittered points (subsample large groups so pooled charts stay legible/fast)
    cap = STRIP_MAX * len(dims)
    strip = pd.concat([sub.sample(min(len(sub), cap), random_state=SEED)
                       for _, sub in df.groupby("group")])
    sns.stripplot(data=strip, x="dim", y="score", hue="group", order=dims,
                  hue_order=HUE_ORDER, dodge=True, jitter=0.18, size=1.8, alpha=0.35,
                  palette={"CORRECT": "#08603a", "INCORRECT": "#7a1410"},
                  legend=False, ax=ax)
    # mean diamonds per group/dim
    x = np.arange(len(dims))
    off = 0.2
    for sign, key in ((-off, "correct"), (off, "incorrect")):
        ax.scatter(x + sign, [stats[d][key][0] for d in dims], marker="D",
                   s=34, color="gold", edgecolor="black", linewidth=0.7, zorder=6)

    star_y = 5.15
    for i, d in enumerate(dims):
        p = stats[d]["p"]
        star = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"
        ax.text(i, star_y, f"{star}\n{stats[d]['delta']:+.2f}",
                ha="center", va="bottom", fontsize=7.5, color="0.25")

    labels = [f"{'◆ ' if d in EMERGENT else ''}{d}\n{(names.get(d) or '')[:16]}" for d in dims]
    ax.set_xticks(x, labels, fontsize=8)
    for t, d in zip(ax.get_xticklabels(), dims):
        if d in EMERGENT:
            t.set_color(EMERGENT_COLOR)
            t.set_fontweight("bold")
    ax.set_ylabel("rubric demand")
    ax.set_ylim(-0.4, 6.0)
    ax.set_yticks(range(6))
    ax.set_title(title, fontsize=12, pad=10)
    ax.margins(x=0.01)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    h = [plt.Rectangle((0, 0), 1, 1, color=PALETTE[g]) for g in HUE_ORDER]
    ax.legend(h, [f"CORRECT (n={n_corr})", f"INCORRECT (n={n_inc})"],
              loc="upper right", fontsize=9)

    fig.text(0.5, 0.005, CAPTION, ha="center", va="bottom", fontsize=8.5, color="0.4")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


def write_csv(path, dims, names, stats, n_corr, n_inc, model):
    with open(path, "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["model", "dim_code", "dim_name", "emergent",
                     "mean_correct", "sem_correct", "mean_incorrect", "sem_incorrect",
                     "delta_inc_minus_corr", "mannwhitney_p", "n_correct", "n_incorrect"])
        for d in dims:
            mc, sc = stats[d]["correct"]
            mi, si = stats[d]["incorrect"]
            wr.writerow([model, d, names.get(d, ""), "yes" if d in EMERGENT else "no",
                         f"{mc:.3f}", f"{sc:.3f}", f"{mi:.3f}", f"{si:.3f}",
                         f"{stats[d]['delta']:+.3f}", f"{stats[d]['p']:.4g}",
                         n_corr, n_inc])


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)

    rubric, names = load_rubric(con)
    dims_all = sorted(names)  # 12 dim codes
    train_ids, test_ids = make_or_load_split(set(rubric))
    correctness = load_correctness(con, train_ids)
    con.close()

    # order models by reasoning accuracy, descending (same order as dim_lines.py)
    model_acc = {m: sum(v.values()) / len(v) for m, v in correctness.items()}
    models = sorted(correctness, key=lambda m: model_acc[m], reverse=True)
    print(f"{len(dims_all)} dims, {len(train_ids)} train items, {len(models)} models")

    # --- pooled "all-models average" chart: defines the global dim order ----------
    pooled_c, pooled_i = [], []  # parallel lists of item_ids (with repeats across models)
    for m in models:
        for item, ok in correctness[m].items():
            (pooled_c if ok else pooled_i).append(item)
    pooled_stats = per_dim_stats(rubric, dims_all, pooled_c, pooled_i)
    dims = sorted(dims_all, key=lambda d: (pooled_stats[d]["p"], -abs(pooled_stats[d]["delta"])))
    print("global dim order (by pooled significance): " + ", ".join(dims))

    # --- render: pooled chart first, then one per model, into PNGs + a single PDF -
    combined_rows = []
    pdf_path = OUTDIR / "charxiv_mann_whitney_all.pdf"
    with PdfPages(pdf_path) as pdf:
        title = (f"CharXiv DeLeAn demand by reasoning correctness — POOLED across "
                 f"{len(models)} models (train, n={len(pooled_c)+len(pooled_i)} obs)")
        fig = make_figure(dims, names, pooled_stats, rubric, pooled_c, pooled_i, title)
        fig.savefig(OUTDIR / "average_all_models.png", dpi=150, bbox_inches="tight")
        pdf.savefig(fig); plt.close(fig)
        write_csv(OUTDIR / "average_all_models.csv", dims, names, pooled_stats,
                  len(pooled_c), len(pooled_i), "POOLED_ALL_MODELS")

        for m in models:
            cids = [i for i, ok in correctness[m].items() if ok]
            iids = [i for i, ok in correctness[m].items() if not ok]
            st = per_dim_stats(rubric, dims, cids, iids)
            title = f"CharXiv DeLeAn demand by reasoning correctness — {m} (train)"
            fig = make_figure(dims, names, st, rubric, cids, iids, title)
            safe = m.replace("/", "_").replace(".", "-")
            fig.savefig(OUTDIR / f"model_{safe}.png", dpi=150, bbox_inches="tight")
            pdf.savefig(fig); plt.close(fig)
            for d in dims:
                mc, sc = st[d]["correct"]; mi, si = st[d]["incorrect"]
                combined_rows.append([m, d, names.get(d, ""), "yes" if d in EMERGENT else "no",
                                      f"{mc:.3f}", f"{sc:.3f}", f"{mi:.3f}", f"{si:.3f}",
                                      f"{st[d]['delta']:+.3f}", f"{st[d]['p']:.4g}",
                                      len(cids), len(iids)])
            sig = sum(st[d]["p"] < 0.05 for d in dims)
            print(f"  {m:28s} corr={len(cids):4d} inc={len(iids):4d}  {sig}/12 dims p<.05")

    with open(OUTDIR / "per_model_long.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["model", "dim_code", "dim_name", "emergent", "mean_correct",
                     "sem_correct", "mean_incorrect", "sem_incorrect",
                     "delta_inc_minus_corr", "mannwhitney_p", "n_correct", "n_incorrect"])
        wr.writerows(combined_rows)

    print(f"\nPDF: {pdf_path}")
    print(f"PNGs + CSVs: {OUTDIR}")


if __name__ == "__main__":
    main()
