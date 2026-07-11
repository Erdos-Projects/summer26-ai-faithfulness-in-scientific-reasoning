"""Per-dimension line graphs: mean demand (±SD) across all 39 models, correct vs incorrect.

For each of the 12 DeLeAn demand dimensions, one figure with the 39 CharXiv models
on the x-axis (labels at 45°) and mean rubric demand on the y-axis. Two lines per
plot — one for each model's CORRECT reasoning answers, one for its INCORRECT — with
SEM error bars. Each marker's SIZE encodes the group's item count for that model
(bigger point = more items behind the mean).

Reuses the cached 800/200 train split and data loaders from
`mann_whitney_analysis.py` (TRAIN ONLY; the 200 test items are never read).
Models are ordered by reasoning accuracy (descending), shared across all 12 plots;
dimensions follow the same global significance order as the Mann-Whitney charts.

    python charxiv_analysis/dim_lines.py
"""
import csv
import sqlite3
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.lines import Line2D

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from mann_whitney_analysis import (  # noqa: E402  reuse identical loaders + split
    DB, EMERGENT, EMERGENT_COLOR, load_rubric, make_or_load_split, load_correctness,
)

OUTDIR = ROOT / "dim_lines"

# Same global dim order as the Mann-Whitney charts (by pooled significance).
DIM_ORDER = ["VL", "AS", "MCr", "MCu", "MA", "VO", "AT", "GS", "QLl", "KNf", "QLq", "CL"]

CORRECT_C = "#2ca25f"
INCORRECT_C = "#de2d26"
SIZE_K = 0.55          # marker area = n * SIZE_K  (n=100 -> 55 pt^2, n=600 -> 330)
SIZE_LEGEND_N = [100, 300, 600]


def group_stats(rubric, dim, item_ids):
    """(mean, sem, n) of `dim` demand over item_ids (SEM = SD / sqrt(n))."""
    v = np.array([rubric[i][dim] for i in item_ids if dim in rubric[i]])
    if len(v) == 0:
        return np.nan, 0.0, 0
    sem = v.std(ddof=1) / np.sqrt(len(v)) if len(v) > 1 else 0.0
    return v.mean(), sem, len(v)


def make_plot(dim, dim_name, models, model_acc, stats):
    """stats[model] = {'correct': (m,sem,n), 'incorrect': (m,sem,n)}."""
    x = np.arange(len(models))
    fig, ax = plt.subplots(figsize=(16, 7))

    for key, color, label in (("correct", CORRECT_C, "CORRECT"),
                              ("incorrect", INCORRECT_C, "INCORRECT")):
        m = np.array([stats[mod][key][0] for mod in models])
        sem = np.array([stats[mod][key][1] for mod in models])
        n = np.array([stats[mod][key][2] for mod in models])
        ax.errorbar(x, m, yerr=sem, fmt="-", color=color, lw=1.4, alpha=0.85,
                    elinewidth=1, capsize=2.5, ecolor=color, zorder=2, label=label)
        ax.scatter(x, m, s=n * SIZE_K, color=color, edgecolor="white", linewidth=0.6,
                   zorder=3)

    emph = dim in EMERGENT
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha="right", fontsize=8)
    ax.set_ylabel(f"mean rubric demand for {dim}  (±SEM)")
    title = f"{dim} — {dim_name}   (demand by reasoning correctness, per model, train)"
    ax.set_title(title, fontsize=13, pad=10,
                 color=EMERGENT_COLOR if emph else "black",
                 fontweight="bold" if emph else "normal")
    ax.set_ylim(bottom=0)
    ax.margins(x=0.01)
    ax.grid(axis="y", alpha=0.3)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

    # line legend + a separate marker-size legend
    line_leg = ax.legend(loc="upper right", fontsize=9, title="line")
    ax.add_artist(line_leg)
    size_handles = [Line2D([0], [0], marker="o", linestyle="", color="0.4",
                           markersize=np.sqrt(nn * SIZE_K), markeredgecolor="white",
                           label=f"{nn} items") for nn in SIZE_LEGEND_N]
    ax.legend(handles=size_handles, loc="upper left", fontsize=8,
              title="marker size = # items", labelspacing=1.3, borderpad=1.0)

    fig.text(0.5, 0.005,
             "Models ordered by reasoning accuracy (left=highest).  "
             "Marker area ∝ # correct/incorrect items behind each mean.  "
             "Error bars = SEM (SD / √n) of demand within group.",
             ha="center", va="bottom", fontsize=8.5, color="0.4")
    fig.tight_layout(rect=[0, 0.04, 1, 1])
    return fig


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    rubric, names = load_rubric(con)
    train_ids, _test = make_or_load_split(set(rubric))
    correctness = load_correctness(con, train_ids)
    con.close()

    models = list(correctness)
    model_acc = {}
    for mod in models:
        vals = list(correctness[mod].values())
        model_acc[mod] = sum(vals) / len(vals)
    models = sorted(models, key=lambda mod: model_acc[mod], reverse=True)

    dims = [d for d in DIM_ORDER if d in names]
    print(f"{len(models)} models, {len(dims)} dims, {len(train_ids)} train items")

    rows = []
    pdf_path = OUTDIR / "charxiv_dim_lines_all.pdf"
    with PdfPages(pdf_path) as pdf:
        for dim in dims:
            stats = {}
            for mod in models:
                cids = [i for i, ok in correctness[mod].items() if ok]
                iids = [i for i, ok in correctness[mod].items() if not ok]
                stats[mod] = {"correct": group_stats(rubric, dim, cids),
                              "incorrect": group_stats(rubric, dim, iids)}
            fig = make_plot(dim, names[dim], models, model_acc, stats)
            fig.savefig(OUTDIR / f"dim_{dim}.png", dpi=150, bbox_inches="tight")
            pdf.savefig(fig)
            plt.close(fig)
            for mod in models:
                cm, csem, cn = stats[mod]["correct"]
                im, isem, in_ = stats[mod]["incorrect"]
                rows.append([dim, names[dim], mod, f"{model_acc[mod]:.4f}",
                             f"{cm:.3f}", f"{csem:.3f}", cn,
                             f"{im:.3f}", f"{isem:.3f}", in_])
            print(f"  {dim:4s} {names[dim][:30]:30s} rendered")

    with open(OUTDIR / "dim_lines_long.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["dim_code", "dim_name", "model", "reasoning_accuracy",
                     "mean_correct", "sem_correct", "n_correct",
                     "mean_incorrect", "sem_incorrect", "n_incorrect"])
        wr.writerows(rows)

    print(f"\nPDF: {pdf_path}")
    print(f"PNGs + CSV: {OUTDIR}")


if __name__ == "__main__":
    main()
