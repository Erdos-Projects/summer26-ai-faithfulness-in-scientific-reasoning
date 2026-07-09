"""Per-item solvability: how many of the 39 models solve each item's reasoning question.

For each of the 1000 CharXiv items, count the models whose reasoning answer is
correct (model_score task='reasoning', score==1). Histogram those counts: x-axis =
#models correct (0-39), y-axis = #items. A right-skewed distribution = most items are
solved by only a few models; the left tail (0) are items no model gets.

Uses ALL 1000 items (this is a pure correctness-count distribution, independent of the
rubric features / train-test split). All 39 leaderboard models are counted, including
the Human ceiling and the GPT-4o-Random baseline.

    python charxiv_analysis/item_difficulty.py
"""
import csv
import sqlite3
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
DB = ROOT.parent / "charxiv_scoring" / "annotations_charxiv.db"
OUTDIR = ROOT / "item_difficulty"


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    n_models = con.execute(
        "SELECT COUNT(DISTINCT model) FROM model_score WHERE task='reasoning'").fetchone()[0]
    rows = con.execute(
        "SELECT item_id, SUM(CASE WHEN score=1 THEN 1 ELSE 0 END) "
        "FROM model_score WHERE task='reasoning' GROUP BY item_id").fetchall()
    con.close()

    counts = np.array([r[1] for r in rows])
    n_items = len(counts)
    print(f"{n_items} items, {n_models} models; solve-count range {counts.min()}-{counts.max()}, "
          f"mean {counts.mean():.2f}, median {int(np.median(counts))}")

    # histogram over integer bins 0..n_models
    edges = np.arange(-0.5, n_models + 1.5, 1)
    hist, _ = np.histogram(counts, bins=edges)

    fig, ax = plt.subplots(figsize=(15, 6.5))
    ax.bar(np.arange(n_models + 1), hist, width=0.9, color="#4878a8", edgecolor="white")
    mean_v, med_v = counts.mean(), float(np.median(counts))
    ax.axvline(mean_v, color="#de2d26", ls="--", lw=1.8, label=f"mean = {mean_v:.1f}")
    ax.axvline(med_v, color="#2ca25f", ls=":", lw=1.8, label=f"median = {med_v:.0f}")
    ax.set_xlabel(f"# models (of {n_models}) that solved the item's reasoning question")
    ax.set_ylabel("# items")
    ax.set_xlim(-0.7, n_models + 0.7)
    ax.set_xticks(range(0, n_models + 1, 2))
    ax.set_title(f"CharXiv per-item solvability — {n_items} items by # of models correct "
                 f"(reasoning task, all {n_models} models)", fontsize=13, pad=10)
    ax.legend(loc="upper right", fontsize=10)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out_png = OUTDIR / "item_solve_count_hist.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)

    # per-item counts + the histogram table
    with open(OUTDIR / "item_solve_counts.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["item_id", "n_models_correct"])
        for iid, c in sorted(rows, key=lambda r: -r[1]):
            w.writerow([iid, c])
    with open(OUTDIR / "solve_count_histogram.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["n_models_correct", "n_items"])
        for k in range(n_models + 1):
            w.writerow([k, int(hist[k])])

    print(f"figure: {out_png}")
    print(f"CSVs: {OUTDIR}")


if __name__ == "__main__":
    main()
