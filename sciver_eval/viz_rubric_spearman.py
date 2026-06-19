"""Spearman correlation of the 12 DeLeAn rubric-demand dims, per confident group.

Reuses the loaders from `viz_rubric_demand` (same 12 full-coverage dims, same
confident paired-correctness labels: entailed run 2 x refuted run 3, 5/5). For
each group (confidently CORRECT, confidently INCORRECT) it computes the
dim x dim Spearman rank-correlation matrix over that group's items, draws a
side-by-side heatmap, and writes the two matrices (long form) to CSV.

    python -m sciver_eval.viz_rubric_spearman [--entailed-run 2 --refuted-run 3]
"""
import argparse
import csv
import sqlite3
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

from sciver_eval import db as sdb
from sciver_eval.viz_rubric_demand import load_rubric, paired_label, RUBRIC_DB, EMERGENT

DEFAULT_OUT = sdb.analysis_dir() / "rubric_spearman_by_correct.png"


def group_matrix(rubric, items, dims):
    """(len(items) x len(dims)) demand matrix, then dim x dim Spearman rho/p."""
    M = np.array([[rubric[i][d] for d in dims] for i in items], dtype=float)
    rho, p = spearmanr(M)                       # column-wise: dims x dims
    rho = np.atleast_2d(rho)
    p = np.atleast_2d(p)
    return rho, p


def draw(dims, names, mats, ns, out):
    fig, axes = plt.subplots(1, 2, figsize=(15, 7.2))
    labels = [f"{'◆' if d in EMERGENT else ''}{d}" for d in dims]
    for ax, (title, rho, n) in zip(axes, mats):
        im = ax.imshow(rho, cmap="RdBu_r", vmin=-1, vmax=1)
        ax.set_xticks(range(len(dims)), labels, fontsize=8, rotation=90)
        ax.set_yticks(range(len(dims)), labels, fontsize=8)
        for t, d in zip(ax.get_xticklabels() + ax.get_yticklabels(), labels + labels):
            if d.startswith("◆"):
                t.set_color("#6a3d9a"); t.set_fontweight("bold")
        for i in range(len(dims)):
            for j in range(len(dims)):
                v = rho[i, j]
                ax.text(j, i, f"{v:.2f}".lstrip("0").replace("-0", "-"),
                        ha="center", va="center", fontsize=6,
                        color="white" if abs(v) > 0.55 else "0.2")
        ax.set_title(f"{title} (n={n})", fontsize=11)
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02, label="Spearman ρ")
    fig.suptitle("Rubric-demand inter-dimension Spearman correlation by confident "
                 "paired correctness (Haiku 5/5, entailed×refuted)", fontsize=12)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(sdb.default_db_path()))
    ap.add_argument("--rubric-db", default=str(RUBRIC_DB))
    ap.add_argument("--entailed-run", type=int, default=2)
    ap.add_argument("--refuted-run", type=int, default=3)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    rubric, dims, names, nitems = load_rubric(args.rubric_db)
    conn = sqlite3.connect(args.db)
    label = paired_label(conn, args.entailed_run, args.refuted_run)
    conn.close()

    groups = {"correct": [], "incorrect": []}
    for item, lab in label.items():
        if item in rubric and all(d in rubric[item] for d in dims):
            groups[lab].append(item)

    rho_c, _ = group_matrix(rubric, groups["correct"], dims)
    rho_i, _ = group_matrix(rubric, groups["incorrect"], dims)
    nc, ni = len(groups["correct"]), len(groups["incorrect"])

    out = Path(args.out)
    draw(dims, names,
         [("confidently CORRECT", rho_c, nc), ("confidently INCORRECT", rho_i, ni)],
         (nc, ni), out)

    csv_path = out.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["group", "dim_i", "dim_j", "spearman_rho"])
        for grp, rho in (("correct", rho_c), ("incorrect", rho_i)):
            for a in range(len(dims)):
                for b in range(len(dims)):
                    w.writerow([grp, dims[a], dims[b], f"{rho[a, b]:.4f}"])

    # console: strongest off-diagonal pairs per group
    print(f"{len(dims)} dims; CORRECT n={nc} INCORRECT n={ni}\ndims: {dims}")
    for grp, rho in (("CORRECT", rho_c), ("INCORRECT", rho_i)):
        pairs = sorted(
            ((rho[a, b], dims[a], dims[b]) for a in range(len(dims)) for b in range(a + 1, len(dims))),
            key=lambda t: -abs(t[0]))
        print(f"\n top |ρ| pairs — {grp}:")
        for r, a, b in pairs[:8]:
            print(f"   {a:>3s}-{b:<3s} ρ={r:+.2f}")
    print(f"\nfigure: {out}\nmatrix: {csv_path}")


if __name__ == "__main__":
    main()
