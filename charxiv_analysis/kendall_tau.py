"""Kendall tau-b inter-dimension dependency of the 12 DeLeAn dims, by reasoning correctness.

CharXiv port of `sciver_eval/viz_rubric_kendall.py`. For each model we split its
reasoning answers into CORRECT vs INCORRECT (train items only), then compute the
12x12 Kendall tau-b matrix of the rubric-demand dimensions within each group and
draw a 3-panel heatmap: CORRECT | INCORRECT | Δ (correct − incorrect). tau-b is
used (not Spearman) because demand is a coarse ordinal 0-5 scale with many ties.

One figure per model (39) plus a POOLED "all-models" panel (every (model,item) train
observation pooled by its correctness label — the same aggregation as the pooled
Mann-Whitney chart). Reuses the cached 800/200 train split (TRAIN ONLY).

    python charxiv_analysis/kendall_tau.py
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
from scipy.stats import kendalltau

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from mann_whitney_analysis import (  # noqa: E402  identical loaders + cached split
    DB, EMERGENT, load_rubric, make_or_load_split, load_correctness,
)

OUTDIR = ROOT / "kendall_tau"
DIM_ORDER = ["VL", "AS", "MCr", "MCu", "MA", "VO", "AT", "GS", "QLl", "KNf", "QLq", "CL"]
MIN_GROUP = 20   # need enough items in a group for a meaningful tau-b


def tau_matrix(rubric, items, dims):
    """12x12 Kendall tau-b (lower triangle computed, mirrored; diagonal 1)."""
    M = np.array([[rubric[i][d] for d in dims] for i in items], dtype=float)
    n = len(dims)
    tau = np.eye(n)
    for a in range(n):
        for b in range(a):
            t = kendalltau(M[:, a], M[:, b]).statistic
            tau[a, b] = tau[b, a] = t if np.isfinite(t) else 0.0
    return tau


def draw(dims, tau_c, tau_i, tau_d, nc, ni, title, out):
    labels = [f"{'◆' if d in EMERGENT else ''}{d}" for d in dims]
    diff_max = float(np.abs(tau_d - np.eye(len(dims))).max()) or 1.0
    panels = [(f"CORRECT (n={nc})", tau_c, 1.0),
              (f"INCORRECT (n={ni})", tau_i, 1.0),
              ("Δ tau-b (correct − incorrect)", tau_d, diff_max)]
    fig, axes = plt.subplots(1, 3, figsize=(21, 7.2))
    for ax, (ptitle, M, vmax) in zip(axes, panels):
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(dims)), labels, fontsize=8, rotation=90)
        ax.set_yticks(range(len(dims)), labels, fontsize=8)
        for t, d in zip(ax.get_xticklabels() + ax.get_yticklabels(), labels + labels):
            if d.startswith("◆"):
                t.set_color("#6a3d9a"); t.set_fontweight("bold")
        for a in range(len(dims)):
            for b in range(len(dims)):
                v = M[a, b]
                ax.text(b, a, f"{v:.2f}".lstrip("0").replace("-0.", "-."),
                        ha="center", va="center", fontsize=6,
                        color="white" if abs(v) > 0.55 * vmax else "0.2")
        ax.set_title(ptitle, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    rubric, names = load_rubric(con)
    train_ids, _test = make_or_load_split(set(rubric))
    correctness = load_correctness(con, train_ids)
    con.close()

    dims = [d for d in DIM_ORDER if d in names]
    model_acc = {m: sum(v.values()) / len(v) for m, v in correctness.items()}
    models = sorted(correctness, key=lambda m: model_acc[m], reverse=True)
    print(f"{len(models)} models, {len(dims)} dims, {len(train_ids)} train items")

    long_rows, diff_rows = [], []
    pdf_path = OUTDIR / "charxiv_kendall_tau_all.pdf"
    with PdfPages(pdf_path) as pdf:
        # --- pooled / all-models panel first -------------------------------------
        pooled_c, pooled_i = [], []
        for m in models:
            for item, ok in correctness[m].items():
                (pooled_c if ok else pooled_i).append(item)
        tc, ti = tau_matrix(rubric, pooled_c, dims), tau_matrix(rubric, pooled_i, dims)
        td = tc - ti
        title = (f"CharXiv rubric-demand Kendall tau-b by reasoning correctness — "
                 f"POOLED across {len(models)} models (train)")
        fig = draw(dims, tc, ti, td, len(pooled_c), len(pooled_i), title,
                   OUTDIR / "pooled_all_models.png")
        pdf.savefig(fig); plt.close(fig)
        _record(long_rows, diff_rows, "POOLED_ALL_MODELS", dims, tc, ti)

        # --- per model -----------------------------------------------------------
        for m in models:
            cids = [i for i, ok in correctness[m].items() if ok]
            iids = [i for i, ok in correctness[m].items() if not ok]
            if len(cids) < MIN_GROUP or len(iids) < MIN_GROUP:
                print(f"  {m:28s} SKIP (corr={len(cids)} inc={len(iids)}, <{MIN_GROUP})")
                continue
            tc, ti = tau_matrix(rubric, cids, dims), tau_matrix(rubric, iids, dims)
            td = tc - ti
            title = (f"CharXiv rubric-demand Kendall tau-b by reasoning correctness — "
                     f"{m} (train)")
            safe = m.replace("/", "_").replace(".", "-")
            fig = draw(dims, tc, ti, td, len(cids), len(iids), title,
                       OUTDIR / f"model_{safe}.png")
            pdf.savefig(fig); plt.close(fig)
            _record(long_rows, diff_rows, m, dims, tc, ti)
            top = max(((dims[a], dims[b], td[a, b]) for a in range(len(dims))
                       for b in range(a + 1, len(dims))), key=lambda r: abs(r[2]))
            print(f"  {m:28s} corr={len(cids):4d} inc={len(iids):4d}  "
                  f"max|Δτ| {top[0]}-{top[1]}={top[2]:+.2f}")

    with open(OUTDIR / "kendall_tau_long.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["model", "group", "dim_i", "dim_j", "tau_b"])
        w.writerows(long_rows)
    with open(OUTDIR / "kendall_tau_diff.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "dim_i", "dim_j", "tau_correct", "tau_incorrect", "delta", "abs_delta"])
        w.writerows(sorted(diff_rows, key=lambda r: (r[0], -abs(float(r[5])))))

    print(f"\nPDF: {pdf_path}")
    print(f"PNGs + CSVs: {OUTDIR}")


def _record(long_rows, diff_rows, model, dims, tc, ti):
    for grp, M in (("correct", tc), ("incorrect", ti)):
        for a in range(len(dims)):
            for b in range(len(dims)):
                long_rows.append([model, grp, dims[a], dims[b], f"{M[a, b]:.4f}"])
    for a in range(len(dims)):
        for b in range(a + 1, len(dims)):
            dlt = tc[a, b] - ti[a, b]
            diff_rows.append([model, dims[a], dims[b], f"{tc[a, b]:.4f}",
                              f"{ti[a, b]:.4f}", f"{dlt:+.4f}", f"{abs(dlt):.4f}"])


if __name__ == "__main__":
    main()
