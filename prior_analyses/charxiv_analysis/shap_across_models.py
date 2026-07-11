"""Across-MODEL SHAP beeswarms: one panel per demand dim, one row per model.

`random_forest.py` draws one beeswarm per model (rows = the 12 demand dims). This
script transposes that view: one beeswarm per DIM, rows = the ~39 models, so you can
read "for demand X, which models' failures are most driven by it." Models are ordered
within each panel by mean|SHAP| for that dim (most sensitive at top); dot colour =
that item's demand level (0-5).

SHAP is recomputed out-of-fold exactly as in random_forest.py (same RF, same
paper-grouped StratifiedGroupKFold, same N_REPEATS), but WITHOUT permutation
importance and WITHOUT the pooled COMBINATION_ALL (a single pooled model is not a
cross-model comparison). Raw out-of-fold SHAP is cached to shap_oof_cache.npz so the
plots can be re-styled without recomputing.

    python charxiv_analysis/shap_across_models.py
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize
import sqlite3

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from mann_whitney_analysis import DB, EMERGENT, load_rubric, make_or_load_split, load_correctness
from random_forest import rf, make_splits, shap_class1, DIM_ORDER, MIN_MINORITY

OUTDIR = ROOT / "shap_across_models"
CACHE = OUTDIR / "shap_oof_cache.npz"
MAX_PER_ROW = 250          # subsample dots per model row for legibility
CMAP = plt.cm.coolwarm     # demand low (blue) -> high (red)


def oof_shap_for(X, y, groups):
    """Out-of-fold TreeSHAP values + the feature rows they explain (no permutation)."""
    sv, ft = [], []
    for tr, te in make_splits(y, groups):
        m = rf().fit(X[tr], y[tr])
        sv.append(shap_class1(m, X[te]))
        ft.append(X[te])
    return np.vstack(sv), np.vstack(ft)


def compute(models, feat, correctness, meta, dims):
    """{model: (oof_shap[n,12], oof_feat[n,12])} for models with enough minority class."""
    out = {}
    for m in models:
        items = list(correctness[m])
        X = np.array([feat[i] for i in items])
        y = np.array([0 if correctness[m][i] else 1 for i in items])
        g = np.array([meta[i] for i in items])
        if min(y.sum(), (y == 0).sum()) < MIN_MINORITY:
            print(f"  {m:28s} SKIP (minority too small)")
            continue
        out[m] = oof_shap_for(X, y, g)
        print(f"  {m:28s} oof rows {out[m][0].shape[0]}")
    return out


def load_or_compute(models, feat, correctness, meta, dims):
    if CACHE.exists():
        z = np.load(CACHE, allow_pickle=True)
        names = list(z["models"])
        data = {n: (z[f"shap_{i}"], z[f"feat_{i}"]) for i, n in enumerate(names)}
        print(f"loaded cached SHAP for {len(names)} models <- {CACHE}")
        return data
    data = compute(models, feat, correctness, meta, dims)
    save = {"models": np.array(list(data))}
    for i, n in enumerate(data):
        save[f"shap_{i}"], save[f"feat_{i}"] = data[n]
    np.savez_compressed(CACHE, **save)
    print(f"cached SHAP -> {CACHE}")
    return data


def draw_dim(dim, di, data, out):
    """Across-model beeswarm for one dim: y = model (by sensitivity), x = SHAP, colour = demand."""
    # rank models by mean|SHAP| for this dim, most sensitive at top
    order = sorted(data, key=lambda m: np.abs(data[m][0][:, di]).mean(), reverse=True)
    rng = np.random.RandomState(0)
    fig, ax = plt.subplots(figsize=(9, max(6, 0.28 * len(order) + 1)))
    for row, m in enumerate(order):
        sv = data[m][0][:, di]
        dem = data[m][1][:, di]
        if len(sv) > MAX_PER_ROW:
            idx = rng.choice(len(sv), MAX_PER_ROW, replace=False)
            sv, dem = sv[idx], dem[idx]
        yv = (len(order) - 1 - row) + (rng.rand(len(sv)) - 0.5) * 0.55
        ax.scatter(sv, yv, c=dem, cmap=CMAP, vmin=0, vmax=5, s=9, alpha=0.6,
                   linewidths=0, zorder=2)
    ax.axvline(0, c="0.3", lw=0.8, zorder=1)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(list(reversed(order)), fontsize=7)
    ax.set_xlabel("SHAP value for this dim  (Δ P(INCORRECT);  right = pushes toward failure)")
    star = " ◆" if dim in EMERGENT else ""
    ax.set_title(f"Per-model SHAP for demand dim {dim}{star}\n(rows = models, top = most "
                 f"{dim}-sensitive; colour = item's {dim} demand 0–5)",
                 color=("#6a3d9a" if dim in EMERGENT else "black"))
    cb = fig.colorbar(ScalarMappable(norm=Normalize(0, 5), cmap=CMAP), ax=ax, pad=0.01)
    cb.set_label("demand level (low → high)", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    rubric, names = load_rubric(con)
    train_ids, _ = make_or_load_split(set(rubric))
    correctness = load_correctness(con, train_ids)
    meta = {r[0]: r[1] for r in con.execute("SELECT item_id, paperid FROM item")}
    con.close()

    dims = [d for d in DIM_ORDER if d in names]
    feat = {i: np.array([rubric[i][d] for d in dims], dtype=float) for i in rubric}
    acc = {m: sum(v.values()) / len(v) for m, v in correctness.items()}
    models = sorted(correctness, key=lambda m: acc[m], reverse=True)
    print(f"{len(models)} models, {len(dims)} dims")

    data = load_or_compute(models, feat, correctness, meta, dims)

    pdf_path = OUTDIR / "shap_across_models.pdf"
    with PdfPages(pdf_path) as pdf:
        for di, dim in enumerate(dims):
            fig = draw_dim(dim, di, data, OUTDIR / f"across_models_{dim}.png")
            pdf.savefig(fig); plt.close(fig)
            print(f"  drew {dim}")
    print(f"\nPDF: {pdf_path}\nPNGs: {OUTDIR}")


if __name__ == "__main__":
    main()
