"""ADeLe-style radial figure: (a) model ability radar, (b) dataset demand histogram.

Styled after Fig. 1 of the General Scales / ADeLe work.

Panel A — ability radar. For each (model, dim) we fit a logistic characteristic
curve P(correct) vs demand level, then define ABILITY = the area under that fitted
curve over the demand range [0,5] (the paper's AUC definition, used instead of the
0.5-crossing because on a hard benchmark the curve's left plateau sits well below 1
and the crossing is undefined — the integral degrades gracefully and stays on the
0-5 demand scale). Ability is a per-MODEL quantity; the main panel shows the MEAN of
the 39 per-model abilities (the correct "average of all models", i.e. a mean of
per-subject abilities — NOT the success-integral of a pooled 39-model mixture, which
on a 28%-accuracy benchmark would read near-zero everywhere). A small-multiples grid
shows individual per-model radars spanning the accuracy range.

Panel B — demand histogram. Polar stacked bars: 12 dims around the circle, each split
by demand level 0-5 (fraction of items at each level), i.e. CharXiv's demand profile.

AUC-ability blends base competence with demand tolerance when the plateau is low; a
floor/ceiling-normalized ability ("demand at which the model loses half of whatever
competence it has") is also written to CSV. The figure uses AUC-as-is.

Train split only (800 items). Output: charxiv_analysis/radial_diagrams/

    python charxiv_analysis/radial_diagrams.py
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
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from mann_whitney_analysis import (  # noqa: E402  identical loaders + cached split
    DB, EMERGENT, EMERGENT_COLOR, load_rubric, make_or_load_split, load_correctness,
)

OUTDIR = ROOT / "radial_diagrams"
DIM_ORDER = ["VL", "AS", "MCr", "MCu", "MA", "VO", "AT", "GS", "QLl", "KNf", "QLq", "CL"]
LEVELS = list(range(6))           # demand 0..5
GRID = np.linspace(0, 5, 51)      # integration grid over the demand scale
TRAPZ = np.trapezoid if hasattr(np, "trapezoid") else np.trapz
# a few models spanning the accuracy range for the per-model small-multiples
SPAN_MODELS = ["Human", "GPT-4o", "Claude-3-5-Sonnet", "InternVL2-26B",
               "ChartGemma", "GPT-4o-Random"]
# panel-b datasets: each scored on the SAME 12 DeLeAn dims (full coverage).
# Demand is a dataset property (no model/correctness, no leakage), so each uses its
# FULL item set, not a train split.
DATASETS = [
    ("CharXiv", ROOT.parent / "charxiv_scoring" / "annotations_charxiv.db"),
    ("SciVer", ROOT.parent / "rubric_scoring" / "annotations_prod.db"),
    ("SciClaimEval", ROOT.parent / "sciclaimeval_scoring" / "annotations_sciclaimeval.db"),
]


def load_demand_freq(db_path, dims):
    """Per-dim fraction of items at each demand level 0-5, from each dim's full-coverage
    pass (mirrors viz_rubric_demand: a dim may be split across passes; take the pass that
    covers ALL items). Returns (freq{dim:{level:frac}}, n_items)."""
    con = sqlite3.connect(db_path)
    nit = con.execute("SELECT COUNT(DISTINCT item_id) FROM annotation WHERE parse_ok=1").fetchone()[0]
    dim_pass = {}
    for d, p, n in con.execute(
        "SELECT dim_code, pass, COUNT(DISTINCT item_id) FROM annotation "
        "WHERE parse_ok=1 GROUP BY dim_code, pass"):
        if n == nit and d in dims:
            dim_pass[d] = p
    freq = {}
    for d in dims:
        scores = [s for (s,) in con.execute(
            "SELECT score FROM annotation WHERE dim_code=? AND pass=? AND parse_ok=1",
            (d, dim_pass[d]))]
        arr = np.array(scores)
        freq[d] = {lv: float((arr == lv).mean()) for lv in LEVELS}
    con.close()
    return freq, nit


def fitted_curve(levels, ys):
    """P(correct) over GRID from a regularized logistic fit; flat at mean if 1 class."""
    levels = np.asarray(levels, float).reshape(-1, 1)
    ys = np.asarray(ys, int)
    if len(np.unique(ys)) < 2:
        return np.full_like(GRID, float(ys.mean()))
    lr = LogisticRegression(C=1.0, solver="lbfgs")
    lr.fit(levels, ys)
    return lr.predict_proba(GRID.reshape(-1, 1))[:, 1]


def abilities(rubric, dims, dem, items_correct):
    """{dim: (auc_ability, norm_ability)} for one model. items_correct: {item: 0/1}."""
    out = {}
    for d in dims:
        L = [dem[d][i] for i in items_correct]
        y = [items_correct[i] for i in items_correct]
        p = fitted_curve(L, y)
        auc = float(TRAPZ(p, GRID))                       # area over [0,5], 0..5 scale
        rng = p.max() - p.min()
        pn = (p - p.min()) / rng if rng > 1e-9 else np.zeros_like(p)
        out[d] = (auc, float(TRAPZ(pn, GRID)))
    return out


def radar(ax, dims, vals, color, rmax, fill=True, lw=2.0, label=None):
    ang = np.linspace(0, 2 * np.pi, len(dims), endpoint=False)
    a = np.concatenate([ang, ang[:1]])
    v = np.concatenate([vals, vals[:1]])
    ax.plot(a, v, color=color, lw=lw, label=label, zorder=3)
    if fill:
        ax.fill(a, v, color=color, alpha=0.18, zorder=2)
    ax.set_xticks(ang)
    ax.set_xticklabels(dims, fontsize=9)
    for t, d in zip(ax.get_xticklabels(), dims):
        if d in EMERGENT:
            t.set_color(EMERGENT_COLOR); t.set_fontweight("bold")
    ax.set_ylim(0, rmax)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(alpha=0.4)


def demand_polar(ax, dims, freq):
    """Stacked polar bars: each dim a wedge, segments = demand levels 0-5 (fractions)."""
    ang = np.linspace(0, 2 * np.pi, len(dims), endpoint=False)
    width = 2 * np.pi / len(dims) * 0.92
    cmap = plt.get_cmap("viridis")
    colors = [cmap(x) for x in np.linspace(0.05, 0.95, len(LEVELS))]
    bottoms = np.zeros(len(dims))
    for lv in LEVELS:
        fr = np.array([freq[d][lv] for d in dims])
        ax.bar(ang, fr, width=width, bottom=bottoms, color=colors[lv],
               edgecolor="white", linewidth=0.5, label=f"level {lv}", zorder=2)
        bottoms += fr
    ax.set_xticks(ang)
    ax.set_xticklabels(dims, fontsize=9)
    for t, d in zip(ax.get_xticklabels(), dims):
        if d in EMERGENT:
            t.set_color(EMERGENT_COLOR); t.set_fontweight("bold")
    ax.set_yticklabels([])
    ax.set_ylim(0, 1.0)
    ax.set_theta_offset(np.pi / 2)
    ax.set_theta_direction(-1)
    ax.grid(alpha=0.3)


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    rubric, names = load_rubric(con)
    train_ids, _test = make_or_load_split(set(rubric))
    correctness = load_correctness(con, train_ids)
    con.close()

    dims = [d for d in DIM_ORDER if d in names]
    dem = {d: {i: rubric[i][d] for i in rubric} for d in dims}
    model_acc = {m: sum(v.values()) / len(v) for m, v in correctness.items()}
    models = sorted(correctness, key=lambda m: model_acc[m], reverse=True)

    # per-model AUC-abilities (+ normalized), and the across-model mean
    ab = {m: abilities(rubric, dims, dem, correctness[m]) for m in models}
    mean_auc = {d: np.mean([ab[m][d][0] for m in models]) for d in dims}
    print(f"{len(models)} models, {len(dims)} dims, {len(train_ids)} train items")
    print("mean AUC-ability per dim:")
    for d in dims:
        print(f"  {d:4s} {mean_auc[d]:.3f}")

    # demand profiles for every dataset (full item set each; same 12 dims)
    demand = {}
    for name, path in DATASETS:
        freq, nit = load_demand_freq(path, dims)
        demand[name] = (freq, nit)
        print(f"demand profile: {name:14s} n={nit}")

    # ---- main figure: panel a (CharXiv ability radar) + panel b (3 demand polars) -
    rmax = max(mean_auc.values()) * 1.15
    fig = plt.figure(figsize=(23, 6.6))
    axA = fig.add_subplot(1, 4, 1, polar=True)
    radar(axA, dims, np.array([mean_auc[d] for d in dims]), "#1f78b4", rmax)
    axA.set_title(f"CharXiv  (mean across {len(models)} models)", fontsize=11, pad=20)
    fig.text(0.06, 0.95, "a   Model ability", fontsize=14, fontweight="bold")
    fig.text(0.30, 0.95, "b   Demand profiles  (fraction of items per demand level)",
             fontsize=14, fontweight="bold")
    last = None
    for k, (name, _path) in enumerate(DATASETS):
        ax = fig.add_subplot(1, 4, k + 2, polar=True)
        freq, nit = demand[name]
        demand_polar(ax, dims, freq)
        ax.set_title(f"{name}  (n={nit})", fontsize=11, pad=20)
        last = ax
    last.legend(loc="center left", bbox_to_anchor=(1.12, 0.5), fontsize=8,
                title="demand level", frameon=False)
    fig.text(0.5, 0.015, "Ability = area under the fitted logistic characteristic curve "
             "(P(correct) vs demand) over [0,5], AUC-as-is, 0-5 demand scale (CharXiv train, "
             f"{len(models)} models).  Demand profiles use each dataset's full item set.  "
             "◆/purple = emergent figure-grounding dims.", ha="center", fontsize=8.5, color="0.4")
    fig.tight_layout(rect=[0, 0.05, 1, 0.93])
    fig.savefig(OUTDIR / "fig1_ability_and_demand.png", dpi=150, bbox_inches="tight")

    # ---- small multiples: per-model radars spanning the range --------------------
    span = [m for m in SPAN_MODELS if m in ab]
    grmax = max(ab[m][d][0] for m in span for d in dims) * 1.12
    fig2 = plt.figure(figsize=(16, 10))
    for k, m in enumerate(span):
        ax = fig2.add_subplot(2, 3, k + 1, polar=True)
        radar(ax, dims, np.array([ab[m][d][0] for d in dims]), "#1f78b4", grmax)
        ax.set_title(f"{m}\n(acc {model_acc[m]:.2f})", fontsize=11, pad=18)
    fig2.suptitle("Per-model ability radars (AUC-ability, shared radial scale) — "
                  "spanning the accuracy range", fontsize=14, y=0.99)
    fig2.tight_layout(rect=[0, 0, 1, 0.96])
    fig2.savefig(OUTDIR / "per_model_ability_radars.png", dpi=150, bbox_inches="tight")

    # ---- all 39 per-model ability radars, paginated, shared radial scale ---------
    all_rmax = max(ab[m][d][0] for m in models for d in dims) * 1.12
    cols, rows = 3, 4
    per_page = cols * rows
    with PdfPages(OUTDIR / "all_model_ability_radars.pdf") as pdf:
        for start in range(0, len(models), per_page):
            chunk = models[start:start + per_page]
            figp = plt.figure(figsize=(13, 16))
            for k, m in enumerate(chunk):
                ax = figp.add_subplot(rows, cols, k + 1, polar=True)
                radar(ax, dims, np.array([ab[m][d][0] for d in dims]), "#1f78b4", all_rmax)
                ax.set_title(f"{m}\n(acc {model_acc[m]:.2f})", fontsize=10, pad=14)
            page = start // per_page + 1
            npages = (len(models) + per_page - 1) // per_page
            figp.suptitle(f"CharXiv per-model ability radars (AUC-ability, shared scale) — "
                          f"accuracy order — page {page}/{npages}", fontsize=13, y=0.995)
            figp.tight_layout(rect=[0, 0, 1, 0.97])
            pdf.savefig(figp)
            plt.close(figp)

    with PdfPages(OUTDIR / "charxiv_radial_all.pdf") as pdf:
        pdf.savefig(fig); pdf.savefig(fig2)
    plt.close(fig); plt.close(fig2)

    # ---- CSVs --------------------------------------------------------------------
    with open(OUTDIR / "ability_long.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "reasoning_acc", "dim", "dim_name", "emergent",
                    "auc_ability", "normalized_ability"])
        for m in models:
            for d in dims:
                auc, nrm = ab[m][d]
                w.writerow([m, f"{model_acc[m]:.4f}", d, names.get(d, ""),
                            "yes" if d in EMERGENT else "no", f"{auc:.4f}", f"{nrm:.4f}"])
        for d in dims:
            w.writerow(["MEAN_ACROSS_MODELS", "", d, names.get(d, ""),
                        "yes" if d in EMERGENT else "no", f"{mean_auc[d]:.4f}", ""])
    with open(OUTDIR / "demand_frequency.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dataset", "n_items", "dim", "dim_name", "emergent"]
                   + [f"level_{lv}_frac" for lv in LEVELS])
        for name, _path in DATASETS:
            freq, nit = demand[name]
            for d in dims:
                w.writerow([name, nit, d, names.get(d, ""), "yes" if d in EMERGENT else "no"]
                           + [f"{freq[d][lv]:.4f}" for lv in LEVELS])

    print(f"\nfigures + PDF + CSVs: {OUTDIR}")


if __name__ == "__main__":
    main()
