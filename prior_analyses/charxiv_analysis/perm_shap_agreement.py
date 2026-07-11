"""Cross-check the two RF importance methods: permutation vs out-of-fold TreeSHAP.

`random_forest.py` produces, for every model, an importance ranking of the 12
DeLeAn demand dims by two independent methods:
  * permutation importance (ΔAUC when a dim is shuffled)   -> rf_importance_long.csv
  * mean |SHAP| over out-of-fold rows (TreeSHAP)           -> rf_shap_importance_long.csv

If a dim is genuinely predictive of failure it should rank high under BOTH; a dim
that only one method flags is a method artifact (permutation importance in
particular over-credits dims that are collinear with a truly-important one).

This script quantifies agreement (per-model Spearman/Kendall rank correlation,
top-1 / top-3 overlap), highlights divergent dims, and writes a 3-panel figure:
  A — distribution of per-model Spearman rho between the two rankings
  B — COMBINATION_ALL: permutation rank vs SHAP rank, one point per dim
  C — how often each dim lands top-3 under each method (where they disagree)

    python charxiv_analysis/perm_shap_agreement.py
"""
import csv
from collections import defaultdict, Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, kendalltau

ROOT = Path(__file__).resolve().parent
RFDIR = ROOT / "random_forest"
PERM_CSV = RFDIR / "rf_importance_long.csv"
SHAP_CSV = RFDIR / "rf_shap_importance_long.csv"
EMERGENT = {"VL", "GS", "MA"}
EMERGENT_COLOR = "#6a3d9a"
STD_COLOR = "#4878a8"


def load(path, valcol):
    """{model: {dim: value}} from a long importance CSV."""
    d = defaultdict(dict)
    with open(path) as f:
        for r in csv.DictReader(f):
            d[r["model"]][r["dim"]] = float(r[valcol])
    return d


def ranked(dimvals):
    """dims sorted high->low by value."""
    return [d for _, d in sorted(((v, k) for k, v in dimvals.items()), reverse=True)]


def color_for(dim):
    return EMERGENT_COLOR if dim in EMERGENT else STD_COLOR


def compute(perm, shap):
    """Per-model agreement stats + pooled top-3 frequency counters."""
    models = [m for m in perm if m in shap]
    rows, rhos, taus, top1, top3 = [], [], [], [], []
    pc, sc = Counter(), Counter()
    for m in models:
        dims = list(perm[m])
        pv = np.array([perm[m][d] for d in dims])
        sv = np.array([shap[m][d] for d in dims])
        rho = spearmanr(pv, sv).statistic
        tau = kendalltau(pv, sv).statistic
        pr, sr = ranked(perm[m]), ranked(shap[m])
        t3 = len(set(pr[:3]) & set(sr[:3]))
        rhos.append(rho); taus.append(tau)
        top1.append(pr[0] == sr[0]); top3.append(t3)
        pc.update(pr[:3]); sc.update(sr[:3])
        rows.append([m, f"{rho:.4f}", f"{tau:.4f}", pr[0], sr[0],
                     "yes" if pr[0] == sr[0] else "no", t3,
                     " ".join(pr[:3]), " ".join(sr[:3])])
    return models, np.array(rhos), np.array(taus), np.array(top1), np.array(top3), pc, sc, rows


def draw(models, rhos, top1, top3, perm, shap, pc, sc, out):
    fig, (axA, axB, axC) = plt.subplots(1, 3, figsize=(18, 5.5))

    # A — per-model Spearman rho distribution
    axA.hist(rhos, bins=np.arange(0, 1.05, 0.1), color="#4878a8", edgecolor="white")
    axA.axvline(rhos.mean(), color="#de2d26", ls="--",
                label=f"mean {rhos.mean():.2f}")
    axA.axvline(0.5, color="0.5", ls=":", label="rho = 0.5")
    axA.set_xlabel("Spearman rho (perm vs SHAP, 12-dim ranking)")
    axA.set_ylabel(f"models (n={len(models)})")
    axA.set_title(f"A. Ranking agreement per model\n{(rhos > 0.5).mean():.0%} of models rho>0.5, "
                  f"top-1 dim matches {top1.mean():.0%}")
    axA.legend(fontsize=8)

    # B — COMBINATION_ALL rank vs rank (1 = most important)
    m = "COMBINATION_ALL"
    dims = list(perm[m])
    prank = {d: i + 1 for i, d in enumerate(ranked(perm[m]))}
    srank = {d: i + 1 for i, d in enumerate(ranked(shap[m]))}
    for d in dims:
        axB.scatter(prank[d], srank[d], s=70, color=color_for(d), zorder=3)
        axB.annotate(("◆" if d in EMERGENT else "") + d, (prank[d], srank[d]),
                     textcoords="offset points", xytext=(6, 4), fontsize=9,
                     color=color_for(d),
                     fontweight="bold" if d in EMERGENT else "normal")
    lim = len(dims) + 1
    axB.plot([1, lim - 1], [1, lim - 1], color="0.6", ls="--", zorder=1,
             label="perfect agreement")
    axB.set_xlim(0, lim); axB.set_ylim(0, lim)
    axB.invert_xaxis(); axB.invert_yaxis()   # rank 1 (most important) at top-right
    axB.set_xlabel("permutation rank (1 = most important)")
    axB.set_ylabel("SHAP |abs| rank")
    axB.set_title("B. COMBINATION_ALL ranking\n(near diagonal = methods concur)")
    axB.legend(fontsize=8, loc="lower left")

    # C — top-3 frequency by method (sorted by combined; shows divergence e.g. VO)
    dim_order = sorted(set(pc) | set(sc), key=lambda d: -(pc[d] + sc[d]))
    x = np.arange(len(dim_order))
    axC.bar(x - 0.2, [pc[d] for d in dim_order], 0.4, label="permutation",
            color="#4878a8")
    axC.bar(x + 0.2, [sc[d] for d in dim_order], 0.4, label="SHAP |abs|",
            color="#9ec6e0")
    axC.set_xticks(x, [("◆" if d in EMERGENT else "") + d for d in dim_order],
                   rotation=0, fontsize=9)
    for t, d in zip(axC.get_xticklabels(), dim_order):
        if d in EMERGENT:
            t.set_color(EMERGENT_COLOR); t.set_fontweight("bold")
    axC.set_ylabel(f"top-3 appearances (of {len(models)} models)")
    axC.set_title("C. How often each dim is top-3, by method\n(perm-only dims = likely "
                  "collinearity artifacts)")
    axC.legend(fontsize=8)

    fig.suptitle("Permutation vs out-of-fold TreeSHAP importance — agreement check "
                 "(◆ purple = emergent figure-grounding dims)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    return fig


def main():
    perm = load(PERM_CSV, "perm_imp_mean")
    shap = load(SHAP_CSV, "shap_mean_abs")
    models, rhos, taus, top1, top3, pc, sc, rows = compute(perm, shap)

    print(f"{len(models)} models compared")
    print(f"Spearman rho : mean {rhos.mean():.3f}  median {np.median(rhos):.3f}  "
          f"min {rhos.min():.3f}  frac>0.5 {(rhos > 0.5).mean():.0%}")
    print(f"Kendall tau  : mean {taus.mean():.3f}  median {np.median(taus):.3f}")
    print(f"Top-1 dim agreement : {top1.sum()}/{len(models)} ({top1.mean():.0%})")
    print(f"Top-3 overlap (avg)  : {top3.mean():.2f}/3")
    print("\nCOMBINATION_ALL:")
    print("  permutation:", " > ".join(ranked(perm["COMBINATION_ALL"])))
    print("  SHAP |abs| :", " > ".join(ranked(shap["COMBINATION_ALL"])))
    print("\nTop-3 frequency (dim: perm / shap):")
    for d in sorted(set(pc) | set(sc), key=lambda d: -(pc[d] + sc[d])):
        flag = "  <- perm-only (artifact?)" if pc[d] >= 4 and sc[d] == 0 else ""
        print(f"  {d:5s}  perm {pc[d]:2d}  |  shap {sc[d]:2d}{flag}")

    out_png = RFDIR / "perm_shap_agreement.png"
    draw(models, rhos, top1, top3, perm, shap, pc, sc, out_png)

    with open(RFDIR / "perm_shap_agreement.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "spearman_rho", "kendall_tau", "perm_top1", "shap_top1",
                    "top1_match", "top3_overlap", "perm_top3", "shap_top3"])
        w.writerows(rows)

    print(f"\nFigure: {out_png} (+ .pdf)")
    print(f"CSV:    {RFDIR / 'perm_shap_agreement.csv'}")


if __name__ == "__main__":
    main()
