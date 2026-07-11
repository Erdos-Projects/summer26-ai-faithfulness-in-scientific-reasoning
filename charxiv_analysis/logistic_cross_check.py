"""Third importance leg: a multivariate LOGISTIC model vs the random forest.

The RF permutation + TreeSHAP views (random_forest.py, perm_shap_agreement.py) agree,
but they are BOTH the same model family — a random forest. This script adds a
fundamentally different model: an L2 logistic regression on the same 12 standardized
demand dims -> P(INCORRECT), cross-validated the same paper-grouped way. It answers
three things the RF views cannot:

  1. Model-independence — does a purely linear/additive model also rank GS/MA on top?
     (If yes, the finding is not an RF artifact.)
  2. Linear vs nonlinear — logistic AUC vs RF AUC. Equal => the (weak) demand->failure
     signal is essentially linear-additive; RF higher => it needs interactions.
  3. Partial effects & collinearity — signed standardized coefficients give each dim's
     effect HOLDING THE OTHERS CONSTANT (unlike univariate Mann-Whitney), and per-fold
     coefficient stability + VIF independently re-test the VO collinearity artifact.

LinearSHAP is included for an apples-to-apples beeswarm, but note mean|SHAP| ~ |coef|
for a linear model, so the standardized coefficients carry the real interpretation.
Headline = COMBINATION_ALL (pooled); a per-model AUC + ranking-agreement summary follows.
Units: LinearSHAP explains the LOGIT (log-odds), not probability like TreeSHAP.

    python charxiv_analysis/logistic_cross_check.py
"""
import csv
import sys
import sqlite3
from collections import defaultdict
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score
from scipy.stats import spearmanr
import shap

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from mann_whitney_analysis import DB, EMERGENT, load_rubric, make_or_load_split, load_correctness
from random_forest import make_splits, DIM_ORDER, MIN_MINORITY

OUTDIR = ROOT / "logistic_cross_check"
RFDIR = ROOT / "random_forest"
EMERGENT_COLOR = "#6a3d9a"
STD_COLOR = "#4878a8"


def lr():
    return LogisticRegression(C=1.0, max_iter=2000)


def oof_logistic(X, y, groups, collect_shap=False):
    """Paper-grouped OOF logistic. Returns per-fold AUCs, standardized coef matrix
    (folds x dims), and optionally out-of-fold LinearSHAP (margin units) + feature rows."""
    aucs, coefs, sv, ft = [], [], [], []
    for tr, te in make_splits(y, groups):
        sc = StandardScaler().fit(X[tr])
        Xtr, Xte = sc.transform(X[tr]), sc.transform(X[te])
        m = lr().fit(Xtr, y[tr])
        aucs.append(roc_auc_score(y[te], m.decision_function(Xte)))
        coefs.append(m.coef_[0])                      # standardized (features were scaled)
        if collect_shap:
            ex = shap.LinearExplainer(m, Xtr)
            sv.append(np.asarray(ex.shap_values(Xte)))
            ft.append(X[te])                          # raw demand for colouring
    out = [np.array(aucs), np.vstack(coefs)]
    if collect_shap:
        out += [np.vstack(sv), np.vstack(ft)]
    return out


def vif(Xstd):
    """Variance inflation factor per dim (regress each dim on the others)."""
    n = Xstd.shape[1]
    v = np.empty(n)
    for j in range(n):
        others = [k for k in range(n) if k != j]
        r2 = LinearRegression().fit(Xstd[:, others], Xstd[:, j]).score(Xstd[:, others], Xstd[:, j])
        v[j] = 1.0 / max(1e-9, 1.0 - r2)
    return v


def load_imp(path, valcol):
    d = defaultdict(dict)
    with open(path) as f:
        for r in csv.DictReader(f):
            d[r["model"]][r["dim"]] = float(r[valcol])
    return d


def ranked(vals):
    return [d for _, d in sorted(((v, k) for k, v in vals.items()), reverse=True)]


def draw(dims, coef_mean, coef_std, sign_frac, lin_shap_abs, oof_shap, oof_feat,
         perm, shap_rf, out):
    fig = plt.figure(figsize=(18, 6))
    gs = fig.add_gridspec(1, 3, width_ratios=[1, 1, 1.1])
    axA, axB, axC = (fig.add_subplot(gs[0, i]) for i in range(3))

    # A — signed standardized coefficients (direction + partial effect), by |coef|
    order = np.argsort(np.abs(coef_mean))[::-1]
    od = [dims[i] for i in order]
    cols = [EMERGENT_COLOR if d in EMERGENT else STD_COLOR for d in od]
    axA.bar(range(len(od)), coef_mean[order], yerr=coef_std[order], capsize=3, color=cols)
    axA.axhline(0, c="0.3", lw=0.8)
    axA.set_xticks(range(len(od)), [("◆" if d in EMERGENT else "") + d for d in od],
                   rotation=90, fontsize=9)
    axA.set_ylabel("standardized logistic coef  (+ = raises failure odds)")
    axA.set_title("A. Partial effect of each demand\n(signed, holds others constant)")

    # B — rank agreement of the THREE methods for COMBINATION_ALL (rank 1 = top)
    lin = {d: lin_shap_abs[i] for i, d in enumerate(dims)}
    methods = {"RF perm": perm, "RF SHAP": shap_rf, "Logistic |coef|":
               {d: abs(coef_mean[i]) for i, d in enumerate(dims)}}
    rank = {name: {d: r + 1 for r, d in enumerate(ranked(v))} for name, v in methods.items()}
    base_order = ranked(methods["RF perm"])
    for x, name in enumerate(methods):
        for d in base_order:
            axB.scatter(x, rank[name][d], color=(EMERGENT_COLOR if d in EMERGENT else STD_COLOR),
                        s=40, zorder=3)
    for d in base_order:
        ys = [rank[name][d] for name in methods]
        axB.plot(range(len(methods)), ys, color="0.8", lw=0.8, zorder=1)
        axB.annotate(("◆" if d in EMERGENT else "") + d, (0, rank["RF perm"][d]),
                     xytext=(-6, 0), textcoords="offset points", ha="right", fontsize=8,
                     color=(EMERGENT_COLOR if d in EMERGENT else "0.3"))
    axB.set_xticks(range(len(methods)), list(methods), fontsize=9)
    axB.invert_yaxis()
    axB.set_ylabel("importance rank (1 = top)")
    axB.set_title("B. Same dim across 3 methods\n(flat line = all agree)")

    # C — LinearSHAP beeswarm for COMBINATION_ALL (log-odds units)
    n = oof_shap.shape[0]
    if n > 4000:
        idx = np.random.RandomState(0).choice(n, 4000, replace=False)
        oof_shap, oof_feat = oof_shap[idx], oof_feat[idx]
    disp = [("◆" if d in EMERGENT else "") + d for d in dims]
    plt.sca(axC)
    shap.summary_plot(oof_shap, features=oof_feat, feature_names=disp, show=False,
                      plot_size=None, sort=True)
    for t in axC.get_yticklabels():
        if t.get_text().startswith("◆"):
            t.set_color(EMERGENT_COLOR); t.set_fontweight("bold")
    axC.set_title("C. LinearSHAP (log-odds)\nout-of-fold, COMBINATION_ALL")

    fig.suptitle("Logistic cross-check vs random forest — COMBINATION_ALL "
                 "(◆ purple = emergent figure-grounding dims)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
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

    rf_auc = {r["model"]: float(r["auc_mean"]) for r in
              csv.DictReader(open(RFDIR / "rf_cv_results.csv"))}
    perm = load_imp(RFDIR / "rf_importance_long.csv", "perm_imp_mean")["COMBINATION_ALL"]
    shap_rf = load_imp(RFDIR / "rf_shap_importance_long.csv", "shap_mean_abs")["COMBINATION_ALL"]

    # ---- COMBINATION_ALL (pooled) -------------------------------------------------
    Xp, yp, gp = [], [], []
    for m in models:
        for item, ok in correctness[m].items():
            Xp.append(feat[item]); yp.append(0 if ok else 1); gp.append(meta[item])
    Xp, yp, gp = np.array(Xp), np.array(yp), np.array(gp)
    auc, coefs, oof_shap, oof_feat = oof_logistic(Xp, yp, gp, collect_shap=True)
    coef_mean, coef_std = coefs.mean(0), coefs.std(0)
    sign_frac = (np.sign(coefs) == np.sign(coef_mean)).mean(0)   # fold sign stability
    lin_abs = np.abs(oof_shap).mean(0)
    vifs = vif(StandardScaler().fit_transform(Xp))

    print(f"COMBINATION_ALL  logistic AUC {auc.mean():.3f}±{auc.std():.3f}  "
          f"vs RF AUC {rf_auc['COMBINATION_ALL']:.3f}")
    print(f"{'dim':5s} {'coef':>8s} {'±sd':>6s} {'sign%':>6s} {'|coef|rk':>8s} "
          f"{'permRk':>7s} {'shapRk':>7s} {'VIF':>6s}")
    prank = {d: i + 1 for i, d in enumerate(ranked(perm))}
    srank = {d: i + 1 for i, d in enumerate(ranked(shap_rf))}
    crank = {d: i + 1 for i, d in enumerate(ranked({d: abs(coef_mean[k]) for k, d in enumerate(dims)}))}
    rows = []
    for k in np.argsort(np.abs(coef_mean))[::-1]:
        d = dims[k]
        print(f"{d:5s} {coef_mean[k]:+8.3f} {coef_std[k]:6.3f} {sign_frac[k]*100:5.0f}% "
              f"{crank[d]:8d} {prank[d]:7d} {srank[d]:7d} {vifs[k]:6.1f}")
        rows.append([d, names.get(d, ""), "yes" if d in EMERGENT else "no",
                     f"{coef_mean[k]:.4f}", f"{coef_std[k]:.4f}", f"{sign_frac[k]:.3f}",
                     crank[d], prank[d], srank[d], f"{lin_abs[k]:.4f}", f"{vifs[k]:.3f}"])

    draw(dims, coef_mean, coef_std, sign_frac, lin_abs, oof_shap, oof_feat,
         perm, shap_rf, OUTDIR / "logistic_cross_check.png")
    with open(OUTDIR / "logistic_combo_dims.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dim", "dim_name", "emergent", "coef_mean", "coef_std", "sign_stability",
                    "logistic_rank", "rf_perm_rank", "rf_shap_rank", "linshap_mean_abs", "vif"])
        w.writerows(rows)

    # ---- per-model: logistic AUC + ranking agreement with RF perm -----------------
    print("\nPer-model logistic vs RF:")
    perm_all = load_imp(RFDIR / "rf_importance_long.csv", "perm_imp_mean")
    msum, rhos = [], []
    for m in models:
        items = list(correctness[m])
        X = np.array([feat[i] for i in items])
        y = np.array([0 if correctness[m][i] else 1 for i in items])
        g = np.array([meta[i] for i in items])
        if min(y.sum(), (y == 0).sum()) < MIN_MINORITY:
            continue
        a, c = oof_logistic(X, y, g)
        cabs = np.abs(c.mean(0))
        pv = np.array([perm_all[m][d] for d in dims]) if m in perm_all else None
        rho = spearmanr(cabs, pv).statistic if pv is not None else np.nan
        rhos.append(rho)
        msum.append([m, f"{a.mean():.4f}", f"{a.std():.4f}", f"{rf_auc.get(m, float('nan')):.4f}",
                     f"{rho:.4f}", ranked({d: cabs[k] for k, d in enumerate(dims)})[0]])
    rhos = np.array([r for r in rhos if not np.isnan(r)])
    print(f"  logistic-vs-RFperm |coef| Spearman: mean {rhos.mean():.3f} median {np.median(rhos):.3f}")
    with open(OUTDIR / "logistic_per_model.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "logistic_auc_mean", "logistic_auc_std", "rf_auc_mean",
                    "coef_vs_rfperm_spearman", "logistic_top_dim"])
        w.writerows(msum)

    print(f"\nFigure: {OUTDIR / 'logistic_cross_check.png'} (+ .pdf)")
    print(f"CSVs:   {OUTDIR}")


if __name__ == "__main__":
    main()
