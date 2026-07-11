"""Predict reasoning failure from the 12 DeLeAn rubric dims — one RF per model + pooled.

CharXiv port of `sciver_eval/viz_rubric_clf.py`. For each model, target y=1 =
INCORRECT (positive class = FAILURE), y=0 = CORRECT, features = the 12 model-agnostic
rubric-demand dims. Leakage-safe: rows are grouped by `item.paperid` (a few papers
contribute >1 figure) with StratifiedGroupKFold, wrapped in repeats so we report
mean±std across folds, never a single split.

A final "COMBINATION_ALL" model pools every (model,item) train row into one dataset
(features identical across models; label is that model's correctness) — the aggregate
failure model. RandomForest only; importance via permutation (held-out folds).
Drop-column importance is omitted here for tractability across 40 models.

Reuses the cached 800/200 train split (TRAIN ONLY).

    python charxiv_analysis/random_forest.py
"""
import csv
import sqlite3
import sys
import textwrap
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, brier_score_loss
import shap

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from mann_whitney_analysis import (  # noqa: E402  identical loaders + cached split
    DB, EMERGENT, load_rubric, make_or_load_split, load_correctness,
)

OUTDIR = ROOT / "random_forest"
DIM_ORDER = ["VL", "AS", "MCr", "MCu", "MA", "VO", "AT", "GS", "QLl", "KNf", "QLq", "CL"]
N_REPEATS = 3
K = 5
PERM_REPEATS = 5
MIN_MINORITY = 25   # need enough of the minority class for grouped stratified CV
SHAP_MAX_POINTS = 4000   # subsample out-of-fold rows for legible/fast beeswarms


def _block(fig, y, heading, paras, width=128, body_size=9.3):
    """Draw a bold heading + wrapped body paragraphs from top y; return new y cursor."""
    if heading:
        fig.text(0.045, y, heading, fontsize=11.5, fontweight="bold", va="top", color="#1a1a1a")
        y -= 0.026
    lines = []
    for p in paras:
        lines += (textwrap.wrap(p, width=width) or [""])
        lines.append("")
    fig.text(0.055, y, "\n".join(lines), fontsize=body_size, va="top", linespacing=1.45)
    return y - len(lines) * 0.0168 - 0.004


def make_cover_fig(n_models=39, n_train=800):
    """One-page reader's guide prepended to the RF PDF."""
    fig = plt.figure(figsize=(13.5, 11))
    fig.text(0.045, 0.975, "CharXiv — Rubric-Demand → Reasoning-Failure Random Forest",
             fontsize=16, fontweight="bold", va="top")
    fig.text(0.045, 0.945, "How to read this document", fontsize=12, style="italic",
             va="top", color="0.35")
    y = 0.905
    y = _block(fig, y, "What this PDF contains", [
        f"Every page after this one is ONE Random Forest model. Each RF tries to predict, from a chart "
        f"item's 12 DeLeAn rubric-demand scores alone, whether a given vision-language model gets that "
        f"item's reasoning question WRONG. Page order: COMBINATION_ALL (all {n_models} models pooled into "
        f"one dataset) first, then one page per model, ordered by reasoning accuracy (strongest first).",
        f"All models are trained and cross-validated on the {n_train}-item TRAIN split only; the 200-item "
        f"test set is untouched."])
    y = _block(fig, y, "Setup (same for every page)", [
        "• Target: y=1 = INCORRECT (failure is the positive class), y=0 = correct.",
        "• Features: the 12 model-agnostic demand dims — VL, AS, MCr, MCu, MA, VO, AT, GS, QLl, KNf, QLq, "
        "CL. ◆ / purple = emergent figure-grounding dims (VL, GS, MA); the rest are standard DeLeAn.",
        "• Validation: paper-grouped StratifiedGroupKFold, 5 folds × 3 repeats = 15 folds (no paper "
        "straddles train/test within a fold). RandomForest, 200 trees. Importance = permutation, 5 "
        "shuffles per fold, scored by ROC-AUC."])
    y = _block(fig, y, "The two panels on each page", [
        "LEFT — AUC distribution. Box-and-whisker of ROC-AUC across the 15 folds (mean marked); dashed "
        "line at 0.5 = chance. Panel title shows mean ± SD. y-axis = ROC-AUC (0.3–1.0).",
        "RIGHT — Permutation importance. One bar per demand dim, sorted high→low. y-axis = ΔAUC: how much "
        "held-out AUC DROPS when that dim's values are randomly shuffled (bigger = more predictive). "
        "Error bars = SD ACROSS THE 15 FOLDS (fold-to-fold wobble) — not SEM, and not the within-shuffle "
        "spread. Bars at/below 0 = the dim carries little or no predictive signal."])
    y = _block(fig, y, "What you can conclude", [
        "• Is demand predictive of failure? If the AUC box sits above 0.5 by more than ~1 SD, that model's "
        "failures are predictable from item demand better than chance (true for 35/39 models here, though "
        "the effect is modest — AUC ≈ 0.59).",
        "• Which demands are most PREDICTIVE (direction-blind — see caveats)? Tallest bars: GS "
        "(Gestalt/Shape), MA (Multi-Element Aggregation), then MCu, VL, AS. But the logistic cross-check "
        "shows GS is PROTECTIVE and MA RF-specific, so read bars as 'informative', not 'raises failure odds'.",
        "• Do models share a signature? Compare the right panel across pages — the same few visual-grounding "
        "dims recur, so the (weak) signal is shared across models, not model-specific quirks.",
        "• COMBINATION_ALL is the aggregate signal; with ~31k rows its AUC has the tightest SD and is the "
        "single most reliable estimate."])
    y = _block(fig, y, "Caveats", [
        "• Weak predictor (AUC ≈ 0.59): use these for EXPLANATION (which dims matter), not deployment.",
        "• Signal is essentially LINEAR: a logistic on the same dims matches/beats this RF (0.603 vs 0.573; "
        "both ≈0.60 even after nested tuning), so the RF's nonlinearity adds nothing. See "
        "logistic_cross_check.py / tuned_model_comparison.py.",
        "• Direction ≠ magnitude. The logistic cross-check resolves sign: GS is PROTECTIVE (coef −0.18, "
        "higher GS demand → FEWER failures, matching Mann-Whitney); MA is RF-SPECIFIC (rank 2 here, 10 under "
        "a linear model). Don't read tall bars as 'failure-driving'.",
        "• Permutation ≈ TreeSHAP (Spearman ≈0.67) on GS/MA; VO is a perm-only blip, but VIFs ≤1.8 rule out "
        "collinearity (just method sensitivity). See perm_shap_agreement.py.",
        "• AUC ± and error bars are population SD (ddof=0) over only 15 folds — itself a noisy SD estimate."],
        body_size=8.7)
    fig.text(0.045, 0.022, "Outputs: charxiv_analysis/random_forest/ — this PDF, 40 PNGs, "
             "rf_cv_results.csv, rf_importance_long.csv; plus charxiv_random_forest_shap.pdf "
             "(out-of-fold TreeSHAP beeswarms), *_shap.png, rf_shap_importance_long.csv.",
             fontsize=8, color="0.45", va="top")
    return fig


def cover_page(pdf, n_models, n_train):
    fig = make_cover_fig(n_models, n_train)
    pdf.savefig(fig)
    plt.close(fig)


def rf():
    return RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                  max_features="sqrt", n_jobs=-1, random_state=0)


def shap_class1(model, X):
    """TreeSHAP values for the positive (FAILURE) class; shape (n_rows, n_dims)."""
    sv = shap.TreeExplainer(model).shap_values(X, check_additivity=False)
    if isinstance(sv, list):       # older shap: [class0, class1]
        return sv[1]
    arr = np.asarray(sv)
    if arr.ndim == 3:              # shap >=0.50: (n_rows, n_dims, n_classes)
        return arr[:, :, 1]
    return arr                     # already (n_rows, n_dims)


def make_splits(y, groups):
    splits = []
    for seed in range(N_REPEATS):
        sgk = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=seed)
        for tr, te in sgk.split(np.zeros(len(y)), y, groups):
            splits.append((tr, te))
    return splits


def evaluate(X, y, groups, dims):
    """Per-fold AUC/bal-acc/Brier + permutation importance matrix (folds x dims),
    plus out-of-fold TreeSHAP values and the feature rows they explain. SHAP is
    computed ONLY on each fold's held-out rows, so the attributions reflect signal
    that generalises rather than memorised training quirks."""
    splits = make_splits(y, groups)
    for tr, te in splits:
        assert not (set(groups[tr]) & set(groups[te])), "paper straddles train/test!"
    aucs, baccs, briers, perms = [], [], [], []
    shap_vals, shap_feat = [], []
    for tr, te in splits:
        m = rf().fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
        baccs.append(balanced_accuracy_score(y[te], m.predict(X[te])))
        briers.append(brier_score_loss(y[te], p))
        pi = permutation_importance(m, X[te], y[te], scoring="roc_auc",
                                    n_repeats=PERM_REPEATS, random_state=0, n_jobs=1)
        perms.append(pi.importances_mean)
        shap_vals.append(shap_class1(m, X[te]))
        shap_feat.append(X[te])
    return (np.array(aucs), np.array(baccs), np.array(briers),
            np.vstack(perms), len(splits),
            np.vstack(shap_vals), np.vstack(shap_feat))


def draw(model, dims, auc, perm_mean, perm_std, base_rate, n, out):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 6))
    ax0.boxplot([auc], tick_labels=[model], showmeans=True)
    ax0.axhline(0.5, ls="--", c="0.5", label="chance (AUC 0.5)")
    ax0.set_ylabel(f"ROC-AUC ({len(auc)} grouped folds)")
    ax0.set_ylim(0.3, 1.0)
    ax0.set_title(f"AUC = {auc.mean():.3f} ± {auc.std():.3f}")
    ax0.legend(fontsize=8)
    order = np.argsort(perm_mean)[::-1]
    od = [dims[i] for i in order]
    colors = ["#6a3d9a" if d in EMERGENT else "#4878a8" for d in od]
    ax1.bar(range(len(od)), perm_mean[order], yerr=perm_std[order], capsize=3, color=colors)
    ax1.axhline(0, c="0.3", lw=0.8)
    ax1.set_xticks(range(len(od)), [f"{'◆' if d in EMERGENT else ''}{d}" for d in od],
                   rotation=90, fontsize=9)
    for t, d in zip(ax1.get_xticklabels(), od):
        if d in EMERGENT:
            t.set_color("#6a3d9a"); t.set_fontweight("bold")
    ax1.set_ylabel("permutation importance (Δ AUC)")
    ax1.set_title("Permutation importance (◆ purple = emergent)")
    fig.suptitle(f"{model} — rubric→failure RF (y=1 INCORRECT; base rate {base_rate:.3f}, "
                 f"n={n}, paper-grouped {K}-fold ×{N_REPEATS})", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig


def draw_beeswarm(model, dims, oof_shap, oof_feat, out):
    """SHAP beeswarm over out-of-fold rows. Each dot = one held-out item; x = its
    SHAP value (right = pushed prediction toward INCORRECT); colour = dim value."""
    n = oof_shap.shape[0]
    if n > SHAP_MAX_POINTS:                       # deterministic subsample for legibility
        idx = np.random.RandomState(0).choice(n, SHAP_MAX_POINTS, replace=False)
        sv, ft = oof_shap[idx], oof_feat[idx]
    else:
        sv, ft = oof_shap, oof_feat
    disp = [f"{'◆' if d in EMERGENT else ''}{d}" for d in dims]
    fig = plt.figure(figsize=(9, 7))
    shap.summary_plot(sv, features=ft, feature_names=disp, show=False, plot_size=None)
    fig = plt.gcf()
    for t in fig.axes[0].get_yticklabels():
        if t.get_text().startswith("◆"):
            t.set_color("#6a3d9a"); t.set_fontweight("bold")
    fig.suptitle(f"{model} — out-of-fold TreeSHAP  (→ right pushes toward INCORRECT; "
                 f"n={n} held-out rows" + (f", {SHAP_MAX_POINTS} shown)" if n > SHAP_MAX_POINTS else ")"),
                 fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig


def run_one(model, X, y, groups, dims, names, pdf, shap_pdf, cv_rows, imp_rows,
            shap_rows, label_out):
    auc, bacc, brier, perms, nfold, oof_shap, oof_feat = evaluate(X, y, groups, dims)
    base_rate = y.mean()
    perm_mean, perm_std = perms.mean(0), perms.std(0)
    safe = model.replace("/", "_").replace(".", "-")
    fig = draw(model, dims, auc, perm_mean, perm_std, base_rate, len(y),
               OUTDIR / f"{label_out}_{safe}.png")
    pdf.savefig(fig); plt.close(fig)
    bee = draw_beeswarm(model, dims, oof_shap, oof_feat,
                        OUTDIR / f"{label_out}_{safe}_shap.png")
    shap_pdf.savefig(bee); plt.close(bee)
    cv_rows.append([model, f"{auc.mean():.4f}", f"{auc.std():.4f}",
                    f"{bacc.mean():.4f}", f"{bacc.std():.4f}",
                    f"{brier.mean():.4f}", f"{brier.std():.4f}",
                    f"{base_rate:.4f}", len(y), nfold])
    order = np.argsort(perm_mean)[::-1]
    for k in order:
        d = dims[k]
        imp_rows.append([model, d, names.get(d, ""), "yes" if d in EMERGENT else "no",
                         f"{perm_mean[k]:.4f}", f"{perm_std[k]:.4f}"])
    # SHAP importance = mean |SHAP| over out-of-fold rows (signed mean kept for direction)
    shap_abs, shap_signed = np.abs(oof_shap).mean(0), oof_shap.mean(0)
    for k in np.argsort(shap_abs)[::-1]:
        d = dims[k]
        shap_rows.append([model, d, names.get(d, ""), "yes" if d in EMERGENT else "no",
                          f"{shap_abs[k]:.5f}", f"{shap_signed[k]:+.5f}"])
    top = [dims[k] for k in order[:4]]
    print(f"  {model:28s} AUC {auc.mean():.3f}±{auc.std():.3f}  base {base_rate:.3f}  "
          f"top: {', '.join(top)}")
    return auc.mean()


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    rubric, names = load_rubric(con)
    train_ids, _test = make_or_load_split(set(rubric))
    correctness = load_correctness(con, train_ids)
    meta = {r[0]: r[1] for r in con.execute("SELECT item_id, paperid FROM item")}
    con.close()

    dims = [d for d in DIM_ORDER if d in names]
    feat = {i: np.array([rubric[i][d] for d in dims], dtype=float) for i in rubric}
    model_acc = {m: sum(v.values()) / len(v) for m, v in correctness.items()}
    models = sorted(correctness, key=lambda m: model_acc[m], reverse=True)
    print(f"{len(models)} models, {len(dims)} dims, {len(train_ids)} train items, "
          f"{len(set(meta.values()))} paper groups")

    cv_rows, imp_rows, shap_rows = [], [], []
    pdf_path = OUTDIR / "charxiv_random_forest_all.pdf"
    shap_pdf_path = OUTDIR / "charxiv_random_forest_shap.pdf"
    with PdfPages(pdf_path) as pdf, PdfPages(shap_pdf_path) as shap_pdf:
        cover_page(pdf, len(models), len(train_ids))   # reader's guide, page 1
        # --- combination of all models (pooled rows) first -----------------------
        Xp, yp, gp = [], [], []
        for m in models:
            for item, ok in correctness[m].items():
                Xp.append(feat[item]); yp.append(0 if ok else 1); gp.append(meta[item])
        run_one("COMBINATION_ALL", np.array(Xp), np.array(yp), np.array(gp),
                dims, names, pdf, shap_pdf, cv_rows, imp_rows, shap_rows, "combo")

        # --- per model -----------------------------------------------------------
        for m in models:
            items = list(correctness[m])
            X = np.array([feat[i] for i in items])
            y = np.array([0 if correctness[m][i] else 1 for i in items])
            g = np.array([meta[i] for i in items])
            minority = min(y.sum(), (y == 0).sum())
            if minority < MIN_MINORITY:
                print(f"  {m:28s} SKIP (minority class n={minority} < {MIN_MINORITY})")
                continue
            run_one(m, X, y, g, dims, names, pdf, shap_pdf, cv_rows, imp_rows,
                    shap_rows, "model")

    with open(OUTDIR / "rf_cv_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "auc_mean", "auc_std", "bal_acc_mean", "bal_acc_std",
                    "brier_mean", "brier_std", "base_rate", "n_rows", "n_folds"])
        w.writerows(cv_rows)
    with open(OUTDIR / "rf_importance_long.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "dim", "dim_name", "emergent", "perm_imp_mean", "perm_imp_std"])
        w.writerows(imp_rows)
    with open(OUTDIR / "rf_shap_importance_long.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "dim", "dim_name", "emergent",
                    "shap_mean_abs", "shap_mean_signed"])
        w.writerows(shap_rows)

    print(f"\nPDF: {pdf_path}")
    print(f"SHAP PDF: {shap_pdf_path}")
    print(f"PNGs + CSVs: {OUTDIR}")


if __name__ == "__main__":
    main()
