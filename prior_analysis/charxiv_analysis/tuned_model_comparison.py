"""Settle it: tuned RF vs tuned logistic on COMBINATION_ALL, nested grouped CV.

The default RF (0.573) lost to an untuned logistic (0.603). This asks whether a
HYPERPARAMETER-TUNED RF can overtake the logistic — fairly. Fair means:
  * nested CV — hyperparameters chosen in an INNER paper-grouped loop, AUC measured
    on a held-out OUTER fold the tuner never saw (no tuning-on-test leakage);
  * symmetric — the logistic's C is tuned the same way, not left at its default;
  * identical outer folds for all four configs (default/tuned x RF/logistic).

To avoid the nested-parallelism trap, the estimators run single-threaded
(n_jobs=1) and GridSearchCV parallelises the search (n_jobs=-1).

    python charxiv_analysis/tuned_model_comparison.py
"""
import csv
import sys
import sqlite3
from collections import Counter
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, StratifiedGroupKFold
from sklearn.metrics import roc_auc_score

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from mann_whitney_analysis import DB, load_rubric, make_or_load_split, load_correctness
from random_forest import make_splits, DIM_ORDER

OUTDIR = ROOT / "tuned_model_comparison"
INNER_K = 3

RF_GRID = {"min_samples_leaf": [5, 20, 50, 100], "max_depth": [None, 5, 10]}
LR_GRID = {"clf__C": [0.03, 0.1, 0.3, 1.0, 3.0, 10.0]}


def rf_base():
    return RandomForestClassifier(n_estimators=200, max_features="sqrt",
                                  n_jobs=1, random_state=0)


def lr_pipe():
    return Pipeline([("sc", StandardScaler()),
                     ("clf", LogisticRegression(max_iter=2000))])


def auc_of(model, Xtr, ytr, Xte, yte):
    model.fit(Xtr, ytr)
    if hasattr(model, "decision_function"):
        s = model.decision_function(Xte)
    else:
        s = model.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, s)


def tuned_auc(estimator, grid, Xtr, ytr, gtr, Xte, yte):
    """Inner grouped GridSearchCV on the outer-train, score on the outer-test."""
    inner = StratifiedGroupKFold(n_splits=INNER_K, shuffle=True, random_state=0)
    gs = GridSearchCV(estimator, grid, scoring="roc_auc", cv=inner, n_jobs=-1, refit=True)
    gs.fit(Xtr, ytr, groups=gtr)
    s = (gs.best_estimator_.decision_function(Xte)
         if hasattr(gs.best_estimator_, "decision_function")
         else gs.best_estimator_.predict_proba(Xte)[:, 1])
    return roc_auc_score(yte, s), gs.best_params_


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
    models = sorted(correctness)

    X, y, g = [], [], []
    for m in models:
        for item, ok in correctness[m].items():
            X.append(feat[item]); y.append(0 if ok else 1); g.append(meta[item])
    X, y, g = np.array(X), np.array(y), np.array(g)
    print(f"COMBINATION_ALL: {len(y)} rows, base rate {y.mean():.3f}, {len(set(g))} paper groups")

    outer = make_splits(y, g)   # same 15 paper-grouped folds used everywhere else
    res = {"RF default": [], "RF tuned": [], "Logistic default": [], "Logistic tuned": []}
    rf_best, lr_best = Counter(), Counter()
    for i, (tr, te) in enumerate(outer):
        Xtr, ytr, gtr, Xte, yte = X[tr], y[tr], g[tr], X[te], y[te]
        res["RF default"].append(auc_of(rf_base().set_params(min_samples_leaf=5),
                                        Xtr, ytr, Xte, yte))
        res["Logistic default"].append(auc_of(lr_pipe().set_params(clf__C=1.0),
                                              Xtr, ytr, Xte, yte))
        a_rf, p_rf = tuned_auc(rf_base(), RF_GRID, Xtr, ytr, gtr, Xte, yte)
        a_lr, p_lr = tuned_auc(lr_pipe(), LR_GRID, Xtr, ytr, gtr, Xte, yte)
        res["RF tuned"].append(a_rf); res["Logistic tuned"].append(a_lr)
        rf_best[str(p_rf)] += 1; lr_best[str(p_lr)] += 1
        print(f"  fold {i+1:2d}/{len(outer)}  RFdef {res['RF default'][-1]:.3f}  "
              f"RFtuned {a_rf:.3f} {p_rf}  LRtuned {a_lr:.3f} {p_lr}")

    print("\n=== outer-fold AUC (nested, paper-grouped) ===")
    rows = []
    for k, v in res.items():
        v = np.array(v)
        print(f"  {k:18s} {v.mean():.4f} ± {v.std():.4f}")
        rows.append([k, f"{v.mean():.4f}", f"{v.std():.4f}"])
    rt, lt = np.array(res["RF tuned"]), np.array(res["Logistic tuned"])
    diff = rt - lt   # paired per outer fold
    print(f"\ntuned RF − tuned Logistic (paired): {diff.mean():+.4f} ± {diff.std():.4f}  "
          f"(RF wins {(diff > 0).sum()}/{len(diff)} folds)")
    print(f"RF best params (modal): {rf_best.most_common(2)}")
    print(f"LR best params (modal): {lr_best.most_common(2)}")

    # figure: paired outer-fold AUCs
    fig, ax = plt.subplots(figsize=(8, 5.5))
    order = ["RF default", "RF tuned", "Logistic default", "Logistic tuned"]
    cols = ["#9ec6e0", "#4878a8", "#f0a878", "#d2691e"]
    data = [res[k] for k in order]
    bp = ax.boxplot(data, tick_labels=order, showmeans=True, patch_artist=True)
    for patch, c in zip(bp["boxes"], cols):
        patch.set_facecolor(c)
    ax.axhline(0.5, ls="--", c="0.5", lw=0.8, label="chance")
    ax.set_ylabel("ROC-AUC (15 paper-grouped outer folds)")
    ax.set_title("Tuned RF vs tuned Logistic — COMBINATION_ALL, nested grouped CV\n"
                 f"tuned RF {rt.mean():.3f} vs tuned Logistic {lt.mean():.3f}")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUTDIR / "tuned_model_comparison.png", dpi=150, bbox_inches="tight")
    fig.savefig(OUTDIR / "tuned_model_comparison.pdf", bbox_inches="tight")

    with open(OUTDIR / "tuned_model_comparison.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["config", "outer_auc_mean", "outer_auc_std"])
        w.writerows(rows)
    print(f"\nFigure + CSV: {OUTDIR}")


if __name__ == "__main__":
    main()
