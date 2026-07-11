"""Predict confident correctness from the 12 DeLeAn rubric dims — leakage-safe.

Same per-item table as the Kendall/bar-chart work (rubric demand scores joined to
confident paired labels: entailed run 2 x refuted run 3, 5/5). Target y=1 for
confidently INCORRECT (positive class = FAILURE), y=0 for CORRECT — so a positive
importance/effect means "more of this dim => model more likely to fail".

Leakage guard: items sharing a paper ID (the entailed×refuted twins live on the
same figure, and some papers contribute >1 figure) must never straddle train/test.
We group by item.paperid and use StratifiedGroupKFold (k=5) so the 148/183 ratio is
preserved per fold AND no paper crosses sides. Wrapped in 10 repeats (seeds 0-9) =>
50 fold scores; we report mean±std across all 50, never a single split.

Two models on the SAME splits: RandomForest + HistGradientBoosting (both sklearn).
Light fixed hyperparameters — at n=331 tuning would just overfit the CV.

Importance: permutation (held-out folds, primary) + drop-column (refit-without-dim,
unique contribution). Gini/impurity importance is NOT reported — the predictive dims
are inter-correlated (tau up to ~0.55) so impurity credit splits arbitrarily.

    python -m sciver_eval.viz_rubric_clf
"""
import argparse
import csv
import sqlite3
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.inspection import permutation_importance
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, brier_score_loss

from sciver_eval import db as sdb
from sciver_eval.viz_rubric_demand import load_rubric, paired_label, RUBRIC_DB, EMERGENT

DEFAULT_OUT = sdb.analysis_dir() / "rf_gbm_importance.png"
N_REPEATS = 10
K = 5
PERM_REPEATS = 10          # inner shuffles per fold for permutation importance


def build_models():
    # modest, fixed — regularized enough to not memorize 265 train rows
    rf = RandomForestClassifier(n_estimators=200, max_depth=None, min_samples_leaf=5,
                                max_features="sqrt", n_jobs=-1, random_state=0)
    gb = HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=200,
                                        l2_regularization=1.0, min_samples_leaf=10,
                                        early_stopping=False, random_state=0)
    return {"RandomForest": rf, "HistGB": gb}


def make_splits(y, groups):
    """50 (train_idx, test_idx) splits: 10 repeats x 5 StratifiedGroupKFolds."""
    splits = []
    for seed in range(N_REPEATS):
        sgk = StratifiedGroupKFold(n_splits=K, shuffle=True, random_state=seed)
        for tr, te in sgk.split(np.zeros(len(y)), y, groups):
            splits.append((tr, te))
    return splits


def evaluate(model, X, y, splits, dims):
    """Per-fold AUC/balanced-acc/Brier + permutation importance (mean over folds)."""
    aucs, baccs, briers, perms = [], [], [], []
    for tr, te in splits:
        m = model.__class__(**model.get_params()).fit(X[tr], y[tr])
        p = m.predict_proba(X[te])[:, 1]
        aucs.append(roc_auc_score(y[te], p))
        baccs.append(balanced_accuracy_score(y[te], m.predict(X[te])))
        briers.append(brier_score_loss(y[te], p))
        pi = permutation_importance(m, X[te], y[te], scoring="roc_auc",
                                    n_repeats=PERM_REPEATS, random_state=0, n_jobs=-1)
        perms.append(pi.importances_mean)
    return (np.array(aucs), np.array(baccs), np.array(briers), np.vstack(perms))


def dropcol_importance(model, X, y, splits, dims):
    """Paired per-fold AUC loss from refitting without each dim (unique contribution)."""
    n = len(dims)
    deltas = np.zeros((len(splits), n))
    for fi, (tr, te) in enumerate(splits):
        full = model.__class__(**model.get_params()).fit(X[tr], y[tr])
        full_auc = roc_auc_score(y[te], full.predict_proba(X[te])[:, 1])
        for d in range(n):
            cols = [c for c in range(n) if c != d]
            red = model.__class__(**model.get_params()).fit(X[tr][:, cols], y[tr])
            red_auc = roc_auc_score(y[te], red.predict_proba(X[te][:, cols])[:, 1])
            deltas[fi, d] = full_auc - red_auc      # positive => dim helps
    return deltas


def draw(dims, res, perm_means, perm_stds, base_rate, out):
    fig, (ax0, ax1) = plt.subplots(1, 2, figsize=(15, 6))
    # (a) AUC distribution across the 50 folds per model
    names = list(res)
    ax0.boxplot([res[m]["auc"] for m in names], tick_labels=names, showmeans=True)
    ax0.axhline(0.5, ls="--", c="0.5", label="chance (AUC 0.5)")
    ax0.set_ylabel("ROC-AUC (50 folds)")
    ax0.set_title("AUC distribution across 50 grouped folds")
    ax0.legend(fontsize=8)
    # (b) permutation importance (use the better model by mean AUC)
    best = max(names, key=lambda m: res[m]["auc"].mean())
    order = np.argsort(perm_means[best])[::-1]
    od = [dims[i] for i in order]
    colors = ["#6a3d9a" if d in EMERGENT else "#4878a8" for d in od]
    ax1.bar(range(len(od)), perm_means[best][order], yerr=perm_stds[best][order],
            capsize=3, color=colors)
    ax1.axhline(0, c="0.3", lw=0.8)
    ax1.set_xticks(range(len(od)), [f"{'◆' if d in EMERGENT else ''}{d}" for d in od],
                   rotation=90, fontsize=9)
    for t, d in zip(ax1.get_xticklabels(), od):
        if d in EMERGENT:
            t.set_color("#6a3d9a"); t.set_fontweight("bold")
    ax1.set_ylabel("permutation importance (Δ AUC)")
    ax1.set_title(f"Permutation importance — {best} (◆ purple = emergent)")
    fig.suptitle(f"Rubric→failure classifier (y=1 INCORRECT; base rate {base_rate:.3f}, "
                 "paper-grouped 5-fold ×10)", fontsize=12)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(sdb.default_db_path()))
    ap.add_argument("--rubric-db", default=str(RUBRIC_DB))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    rubric, dims, names_d, nitems = load_rubric(args.rubric_db)
    conn = sqlite3.connect(args.db)
    label = paired_label(conn, 2, 3)
    meta = {r[0]: r[1] for r in conn.execute("SELECT item_id, paperid FROM item")}
    conn.close()

    items = [i for i in label if i in rubric and all(d in rubric[i] for d in dims)]
    missing = [i for i in items if i not in meta or not meta[i]]
    if missing:
        raise SystemExit(f"STOP: {len(missing)} items lack a paperid grouping key: {missing[:5]}")

    X = np.array([[rubric[i][d] for d in dims] for i in items], dtype=float)
    y = np.array([1 if label[i] == "incorrect" else 0 for i in items])   # 1 = FAILURE
    groups = np.array([meta[i] for i in items])
    base_rate = y.mean()
    ngrp = len(set(groups))
    print(f"{len(items)} items, {len(dims)} dims, {ngrp} paper groups | "
          f"y=1 INCORRECT n={y.sum()} y=0 CORRECT n={(y==0).sum()} | base rate {base_rate:.3f}")
    print(f"grouping key = item.paperid; positive class = confident FAILURE\n")

    splits = make_splits(y, groups)
    # leakage assertion: no paper appears on both sides of any split
    for tr, te in splits:
        assert not (set(groups[tr]) & set(groups[te])), "paper straddles train/test!"

    models = build_models()
    res, perm_means, perm_stds, drop_means, drop_stds = {}, {}, {}, {}, {}
    for name, model in models.items():
        print(f"fitting {name}: CV+permutation ...", flush=True)
        auc, bacc, brier, perms = evaluate(model, X, y, splits, dims)
        print(f"  {name} AUC {auc.mean():.3f}; drop-column ({len(dims)} dims x {len(splits)} folds) ...", flush=True)
        drops = dropcol_importance(model, X, y, splits, dims)
        res[name] = {"auc": auc, "bacc": bacc, "brier": brier}
        perm_means[name], perm_stds[name] = perms.mean(0), perms.std(0)
        drop_means[name], drop_stds[name] = drops.mean(0), drops.std(0)

    adir = Path(args.out).parent
    # results table
    with open(adir / "rf_gbm_cv_results.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "auc_mean", "auc_std", "bal_acc_mean", "bal_acc_std",
                    "brier_mean", "brier_std", "base_rate", "n_folds"])
        for m in res:
            r = res[m]
            w.writerow([m, f"{r['auc'].mean():.4f}", f"{r['auc'].std():.4f}",
                        f"{r['bacc'].mean():.4f}", f"{r['bacc'].std():.4f}",
                        f"{r['brier'].mean():.4f}", f"{r['brier'].std():.4f}",
                        f"{base_rate:.4f}", len(splits)])
    # importance table (sorted by permutation, per model)
    with open(adir / "rf_gbm_importance.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["model", "dim", "dim_name", "emergent", "perm_imp_mean", "perm_imp_std",
                    "dropcol_auc_loss_mean", "dropcol_auc_loss_std"])
        for m in res:
            order = np.argsort(perm_means[m])[::-1]
            for k in order:
                d = dims[k]
                w.writerow([m, d, names_d.get(d, ""), "yes" if d in EMERGENT else "no",
                            f"{perm_means[m][k]:.4f}", f"{perm_stds[m][k]:.4f}",
                            f"{drop_means[m][k]:.4f}", f"{drop_stds[m][k]:.4f}"])

    out = Path(args.out)
    draw(dims, res, perm_means, perm_stds, base_rate, out)

    # ---- printed report ----
    print("\n=== CV results (mean ± std over 50 paper-grouped folds) ===")
    print(f"base rate (majority/plain-acc floor) = {base_rate:.3f}; balanced-acc & AUC chance = 0.500")
    for m in res:
        r = res[m]
        print(f"  {m:13s} AUC {r['auc'].mean():.3f}±{r['auc'].std():.3f} | "
              f"bal-acc {r['bacc'].mean():.3f}±{r['bacc'].std():.3f} | "
              f"Brier {r['brier'].mean():.3f}±{r['brier'].std():.3f}")

    best = max(res, key=lambda m: res[m]["auc"].mean())
    print(f"\n=== Importance ({best}, sorted by permutation) ===")
    print(f"  {'dim':<5} {'perm ΔAUC':>14} {'dropcol ΔAUC':>16}  flag")
    order = np.argsort(perm_means[best])[::-1]
    # rank disagreement flag: dim's rank differs a lot between the two methods
    perm_rank = {dims[k]: r for r, k in enumerate(np.argsort(perm_means[best])[::-1])}
    drop_rank = {dims[k]: r for r, k in enumerate(np.argsort(drop_means[best])[::-1])}
    for k in order:
        d = dims[k]
        flag = ""
        if abs(perm_rank[d] - drop_rank[d]) >= 4:
            flag = f"⚠ rank {perm_rank[d]}↔{drop_rank[d]} (correlation masking)"
        em = " ◆" if d in EMERGENT else ""
        print(f"  {d:<5}{em:<2}{perm_means[best][k]:>+7.3f}±{perm_stds[best][k]:.3f}"
              f"{drop_means[best][k]:>+9.3f}±{drop_stds[best][k]:.3f}  {flag}")

    # ---- plain-language verdict ----
    print("\n=== VERDICT ===")
    for m in res:
        a = res[m]["auc"]
        margin = a.mean() - 0.5
        beats = margin > a.std()
        print(f"  {m}: AUC {a.mean():.3f}, {margin:+.3f} over chance, fold std {a.std():.3f} "
              f"-> {'BEATS chance by >1 std' if beats else 'within 1 std of chance (weak)'}")
    top = [dims[k] for k in order[:4]]
    em_pos = {d: perm_means[best][dims.index(d)] for d in EMERGENT}
    em_near0 = all(abs(v) < 0.01 for v in em_pos.values())
    print(f"  carried by: {', '.join(top)} (top-4 permutation dims for {best})")
    print(f"  emergent VL/MA/GS permutation imp: " +
          ", ".join(f"{d}={em_pos[d]:+.3f}" for d in ("VL", "MA", "GS")) +
          (" -> all near zero, as expected" if em_near0 else " -> NOT all near zero"))
    print(f"\nfigure: {out}")
    for n in ("rf_gbm_cv_results.csv", "rf_gbm_importance.csv"):
        print(f"csv:    {adir / n}")


if __name__ == "__main__":
    main()
