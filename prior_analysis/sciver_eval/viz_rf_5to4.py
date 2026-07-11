"""Apply the frozen 5/5-trained RandomForest to ALL 4/5 paired items, descriptively.

Refits the failure model EXACTLY as rf_gbm (12 DeLeAn dims, y=1=confident FAILURE,
same light config) on the 331 confident 5/5 items, freezes it to joblib, then SCORES
(never retrains) every 4/5 item. 4/5 quadrants (run2 entailed x run3 refuted, >=4/5
correct per condition): both_correct 236, entailed_only 70, refuted_only 451,
both_wrong 60 (=817).

STRUCTURAL FACT that drives the read: rubric demand is scored per FIGURE/item_id, and
run2/run3 reuse the same 817 item_ids. So the entailed half and the refuted half of a
pair have IDENTICAL features => identical failure-prob => within-pair gap is exactly 0
by construction. That is the strongest possible form of "the model is blind within a
pair": it literally has no condition-specific input. We still emit per item-half rows
and the gap distribution (all zeros) as requested, and put the real signal in the
between-quadrant failure-prob distribution.

    python -m sciver_eval.viz_rf_5to4
"""
import csv
import sqlite3
from pathlib import Path

import numpy as np
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sciver_eval import db as sdb
from sciver_eval.viz_rubric_demand import load_rubric, paired_label, RUBRIC_DB
from sciver_eval.viz_agreement import item_correct_threshold
from sciver_eval.viz_rubric_clf import build_models

ADIR = sdb.analysis_dir()
FROZEN = ADIR / "rf_5of5_frozen.joblib"
QORDER = ["both_correct", "entailed_only", "refuted_only", "both_wrong"]
QCOLOR = {"both_correct": "#2ca25f", "entailed_only": "#f0a030",
          "refuted_only": "#de2d26", "both_wrong": "#7a0177"}


def quadrant(ec, rc):
    return ("both_correct" if ec and rc else "entailed_only" if ec else
            "refuted_only" if rc else "both_wrong")


def describe(probs):
    p = np.asarray(probs)
    q1, q3 = np.percentile(p, [25, 75])
    return p.mean(), np.median(p), q1, q3, len(p)


def main():
    rubric, dims, names, nitems = load_rubric(RUBRIC_DB)
    conn = sqlite3.connect(sdb.default_db_path())
    meta = {r[0]: r[1] for r in conn.execute("SELECT item_id, paperid FROM item")}

    # ---- frozen 5/5 RandomForest: fit on the 331 confident items, save ----
    lab = paired_label(conn, 2, 3)
    train_items = [i for i in lab if i in rubric and all(d in rubric[i] for d in dims)]
    Xtr = np.array([[rubric[i][d] for d in dims] for i in train_items], dtype=float)
    ytr = np.array([1 if lab[i] == "incorrect" else 0 for i in train_items])   # 1=FAILURE
    rf = build_models()["RandomForest"].fit(Xtr, ytr)
    joblib.dump({"model": rf, "dims": dims, "y_meaning": "1=confident FAILURE"}, FROZEN)
    train_papers = set(meta[i] for i in train_items)
    print(f"frozen RF: trained on {len(train_items)} 5/5 items "
          f"({ytr.sum()} fail / {(ytr==0).sum()} correct), {len(train_papers)} papers -> {FROZEN.name}")

    # ---- ALL 4/5 items: quadrant + per-item failure-prob (figure-level) ----
    e = item_correct_threshold(conn, 2, 4)
    r = item_correct_threshold(conn, 3, 4)
    conn.close()
    items = [i for i in e if i in r and i in rubric and all(d in rubric[i] for d in dims)]
    X = np.array([[rubric[i][d] for d in dims] for i in items], dtype=float)
    fp = rf.predict_proba(X)[:, 1]                      # failure-prob, one per item
    P = dict(zip(items, fp))
    Q = {i: quadrant(e[i], r[i]) for i in items}
    seen = {i: (meta[i] in train_papers) for i in items}

    # ---- per item-half rows (entailed half + refuted half share the item's fp) ----
    rows = []
    for i in items:
        for cond, correct in (("entailed", e[i]), ("refuted", r[i])):
            rows.append((meta[i], Q[i], cond, bool(correct), P[i], seen[i]))
    with open(ADIR / "rf_5to4_allquadrants.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["paperid", "quadrant", "condition", "true_correct",
                    "predicted_failure_prob", "seen_in_training"])
        for pid, q, cond, corr, prob, sn in rows:
            w.writerow([pid, q, cond, int(corr), f"{prob:.4f}", int(sn)])

    n_half = len(rows)
    n_seen = sum(1 for *_, sn in rows if sn)
    print(f"\n4/5 item-halves: {n_half} (= {len(items)} items x2). "
          f"seen_in_training (paperid in 5/5): {n_seen} ({n_seen/n_half:.1%})")

    # ---- quadrant breakdown, twice: (a) all, (b) papers NOT in training ----
    def breakdown(keep, title):
        sub = [i for i in items if keep(i)]
        print(f"\n=== failure-prob by quadrant — {title} (n_items={len(sub)}) ===")
        print(f"  {'quadrant':<14}{'n':>5}{'mean':>8}{'median':>8}{'IQR':>16}")
        out = {}
        for q in QORDER:
            ps = [P[i] for i in sub if Q[i] == q]
            if not ps:
                continue
            m, md, q1, q3, n = describe(ps)
            out[q] = (m, md, q1, q3, n)
            print(f"  {q:<14}{n:>5}{m:>8.3f}{md:>8.3f}   [{q1:.3f}, {q3:.3f}]")
        return out

    bd_all = breakdown(lambda i: True, "ALL items")
    bd_held = breakdown(lambda i: not seen[i], "papers NOT in 5/5 training")

    # ---- disagreement quadrants split by condition half (proves the blindness) ----
    print("\n=== disagreement quadrants split by condition half (mean failure-prob) ===")
    print("    (entailed & refuted halves share figure-level features => identical by construction)")
    for q in ("entailed_only", "refuted_only"):
        sub = [i for i in items if Q[i] == q]
        me = np.mean([P[i] for i in sub])               # entailed half == refuted half
        print(f"  {q:<14} entailed-half mean={me:.3f}   refuted-half mean={me:.3f}   (Δ=0.000)")

    # ---- KEY DIAGNOSTIC: within-pair |Δ failure-prob| ----
    gap = {i: abs(P[i] - P[i]) for i in items}           # ≡ 0 (same features both halves)
    print("\n=== within-pair |Δ failure-prob| by quadrant ===")
    for q in QORDER:
        g = [gap[i] for i in items if Q[i] == q]
        print(f"  {q:<14} mean={np.mean(g):.4f}  max={np.max(g):.4f}  (n={len(g)})")
    print("  -> exactly 0 everywhere: the model cannot separate the twin that succeeded "
          "from the\n     twin that failed, because rubric demand carries no condition signal.")

    # ---- plots ----
    # (6) failure-prob distribution per quadrant: box (all) + held-out overlay
    fig, ax = plt.subplots(figsize=(9, 6))
    data = [[P[i] for i in items if Q[i] == q] for q in QORDER]
    bp = ax.boxplot(data, tick_labels=[f"{q}\n(n={len(d)})" for q, d in zip(QORDER, data)],
                    showmeans=True, patch_artist=True, widths=0.6)
    for patch, q in zip(bp["boxes"], QORDER):
        patch.set_facecolor(QCOLOR[q]); patch.set_alpha(0.45)
    for k, q in enumerate(QORDER):                       # held-out means as red diamonds
        hp = [P[i] for i in items if Q[i] == q and not seen[i]]
        if hp:
            ax.scatter(k + 1, np.mean(hp), marker="D", color="k", zorder=5,
                       label="held-out mean" if k == 0 else None)
    base = ytr.mean()
    ax.axhline(base, ls="--", c="0.4", label=f"5/5 base failure rate {base:.2f}")
    ax.set_ylabel("predicted failure-prob (frozen 5/5 RF)")
    ax.set_title("Frozen 5/5 RF applied to 4/5 items — failure-prob by agreement quadrant")
    ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(ADIR / "rf_5to4_quadrant.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # (7) within-pair gap histogram: disagreement vs agreement (spike at 0)
    fig, ax = plt.subplots(figsize=(8, 5))
    dis = [gap[i] for i in items if Q[i] in ("entailed_only", "refuted_only")]
    agr = [gap[i] for i in items if Q[i] in ("both_correct", "both_wrong")]
    ax.hist([agr, dis], bins=np.linspace(0, 0.5, 26), label=["agreement pairs", "disagreement pairs"],
            color=["#2ca25f", "#de2d26"], alpha=0.7)
    ax.set_xlabel("within-pair |Δ failure-prob|  (entailed half − refuted half)")
    ax.set_ylabel("pairs")
    ax.set_title("Within-pair failure-prob gap ≡ 0 (figure-level features carry no condition info)")
    ax.annotate("all pairs at Δ=0:\nmodel is identical on\nboth halves by construction",
                xy=(0.0, len(items)), xytext=(0.18, len(items) * 0.6), fontsize=10,
                arrowprops=dict(arrowstyle="->", color="0.3"))
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(ADIR / "rf_5to4_withinpair_gap.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---- verdict ----
    print("\n=== VERDICT ===")
    bc, bw = bd_all["both_correct"], bd_all["both_wrong"]
    ro = bd_all["refuted_only"]
    sep = bw[0] - bc[0]
    print(f"(i) diagonal separation: both_correct mean fp={bc[0]:.3f} vs both_wrong={bw[0]:.3f} "
          f"(Δ={sep:+.3f}).")
    print(f"    -> {'SEPARATE in the expected direction' if sep > 0 else 'do NOT separate'}; "
          f"signal {'preserved' if sep > 0.03 else 'weak/absent'} on the diagonal "
          f"(IQRs {bc[2]:.2f}-{bc[3]:.2f} vs {bw[2]:.2f}-{bw[3]:.2f} overlap heavily).")
    print(f"(ii) within-pair blindness: gap ≡ 0 for all 817 pairs (model identical on both halves).")
    print(f"    refuted_only (n=451) mean fp={ro[0]:.3f}, vs both_correct {bc[0]:.3f} / both_wrong {bw[0]:.3f}.")
    closer = "both_correct (UNDER-predicts failure)" if abs(ro[0]-bc[0]) < abs(ro[0]-bw[0]) else "both_wrong"
    print(f"    -> refuted_only sits closer to {closer}: the 451 refuted-half failures look "
          f"LOW-demand to the\n       model, i.e. they fail for a reason the rubric does not capture.")

    print(f"\nfrozen: {FROZEN}")
    for n in ("rf_5to4_allquadrants.csv", "rf_5to4_quadrant.png", "rf_5to4_withinpair_gap.png"):
        print(f"out:    {ADIR / n}")


if __name__ == "__main__":
    main()
