"""Kendall's tau-b inter-dimension dependency of the 12 rubric-demand dims, per group.

Same item-level source as viz_rubric_demand / viz_rubric_spearman (per-item DeLeAn
demand scores, confident paired-correctness labels: entailed run 2 x refuted run 3,
5/5). Tau-b is computed from the RAW per-item scores — the rubric is a coarse ordinal
scale (~0-4) with many ties, which tau-b handles more honestly than Spearman at this n.

Outputs (analysis dir):
  kendall_tau_by_correct.csv     long: group, dim_i, dim_j, tau_b
  kendall_tau_diff.csv           long: dim_i, dim_j, tau_correct, tau_incorrect, delta
  kendall_tau_pair_changes.csv   ranked by |delta| (correct - incorrect)
  kendall_tau_by_correct.png     3-panel heatmap: correct | incorrect | diff

    python -m sciver_eval.viz_rubric_kendall [--entailed-run 2 --refuted-run 3]
"""
import argparse
import csv
import sqlite3
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import kendalltau

from sciver_eval import db as sdb
from sciver_eval.viz_rubric_demand import load_rubric, paired_label, RUBRIC_DB, EMERGENT

DEFAULT_OUT = sdb.analysis_dir() / "kendall_tau_by_correct.png"
SPEARMAN_CSV = sdb.analysis_dir() / "rubric_spearman_by_correct.csv"


def tau_matrix(rubric, items, dims):
    """12x12 Kendall tau-b; lower triangle computed, mirrored. Diagonal = 1."""
    M = np.array([[rubric[i][d] for d in dims] for i in items], dtype=float)
    n = len(dims)
    tau = np.eye(n)
    for a in range(n):
        for b in range(a):
            t = kendalltau(M[:, a], M[:, b]).statistic   # tau-b, ties handled
            tau[a, b] = tau[b, a] = t
    return tau


def perm_test(rubric, items_c, items_i, dims, observed, n_perm, seed):
    """Label-shuffle null for Δτ per pair. Pools the 148+183 items, reshuffles the
    correct/incorrect split (sizes fixed), recomputes tau_b in each side, diffs.
    Returns {(dim_i, dim_j): p_perm}. observed: {(i,j): |Δτ|}."""
    items = items_c + items_i
    nc = len(items_c)
    M = np.array([[rubric[i][d] for d in dims] for i in items], dtype=float)
    cols = {d: M[:, k] for k, d in enumerate(dims)}
    pairs = [(dims[a], dims[b]) for a in range(len(dims)) for b in range(a + 1, len(dims))]
    counts = {p: 0 for p in pairs}
    rng = np.random.default_rng(seed)
    for _ in range(n_perm):
        perm = rng.permutation(len(items))
        ci, ii = perm[:nc], perm[nc:]            # shuffled "correct"/"incorrect" indices
        for di, dj in pairs:
            xa, xb = cols[di], cols[dj]
            d_perm = (kendalltau(xa[ci], xb[ci]).statistic
                      - kendalltau(xa[ii], xb[ii]).statistic)
            if abs(d_perm) >= observed[(di, dj)]:
                counts[(di, dj)] += 1
    # add-one finite-sample p (fraction of |Δτ_perm| >= |Δτ_obs|; never exactly 0)
    return {p: (counts[p] + 1) / (n_perm + 1) for p in pairs}


def bh_fdr(pvals):
    """Benjamini-Hochberg q-values for a list of p, preserving input order."""
    n = len(pvals)
    order = sorted(range(n), key=lambda k: pvals[k])
    q = [0.0] * n
    prev = 1.0
    for rank, idx in enumerate(reversed(order), start=1):      # largest p first
        i = n - rank + 1                                        # BH rank (1-based)
        prev = min(prev, pvals[idx] * n / i)
        q[idx] = prev
    return q


def draw(dims, tau_c, tau_i, tau_d, ns, out):
    labels = [f"{'◆' if d in EMERGENT else ''}{d}" for d in dims]
    panels = [("confidently CORRECT", tau_c, ns[0]),
              ("confidently INCORRECT", tau_i, ns[1]),
              ("Δ tau-b (correct − incorrect)", tau_d, None)]
    fig, axes = plt.subplots(1, 3, figsize=(21, 7.2))
    for ax, (title, M, n) in zip(axes, panels):
        # diff panel has a tighter scale so small dependency shifts are visible
        vmax = 1.0 if n is not None else float(np.abs(tau_d - np.eye(len(dims))).max())
        im = ax.imshow(M, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        ax.set_xticks(range(len(dims)), labels, fontsize=8, rotation=90)
        ax.set_yticks(range(len(dims)), labels, fontsize=8)
        for t, d in zip(ax.get_xticklabels() + ax.get_yticklabels(), labels + labels):
            if d.startswith("◆"):
                t.set_color("#6a3d9a"); t.set_fontweight("bold")
        for a in range(len(dims)):
            for b in range(len(dims)):
                v = M[a, b]
                ax.text(b, a, f"{v:.2f}".lstrip("0").replace("-0", "-"),
                        ha="center", va="center", fontsize=6,
                        color="white" if abs(v) > 0.55 * vmax else "0.2")
        ttl = f"{title} (n={n})" if n is not None else title
        ax.set_title(ttl, fontsize=11)
        fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle("Rubric-demand inter-dimension Kendall tau-b by confident paired "
                 "correctness (Haiku 5/5, entailed×refuted)", fontsize=13)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def load_spearman(path):
    """{(group, frozenset{i,j}): rho} from the existing Spearman CSV, if present."""
    if not path.exists():
        return {}
    out = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            if r["dim_i"] != r["dim_j"]:
                out[(r["group"], frozenset((r["dim_i"], r["dim_j"])))] = float(r["spearman_rho"])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(sdb.default_db_path()))
    ap.add_argument("--rubric-db", default=str(RUBRIC_DB))
    ap.add_argument("--entailed-run", type=int, default=2)
    ap.add_argument("--refuted-run", type=int, default=3)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--n-perm", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rubric, dims, names, nitems = load_rubric(args.rubric_db)
    conn = sqlite3.connect(args.db)
    label = paired_label(conn, args.entailed_run, args.refuted_run)
    conn.close()

    groups = {"correct": [], "incorrect": []}
    for item, lab in label.items():
        if item in rubric and all(d in rubric[item] for d in dims):
            groups[lab].append(item)
    nc, ni = len(groups["correct"]), len(groups["incorrect"])

    tau_c = tau_matrix(rubric, groups["correct"], dims)
    tau_i = tau_matrix(rubric, groups["incorrect"], dims)
    tau_d = tau_c - tau_i

    out = Path(args.out)
    draw(dims, tau_c, tau_i, tau_d, (nc, ni), out)
    adir = out.parent

    # tau-b long format (both groups, full symmetric matrix incl. diagonal)
    with open(adir / "kendall_tau_by_correct.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["group", "dim_i", "dim_j", "tau_b"])
        for grp, M in (("correct", tau_c), ("incorrect", tau_i)):
            for a in range(len(dims)):
                for b in range(len(dims)):
                    w.writerow([grp, dims[a], dims[b], f"{M[a, b]:.4f}"])

    # difference matrix + ranked pair-change table (off-diagonal, unordered pairs)
    pairs = [(dims[a], dims[b], tau_c[a, b], tau_i[a, b], tau_c[a, b] - tau_i[a, b])
             for a in range(len(dims)) for b in range(a + 1, len(dims))]
    with open(adir / "kendall_tau_diff.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["dim_i", "dim_j", "tau_correct", "tau_incorrect", "delta"])
        for di, dj, tc, ti, dlt in pairs:
            w.writerow([di, dj, f"{tc:.4f}", f"{ti:.4f}", f"{dlt:+.4f}"])

    # permutation test on Δτ per pair + BH-FDR across all 66
    observed = {(di, dj): abs(dlt) for di, dj, _, _, dlt in pairs}
    print(f"running {args.n_perm} permutations x 66 pairs (seed={args.seed})...")
    p_perm = perm_test(rubric, groups["correct"], groups["incorrect"], dims,
                       observed, args.n_perm, args.seed)
    keys = [(di, dj) for di, dj, *_ in pairs]
    q_bh = dict(zip(keys, bh_fdr([p_perm[k] for k in keys])))

    ranked = sorted(pairs, key=lambda p: -abs(p[4]))
    with open(adir / "kendall_tau_pair_changes.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dim_i", "dim_j", "tau_correct", "tau_incorrect", "delta",
                    "abs_delta", "p_perm", "q_bh", "significant_q05"])
        for di, dj, tc, ti, dlt in ranked:
            pv, qv = p_perm[(di, dj)], q_bh[(di, dj)]
            w.writerow([di, dj, f"{tc:.4f}", f"{ti:.4f}", f"{dlt:+.4f}", f"{abs(dlt):.4f}",
                        f"{pv:.4g}", f"{qv:.4g}", qv < 0.05])

    print(f"{len(dims)} dims; CORRECT n={nc} INCORRECT n={ni}\n")
    survivors = [(di, dj, tc, ti, dlt) for di, dj, tc, ti, dlt in ranked
                 if q_bh[(di, dj)] < 0.05]
    if survivors:
        print(f"Pairs surviving BH q < 0.05 ({len(survivors)}/66), by |Δτ|:")
        print(f"  {'pair':<9} {'tau_corr':>8} {'tau_inc':>8} {'delta':>8} "
              f"{'p_perm':>8} {'q_bh':>8}")
        for di, dj, tc, ti, dlt in survivors:
            print(f"  {di+'-'+dj:<9} {tc:>+8.2f} {ti:>+8.2f} {dlt:>+8.2f} "
                  f"{p_perm[(di,dj)]:>8.4f} {q_bh[(di,dj)]:>8.4f}")
    else:
        print("No pair survives BH q < 0.05. The cross-group dependency shifts are "
              "not distinguishable from label-shuffle noise at this n (148/183).")
    smallest = min(ranked, key=lambda r: q_bh[(r[0], r[1])])
    print(f"  smallest q: {smallest[0]}-{smallest[1]} "
          f"(Δτ={smallest[4]:+.2f}, p={p_perm[(smallest[0],smallest[1])]:.4f}, "
          f"q={q_bh[(smallest[0],smallest[1])]:.3f})\n")

    print("Top 15 pairs by |Δ tau-b| (correct − incorrect):")
    print(f"  {'pair':<9} {'tau_corr':>8} {'tau_inc':>8} {'delta':>8} {'|Δ|':>6}")
    for di, dj, tc, ti, dlt in ranked[:15]:
        print(f"  {di+'-'+dj:<9} {tc:>+8.2f} {ti:>+8.2f} {dlt:>+8.2f} {abs(dlt):>6.2f}")

    # cross-check vs existing Spearman: flag pairs where tie-handling moved the read
    sp = load_spearman(SPEARMAN_CSV)
    if sp:
        print(f"\nCross-check vs {SPEARMAN_CSV.name} — pairs with |rho − tau_b| > 0.10:")
        flagged = []
        for grp, M in (("correct", tau_c), ("incorrect", tau_i)):
            for a in range(len(dims)):
                for b in range(a + 1, len(dims)):
                    key = (grp, frozenset((dims[a], dims[b])))
                    if key in sp and abs(sp[key] - M[a, b]) > 0.10:
                        flagged.append((grp, dims[a], dims[b], sp[key], M[a, b]))
        if flagged:
            print(f"  {'group':<10} {'pair':<9} {'rho':>6} {'tau_b':>6} {'diff':>6}")
            for grp, di, dj, rho, tb in sorted(flagged, key=lambda x: -abs(x[3] - x[4])):
                print(f"  {grp:<10} {di+'-'+dj:<9} {rho:>+6.2f} {tb:>+6.2f} {rho-tb:>+6.2f}")
        else:
            print("  none — tau-b and Spearman agree within 0.10 everywhere.")
        # tau-b is mechanically smaller-magnitude than rho; report typical shrink
        diffs = [abs(sp[(g, frozenset((dims[a], dims[b])))] - M[a, b])
                 for g, M in (("correct", tau_c), ("incorrect", tau_i))
                 for a in range(len(dims)) for b in range(a + 1, len(dims))
                 if (g, frozenset((dims[a], dims[b]))) in sp]
        print(f"  median |rho − tau_b| across all pairs/groups: {np.median(diffs):.3f}")
    else:
        print(f"\n(no Spearman CSV at {SPEARMAN_CSV} — skipping cross-check)")

    print(f"\nfigure: {out}")
    for n in ("kendall_tau_by_correct.csv", "kendall_tau_diff.csv", "kendall_tau_pair_changes.csv"):
        print(f"csv:    {adir / n}")


if __name__ == "__main__":
    main()
