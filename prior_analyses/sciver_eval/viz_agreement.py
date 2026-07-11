"""Paired accuracy-AGREEMENT matrix: do the entailed and refuted conditions agree
on whether the model gets an item right?

For each of the 817 items we have correct-on-entailed and correct-on-refuted
(default = MAJORITY VOTE across each run's trials; --trial N for a single trial).
The 2x2 records agreement (both right / both wrong = the diagonal) vs disagreement
(right in only one condition = off-diagonal), and reports observed agreement plus
Cohen's kappa (agreement beyond chance).

    python -m sciver_eval.viz_agreement [--entailed-run 2] [--refuted-run 3] [--trial N]
"""
import argparse
import csv
import sqlite3
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sciver_eval import db as sdb
from sciver_eval.viz_venn import item_correct, run_meta

DEFAULT_OUT = sdb.analysis_dir() / "agreement_entailed_refuted.png"


def item_correct_threshold(conn, run, min_correct):
    """{item_id: correct?} where an item counts as correct iff it was correct on at
    least `min_correct` of its trials (5/5 = unanimous, 4/5 strict, 3/5 = majority)."""
    rows = conn.execute(
        "SELECT item_id, SUM(correct) k FROM prediction WHERE run=? GROUP BY item_id", (run,)
    ).fetchall()
    return {iid: (k >= min_correct) for iid, k in rows}


def counts(conn, e_run, r_run, trial=None, min_correct=None):
    if min_correct is not None:
        e = item_correct_threshold(conn, e_run, min_correct)
        r = item_correct_threshold(conn, r_run, min_correct)
    else:
        e = item_correct(conn, e_run, trial)
        r = item_correct(conn, r_run, trial)
    common = [i for i in e if i in r]
    both = sum(1 for i in common if e[i] and r[i])        # agree: both correct
    e_only = sum(1 for i in common if e[i] and not r[i])  # disagree
    r_only = sum(1 for i in common if not e[i] and r[i])  # disagree
    neither = sum(1 for i in common if not e[i] and not r[i])  # agree: both wrong
    return both, e_only, r_only, neither, len(common)


def agreement_stats(both, e_only, r_only, neither, n):
    po = (both + neither) / n  # observed agreement
    e_corr, r_corr = (both + e_only) / n, (both + r_only) / n
    pe = e_corr * r_corr + (1 - e_corr) * (1 - r_corr)  # chance agreement
    kappa = (po - pe) / (1 - pe) if pe != 1 else 0.0
    return po, kappa


def draw(M, n, po, kappa, out, e_run, r_run, mlabel, agg):
    both, e_only, r_only, neither = M
    grid = [[both, e_only], [r_only, neither]]
    # diagonal = agreement (+), off-diagonal = disagreement (-); diverging colour, 0=neutral
    signed = np.array([[both, -e_only], [-r_only, neither]], float)
    lim = max(both, e_only, r_only, neither, 1)

    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    ax.imshow(signed, cmap="RdYlGn", vmin=-lim, vmax=lim)
    labels = [["both correct\n(agree)", "entailed only\n(disagree)"],
              ["refuted only\n(disagree)", "both wrong\n(agree)"]]
    for i in range(2):
        for j in range(2):
            v = grid[i][j]
            ax.text(j, i, f"{v}\n{v / n:.1%}\n{labels[i][j]}",
                    ha="center", va="center", fontsize=12,
                    color="0.1", fontweight="bold")
    ax.set_xticks([0, 1], [f"Refuted (run {r_run})\ncorrect", "wrong"])
    ax.set_yticks([0, 1], [f"Entailed (run {e_run})\ncorrect", "wrong"])
    ax.set_xlabel("refuted-condition correctness")
    ax.set_ylabel("entailed-condition correctness")
    ax.set_title(f"{mlabel} paired accuracy agreement ({agg})", fontsize=13, pad=14)
    ax.annotate(
        f"observed agreement = {po:.1%}   •   Cohen's κ = {kappa:.3f}   •   N = {n}\n"
        f"(diagonal = conditions agree; off-diagonal = disagree)",
        xy=(0.5, -0.16), xycoords="axes fraction", ha="center", va="top", fontsize=10, color="0.3",
    )
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(sdb.default_db_path()))
    ap.add_argument("--entailed-run", type=int, default=2)
    ap.add_argument("--refuted-run", type=int, default=3)
    ap.add_argument("--trial", type=int, default=None,
                    help="single trial to use; default = MAJORITY VOTE across all trials")
    ap.add_argument("--min-correct", type=int, default=None,
                    help="item counts as correct on a run iff correct on >= this many trials "
                         "(e.g. 5,4,3 for 5/5,4/5,3/5); overrides the majority-vote default")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    both, e_only, r_only, neither, n = counts(
        conn, args.entailed_run, args.refuted_run, args.trial, args.min_correct)
    mlabel, ntrials = run_meta(conn, args.entailed_run)
    conn.close()

    if args.min_correct is not None:
        agg = f"≥{args.min_correct}/{ntrials} trials correct"
    elif args.trial is not None:
        agg = f"trial {args.trial}"
    elif ntrials > 1:
        agg = f"maj. vote of {ntrials} trials"
    else:
        agg = "single trial"
    po, kappa = agreement_stats(both, e_only, r_only, neither, n)

    out = Path(args.out)
    draw((both, e_only, r_only, neither), n, po, kappa, out, args.entailed_run, args.refuted_run, mlabel, agg)

    csv_path = out.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["cell", "entailed", "refuted", "kind", "count", "frequency"])
        rows = [("both_correct", "correct", "correct", "agree", both),
                ("entailed_only", "correct", "wrong", "disagree", e_only),
                ("refuted_only", "wrong", "correct", "disagree", r_only),
                ("both_wrong", "wrong", "wrong", "agree", neither)]
        for name, ec, rc, kind, v in rows:
            w.writerow([name, ec, rc, kind, v, f"{v / n:.4f}"])
        w.writerow(["observed_agreement", "", "", "", both + neither, f"{po:.4f}"])
        w.writerow(["cohen_kappa", "", "", "", "", f"{kappa:.4f}"])
        w.writerow(["TOTAL", "", "", "", n, "1.0000"])

    print(f"{mlabel}  {agg}  (entailed run {args.entailed_run} x refuted run {args.refuted_run})")
    print(f"  both correct (agree)  : {both:4d}  ({both / n:.1%})")
    print(f"  entailed only (disagr): {e_only:4d}  ({e_only / n:.1%})")
    print(f"  refuted only  (disagr): {r_only:4d}  ({r_only / n:.1%})")
    print(f"  both wrong   (agree)  : {neither:4d}  ({neither / n:.1%})")
    print(f"  observed agreement    : {po:.1%}   Cohen's kappa = {kappa:.3f}")
    print(f"figure: {out}")
    print(f"counts: {csv_path}")


if __name__ == "__main__":
    main()
