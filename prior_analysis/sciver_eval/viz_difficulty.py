"""Run-1 multi-trial difficulty view: per-item p(correct) over the 5 trials.

Run 1 verified each of the 817 chart items `--trials` times (default detected from
the DB). For each item, p(correct) = (# correct trials) / (# trials) — an empirical
per-item difficulty estimate. The histogram shows how items distribute across
p(correct) buckets: the 0.0 bar = always-wrong, 1.0 = always-right, and everything
between = stochastic items that flip across trials (the signal multi-trial buys).

Usage:
    python -m sciver_eval.viz_difficulty [--run 1] [--out PATH] [--db PATH]
"""
import argparse
import csv
import sqlite3
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sciver_eval import db as sdb

DEFAULT_OUT = sdb.analysis_dir() / "run1_pcorrect_hist.png"


def per_item_pcorrect(conn, run):
    """Return {item_id: (k_correct, n_trials)} for the run."""
    rows = conn.execute(
        "SELECT item_id, SUM(correct) k, COUNT(*) t FROM prediction WHERE run=? GROUP BY item_id",
        (run,),
    ).fetchall()
    return {iid: (k, t) for iid, k, t in rows}


def draw(buckets, ntrials, nitems, out_path, run):
    ps = sorted(buckets)
    counts = [buckets[p] for p in ps]
    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    bars = ax.bar([f"{p:.1f}" for p in ps], counts, color="#4c9be8", edgecolor="0.3", width=0.8)
    for b, val in zip(bars, counts):
        ax.text(b.get_x() + b.get_width() / 2, val, f"{val}\n({val / nitems:.1%})",
                ha="center", va="bottom", fontsize=9)
    ax.set_xlabel(f"p(correct) over {ntrials} trials  (0.0 = always wrong, 1.0 = always right)")
    ax.set_ylabel("number of items")
    ax.set_title(f"Run {run} per-item difficulty — Haiku 4.5 on {nitems} SciVer chart items", fontsize=12, pad=12)
    ax.set_ylim(0, max(counts) * 1.15)
    # summary annotation: always-wrong / stochastic / always-right
    aw = buckets.get(0.0, 0)
    ar = buckets.get(1.0, 0)
    stoch = nitems - aw - ar
    ax.annotate(
        f"always-wrong: {aw} ({aw/nitems:.1%})   |   stochastic: {stoch} ({stoch/nitems:.1%})   "
        f"|   always-right: {ar} ({ar/nitems:.1%})",
        xy=(0.5, -0.18), xycoords="axes fraction", ha="center", va="top", fontsize=10, color="0.3",
    )
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(sdb.default_db_path()))
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    items = per_item_pcorrect(conn, args.run)
    conn.close()
    if not items:
        raise SystemExit(f"no predictions for run {args.run} in {args.db}")

    ntrials = max(t for _, t in items.values())
    nitems = len(items)
    buckets = Counter(round(k / t, 1) for k, t in items.values())
    # ensure every p-level the run can produce is present (0 fill) for a clean axis
    for i in range(ntrials + 1):
        buckets.setdefault(round(i / ntrials, 1), 0)

    out = Path(args.out)
    draw(buckets, ntrials, nitems, out, args.run)

    csv_path = out.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["p_correct", "count", "frequency"])
        for p in sorted(buckets):
            w.writerow([f"{p:.1f}", buckets[p], f"{buckets[p] / nitems:.4f}"])
        w.writerow(["TOTAL", nitems, "1.0000"])

    for p in sorted(buckets):
        print(f"p(correct)={p:.1f}: {buckets[p]:4d}  ({buckets[p] / nitems:.1%})")
    print(f"items={nitems}  trials={ntrials}")
    print(f"figure written: {out}")
    print(f"counts  written: {csv_path}")


if __name__ == "__main__":
    main()
