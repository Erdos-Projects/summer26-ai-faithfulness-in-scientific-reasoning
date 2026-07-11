"""Per-item trial-agreement (vote consistency) for the SciVer eval.

Each item is verified over `t` trials (5). For a fixed item+run the truth is
fixed, so the `t` answers split into a majority and a minority. The agreement
level is the size of the majority block:

  * 5/5  — unanimous (all trials gave the same answer)
  * 4/5  — one trial dissented
  * 3/5  — bare majority (2 trials dissented)

(With 5 binary votes those are the only possible levels — a tie is impossible.)
This quantifies how *stable* the model is per item, independent of correctness,
and we additionally split each level by whether that majority answer was correct.

Agreement is computed on the raw `predicted` answer, NOT on `correct`, but the
two give the same block sizes since truth is constant within an item+run.

Usage:
    python -m sciver_eval.viz_consistency [--runs 2 3] [--out PATH] [--db PATH]
"""
import argparse
import csv
import sqlite3
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sciver_eval import db as sdb

DEFAULT_OUT = sdb.analysis_dir() / "trial_consistency.png"

RUN_NAME = {1: "difficulty", 2: "entailed", 3: "refuted"}


def consistency(conn, run):
    """For `run`, return (ntrials, {level: {"correct": n, "wrong": n}}) keyed by
    the majority-block size (e.g. 5, 4, 3 for 5 trials)."""
    rows = conn.execute(
        "SELECT item_id, SUM(predicted) p, SUM(correct) k, COUNT(*) t "
        "FROM prediction WHERE run=? GROUP BY item_id",
        (run,),
    ).fetchall()
    ntrials = max((t for _, _, _, t in rows), default=0)
    out = defaultdict(lambda: {"correct": 0, "wrong": 0})
    for _iid, p, k, t in rows:
        majority = max(p, t - p)            # size of the larger answer block
        maj_correct = (2 * k > t)           # did the majority answer match truth?
        out[majority]["correct" if maj_correct else "wrong"] += 1
    return ntrials, out


def draw(per_run, out_path):
    runs = list(per_run)
    fig, axes = plt.subplots(1, len(runs), figsize=(5.2 * len(runs), 5.2), squeeze=False)
    axes = axes[0]
    for ax, run in zip(axes, runs):
        ntrials, dist = per_run[run]
        levels = sorted(dist, reverse=True)          # 5, 4, 3
        labels = [f"{lv}/{ntrials}" for lv in levels]
        corr = [dist[lv]["correct"] for lv in levels]
        wrong = [dist[lv]["wrong"] for lv in levels]
        n = sum(corr) + sum(wrong)
        b1 = ax.bar(labels, corr, color="#4c9be8", edgecolor="0.3", label="majority correct")
        ax.bar(labels, wrong, bottom=corr, color="#e8954c", edgecolor="0.3", label="majority wrong")
        for x, (c, w) in enumerate(zip(corr, wrong)):
            tot = c + w
            ax.text(x, tot, f"{tot}\n({tot / n:.0%})", ha="center", va="bottom", fontsize=9)
        ax.set_title(f"run {run} — {RUN_NAME.get(run, '?')}  (n={n})", fontsize=12, pad=10)
        ax.set_xlabel(f"agreement (majority block of {ntrials} trials)")
        ax.set_ylabel("number of items")
        ax.set_ylim(0, n * 0.85)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
        ax.legend(fontsize=8, loc="upper right")
    fig.suptitle("Haiku 4.5 per-item trial agreement on SciVer chart items", fontsize=13)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(sdb.default_db_path()))
    ap.add_argument("--runs", type=int, nargs="+", default=None,
                    help="runs to analyse (default: all runs present, sorted)")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    runs = args.runs or [r for (r,) in conn.execute(
        "SELECT DISTINCT run FROM prediction ORDER BY run").fetchall()]
    per_run = {run: consistency(conn, run) for run in runs}
    conn.close()

    out = Path(args.out)
    draw(per_run, out)

    csv_path = out.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["run", "run_name", "agreement", "majority_correct",
                    "majority_wrong", "n_items", "frequency"])
        for run, (ntrials, dist) in per_run.items():
            n = sum(d["correct"] + d["wrong"] for d in dist.values())
            for lv in sorted(dist, reverse=True):
                c, wr = dist[lv]["correct"], dist[lv]["wrong"]
                w.writerow([run, RUN_NAME.get(run, "?"), f"{lv}/{ntrials}",
                            c, wr, c + wr, f"{(c + wr) / n:.4f}"])

    for run, (ntrials, dist) in per_run.items():
        n = sum(d["correct"] + d["wrong"] for d in dist.values())
        print(f"run {run} ({RUN_NAME.get(run, '?')}), n={n}, trials={ntrials}")
        for lv in sorted(dist, reverse=True):
            c, wr = dist[lv]["correct"], dist[lv]["wrong"]
            tot = c + wr
            print(f"  {lv}/{ntrials}: {tot:4d} ({tot / n:5.1%})  "
                  f"[maj correct {c}, maj wrong {wr}]")
    print(f"figure written: {out}")
    print(f"counts  written: {csv_path}")


if __name__ == "__main__":
    main()
