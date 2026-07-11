"""Venn diagram of paired entailed/refuted correctness for the SciVer eval.

Each of the 817 chart items was verified under two conditions:
  * entailed run (default run 2): claim is an ENTAILED statement -> truth = "yes",
    so `correct` <=> the model predicted "yes".
  * refuted  run (default run 3): claim is a REFUTED statement  -> truth = "no",
    so `correct` <=> the model predicted "no".

Joining the two runs per item partitions the dataset into four regions:
  * E only  : correct on entailed, wrong on refuted  -> model said "yes" both times  (always-entailed)
  * R only  : correct on refuted, wrong on entailed  -> model said "no"  both times  (always-refuted / skeptic)
  * E & R   : correct on both                        -> genuinely discriminates      (correct either way)
  * neither : wrong on both                          -> model said "no" then "yes"    (anti / wrong either way)

Correctness per item defaults to a MAJORITY VOTE across all trials of each run
(an item counts as correct iff correct on >half its trials, i.e. >=3/5). Pass
`--trial N` to instead use a single trial (the old single-shot behaviour).

Usage:
    python -m sciver_eval.viz_venn [--entailed-run 2] [--refuted-run 3] [--trial N] [--out PATH] [--db PATH]
"""
import argparse
import csv
import sqlite3
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sciver_eval import db as sdb

# Outputs (figure + counts CSV) live in the canonical analysis folder (see
# db.analysis_dir()) so derived artifacts stay co-located with predictions.db.
DEFAULT_OUT = sdb.analysis_dir() / "venn_entailed_refuted.png"


def _tally(rows):
    both = sum(1 for ce, cr in rows if ce and cr)
    e_only = sum(1 for ce, cr in rows if ce and not cr)
    r_only = sum(1 for ce, cr in rows if cr and not ce)
    neither = sum(1 for ce, cr in rows if not ce and not cr)
    return dict(n=len(rows), both=both, e_only=e_only, r_only=r_only, neither=neither)


def item_correct(conn, run, trial=None):
    """Map {item_id: bool} of whether the model was correct on `run`.

    trial=None (default): MAJORITY VOTE across all trials of the run — an item is
    "correct" iff it was correct on more than half its trials (>=3/5 for 5 trials).
    trial=N: correctness on that single trial only (reproduces the old behaviour).
    """
    if trial is not None:
        rows = conn.execute(
            "SELECT item_id, correct FROM prediction WHERE run=? AND trial=?",
            (run, trial),
        ).fetchall()
        return {iid: bool(c) for iid, c in rows}
    rows = conn.execute(
        "SELECT item_id, SUM(correct) k, COUNT(*) t FROM prediction WHERE run=? GROUP BY item_id",
        (run,),
    ).fetchall()
    return {iid: (2 * k > t) for iid, k, t in rows}


def paired_counts(conn, entailed_run, refuted_run, trial=None):
    e = item_correct(conn, entailed_run, trial)
    r = item_correct(conn, refuted_run, trial)
    rows = [(e[i], r[i]) for i in e if i in r]
    return _tally(rows)


def paired_counts_by_claim_type(conn, entailed_run, refuted_run, trial=None):
    e = item_correct(conn, entailed_run, trial)
    r = item_correct(conn, refuted_run, trial)
    ct = {i: t for i, t in conn.execute("SELECT item_id, claim_type FROM item").fetchall()}
    common = [i for i in e if i in r]
    by_type = {}
    for t in sorted({ct.get(i) for i in common}):
        by_type[t] = _tally([(e[i], r[i]) for i in common if ct.get(i) == t])
    return by_type


def _render_venn(ax, c, entailed_run, refuted_run, fs=11, agg=""):
    """Draw one paired entailed/refuted Venn onto `ax`. Returns the 'wrong either
    way' caption (count outside both circles) for the caller to place. `agg` is a
    short descriptor of the trial aggregation (e.g. 'maj. vote' or 'trial 1')."""
    n, both, e_only, r_only, neither = (c["n"], c["both"], c["e_only"], c["r_only"], c["neither"])
    e_tot, r_tot = both + e_only, both + r_only
    pct = lambda k: f"{k / n:.1%}" if n else "0.0%"
    tag = f", {agg}" if agg else ""
    set_labels = (
        f"Correct on ENTAILED\n(run {entailed_run}{tag})  n={e_tot} ({pct(e_tot)})",
        f"Correct on REFUTED\n(run {refuted_run}{tag})  n={r_tot} ({pct(r_tot)})",
    )
    try:
        from matplotlib_venn import venn2, venn2_circles
        v = venn2(subsets=(e_only, r_only, both), set_labels=set_labels,
                  set_colors=("#4c9be8", "#e8954c"), alpha=0.55, ax=ax)
        venn2_circles(subsets=(e_only, r_only, both), linewidth=1.0, color="0.3", ax=ax)
        meanings = {
            "10": ('always "yes"\n(both entailed)', e_only),
            "01": ('always "no"\n(both refuted)', r_only),
            "11": ("correct\neither way", both),
        }
        for key, (label, val) in meanings.items():
            lbl = v.get_label_by_id(key)
            if lbl is not None:
                lbl.set_text(f"{val} ({pct(val)})\n{label}")
                lbl.set_fontsize(fs)
        for sid in ("A", "B"):
            t = v.get_label_by_id(sid)
            if t is not None:
                t.set_fontsize(fs)
    except ImportError:  # manual fallback: two overlapping circles
        from matplotlib.patches import Circle
        ax.add_patch(Circle((0.38, 0.5), 0.30, color="#4c9be8", alpha=0.45))
        ax.add_patch(Circle((0.62, 0.5), 0.30, color="#e8954c", alpha=0.45))
        ax.text(0.22, 0.5, f'{e_only} ({pct(e_only)})\nalways "yes"\n(both entailed)', ha="center", va="center", fontsize=fs)
        ax.text(0.78, 0.5, f'{r_only} ({pct(r_only)})\nalways "no"\n(both refuted)', ha="center", va="center", fontsize=fs)
        ax.text(0.50, 0.5, f"{both} ({pct(both)})\ncorrect\neither way", ha="center", va="center", fontsize=fs)
        ax.text(0.30, 0.86, set_labels[0].replace("\n", " "), ha="center", fontsize=fs)
        ax.text(0.70, 0.14, set_labels[1].replace("\n", " "), ha="center", fontsize=fs)
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_aspect("equal"); ax.axis("off")
    return f'wrong either way (said "no" then "yes"): {neither} ({pct(neither)})'


def draw(c, out_path, entailed_run, refuted_run, agg="", model_label="Haiku 4.5"):
    fig, ax = plt.subplots(figsize=(8, 6.5))
    caption = _render_venn(ax, c, entailed_run, refuted_run, agg=agg)
    sub = f" ({agg})" if agg else ""
    ax.set_title(
        f"SciVer chart items (N={c['n']}) — {model_label} paired entailed/refuted correctness{sub}",
        fontsize=12, pad=16,
    )
    ax.annotate(caption, xy=(0.5, -0.02), xycoords="axes fraction",
                ha="center", va="top", fontsize=10, color="0.3")
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def draw_by_claim_type(by_type, out_path, entailed_run, refuted_run, agg="", model_label="Haiku 4.5"):
    """One Venn panel per claim_type (direct / analytical), side by side."""
    types = list(by_type)
    fig, axes = plt.subplots(1, len(types), figsize=(7.5 * len(types), 6.8))
    if len(types) == 1:
        axes = [axes]
    for ax, ct in zip(axes, types):
        c = by_type[ct]
        caption = _render_venn(ax, c, entailed_run, refuted_run, fs=10, agg=agg)
        ax.set_title(f"{ct}  (n={c['n']})", fontsize=12, pad=12)
        ax.annotate(caption, xy=(0.5, -0.04), xycoords="axes fraction",
                    ha="center", va="top", fontsize=9, color="0.3")
    sub = f" ({agg})" if agg else ""
    fig.suptitle(f"{model_label} paired entailed/refuted correctness by claim_type{sub}",
                 fontsize=13, y=1.0)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


REGION_DESC = [
    ("correct_either_way", "overlap (E&R): said 'yes' then 'no'", "both"),
    ("both_entailed", "E only: always 'yes'", "e_only"),
    ("both_refuted", "R only: always 'no'", "r_only"),
    ("wrong_either_way", "outside: said 'no' then 'yes'", "neither"),
]

MODEL_LABELS = {"claude-haiku-4-5": "Haiku 4.5", "claude-sonnet-4-6": "Sonnet 4.6"}


def run_meta(conn, run):
    """(display model label, n_trials) for a run — used to label figures correctly
    regardless of which model/run the Venn is built from."""
    row = conn.execute("SELECT model FROM run WHERE run_id=?", (run,)).fetchone()
    model = row[0] if row else "?"
    ntrials = conn.execute(
        "SELECT COUNT(DISTINCT trial) FROM prediction WHERE run=?", (run,)).fetchone()[0] or 0
    return MODEL_LABELS.get(model, model), ntrials


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(sdb.default_db_path()))
    ap.add_argument("--entailed-run", type=int, default=2)
    ap.add_argument("--refuted-run", type=int, default=3)
    ap.add_argument("--trial", type=int, default=None,
                    help="single trial to use; default = MAJORITY VOTE across all trials")
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--by-claim-type", action="store_true",
                    help="also emit venn_by_claim_type.png/.csv split by direct/analytical")
    args = ap.parse_args()

    conn = sqlite3.connect(args.db)
    c = paired_counts(conn, args.entailed_run, args.refuted_run, args.trial)
    by_type = (paired_counts_by_claim_type(conn, args.entailed_run, args.refuted_run, args.trial)
               if args.by_claim_type else None)
    mlabel, ntrials = run_meta(conn, args.entailed_run)
    conn.close()

    if args.trial is not None:
        agg = f"trial {args.trial}"
    elif ntrials > 1:
        agg = f"maj. vote of {ntrials} trials"
    else:
        agg = "single trial"
    out = Path(args.out)
    draw(c, out, args.entailed_run, args.refuted_run, agg, mlabel)
    src = f"entailed run {args.entailed_run} x refuted run {args.refuted_run}, {mlabel}, {agg}"

    n = c["n"]
    csv_path = out.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["region", "description", "count", "frequency"])
        for key, desc, attr in REGION_DESC:
            w.writerow([key, desc, c[attr], f"{c[attr] / n:.4f}"])
        w.writerow(["TOTAL", f"N items ({src})", n, "1.0000"])

    for key, desc, attr in REGION_DESC:
        print(f"{key:20s} {c[attr]:4d}  ({c[attr] / n:.1%})  {desc}")
    print(f"{'TOTAL':20s} {n:4d}")
    print(f"figure written: {out}")
    print(f"counts  written: {csv_path}")

    if by_type is not None:
        ct_png = out.with_name(out.stem + "_by_claim_type.png")
        ct_csv = ct_png.with_suffix(".csv")
        draw_by_claim_type(by_type, ct_png, args.entailed_run, args.refuted_run, agg, mlabel)
        with open(ct_csv, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["claim_type", "region", "count", "frequency"])
            for ct, cc in by_type.items():
                for key, _desc, attr in REGION_DESC:
                    w.writerow([ct, key, cc[attr], f"{cc[attr] / cc['n']:.4f}"])
                w.writerow([ct, "TOTAL", cc["n"], "1.0000"])
        print(f"by-claim-type figure: {ct_png}")
        print(f"by-claim-type counts: {ct_csv}")


if __name__ == "__main__":
    main()
