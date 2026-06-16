"""Rubric demand vs CONFIDENT paired correctness.

Joins SciVer predictions (entailed run 2 + refuted run 3, 5 trials each) with the
DeLeAn rubric demand scores from `rubric_scoring/annotations_prod.db`. Uses ALL
fully-scored dimensions (every dim annotated on all 817 items) — pass 1 (AS, AT,
MCr, QLl, QLq, VO) + pass 3 (CL, GS, KNf, MA, MCu, VL) = 12 dims. Partial dims
(SNs @200, the 3-item pass-2 pilot) are excluded automatically.

Each item is labelled by its confident (unanimous, 5/5) paired behaviour:
  * CORRECT   : 5/5 correct on BOTH conditions (yes-entailed AND no-refuted).
  * INCORRECT : unanimous (5/5) per condition but NOT aligned with both (dominated
                by always-"no" skeptic items; strict both-wrong is n=1).
  * excluded  : stochastic items (any condition with 1-4 of 5 correct).

Compares mean rubric demand per dimension between the two groups, SEM error bars +
Mann-Whitney U per dimension.

    python -m sciver_eval.viz_rubric_demand [--entailed-run 2 --refuted-run 3]
"""
import argparse
import collections
import csv
import sqlite3
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import mannwhitneyu

from sciver_eval import db as sdb

DEFAULT_OUT = sdb.analysis_dir() / "rubric_demand_by_correct.png"
RUBRIC_DB = Path(__file__).resolve().parent.parent / "rubric_scoring" / "annotations_prod.db"
# Authored figure-grounding dims (rubric_scoring.extract_rubrics.EMERGENT_DIMS);
# the rest are standard DeLeAn. Highlighted in the chart.
EMERGENT = {"VL", "GS", "MA"}
EMERGENT_COLOR = "#6a3d9a"


def load_rubric(db_path):
    """{item_id: {dim_code: score}}, ordered dim list, {dim_code: dim_name} — only
    dimensions scored on ALL items (full coverage), taking each dim from the pass
    that fully covers it."""
    con = sqlite3.connect(db_path)
    nitems = con.execute("SELECT COUNT(DISTINCT item_id) FROM annotation").fetchone()[0]
    dim_pass, names = {}, {}
    for code, p, c, name in con.execute(
            """SELECT dim_code, pass, COUNT(DISTINCT item_id), MAX(dim_name)
               FROM annotation WHERE parse_ok=1 GROUP BY dim_code, pass"""):
        if c == nitems:                       # full-coverage dim
            dim_pass[code] = p
            names[code] = name
    rubric = collections.defaultdict(dict)
    for code, p in dim_pass.items():
        for item_id, score in con.execute(
                "SELECT item_id, score FROM annotation WHERE dim_code=? AND pass=? AND parse_ok=1",
                (code, p)):
            rubric[item_id][code] = float(score)
    con.close()
    dims = sorted(dim_pass, key=lambda c: (dim_pass[c], c))   # group by pass, then code
    return rubric, dims, names, nitems


def paired_label(conn, e_run, r_run):
    def kc(run):
        return dict(conn.execute(
            "SELECT item_id, SUM(correct) FROM prediction WHERE run=? GROUP BY item_id", (run,)))
    e, r = kc(e_run), kc(r_run)
    label = {}
    for i in e:
        if i not in r:
            continue
        ec, rc = e[i], r[i]
        if ec in (0, 5) and rc in (0, 5):          # confident in both conditions
            label[i] = "correct" if (ec == 5 and rc == 5) else "incorrect"
    return label


def draw(dims, names, stats, n_corr, n_inc, out):
    x = np.arange(len(dims))
    w = 0.4
    cm = [stats[d]["correct"][0] for d in dims]
    ce = [stats[d]["correct"][1] for d in dims]
    im = [stats[d]["incorrect"][0] for d in dims]
    ie = [stats[d]["incorrect"][1] for d in dims]

    fig, ax = plt.subplots(figsize=(14, 6.5))
    # shade emergent (authored figure-grounding) dimensions
    for i, d in enumerate(dims):
        if d in EMERGENT:
            ax.axvspan(i - 0.5, i + 0.5, color=EMERGENT_COLOR, alpha=0.08, zorder=0)
    ax.bar(x - w / 2, cm, w, yerr=ce, capsize=3, color="#2ca25f",
           label=f"confidently CORRECT (n={n_corr})")
    ax.bar(x + w / 2, im, w, yerr=ie, capsize=3, color="#de2d26",
           label=f"confidently INCORRECT (n={n_inc})")
    top = max(max(cm), max(im)) + max(max(ce), max(ie))
    for i, d in enumerate(dims):
        p = stats[d]["p"]
        star = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else "ns"
        ax.text(x[i], top + 0.04, f"{star}\n{im[i] - cm[i]:+.2f}",
                ha="center", va="bottom", fontsize=8, color="0.25")
    labels = [f"{'◆ ' if d in EMERGENT else ''}{d}\n{(names.get(d) or '')[:16]}" for d in dims]
    ax.set_xticks(x, labels, fontsize=8)
    for t, d in zip(ax.get_xticklabels(), dims):
        if d in EMERGENT:
            t.set_color(EMERGENT_COLOR)
            t.set_fontweight("bold")
    ax.set_ylabel("mean rubric demand")
    ax.set_ylim(0, top + 0.5)
    ax.set_title("DeLeAn rubric demand (12 dims) by confident paired correctness "
                 "(Haiku 5/5, entailed×refuted)", fontsize=12, pad=12)
    ax.legend(loc="upper right", fontsize=9)
    ax.annotate("Δ shown = incorrect − correct (positive ⇒ failed items higher-demand).  "
                "Mann-Whitney U: *** p<.001, ** p<.01, * p<.05.  Dims ordered by significance (p asc).  "
                "◆ / purple = emergent (authored figure-grounding) dim; rest are DeLeAn.",
                xy=(0.5, -0.17), xycoords="axes fraction", ha="center", va="top",
                fontsize=8.5, color="0.4")
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(sdb.default_db_path()))
    ap.add_argument("--rubric-db", default=str(RUBRIC_DB))
    ap.add_argument("--entailed-run", type=int, default=2)
    ap.add_argument("--refuted-run", type=int, default=3)
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    args = ap.parse_args()

    rubric, dims, names, nitems = load_rubric(args.rubric_db)
    conn = sqlite3.connect(args.db)
    label = paired_label(conn, args.entailed_run, args.refuted_run)
    conn.close()

    groups = {"correct": [], "incorrect": []}
    for item, lab in label.items():
        if item in rubric:
            groups[lab].append(item)
    n_corr, n_inc = len(groups["correct"]), len(groups["incorrect"])

    stats = {}
    sem = lambda a: a.std(ddof=1) / np.sqrt(len(a)) if len(a) > 1 else 0.0
    for d in dims:
        vc = np.array([rubric[i][d] for i in groups["correct"] if d in rubric[i]])
        vi = np.array([rubric[i][d] for i in groups["incorrect"] if d in rubric[i]])
        try:
            p = mannwhitneyu(vc, vi, alternative="two-sided").pvalue
        except ValueError:
            p = 1.0
        stats[d] = {"correct": (vc.mean(), sem(vc)), "incorrect": (vi.mean(), sem(vi)), "p": p}

    # order dimensions by significance (p ascending; ties -> larger effect first)
    dims = sorted(dims, key=lambda d: (stats[d]["p"],
                                       -(stats[d]["incorrect"][0] - stats[d]["correct"][0])))

    out = Path(args.out)
    draw(dims, names, stats, n_corr, n_inc, out)

    csv_path = out.with_suffix(".csv")
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["dim_code", "dim_name", "emergent", "mean_correct", "sem_correct",
                    "mean_incorrect", "sem_incorrect", "delta_inc_minus_corr", "mannwhitney_p"])
        for d in dims:
            mc, sc = stats[d]["correct"]; mi, si = stats[d]["incorrect"]
            w.writerow([d, names.get(d, ""), "yes" if d in EMERGENT else "no",
                        f"{mc:.3f}", f"{sc:.3f}", f"{mi:.3f}",
                        f"{si:.3f}", f"{mi - mc:+.3f}", f"{stats[d]['p']:.4g}"])
        w.writerow(["N", "", f"correct={n_corr}", f"incorrect={n_inc}", "", "", "", ""])

    print(f"{len(dims)} full-coverage dims (of {nitems} items); CORRECT n={n_corr} INCORRECT n={n_inc}")
    for d in dims:
        mc = stats[d]["correct"][0]; mi = stats[d]["incorrect"][0]; p = stats[d]["p"]
        flag = "***" if p < 1e-3 else "**" if p < 1e-2 else "*" if p < 5e-2 else ""
        print(f"  {d:4s} {(names.get(d) or '')[:30]:30s} corr={mc:.2f} inc={mi:.2f} Δ={mi-mc:+.2f} p={p:.3g} {flag}")
    print(f"figure: {out}")
    print(f"counts: {csv_path}")


if __name__ == "__main__":
    main()
