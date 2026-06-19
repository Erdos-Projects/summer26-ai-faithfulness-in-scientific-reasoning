"""Tier-1 exploratory analysis of pass-1 DeLeAn demand scores (annotation-only, no model runs).

Produces, for the 6 active dims (AS, QLq, QLl, MCr, AT, VO) over the 817 SciVer chart items:
  1. per-dimension summary table (mean/sd/min/max + `sensitive`/`flat` flags) -> stdout + CSV
  2. violin+strip grid (2x3), shared y, mean marked -> PNG
  3. stacked horizontal demand-profile bars (level-frequency segments) -> PNG
  4. side-by-side Pearson & Spearman correlation heatmaps -> PNG
  5. AT/VO separation by gold label (refuted vs entailed): stats + Mann-Whitney + split violins -> CSV + PNG
Then prints a plain-language readout.

Run:  python -m rubric_scoring.analysis_pass1 [--pass N]
Axis note: a handful of level-0 scores exist (QLq, MCr), so axes/segments span 0-5 (not 1-5)
to avoid hiding the floor; the paper's sensitivity criterion is unaffected.
"""
import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.stats import mannwhitneyu

from rubric_scoring import config, db

# Active dims in a deliberate order: elemental scan/quant/logic/relevance, then the two extraneous.
DIMS = ["AS", "QLq", "QLl", "MCr", "AT", "VO"]
DIM_LABEL = {
    "AS": "AS\nAttention & Scan",
    "QLq": "QLq\nQuantitative",
    "QLl": "QLl\nLogical",
    "MCr": "MCr\nRelevant Info",
    "AT": "AT\nAtypicality",
    "VO": "VO\nVolume",
}
LEVELS = list(range(6))  # 0-5 inclusive (0s present on QLq/MCr; 5 present on AT)
SENS_MEAN, SENS_SD = 2.0, 1.0  # paper's sensitivity criterion: mean >= 2 AND s.d. >= 1.0


def load(conn, pass_no):
    """Long dataframe: one row per scored cell, with the item's gold label joined."""
    rows = conn.execute(
        """SELECT a.item_id, a.dim_code, a.score, i.label
           FROM annotation a JOIN item i ON a.item_id = i.item_id
           WHERE a.pass = ? AND a.parse_ok = 1""",
        (pass_no,),
    ).fetchall()
    df = pd.DataFrame([tuple(r) for r in rows], columns=["item_id", "dim_code", "score", "label"])
    df["score"] = df["score"].astype(int)
    df["verdict"] = df["label"].map({0: "refuted", 1: "entailed"})
    return df


def summary_table(df, out_csv):
    recs = []
    for d in DIMS:
        s = df.loc[df.dim_code == d, "score"]
        mean, sd = s.mean(), s.std(ddof=1)
        recs.append(
            {
                "dim_code": d,
                "n": int(s.size),
                "mean": round(mean, 3),
                "sd": round(sd, 3),
                "min": int(s.min()),
                "max": int(s.max()),
                "sensitive": bool(mean >= SENS_MEAN and sd >= SENS_SD),
                "flat": bool(sd < SENS_SD),
            }
        )
    tbl = pd.DataFrame(recs)
    tbl.to_csv(out_csv, index=False)
    return tbl


def violin_grid(df, out_png):
    fig, axes = plt.subplots(2, 3, figsize=(15, 9), sharey=True)
    for ax, d in zip(axes.flat, DIMS):
        sub = df[df.dim_code == d]
        sns.violinplot(y=sub.score, ax=ax, color="#9ecae1", cut=0, inner=None, linewidth=1)
        sns.stripplot(y=sub.score, ax=ax, color="#08306b", size=2, alpha=0.25, jitter=0.28)
        m = sub.score.mean()
        ax.axhline(m, color="crimson", lw=2, ls="--")
        ax.text(0.97, m, f" mean {m:.2f}", color="crimson", va="bottom", ha="right",
                transform=ax.get_yaxis_transform(), fontsize=10, fontweight="bold")
        ax.set_title(DIM_LABEL[d], fontsize=11)
        ax.set_ylim(-0.3, 5.3)
        ax.set_yticks(LEVELS)
        ax.set_xlabel("")
        ax.set_ylabel("demand level" if ax in (axes[0, 0], axes[1, 0]) else "")
    fig.suptitle("Pass-1 demand-level distributions by dimension (violin + strip, mean dashed)",
                 fontsize=14, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def demand_profile(df, out_png):
    cmap = plt.get_cmap("YlOrRd")
    colors = {lv: cmap(x) for lv, x in zip(LEVELS, np.linspace(0.12, 0.95, len(LEVELS)))}
    fig, ax = plt.subplots(figsize=(11, 6))
    y = np.arange(len(DIMS))
    for d_i, d in enumerate(DIMS):
        counts = df[df.dim_code == d].score.value_counts(normalize=True)
        left = 0.0
        for lv in LEVELS:
            frac = float(counts.get(lv, 0.0))
            if frac <= 0:
                continue
            ax.barh(d_i, frac, left=left, color=colors[lv], edgecolor="white")
            if frac >= 0.05:
                ax.text(left + frac / 2, d_i, f"{frac*100:.0f}", va="center", ha="center",
                        fontsize=9, color="black")
            left += frac
    ax.set_yticks(y)
    ax.set_yticklabels([DIM_LABEL[d].replace("\n", " ") for d in DIMS])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of items at each demand level")
    ax.set_title("Pass-1 demand profile per dimension (darker = higher level)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[lv]) for lv in LEVELS]
    ax.legend(handles, [f"level {lv}" for lv in LEVELS], ncol=len(LEVELS),
              loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def correlations(df, out_png):
    wide = df.pivot_table(index="item_id", columns="dim_code", values="score")[DIMS]
    pear = wide.corr(method="pearson")
    spear = wide.corr(method="spearman")
    fig, axes = plt.subplots(1, 2, figsize=(15, 6.5))
    for ax, mat, name in zip(axes, [pear, spear], ["Pearson", "Spearman"]):
        sns.heatmap(mat, ax=ax, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1,
                    square=True, cbar_kws={"shrink": 0.8}, annot_kws={"fontsize": 10})
        ax.set_title(f"{name} correlation of demand scales", fontsize=12)
    fig.suptitle("Inter-dimension correlation (are the 6 scales redundant or independent?)",
                 fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return pear, spear


def label_separation(df, out_csv, out_png):
    """AT/VO by gold label: the shortcut-artifact diagnostic the paper never ran on SciVer."""
    recs = []
    for d in ["AT", "VO"]:
        sub = df[df.dim_code == d]
        ref = sub.loc[sub.verdict == "refuted", "score"].to_numpy()
        ent = sub.loc[sub.verdict == "entailed", "score"].to_numpy()
        # Mann-Whitney U (refuted > entailed?), one-sided 'greater'; rank-biserial effect size.
        u, p = mannwhitneyu(ref, ent, alternative="greater")
        rbc = 2.0 * u / (len(ref) * len(ent)) - 1.0  # rank-biserial correlation
        recs.append(
            {
                "dim_code": d,
                "mean_refuted": round(ref.mean(), 3),
                "mean_entailed": round(ent.mean(), 3),
                "delta": round(ref.mean() - ent.mean(), 3),
                "n_refuted": len(ref),
                "n_entailed": len(ent),
                "mwu_p_refuted_gt_entailed": round(float(p), 5),
                "rank_biserial": round(float(rbc), 3),
            }
        )
    tbl = pd.DataFrame(recs)
    tbl.to_csv(out_csv, index=False)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)
    for ax, d in zip(axes, ["AT", "VO"]):
        sub = df[df.dim_code == d]
        sns.violinplot(data=sub, x="verdict", y="score", hue="verdict", order=["refuted", "entailed"],
                       palette={"refuted": "#d6604d", "entailed": "#4393c3"}, cut=0,
                       inner="quartile", legend=False, ax=ax)
        sns.stripplot(data=sub, x="verdict", y="score", order=["refuted", "entailed"],
                      color="black", size=2, alpha=0.2, jitter=0.3, ax=ax)
        for xi, v in enumerate(["refuted", "entailed"]):
            m = sub.loc[sub.verdict == v, "score"].mean()
            ax.plot(xi, m, marker="D", color="gold", markeredgecolor="black", markersize=9, zorder=5)
        ax.set_title(DIM_LABEL[d].replace("\n", " "))
        ax.set_ylim(-0.3, 5.3)
        ax.set_yticks(LEVELS)
        ax.set_xlabel("")
        ax.set_ylabel("demand level" if d == "AT" else "")
    fig.suptitle("Extraneous-dimension separation by gold label (refuted vs entailed) — shortcut-artifact test",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return tbl


def readout(summary, pear, spear, sep):
    sens = summary.loc[summary.sensitive, "dim_code"].tolist()
    flat = summary.loc[summary.flat, "dim_code"].tolist()
    # strongest off-diagonal |r| in either matrix
    def max_offdiag(m):
        a = m.to_numpy().copy()
        np.fill_diagonal(a, 0.0)
        i, j = np.unravel_index(np.argmax(np.abs(a)), a.shape)
        return m.index[i], m.columns[j], a[i, j]
    pi, pj, pv = max_offdiag(pear)
    si, sj, sv = max_offdiag(spear)
    redundant = abs(pv) > 0.8 or abs(sv) > 0.8
    print("\n" + "=" * 78 + "\nPLAIN-LANGUAGE READOUT\n" + "=" * 78)
    print(
        f"Sensitive scales (mean>=2 & s.d.>=1.0): {', '.join(sens) or 'none'}. "
        f"Flat scales (s.d.<1.0): {', '.join(flat) or 'none'}. The flat ones concentrate demand "
        f"in a narrow band, so on these SciVer charts they barely discriminate — mirroring the "
        f"paper's 'lack of sensitivity' concern."
    )
    print(
        f"No scale pair is redundant: the strongest association is Pearson {pv:+.2f} "
        f"({pi}-{pj}) and Spearman {sv:+.2f} ({si}-{sj}), both "
        f"{'ABOVE' if redundant else 'well below'} the |r|>0.8 redundancy threshold — the six "
        f"scales behave as largely independent axes of chart difficulty."
    )
    at = sep.set_index("dim_code").loc["AT"]
    vo = sep.set_index("dim_code").loc["VO"]
    def verdict(row):
        if row.mwu_p_refuted_gt_entailed < 0.05 and row.delta > 0:
            return f"refuted DO score higher (Δ={row.delta:+.2f}, p={row.mwu_p_refuted_gt_entailed:.3g}, rank-biserial={row.rank_biserial:+.2f})"
        return f"distributions overlap (Δ={row.delta:+.2f}, p={row.mwu_p_refuted_gt_entailed:.3g})"
    print(
        f"Shortcut-artifact test — Atypicality: {verdict(at)}; Volume: {verdict(vo)}. "
        f"A clear refuted>entailed gap is a measurable fingerprint of the construction artifact; "
        f"overlap is evidence the edited claims are not trivially artifact-detectable on these scales."
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="pass_no", type=int, default=1)
    ap.add_argument("--outdir", default=os.path.join(config.repo_root(), "rubric_scoring", "analysis"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    conn = db.connect(config.db_path())
    df = load(conn, a.pass_no)
    print(f"loaded {len(df)} scored cells for pass {a.pass_no} "
          f"({df.item_id.nunique()} items x {df.dim_code.nunique()} dims); "
          f"labels: {(df.label==0).sum()//len(DIMS)} refuted / {(df.label==1).sum()//len(DIMS)} entailed")

    summary = summary_table(df, os.path.join(a.outdir, "pass1_summary.csv"))
    print("\nPER-DIMENSION SUMMARY\n" + summary.to_string(index=False))

    violin_grid(df, os.path.join(a.outdir, "pass1_violins.png"))
    demand_profile(df, os.path.join(a.outdir, "pass1_demand_profile.png"))
    pear, spear = correlations(df, os.path.join(a.outdir, "pass1_correlations.png"))
    sep = label_separation(df, os.path.join(a.outdir, "pass1_label_separation.csv"),
                           os.path.join(a.outdir, "pass1_label_separation.png"))
    print("\nAT/VO SEPARATION BY LABEL\n" + sep.to_string(index=False))

    readout(summary, pear, spear, sep)
    print(f"\nfigures + CSVs written to {a.outdir}")


if __name__ == "__main__":
    main()
