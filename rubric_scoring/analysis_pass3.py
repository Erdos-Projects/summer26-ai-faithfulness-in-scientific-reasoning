"""Tier-1 exploratory analysis of pass-3 demand scores (7 dims over 200 SciVer chart items).
Mirrors analysis_pass1.py: per-dim summary, violin+strip grid, stacked demand profile,
Pearson/Spearman correlation heatmaps, and label-association (refuted vs entailed) across ALL 7
dims. NOTE: scoring is verdict-blind (the agent sees only claim+figure, never the label), so this
is NOT a "leak" test — it measures whether the refuted vs entailed claim-figure pairs genuinely
differ in demand, i.e. a dataset construction/selection artifact, not anything the annotator saw.
Outputs into rubric_scoring/analysis_pass3/.

Run:  python -m rubric_scoring.analysis_pass3 [--pass 3]
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

# pass-3 dims: 4 DeLeAn (MCu, CL, SNs, KNf) + 3 emergent (VL, GS, MA)
DIMS = ["MCu", "CL", "SNs", "KNf", "VL", "GS", "MA"]
DIM_LABEL = {
    "MCu": "MCu\nCalibrating Knowns/Unknowns",
    "CL": "CL\nConceptualisation/Learning",
    "SNs": "SNs\nSpatio-physical",
    "KNf": "KNf\nFormal Sciences",
    "VL": "VL\nVisual Localization",
    "GS": "GS\nGestalt/Shape",
    "MA": "MA\nMulti-Element Aggregation",
}
LEVELS = list(range(6))
SENS_MEAN, SENS_SD = 2.0, 1.0
NROW, NCOL = 2, 4  # 7 dims -> 2x4 grid, last cell blank


def load(conn, pass_no):
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


def _blank_extra(axes, n):
    for ax in axes.flat[n:]:
        ax.axis("off")


def summary_table(df, out_csv):
    recs = []
    for d in DIMS:
        s = df.loc[df.dim_code == d, "score"]
        mean, sd = s.mean(), s.std(ddof=1)
        recs.append({"dim_code": d, "n": int(s.size), "mean": round(mean, 3), "sd": round(sd, 3),
                     "min": int(s.min()), "max": int(s.max()),
                     "sensitive": bool(mean >= SENS_MEAN and sd >= SENS_SD),
                     "flat": bool(sd < SENS_SD)})
    tbl = pd.DataFrame(recs)
    tbl.to_csv(out_csv, index=False)
    return tbl


def violin_grid(df, out_png):
    fig, axes = plt.subplots(NROW, NCOL, figsize=(20, 10), sharey=True)
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
        ax.set_ylabel("demand level")
    _blank_extra(axes, len(DIMS))
    fig.suptitle("Pass-3 demand-level distributions by dimension (violin + strip, mean dashed)",
                 fontsize=15, y=0.99)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)


def demand_profile(df, out_png):
    cmap = plt.get_cmap("YlOrRd")
    colors = {lv: cmap(x) for lv, x in zip(LEVELS, np.linspace(0.12, 0.95, len(LEVELS)))}
    fig, ax = plt.subplots(figsize=(11, 6))
    for d_i, d in enumerate(DIMS):
        counts = df[df.dim_code == d].score.value_counts(normalize=True)
        left = 0.0
        for lv in LEVELS:
            frac = float(counts.get(lv, 0.0))
            if frac <= 0:
                continue
            ax.barh(d_i, frac, left=left, color=colors[lv], edgecolor="white")
            if frac >= 0.05:
                ax.text(left + frac / 2, d_i, f"{frac*100:.0f}", va="center", ha="center", fontsize=9)
            left += frac
    ax.set_yticks(range(len(DIMS)))
    ax.set_yticklabels([DIM_LABEL[d].replace("\n", " ") for d in DIMS])
    ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xlabel("fraction of items at each demand level")
    ax.set_title("Pass-3 demand profile per dimension (darker = higher level)")
    handles = [plt.Rectangle((0, 0), 1, 1, color=colors[lv]) for lv in LEVELS]
    ax.legend(handles, [f"level {lv}" for lv in LEVELS], ncol=len(LEVELS),
              loc="upper center", bbox_to_anchor=(0.5, -0.1), frameon=False)
    fig.tight_layout()
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)


def correlations(df, out_png):
    wide = df.pivot_table(index="item_id", columns="dim_code", values="score")[DIMS]
    pear, spear = wide.corr("pearson"), wide.corr("spearman")
    fig, axes = plt.subplots(1, 2, figsize=(17, 7))
    for ax, mat, name in zip(axes, [pear, spear], ["Pearson", "Spearman"]):
        sns.heatmap(mat, ax=ax, annot=True, fmt=".2f", cmap="vlag", vmin=-1, vmax=1,
                    square=True, cbar_kws={"shrink": 0.8}, annot_kws={"fontsize": 9})
        ax.set_title(f"{name} correlation of pass-3 demand scales", fontsize=12)
    fig.suptitle("Inter-dimension correlation (pass 3, 7 dims)", fontsize=14)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return pear, spear


def label_separation(df, out_csv, out_png):
    """refuted vs entailed across all 7 dims — two-sided label-ASSOCIATION test.
    Scoring is verdict-blind, so a significant difference is NOT a leak; it means the refuted and
    entailed claim-figure pairs genuinely differ in demand (a dataset construction/selection signal).
    Bonferroni threshold for 7 dims is 0.05/7 = 0.0071."""
    BONF = 0.05 / len(DIMS)
    recs = []
    for d in DIMS:
        sub = df[df.dim_code == d]
        ref = sub.loc[sub.verdict == "refuted", "score"].to_numpy()
        ent = sub.loc[sub.verdict == "entailed", "score"].to_numpy()
        u, p = mannwhitneyu(ref, ent, alternative="two-sided")
        rbc = 2.0 * u / (len(ref) * len(ent)) - 1.0
        recs.append({"dim_code": d, "mean_refuted": round(ref.mean(), 3),
                     "mean_entailed": round(ent.mean(), 3), "delta": round(ref.mean() - ent.mean(), 3),
                     "n_refuted": len(ref), "n_entailed": len(ent),
                     "mwu_p_two_sided": round(float(p), 5), "rank_biserial": round(float(rbc), 3),
                     "assoc_p05": bool(p < 0.05), "assoc_bonferroni": bool(p < BONF)})
    tbl = pd.DataFrame(recs)
    tbl.to_csv(out_csv, index=False)

    fig, axes = plt.subplots(NROW, NCOL, figsize=(20, 10), sharey=True)
    for ax, d in zip(axes.flat, DIMS):
        sub = df[df.dim_code == d]
        sns.violinplot(data=sub, x="verdict", y="score", hue="verdict", order=["refuted", "entailed"],
                       palette={"refuted": "#d6604d", "entailed": "#4393c3"}, cut=0,
                       inner="quartile", legend=False, ax=ax)
        for xi, v in enumerate(["refuted", "entailed"]):
            m = sub.loc[sub.verdict == v, "score"].mean()
            ax.plot(xi, m, marker="D", color="gold", markeredgecolor="black", markersize=9, zorder=5)
        row = tbl.set_index("dim_code").loc[d]
        flag = "  *assoc(Bonf)" if row.assoc_bonferroni else ("  assoc(p<.05)" if row.assoc_p05 else "")
        ax.set_title(f"{d}  Δ={row.delta:+.2f} p={row.mwu_p_two_sided:.3g}{flag}", fontsize=11)
        ax.set_ylim(-0.3, 5.3)
        ax.set_yticks(LEVELS)
        ax.set_xlabel("")
        ax.set_ylabel("demand level")
    _blank_extra(axes, len(DIMS))
    fig.suptitle("Pass-3 demand by gold label (refuted vs entailed) — label-ASSOCIATION (scoring is "
                 "verdict-blind; a gap = the data's refuted/entailed pairs differ in demand, not a leak)",
                 fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    return tbl


def readout(summary, pear, spear, sep, df):
    sens = summary.loc[summary.sensitive, "dim_code"].tolist()
    flat = summary.loc[summary.flat, "dim_code"].tolist()

    def max_offdiag(m):
        a = m.to_numpy().copy()
        np.fill_diagonal(a, 0.0)
        i, j = np.unravel_index(np.argmax(np.abs(a)), a.shape)
        return m.index[i], m.columns[j], a[i, j]
    pi, pj, pv = max_offdiag(pear)
    si, sj, sv = max_offdiag(spear)
    assoc05 = sep.loc[sep.assoc_p05, "dim_code"].tolist()
    assocB = sep.loc[sep.assoc_bonferroni, "dim_code"].tolist()
    nref = int((df[df.dim_code == DIMS[0]].verdict == "refuted").sum())
    nent = int((df[df.dim_code == DIMS[0]].verdict == "entailed").sum())
    print("\n" + "=" * 84 + "\nPLAIN-LANGUAGE READOUT (pass 3)\n" + "=" * 84)
    print(f"Item label split (of {nref+nent}): {nref} refuted / {nent} entailed (imbalanced).")
    print(f"Sensitive scales (mean>=2 & s.d.>=1.0): {', '.join(sens) or 'none'}. "
          f"Flat scales (s.d.<1.0, low resolution on these charts): {', '.join(flat) or 'none'}.")
    print(f"Strongest correlation: Pearson {pv:+.2f} ({pi}-{pj}), Spearman {sv:+.2f} ({si}-{sj}) — "
          f"{'REDUNDANT (|r|>0.8)' if abs(pv) > 0.8 or abs(sv) > 0.8 else 'below the 0.8 redundancy bar'}.")
    print("Label association (refuted vs entailed). Scoring is VERDICT-BLIND, so this is NOT a leak: "
          "a gap means the dataset's refuted vs entailed claim-figure pairs genuinely differ in demand "
          "(a construction/selection signal a downstream predictor could exploit).")
    print(f"  significant at p<.05: {', '.join(assoc05) or 'none'};  surviving Bonferroni (p<.0071): "
          f"{', '.join(assocB) or 'none'} (refuted scores higher on each). Confirm on actual twin pairs "
          f"(same claim family, original vs perturbed) before reading too much into the marginal split.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="pass_no", type=int, default=3)
    ap.add_argument("--outdir", default=os.path.join(config.repo_root(), "rubric_scoring", "analysis_pass3"))
    a = ap.parse_args()
    os.makedirs(a.outdir, exist_ok=True)
    sns.set_theme(style="whitegrid", context="talk")

    conn = db.connect(config.db_path())
    df = load(conn, a.pass_no)
    print(f"loaded {len(df)} scored cells for pass {a.pass_no} "
          f"({df.item_id.nunique()} items x {df.dim_code.nunique()} dims)")

    summary = summary_table(df, os.path.join(a.outdir, "pass3_summary.csv"))
    print("\nPER-DIMENSION SUMMARY\n" + summary.to_string(index=False))
    violin_grid(df, os.path.join(a.outdir, "pass3_violins.png"))
    demand_profile(df, os.path.join(a.outdir, "pass3_demand_profile.png"))
    pear, spear = correlations(df, os.path.join(a.outdir, "pass3_correlations.png"))
    sep = label_separation(df, os.path.join(a.outdir, "pass3_label_separation.csv"),
                           os.path.join(a.outdir, "pass3_label_separation.png"))
    print("\nLABEL ASSOCIATION (verdict-blind scoring; data signal, not a leak)\n" + sep.to_string(index=False))
    readout(summary, pear, spear, sep, df)
    print(f"\nfigures + CSVs written to {a.outdir}")


if __name__ == "__main__":
    main()
