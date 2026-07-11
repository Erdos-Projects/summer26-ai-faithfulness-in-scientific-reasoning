"""Paired entailed-vs-refuted analysis: join run 2 (entailed) and run 3 (refuted)
on item_id and report the accuracy gap, McNemar on discordant pairs, and pooled
signal-detection d'/criterion. Pure stats use stdlib statistics.NormalDist.

    python -m sciver_eval.analyze_paired --ent 2 --ref 3
"""
import argparse
from statistics import NormalDist

from sciver_eval import db

_Z = NormalDist().inv_cdf


def dprime_criterion(hit, fa, n_signal=None, n_noise=None):
    """Signal-detection d' and criterion c. Clamps 0/1 rates (log-linear style)
    using the per-class N when given, else a small epsilon, so z stays finite."""
    def clamp(p, n):
        eps = (1.0 / (2 * n)) if n else 1e-3
        return min(1 - eps, max(eps, p))
    zh, zf = _Z(clamp(hit, n_signal)), _Z(clamp(fa, n_noise))
    return zh - zf, -0.5 * (zh + zf)


def mcnemar(b, c):
    """McNemar chi-square with continuity correction: max(0,|b-c|-1)^2/(b+c).
    Clamping the corrected difference at 0 makes a symmetric table (b==c) score
    exactly 0. Returns (statistic, dof); b,c are the discordant-pair counts."""
    n = b + c
    if n == 0:
        return 0.0, 1
    corr = max(0.0, abs(b - c) - 1)
    return corr * corr / n, 1


def _per_item_correct(conn, run, model):
    """item_id -> mean(correct) over parsed trials for one run."""
    rows = conn.execute(
        "SELECT item_id, AVG(correct) FROM prediction "
        "WHERE run=? AND model=? AND parse_ok=1 GROUP BY item_id", (run, model)).fetchall()
    return {i: a for i, a in rows}


def paired_metrics(conn, run_ent, run_ref, model):
    ent = _per_item_correct(conn, run_ent, model)
    ref = _per_item_correct(conn, run_ref, model)
    shared = sorted(set(ent) & set(ref))
    acc_e = sum(ent[i] for i in shared) / len(shared) if shared else 0.0
    acc_r = sum(ref[i] for i in shared) / len(shared) if shared else 0.0
    # per-item binary (majority-correct over trials) for discordant pairs
    be = {i: ent[i] >= 0.5 for i in shared}
    br = {i: ref[i] >= 0.5 for i in shared}
    b = sum(1 for i in shared if be[i] and not br[i])   # entailed-only correct
    c = sum(1 for i in shared if br[i] and not be[i])   # refuted-only correct
    stat, dof = mcnemar(b, c)
    hit = acc_e                                         # P(yes | entailed) = entailed accuracy
    fa = 1 - acc_r                                      # P(yes | refuted)  = 1 - refuted accuracy
    dprime, crit = dprime_criterion(hit, fa, n_signal=len(shared), n_noise=len(shared))
    return {"n_items": len(shared), "acc_entailed": acc_e, "acc_refuted": acc_r,
            "gap": acc_e - acc_r, "mcnemar_b": b, "mcnemar_c": c,
            "mcnemar_stat": stat, "mcnemar_dof": dof, "dprime": dprime, "criterion": crit}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ent", type=int, default=2)
    ap.add_argument("--ref", type=int, default=3)
    ap.add_argument("--model", default="claude-haiku-4-5")
    a = ap.parse_args()
    conn = db.connect(db.default_db_path())
    m = paired_metrics(conn, a.ent, a.ref, a.model)
    print(f"n={m['n_items']}  acc_entailed={m['acc_entailed']:.3f}  "
          f"acc_refuted={m['acc_refuted']:.3f}  gap={m['gap']:+.3f}")
    print(f"McNemar b={m['mcnemar_b']} c={m['mcnemar_c']} chi2={m['mcnemar_stat']:.3f}")
    print(f"SDT  d'={m['dprime']:.3f}  criterion c={m['criterion']:+.3f}  "
          f"(c>0 = skeptic bias toward 'no')")
