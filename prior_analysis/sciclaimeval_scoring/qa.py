"""Per-pass QA: parse rate, thinking==0, score spread per dim, degenerate-distribution flags."""
from collections import defaultdict

def run_qa(conn, pass_no):
    rows = conn.execute(
        "SELECT dim_code,score,parse_ok,output_thinking_tokens FROM annotation WHERE pass=?",
        (pass_no,)).fetchall()
    n = len(rows)
    parse_ok = sum(r[2] for r in rows)
    think_total = sum(r[3] or 0 for r in rows)
    per = defaultdict(list)
    for dim, score, ok, _ in rows:
        if ok: per[dim].append(score)
    per_dim = {d: {"n": len(v), "distinct_scores": len(set(v)),
                   "min": min(v), "max": max(v),
                   "flat": len(set(v)) <= 1} for d, v in per.items()}
    return {"pass": pass_no, "n": n,
            "parse_rate": (parse_ok / n) if n else 0.0,
            "thinking_tokens_total": think_total,
            "thinking_clean": think_total == 0,
            "per_dim": per_dim}
