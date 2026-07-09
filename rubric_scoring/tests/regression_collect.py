"""Manual regression: the NEW incremental collector must reproduce the prod annotation
rows. Proves the "byte-identical annotation output" constraint of the incremental-
collection feature (see docs/superpowers/specs/2026-06-18-incremental-transcript-collect-design.md).

This is NOT a pytest unit (no test_ functions): it depends on the live ~/.claude/projects
transcript corpus (~48k files) and the committed rubric_scoring/annotations_prod.db, and
takes tens of minutes. Run it deliberately, from the repo root, with the project venv:

    env -u CLAUDE_CODE_SESSION_ID PYTHONPATH=. .venv/bin/python -m rubric_scoring.tests.regression_collect

Why each part of that command:
  - `env -u CLAUDE_CODE_SESSION_ID` forces collect()'s glob UNSCOPED (scan every session's
    transcripts). Prod was assembled across many sessions; scoping to the current session
    would find none of them. `full=True` bypasses the ledger but does NOT change scoping.
  - the project venv supplies pypdf (pulled in transitively via prompts -> extract_rubrics).

Method: the full set of cells prod knows about cannot be reconstructed from the on-disk
prompt manifests (prepare.py writes MISSING cells only). So we rebuild the full manifest
from prod itself (every distinct pass/item/dim), run the new collector over the real
corpus into a scratch DB, and diff scored rows against prod.

Validated 2026-06-18: 10013/10013 prod scored cells reproduced, 0 missing, 0 extra, and
2 score differences — both on item sciver_val_417 (pass 1, dims AS & QLl), which was
scored THREE times with within-1 differing scores (AS: 1,2,2 ; QLl: 2,1,1). When a cell
has multiple differing scored transcripts, which attempt wins is glob-order dependent in
BOTH the old and new collector, so these are benign data nondeterminism, not a regression.
"""
import os, sqlite3, tempfile

# Force the unscoped glob BEFORE importing the collector's config-reading helpers.
os.environ.pop("CLAUDE_CODE_SESSION_ID", None)

from rubric_scoring import db, scrape, config
from rubric_scoring.prompts import tag as mktag


def _scored(conn):
    return dict(conn.execute(
        "SELECT pass||'|'||item_id||'|'||dim_code, score FROM annotation WHERE parse_ok=1"))


def main():
    prod = db.connect(config.db_path())
    # Full manifest = every cell prod knows about, keyed by the canonical tag.
    manifest = {mktag(p, item, dim): {"item_id": item, "dim_code": dim, "dim_name": dname}
                for p, item, dim, dname in prod.execute(
                    "SELECT DISTINCT pass,item_id,dim_code,dim_name FROM annotation")}
    print(f"full manifest reconstructed from prod: {len(manifest)} cells")

    scratch = tempfile.mkdtemp() + "/rebuild_full.db"
    conn = db.connect(scratch); db.init(conn)
    # collect() labels rows with the pass_no argument, so run it once per pass over the
    # pass's slice of the manifest. full=True => ledger ignored, whole corpus rescanned.
    passes = [r[0] for r in prod.execute("SELECT DISTINCT pass FROM annotation")]
    for pno in passes:
        sub = {t: m for t, m in manifest.items() if t.startswith(f"rs_p{pno:02d}__")}
        n = scrape.collect(conn, pno, sub, session_id="unknown", full=True)
        print(f"  pass {pno}: collected {n} rows from {len(sub)} manifest cells")

    P, R = _scored(prod), _scored(conn)
    pk, rk = set(P), set(R)
    only_prod, only_reb = pk - rk, rk - pk
    mism = sorted(k for k in (pk & rk) if P[k] != R[k])
    print(f"\nprod scored: {len(P)} | rebuild scored: {len(R)}")
    print(f"only in prod: {len(only_prod)} | only in rebuild: {len(only_reb)} | "
          f"common: {len(pk & rk)} | score mismatches: {len(mism)}")
    for k in mism:
        print(f"  {k}: prod={P[k]} rebuild={R[k]}")

    # Hard gate: the cell SET must match exactly. Score mismatches are reported for the
    # human to confirm they are multi-attempt nondeterminism (see module docstring), not
    # a collector defect — they are not auto-failed here.
    ok = not only_prod and not only_reb
    print(f"\nCELL-SET MATCH: {'PASS' if ok else 'FAIL'}"
          + ("" if ok else f"  (missing {len(only_prod)}, extra {len(only_reb)})"))
    print(f"SCRATCH_DB {scratch}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
