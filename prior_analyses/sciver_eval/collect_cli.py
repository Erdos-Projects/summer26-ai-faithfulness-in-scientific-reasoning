"""Scrape this session's subagent transcripts into `prediction` for a run.

    python -m sciver_eval.collect_cli --run 1 --manifest <path to run_NN_manifest.json>

Safe to re-run (upsert). Token usage is read from the transcripts, so it is exact
and credit-free.
"""
import argparse
import json
import os

from sciver_eval import db
from sciver_eval.scrape import collect

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", dest="run_no", type=int, default=1)
    ap.add_argument("--manifest", required=True)
    a = ap.parse_args()

    conn = db.connect(db.default_db_path())
    db.init(conn)
    with open(a.manifest) as f:
        manifest = json.load(f)
    row = conn.execute("SELECT model FROM run WHERE run_id=?", (a.run_no,)).fetchone()
    model = row[0] if row else "claude-haiku-4-5"
    session_id = os.environ.get("CLAUDE_CODE_SESSION_ID", "unknown")

    n = collect(conn, a.run_no, model, manifest, session_id=session_id)
    done = len(db.completed_cells(conn, a.run_no, model))
    print(f"collected {n} predictions this session; run {a.run_no} now has {done} completed cells")
