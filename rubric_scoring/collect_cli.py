"""Scrape this session's subagent transcripts into the DB for a pass."""
import argparse, json, os
from rubric_scoring import config, db
from rubric_scoring.scrape import collect

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="pass_no", type=int, required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--full", action="store_true",
                    help="ignore the ingestion ledger and rescan every transcript")
    a = ap.parse_args()
    conn = db.connect(config.db_path()); db.init(conn)
    with open(a.manifest) as f:
        manifest = json.load(f)
    session_id = os.environ.get("CLAUDE_CODE_SESSION_ID", "unknown")
    n = collect(conn, a.pass_no, manifest, session_id=session_id, full=a.full)
    done = len(db.completed_cells(conn, a.pass_no))
    print(f"collected {n} cells this session; pass {a.pass_no} now has {done} completed cells")
