"""SQLite layer for the SciClaimEval prediction eval.

`item` matches the schema in `sciclaimeval_scoring.db` (same `item_id`, `domain`,
and `operation` columns), so the two DBs join on `item_id` for downstream
`r(demand, p_correct)` analysis. `run`/`prediction` are the verdict-aware tables
that record per-call Supported/Refuted predictions and token provenance.
"""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _now():
    return datetime.now(timezone.utc).isoformat()


# Runs go through the harness-subagent agent() path (see wf_dispatch.js), which
# exposes no temperature argument. The effective sampling temperature is the
# Claude Code / Workflow default and is NOT controlled or verified by this code.
# Record temperature as NULL and state the caveat in `deviations` rather than
# logging a hardcoded value that implies a parameter we set.
TEMPERATURE_UNCONTROLLED_NOTE = (
    "temperature not controlled: harness-subagent agent() path exposes no "
    "temperature arg; effective value = Claude Code/Workflow default "
    "(unverified, assumed ~1.0, not enforced)"
)


SCHEMA = """
CREATE TABLE IF NOT EXISTS item(
  item_id TEXT PRIMARY KEY, source TEXT, paperid TEXT, claim_type TEXT,
  vtype TEXT, image_path TEXT, claim TEXT, label INTEGER, domain TEXT, operation TEXT);
CREATE TABLE IF NOT EXISTS run(
  run_id INTEGER PRIMARY KEY, created TEXT, skill_version TEXT, model TEXT,
  prompt_version TEXT, temperature REAL, item_source TEXT, n_items INTEGER,
  params TEXT, deviations TEXT);
CREATE TABLE IF NOT EXISTS prediction(
  id INTEGER PRIMARY KEY AUTOINCREMENT, run INTEGER, session_id TEXT,
  item_id TEXT, model TEXT, trial INTEGER, agent_id TEXT, raw_output TEXT, answer TEXT,
  predicted INTEGER, parse_ok INTEGER, correct INTEGER,
  input_tokens INTEGER, cache_creation_tokens INTEGER, cache_read_tokens INTEGER,
  output_tokens INTEGER, total_input_tokens INTEGER, output_thinking_tokens INTEGER,
  wall_clock_s REAL, created TEXT,
  UNIQUE(run, item_id, model, trial));
"""

_PRED_FIELDS = ["run", "session_id", "item_id", "model", "trial", "agent_id", "raw_output",
                "answer", "predicted", "parse_ok", "correct", "input_tokens",
                "cache_creation_tokens", "cache_read_tokens", "output_tokens",
                "total_input_tokens", "output_thinking_tokens", "wall_clock_s"]


def default_db_path() -> Path:
    return Path(__file__).resolve().parent / "predictions_sciclaimeval.db"


def analysis_dir() -> Path:
    """Canonical home for derived figures/tables over the runs in predictions.db.

    All trial-analysis scripts (viz_venn, viz_difficulty, ...) write here so the
    outputs stay co-located with the data. Created on demand.
    """
    d = Path(__file__).resolve().parent / "trial_analysis"
    d.mkdir(parents=True, exist_ok=True)
    return d


def connect(path) -> sqlite3.Connection:
    return sqlite3.connect(str(Path(path)))


def init(conn):
    conn.executescript(SCHEMA)
    conn.commit()


def upsert_item(conn, it):
    """`it` is a sciclaimeval_scoring.items.Item."""
    conn.execute("INSERT OR REPLACE INTO item VALUES(?,?,?,?,?,?,?,?,?,?)",
                 (it.item_id, it.source, it.paperid, it.claim_type, it.vtype,
                  it.image_path, it.claim, 1 if it.label else 0, it.domain, it.operation))


def register_run(conn, run_no, *, skill_version, model, prompt_version, temperature,
                 item_source, n_items, params="", deviations="", created=None):
    conn.execute("INSERT OR REPLACE INTO run VALUES(?,?,?,?,?,?,?,?,?,?)",
                 (run_no, created or _now(), skill_version, model, prompt_version,
                  temperature, item_source, n_items, params, deviations))
    conn.commit()


def upsert_prediction(conn, row: dict):
    cols = ",".join(_PRED_FIELDS + ["created"])
    ph = ",".join(["?"] * (len(_PRED_FIELDS) + 1))
    upd = (",".join(f"{c}=excluded.{c}" for c in _PRED_FIELDS
                    if c not in ("run", "item_id", "model", "trial")) + ",created=excluded.created")
    vals = [row.get(f) for f in _PRED_FIELDS] + [row.get("created") or _now()]
    conn.execute(
        f"INSERT INTO prediction({cols}) VALUES({ph}) "
        f"ON CONFLICT(run,item_id,model,trial) DO UPDATE SET {upd}", vals)
    conn.commit()


def completed_cells(conn, run_no, model) -> set:
    """{(item_id, trial)} already scored for this run/model (resume key)."""
    return {(i, t) for i, t in conn.execute(
        "SELECT item_id, trial FROM prediction WHERE run=? AND model=? AND parse_ok=1",
        (run_no, model))}
