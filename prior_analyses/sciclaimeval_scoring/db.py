import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _now(): return datetime.now(timezone.utc).isoformat()

SCHEMA = """
CREATE TABLE IF NOT EXISTS item(
  item_id TEXT PRIMARY KEY, source TEXT, paperid TEXT, claim_type TEXT,
  vtype TEXT, image_path TEXT, claim TEXT, label INTEGER, domain TEXT, operation TEXT);
CREATE TABLE IF NOT EXISTS pass(
  pass INTEGER PRIMARY KEY, created TEXT, skill_version TEXT, model TEXT,
  active_dims TEXT, item_source TEXT, n_items INTEGER, params TEXT, deviations TEXT);
CREATE TABLE IF NOT EXISTS annotation(
  id INTEGER PRIMARY KEY AUTOINCREMENT, pass INTEGER, session_id TEXT,
  item_id TEXT, dim_code TEXT, dim_name TEXT, agent_id TEXT, raw_output TEXT,
  score INTEGER, parse_ok INTEGER, input_tokens INTEGER, cache_creation_tokens INTEGER,
  cache_read_tokens INTEGER, output_tokens INTEGER, total_input_tokens INTEGER,
  output_thinking_tokens INTEGER, wall_clock_s REAL, created TEXT,
  UNIQUE(pass, item_id, dim_code));
"""

_FIELDS = ["pass", "session_id", "item_id", "dim_code", "dim_name", "agent_id", "raw_output",
           "score", "parse_ok", "input_tokens", "cache_creation_tokens", "cache_read_tokens",
           "output_tokens", "total_input_tokens", "output_thinking_tokens", "wall_clock_s"]


def connect(path) -> sqlite3.Connection:
    return sqlite3.connect(str(Path(path)))


def init(conn):
    conn.executescript(SCHEMA)
    conn.commit()


def register_pass(conn, pass_no, *, skill_version, model, active_dims, item_source,
                  n_items, params, deviations, created=None):
    conn.execute("INSERT OR REPLACE INTO pass VALUES(?,?,?,?,?,?,?,?,?)",
                 (pass_no, created if created is not None else _now(), skill_version, model,
                  active_dims, item_source, n_items, params, deviations))
    conn.commit()


def upsert_item(conn, it):
    conn.execute("INSERT OR REPLACE INTO item VALUES(?,?,?,?,?,?,?,?,?,?)",
                 (it.item_id, it.source, it.paperid, it.claim_type, it.vtype,
                  it.image_path, it.claim, 1 if it.label else 0, it.domain, it.operation))
    conn.commit()


def upsert_annotation(conn, row: dict):
    cols = ",".join(_FIELDS + ["created"])
    ph = ",".join(["?"] * (len(_FIELDS) + 1))
    upd = (",".join(f"{c}=excluded.{c}" for c in _FIELDS if c not in ("pass", "item_id", "dim_code"))
           + ",created=excluded.created")
    vals = [row.get(f) for f in _FIELDS] + [row.get("created") or _now()]
    conn.execute(
        f"INSERT INTO annotation({cols}) VALUES({ph}) "
        f"ON CONFLICT(pass,item_id,dim_code) DO UPDATE SET {upd}", vals)
    conn.commit()


def completed_cells(conn, pass_no) -> set:
    return {(i, d) for i, d in conn.execute(
        "SELECT item_id,dim_code FROM annotation WHERE pass=? AND parse_ok=1", (pass_no,))}
