"""SQLite store for the CharXiv rubric pass + the released correctness scores.

Three concerns in one file:
  * item        — the 1000 val chart-QA items (figure + reasoning question + metadata)
  * pass / annotation — the resume-aware DeLeAn demand pass (one row per item×dim cell)
  * model_score — CharXiv's RELEASED per-model correctness (reasoning + descriptive), so the
                  correctness side needs no eval run; joins to item on item_id.
"""
import sqlite3
from datetime import datetime, timezone
from pathlib import Path


def _now(): return datetime.now(timezone.utc).isoformat()

SCHEMA = """
CREATE TABLE IF NOT EXISTS item(
  item_id TEXT PRIMARY KEY, source TEXT, paperid TEXT, figure_id INTEGER,
  image_path TEXT, query TEXT, answer TEXT, inst_category INTEGER,
  category TEXT, year TEXT, title TEXT);
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
CREATE TABLE IF NOT EXISTS model_score(
  model TEXT, task TEXT, item_id TEXT, sub_q INTEGER, figure_id INTEGER,
  extracted_answer TEXT, score INTEGER,
  PRIMARY KEY(model, task, item_id, sub_q));
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
    conn.execute("INSERT OR REPLACE INTO item VALUES(?,?,?,?,?,?,?,?,?,?,?)",
                 (it.item_id, it.source, it.paperid, it.figure_id, it.image_path,
                  it.query, it.answer, it.inst_category, it.category, it.year, it.title))
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


def upsert_model_score(conn, model, task, item_id, sub_q, figure_id, extracted_answer, score):
    conn.execute(
        "INSERT OR REPLACE INTO model_score VALUES(?,?,?,?,?,?,?)",
        (model, task, item_id, sub_q, figure_id, extracted_answer, score))


def completed_cells(conn, pass_no) -> set:
    return {(i, d) for i, d in conn.execute(
        "SELECT item_id,dim_code FROM annotation WHERE pass=? AND parse_ok=1", (pass_no,))}
