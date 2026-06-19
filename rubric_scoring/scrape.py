"""Recover per-cell scores + exact token usage from subagent transcripts.
Transcripts: <projects>/**/subagents/agent-*.jsonl ; matched to cells by the tag in their text."""
import glob, json, os, re
from datetime import datetime
from rubric_scoring import db, config
from rubric_scoring.parse import parse_score

# Tags emit newline-bounded as "PILOT_TAG: {tag}\n"; the trailing [A-Za-z]+ for the dim
# code is intentionally unbounded — safe because a tag is never immediately followed by a letter.
_TAG_RE = re.compile(r"rs_p\d+__\S+?__[A-Za-z]+")

def extract_tags(text):
    return sorted(set(_TAG_RE.findall(text or "")))

def _ts(s):
    try: return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception: return None

def scan_transcript(path):
    inp = cc = cr = out = think = 0; ts = []; agent_id = None; final = ""
    with open(path, errors="ignore") as f:
        raw = f.read()
    for line in raw.splitlines():
        try: o = json.loads(line)
        except Exception: continue
        agent_id = o.get("agentId", agent_id)
        if o.get("timestamp"):
            t = _ts(o["timestamp"])
            if t: ts.append(t)
        m = o.get("message") if isinstance(o.get("message"), dict) else {}
        u = m.get("usage") or {}
        inp += u.get("input_tokens", 0); out += u.get("output_tokens", 0)
        cc += u.get("cache_creation_input_tokens", 0); cr += u.get("cache_read_input_tokens", 0)
        c = m.get("content")
        if isinstance(c, list):
            for b in c:
                if isinstance(b, dict):
                    if b.get("type") == "thinking": think += len(b.get("thinking", ""))  # char-length proxy; ==0 reliably means no extended reasoning occurred
                    if b.get("type") == "text" and m.get("role") == "assistant": final = b["text"]
    dur = (max(ts) - min(ts)).total_seconds() if len([x for x in ts if x]) >= 2 else None
    tags = extract_tags(raw)
    return {"agent_id": agent_id, "input_tokens": inp, "cache_creation_tokens": cc,
            "cache_read_tokens": cr, "output_tokens": out, "total_input_tokens": inp + cc + cr,
            "output_thinking_tokens": think, "wall_clock_s": dur,
            "score": parse_score(final), "raw_output": final, "tags": tags}

def collect(conn, pass_no, manifest, projects_dir=None, session_id="unknown", full=False):
    projects_dir = projects_dir or config.claude_projects_dir()
    # "**" after subagents so we also match agents nested under workflows/<run>/ (Workflow path),
    # not just direct subagents/agent-*.jsonl (manual Agent-tool path). "**" matches zero+ dirs.
    if session_id and session_id != "unknown":
        pattern = os.path.join(str(projects_dir), "**", session_id, "subagents", "**", "agent-*.jsonl")
    else:
        pattern = os.path.join(str(projects_dir), "**", "subagents", "**", "agent-*.jsonl")
    files = glob.glob(pattern, recursive=True)
    seen = {} if full else db.ledger_seen(conn)
    index = {}  # tag -> scan result; prefer a transcript that parsed to a score
    for f in files:
        try:
            st = os.stat(f)
        except OSError:
            continue
        if seen.get(f) == (st.st_size, st.st_mtime):
            continue  # already ingested, unchanged -> skip the read entirely
        scan = scan_transcript(f)
        for tag in scan["tags"]:
            cur = index.get(tag)
            if cur is None or (cur["score"] is None and scan["score"] is not None):
                index[tag] = scan
        db.mark_ingested(conn, f, st.st_size, st.st_mtime)
    done = db.completed_cells(conn, pass_no)  # (item_id, dim_code) already parse_ok=1
    written = 0
    for tag, meta in manifest.items():
        info = index.get(tag)
        if info is None:
            continue  # tag not in any newly-ingested transcript this run
        if info["score"] is None and (meta["item_id"], meta["dim_code"]) in done:
            continue  # never regress: a late failed retry must not clobber a good score
        row = {"pass": pass_no, "session_id": session_id, "item_id": meta["item_id"],
               "dim_code": meta["dim_code"], "dim_name": meta["dim_name"],
               "agent_id": info["agent_id"], "raw_output": info["raw_output"],
               "score": info["score"], "parse_ok": 1 if info["score"] is not None else 0,
               "input_tokens": info["input_tokens"], "cache_creation_tokens": info["cache_creation_tokens"],
               "cache_read_tokens": info["cache_read_tokens"], "output_tokens": info["output_tokens"],
               "total_input_tokens": info["total_input_tokens"],
               "output_thinking_tokens": info["output_thinking_tokens"],
               "wall_clock_s": info["wall_clock_s"]}
        db.upsert_annotation(conn, row); written += 1
    return written
