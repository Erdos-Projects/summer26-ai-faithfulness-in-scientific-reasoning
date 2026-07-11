"""Recover per-cell scores + exact token usage from subagent transcripts.
Transcripts: <projects>/**/subagents/agent-*.jsonl ; matched to cells by the tag in their text."""
import glob, json, os
from datetime import datetime
from sciclaimeval_scoring import db, config
from sciclaimeval_scoring.parse import parse_score

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
    return {"agent_id": agent_id, "input_tokens": inp, "cache_creation_tokens": cc,
            "cache_read_tokens": cr, "output_tokens": out, "total_input_tokens": inp + cc + cr,
            "output_thinking_tokens": think, "wall_clock_s": dur,
            "score": parse_score(final), "raw_output": final}

def collect(conn, pass_no, manifest, projects_dir=None, session_id="unknown"):
    projects_dir = projects_dir or config.claude_projects_dir()
    # "**" after subagents so we also match agents nested under workflows/<run>/ (Workflow path),
    # not just direct subagents/agent-*.jsonl (manual Agent-tool path). "**" matches zero+ dirs.
    if session_id and session_id != "unknown":
        pattern = os.path.join(str(projects_dir), "**", session_id, "subagents", "**", "agent-*.jsonl")
    else:
        pattern = os.path.join(str(projects_dir), "**", "subagents", "**", "agent-*.jsonl")
    files = glob.glob(pattern, recursive=True)
    contents = {}
    for f in files:
        with open(f, errors="ignore") as fh:
            contents[f] = fh.read()
    written = 0
    for tag, meta in manifest.items():
        # A tag can appear in MORE than one transcript: a failed attempt (e.g. rate-limited —
        # the tag is in the prompt even though no score was produced) plus a later successful
        # retry. Pick a transcript that actually parses to a score; fall back to any match so a
        # genuinely-unscored cell is still recorded as parse_ok=0 rather than silently dropped.
        matches = [f for f, txt in contents.items() if tag in txt]
        if not matches: continue
        scanned = [scan_transcript(f) for f in matches]
        info = next((s for s in scanned if s["score"] is not None), scanned[0])
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
