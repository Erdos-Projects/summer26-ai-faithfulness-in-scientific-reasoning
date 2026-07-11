"""Recover per-item verdicts + exact token usage from this session's subagent
transcripts, matched to items by the TASK_TAG embedded in each prompt.

Mirrors rubric_scoring.scrape but is verdict-AWARE: it parses 'Answer: yes/no',
maps it to predicted (1=entailed), and scores `correct` against the item's label
(label comes from the manifest, never from the prompt the subagent saw).
"""
import glob
import json
import os
from datetime import datetime

from sciclaimeval_scoring import config  # reuse projects-dir + path resolution
from sciclaimeval_eval import db
from sciclaimeval_eval.parse import parse_answer


def _ts(s):
    try:
        return datetime.fromisoformat(s.replace("Z", "+00:00"))
    except Exception:
        return None


def scan_transcript(path):
    inp = cc = cr = out = think = 0
    ts = []
    agent_id = None
    final = ""
    with open(path, errors="ignore") as f:
        raw = f.read()
    for line in raw.splitlines():
        try:
            o = json.loads(line)
        except Exception:
            continue
        agent_id = o.get("agentId", agent_id)
        if o.get("timestamp"):
            t = _ts(o["timestamp"])
            if t:
                ts.append(t)
        m = o.get("message") if isinstance(o.get("message"), dict) else {}
        u = m.get("usage") or {}
        inp += u.get("input_tokens", 0)
        out += u.get("output_tokens", 0)
        cc += u.get("cache_creation_input_tokens", 0)
        cr += u.get("cache_read_input_tokens", 0)
        c = m.get("content")
        if isinstance(c, list):
            for b in c:
                if isinstance(b, dict):
                    if b.get("type") == "thinking":
                        think += len(b.get("thinking", ""))  # ==0 reliably means no extended reasoning
                    if b.get("type") == "text" and m.get("role") == "assistant":
                        final = b["text"]
    dur = (max(ts) - min(ts)).total_seconds() if len([x for x in ts if x]) >= 2 else None
    return {"agent_id": agent_id, "input_tokens": inp, "cache_creation_tokens": cc,
            "cache_read_tokens": cr, "output_tokens": out, "total_input_tokens": inp + cc + cr,
            "output_thinking_tokens": think, "wall_clock_s": dur, "raw_output": final}


def collect(conn, run_no, model, manifest, projects_dir=None, session_id="unknown"):
    projects_dir = projects_dir or config.claude_projects_dir()
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
        # A tag can appear in a failed attempt + a successful retry; prefer one that
        # actually parsed a verdict, else record the best available as parse_ok=0.
        matches = [f for f, txt in contents.items() if tag in txt]
        if not matches:
            continue
        scanned = [scan_transcript(f) for f in matches]
        info = next((s for s in scanned if parse_answer(s["raw_output"]) is not None), scanned[0])
        ans = parse_answer(info["raw_output"])
        if ans is not None:
            predicted, parse_ok = (1 if ans == "yes" else 0), 1
        else:  # fallback to the paper's substring rule, flagged for QA
            predicted, parse_ok = (1 if "yes" in (info["raw_output"] or "").lower() else 0), 0
        label = int(meta["label"])
        row = {"run": run_no, "session_id": session_id, "item_id": meta["item_id"],
               "model": model, "trial": int(meta["trial"]),
               "agent_id": info["agent_id"], "raw_output": info["raw_output"],
               "answer": ans, "predicted": predicted, "parse_ok": parse_ok,
               "correct": 1 if predicted == label else 0,
               "input_tokens": info["input_tokens"], "cache_creation_tokens": info["cache_creation_tokens"],
               "cache_read_tokens": info["cache_read_tokens"], "output_tokens": info["output_tokens"],
               "total_input_tokens": info["total_input_tokens"],
               "output_thinking_tokens": info["output_thinking_tokens"],
               "wall_clock_s": info["wall_clock_s"]}
        db.upsert_prediction(conn, row)
        written += 1
    return written
