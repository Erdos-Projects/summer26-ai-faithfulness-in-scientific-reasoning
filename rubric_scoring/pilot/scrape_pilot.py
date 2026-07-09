"""Scrape per-agent token usage + timing from subagent transcripts, parse scores,
write annotation rows to the SQLite store, and print a summary + full-run projection.

Maps transcript -> (item, dim) by the unique PILOT_TAG embedded in each prompt.
"""
import json, os, re, glob, sqlite3, sys
from datetime import datetime

RUN_ID = sys.argv[1] if len(sys.argv) > 1 else "pilot_0001"
TD = "/home/awndre/.claude/projects/-home-awndre-projects-Erdos-Annotations"
RES = "/home/awndre/projects/Erdos/Annotations/results"
manifest = json.load(open(os.path.join(RES, f"{RUN_ID}_manifest.json")))
db = sqlite3.connect(os.path.join(RES, "annotations.db"))

def parse_ts(s):
    try: return datetime.fromisoformat(s.replace("Z","+00:00"))
    except: return None

def parse_score(text):
    if not text: return None
    m = re.findall(r"is:\s*\*?\*?\s*([0-5])\b", text)
    if m: return int(m[-1])
    m = re.findall(r"level .*? is:?\s*([0-5])\b", text, re.I)
    return int(m[-1]) if m else None

# index transcripts by the pilot tag they contain
tag_to_file = {}
for fp in glob.glob(os.path.join(TD, "**", "agent-*.jsonl"), recursive=True):
    txt = open(fp, errors="ignore").read()
    for tag in manifest:
        if tag in txt:
            tag_to_file[tag] = fp

db.execute("DELETE FROM annotation WHERE run_id=?",(RUN_ID,))  # idempotent re-scrape
rows = []
for tag, fp in tag_to_file.items():
    info = manifest[tag]
    inp=cc=cr=out=0; ts=[]; agent_id=None; final_text=""
    for line in open(fp, errors="ignore"):
        try: o=json.loads(line)
        except: continue
        agent_id = o.get("agentId", agent_id)
        if o.get("timestamp"):
            t=parse_ts(o["timestamp"])
            if t: ts.append(t)
        msg=o.get("message",{})
        u=msg.get("usage") if isinstance(msg,dict) else None
        if u:
            inp+=u.get("input_tokens",0); out+=u.get("output_tokens",0)
            cc+=u.get("cache_creation_input_tokens",0); cr+=u.get("cache_read_input_tokens",0)
        # capture last assistant text for score parse
        if isinstance(msg,dict) and msg.get("role")=="assistant":
            c=msg.get("content")
            if isinstance(c,list):
                for blk in c:
                    if isinstance(blk,dict) and blk.get("type")=="text":
                        final_text=blk["text"]
            elif isinstance(c,str):
                final_text=c
    total_in = inp+cc+cr
    dur = (max(ts)-min(ts)).total_seconds() if len(ts)>=2 else None
    score = parse_score(final_text)
    rows.append(dict(tag=tag, item=info["item_id"], dim=info["dim_code"], agent_id=agent_id,
                     inp=inp, cc=cc, cr=cr, out=out, total_in=total_in, dur=dur,
                     score=score, parse_ok=score is not None, raw=final_text))
    db.execute("""INSERT INTO annotation(run_id,item_id,dim_code,dim_name,prompt_tag,agent_id,
        raw_output,score,parse_ok,input_tokens,cache_creation_tokens,cache_read_tokens,
        output_tokens,total_input_tokens,wall_clock_s,created)
        VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
        (RUN_ID, info["item_id"], info["dim_code"], info["dim_name"], tag, agent_id,
         final_text, score, 1 if score is not None else 0, inp, cc, cr, out, total_in, dur,
         "2026-06-13"))
db.commit()

rows.sort(key=lambda r:(r["item"],r["dim"]))
print(f"{'tag':<34}{'score':>6}{'in':>8}{'cache_cr':>10}{'cache_rd':>10}{'out':>7}{'TOTAL':>9}{'sec':>7}")
for r in rows:
    print(f"{r['dim']+'@'+r['item']:<34}{str(r['score']):>6}{r['inp']:>8}{r['cc']:>10}{r['cr']:>10}{r['out']:>7}{r['total_in']+r['out']:>9}{r['dur'] or 0:>7.1f}")

n=len(rows)
avg_tot = sum(r["total_in"]+r["out"] for r in rows)/n     # total processed incl cache re-reads
avg_un  = sum(r["inp"] for r in rows)/n                    # uncached input
avg_cc  = sum(r["cc"] for r in rows)/n                     # cache writes (overhead+rubric, once/agent)
avg_cr  = sum(r["cr"] for r in rows)/n                     # cache reads (re-read each turn)
avg_out = sum(r["out"] for r in rows)/n
avg_new = avg_un+avg_cc+avg_out                            # "new" tokens (no re-read double count)
avg_dur = sum((r["dur"] or 0) for r in rows)/n
parse_rate = sum(r["parse_ok"] for r in rows)/n
print(f"\nN={n}  parse_rate={parse_rate:.0%}")
print(f"avg/agent: total_processed={avg_tot:,.0f}  new_tokens={avg_new:,.0f}  "
      f"(cache_write={avg_cc:,.0f} cache_read={avg_cr:,.0f} out={avg_out:,.0f})  dur={avg_dur:.1f}s")

# Sonnet 4.x API-equivalent rates ($/Mtok): input 3.00, cache-write 3.75, cache-read 0.30, output 15.00
cost_per = avg_un*3.00/1e6 + avg_cc*3.75/1e6 + avg_cr*0.30/1e6 + avg_out*15.00/1e6
print(f"API-equivalent per agent ~${cost_per:.4f}  (Max-plan actual $ = 0)")
print("\n--- full-run projection (6 rubrics x N items) ---")
for N in (654, 750, 1500):
    calls=N*6
    secs_seq = calls*avg_dur
    print(f"N={N:>4} items -> {calls:,} agents | ~{calls*avg_tot/1e6:,.0f}M tok processed "
          f"({calls*avg_new/1e6:,.0f}M new) | API-equiv ~${calls*cost_per:,.0f} | "
          f"seq ~{secs_seq/3600:,.0f}h | @8-conc ~{secs_seq/8/3600:,.1f}h | @16-conc ~{secs_seq/16/3600:,.1f}h")
