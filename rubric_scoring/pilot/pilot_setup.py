"""Pilot scaffold for the DeLeAn rubric-scoring pass (credit-free subagent path).

Extracts the 6 chosen rubrics verbatim from the supplementary PDF, loads 3 SciVer
chart items, creates the SQLite system-of-record, and emits one tagged prompt file
per (item x rubric) = 18 isolated scoring tasks. Verdict-blind: only claim + figure
go into prompts; label/origin/perturbed/explanation stay in the DB, never the prompt.
"""
import json, os, re, sqlite3, hashlib, sys
from pypdf import PdfReader

ROOT = "/home/awndre/projects/Erdos"
ANN  = os.path.join(ROOT, "Annotations")
RES  = os.path.join(ANN, "results")
RUB  = os.path.join(ANN, "rubrics")
PROMPTS = os.path.join(RES, "pilot_prompts")
for d in (RES, RUB, PROMPTS): os.makedirs(d, exist_ok=True)

RUN_ID = sys.argv[1] if len(sys.argv) > 1 else "pilot_0001"
SKILL_VERSION = "pilot-0.1"
MODEL = "claude-sonnet-4-6"
EFFORT = "standard"

# code -> (display name, pdf page 1-based, trim-start marker)
DIMS = {
    "AS":  ("Attention and Scan",                 78, "Attention and Scan (AS)"),
    "MCr": ("Identifying Relevant Information",    84, "R3. Identifying Relevant Information (MCr)"),
    "QLq": ("Quantitative Reasoning",             86, "R1. Quantitative Reasoning (QLq)"),
    "QLl": ("Logical Reasoning",                  87, "R2. Logical Reasoning (QLl)"),
    "VO":  ("Volume",                             96, "Volume (VO)"),
    "AT":  ("Atypicality",                        97, "Atypicality (AT)"),
}

# ---- 1. extract + save rubrics (byte-faithful from PDF; strip trailing page-number footer) ----
reader = PdfReader(os.path.join(ANN, "Notes/ADeLesupplementary.pdf"))
rubrics = {}
for code, (name, page, marker) in DIMS.items():
    txt = reader.pages[page-1].extract_text() or ""
    i = txt.find(marker)
    if i > 0: txt = txt[i:]
    txt = re.sub(r"\n\s*\d{1,3}\s*$", "", txt.rstrip())  # drop trailing page number
    rubrics[code] = txt.strip()
    with open(os.path.join(RUB, f"{code}.txt"), "w") as f: f.write(rubrics[code])
    print(f"rubric {code}: {len(rubrics[code])} chars (~{len(rubrics[code])//4} tok)")

# ---- 2. load 3 SciVer chart items (deterministic: first 3 chart-type in valset) ----
val = json.load(open(os.path.join(ROOT, "SciVer/valset.json")))
items = []
for idx, x in enumerate(val):
    if x.get("type") == "chart":
        items.append((idx, x))
    if len(items) == 3: break

# ---- 3. SQLite system-of-record ----
db = sqlite3.connect(os.path.join(RES, "annotations.db"))
db.executescript("""
CREATE TABLE IF NOT EXISTS item(
  item_id TEXT PRIMARY KEY, source TEXT, paperid TEXT, claim_type TEXT, vtype TEXT,
  image_path TEXT, claim TEXT, label INTEGER);
CREATE TABLE IF NOT EXISTS run(
  run_id TEXT PRIMARY KEY, created TEXT, skill_version TEXT, model TEXT, effort TEXT,
  temperature_note TEXT, max_output_instructed INTEGER, rubric_set TEXT,
  item_source TEXT, n_items INTEGER, deviations TEXT);
CREATE TABLE IF NOT EXISTS annotation(
  id INTEGER PRIMARY KEY AUTOINCREMENT, run_id TEXT, item_id TEXT, dim_code TEXT,
  dim_name TEXT, prompt_tag TEXT, agent_id TEXT, raw_output TEXT, score INTEGER,
  parse_ok INTEGER, input_tokens INTEGER, cache_creation_tokens INTEGER,
  cache_read_tokens INTEGER, output_tokens INTEGER, total_input_tokens INTEGER,
  wall_clock_s REAL, started TEXT, ended TEXT, created TEXT);
""")
db.execute("INSERT OR REPLACE INTO run VALUES(?,?,?,?,?,?,?,?,?,?,?)", (
    RUN_ID, "2026-06-13", SKILL_VERSION, MODEL, EFFORT,
    "harness-fixed (temperature 0 NOT controllable on subagent path)", 1000,
    ",".join(DIMS), "SciVer/valset.json chart items [0:3]", len(items),
    "temp!=0; output cap instructed not enforced; rubric anchors are text-task (DeLeAn v1.0 textual-only); no caption"))

ITEMS_META = {}
for idx, x in items:
    item_id = f"sciver_val_{idx}"
    ITEMS_META[item_id] = {"image_path": x["image_path"].replace("./",""), "claim": x["claim"]}
    db.execute("INSERT OR REPLACE INTO item VALUES(?,?,?,?,?,?,?,?)", (
        item_id, "sciver", x["paperid"], x["claim_type"], x["type"],
        x["image_path"].replace("./",""), x["claim"], 1 if x["label"] else 0))
db.commit()

# ---- 4. build 18 tagged prompt files (verdict-blind) ----
TEMPLATE = '''ROLE: You are a demand-level annotator. Your only job is to score ONE rubric on ONE
task instance and then STOP. Do not verify the claim. Do not determine if it is true.
Use the Read tool to view the figure image referenced below, reason briefly, then output
the required conclusion line and finish immediately. Use no tools other than Read.

PILOT_TAG: {tag}

QUERY:
The following rubric describes six distinct levels of *{name}* required by different tasks:

{rubric}

TASK INSTANCE:
This is a scientific claim-verification task: the task is to determine whether the claim
below is supported or refuted by the figure. (You only assess the rubric demand of this
task; you do NOT perform the verification.)

Claim: "{claim}"

Figure: a chart image for this task is provided. View it with the Read tool before scoring.
Figure path: {fig}

INSTRUCTION:
Score the level of *{name}* demanded by the given TASK INSTANCE using a discrete value
from 0 to 5. Use CHAIN-OF-THOUGHTS REASONING to reason step by step before assigning the
score. After the CHAIN-OF-THOUGHTS REASONING STEPS, conclude your assessment with the
statement:

"Thus, the level of *{name}* demanded by the given TASK INSTANCE is: SCORE",

where 'SCORE' is the integer score you have determined. Then STOP.
'''

manifest = {}
for item_id, meta in ITEMS_META.items():
    figabs = os.path.join(ROOT, meta["image_path"])
    for code,(name,_,_) in DIMS.items():
        tag = f"{RUN_ID}__{item_id}__{code}"
        prompt = TEMPLATE.format(tag=tag, name=name, rubric=rubrics[code],
                                 claim=meta["claim"], fig=figabs)
        path = os.path.join(PROMPTS, f"{tag}.txt")
        with open(path,"w") as f: f.write(prompt)
        manifest[tag] = {"item_id": item_id, "dim_code": code, "dim_name": name,
                         "prompt_file": path, "prompt_tokens_est": len(prompt)//4}
json.dump(manifest, open(os.path.join(RES,f"{RUN_ID}_manifest.json"),"w"), indent=2)

print(f"\nitems: {list(ITEMS_META)}")
print(f"prompts written: {len(manifest)}  -> {PROMPTS}")
print(f"db: {os.path.join(RES,'annotations.db')}")
print(f"sample prompt tokens est: {[manifest[t]['prompt_tokens_est'] for t in list(manifest)[:6]]}")
