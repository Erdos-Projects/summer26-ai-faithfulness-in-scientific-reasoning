# SciClaimEval Rubric-Scoring + Prediction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Port the SciVer demand-scoring (`rubric_scoring`) and prediction (`sciver_eval`) pipelines to the SciClaimEval dev figure items as two fully isolated sibling modules with their own DBs.

**Architecture:** Full standalone copies. `sciclaimeval_scoring/` clones `rubric_scoring/` (DeLeAn demand annotation, one isolated Sonnet subagent per item×dimension, verdict-blind, resumable). `sciclaimeval_eval/` clones `sciver_eval/` (Haiku Supported/Refuted prediction over trials) and imports `config`/`items` from `sciclaimeval_scoring` (mirroring how `sciver_eval` imports `rubric_scoring`). The only real rewrite is the item loader; everything else is a copy with package-name and a few string edits. Both DBs join on `item_id`.

**Tech Stack:** Python 3, SQLite (stdlib `sqlite3`), pytest. Credit-free Claude Code subagent path (Agent tool / Workflows) for the actual annotation/prediction runs.

## Global Constraints

- **Verdict-blind (scoring):** an annotator prompt receives ONLY `claim` + figure path. `label`, `operation`, `caption`, `context`, `claim_id_pair` are stored in the DB and NEVER interpolated into a prompt. (`operation` leaks the verdict: `Supported_claim_only` ⇒ Supported.)
- **Verdict-clean (prediction):** a prediction prompt may show `claim` + `caption` + `context` + figure (the real task), but NEVER `label` or `operation`.
- **Isolation from SciVer:** no imports from `rubric_scoring` or `sciver_eval`; no writes to `annotations_prod.db` or `predictions.db`. Intra-SciClaimEval imports (`sciclaimeval_eval` → `sciclaimeval_scoring`) are allowed.
- **Item universe:** exactly the **265** `evi_type=="figure"` items of `SciClaimEval/sciclaimeval-shared-task/data/dev_task1_release.json`. `item_id = "scev_" + claim_id`.
- **Tag namespaces (no collision with the SciVer DBs):** scoring `rsc_p{pass:02d}__{item_id}__{code}`; prediction `sce_r{run:02d}__{item_id}__t{trial:02d}`.
- **Run location:** all commands from the repo root `summer26-ai-science-reasoning/` with the venv active (`source .venv/bin/activate`). Tests: `python -m pytest`.
- **Defaults:** scoring dims `AS,QLq,QLl,MCr,AT,VO`; prediction model `claude-haiku-4-5`.
- **No-guessed-numbers:** the counts asserted in tests (265 items; 149 Supported / 116 Refuted) are verified against the dataset; if a future dataset revision changes them, reconcile the test and the spec together.

---

# Part A — `sciclaimeval_scoring` (demand rubrics)

### Task 1: Scaffold module, config, rubric library

**Files:**
- Create: `sciclaimeval_scoring/__init__.py` (empty)
- Create: `sciclaimeval_scoring/config.py`
- Create: `sciclaimeval_scoring/extract_rubrics.py` (copy + repoint)
- Create: `sciclaimeval_scoring/rubrics/*.txt` (copy of all 21)
- Test: `sciclaimeval_scoring/tests/__init__.py` (empty), `sciclaimeval_scoring/tests/test_config.py`

**Interfaces:**
- Produces: `config.repo_root()`, `config.sciclaimeval_dir()`, `config.rubrics_dir()`, `config.db_path()`, `config.claude_projects_dir()` → all `pathlib.Path`. `extract_rubrics.ALL_DIMS` (dict code→(name,page,marker)), `extract_rubrics.EMERGENT_DIMS` (dict code→name).

- [ ] **Step 1: Create the folders and copy the rubric assets + extract_rubrics**

```bash
mkdir -p sciclaimeval_scoring/tests sciclaimeval_scoring/rubrics sciclaimeval_scoring/prompts
touch sciclaimeval_scoring/__init__.py sciclaimeval_scoring/tests/__init__.py
cp rubric_scoring/rubrics/*.txt sciclaimeval_scoring/rubrics/
cp rubric_scoring/extract_rubrics.py sciclaimeval_scoring/extract_rubrics.py
sed -i 's/rubric_scoring/sciclaimeval_scoring/g' sciclaimeval_scoring/extract_rubrics.py
ls sciclaimeval_scoring/rubrics/*.txt | wc -l   # expect 21
```

- [ ] **Step 2: Write `sciclaimeval_scoring/config.py`**

```python
"""Path resolution for the SciClaimEval rubric-scoring module.
Everything is relative or env-overridable — no hardcoded absolutes."""
import os
from pathlib import Path

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def sciclaimeval_dir() -> Path:
    """SciClaimEval data root holding dev_task1_release.json + figures/."""
    env = os.environ.get("SCICLAIMEVAL_DIR")
    candidates = [Path(env)] if env else []
    candidates += [
        repo_root() / "data" / "raw" / "SciClaimEval" / "sciclaimeval-shared-task" / "data",
        repo_root().parent / "SciClaimEval" / "sciclaimeval-shared-task" / "data",
    ]
    for c in candidates:
        if (c / "dev_task1_release.json").is_file():
            return c
    return candidates[-1]  # best guess; CLIs/tests surface a clear error if missing

def rubrics_dir() -> Path:
    return Path(__file__).resolve().parent / "rubrics"

def db_path() -> Path:
    env = os.environ.get("SCICLAIMEVAL_SCORING_DB")
    return Path(env) if env else repo_root() / "sciclaimeval_scoring" / "annotations_sciclaimeval.db"

def claude_projects_dir() -> Path:
    env = os.environ.get("RUBRIC_SCORING_PROJECTS")
    return Path(env) if env else Path.home() / ".claude" / "projects"
```

- [ ] **Step 3: Write the failing test `sciclaimeval_scoring/tests/test_config.py`**

```python
from sciclaimeval_scoring import config

def test_data_root_resolves_and_has_release_file():
    sd = config.sciclaimeval_dir()
    assert (sd / "dev_task1_release.json").is_file(), sd

def test_rubric_library_has_21_files():
    txts = list(config.rubrics_dir().glob("*.txt"))
    assert len(txts) == 21, [p.name for p in txts]

def test_db_path_is_module_local_and_sciclaimeval_named():
    p = config.db_path()
    assert p.name == "annotations_sciclaimeval.db"
    assert "sciclaimeval_scoring" in str(p)
```

- [ ] **Step 4: Run the test**

Run: `python -m pytest sciclaimeval_scoring/tests/test_config.py -v`
Expected: PASS (3 passed). If `test_data_root_resolves` fails, the SciClaimEval data is not at `../SciClaimEval/...`; set `SCICLAIMEVAL_DIR`.

- [ ] **Step 5: Commit**

```bash
git add sciclaimeval_scoring/__init__.py sciclaimeval_scoring/config.py \
        sciclaimeval_scoring/extract_rubrics.py sciclaimeval_scoring/rubrics \
        sciclaimeval_scoring/tests/__init__.py sciclaimeval_scoring/tests/test_config.py
git commit -m "feat(sciclaimeval_scoring): scaffold module, config, rubric library"
```

---

### Task 2: Item loader (the one real rewrite)

**Files:**
- Create: `sciclaimeval_scoring/items.py`
- Test: `sciclaimeval_scoring/tests/test_items.py`

**Interfaces:**
- Consumes: `config.sciclaimeval_dir()`.
- Produces: `items.Item` (frozen dataclass: `item_id, source, paperid, claim_type, vtype, image_path, claim, label, domain, operation`); `items.load_items() -> list[Item]`.

- [ ] **Step 1: Write the failing test `sciclaimeval_scoring/tests/test_items.py`**

```python
import os
from sciclaimeval_scoring.items import load_items, Item

def test_loads_265_figures():
    items = load_items()
    assert len(items) == 265
    assert all(isinstance(i, Item) for i in items)

def test_fields_and_id_scheme():
    i = load_items()[0]
    assert i.vtype == "figure"
    assert i.source == "sciclaimeval"
    assert i.claim and i.image_path.endswith(".png")
    assert i.item_id.startswith("scev_")
    assert i.domain in {"ml", "nlp", "peerj"}

def test_label_mapping_counts():
    items = load_items()
    supported = sum(1 for i in items if i.label is True)
    refuted = sum(1 for i in items if i.label is False)
    assert (supported, refuted) == (149, 116)

def test_item_ids_unique():
    ids = [i.item_id for i in load_items()]
    assert len(ids) == len(set(ids))

def test_image_files_exist_for_sample():
    for i in load_items()[:25]:
        assert os.path.isfile(i.image_path), i.image_path
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest sciclaimeval_scoring/tests/test_items.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sciclaimeval_scoring.items'`.

- [ ] **Step 3: Write `sciclaimeval_scoring/items.py`**

```python
"""Canonical item set: every SciClaimEval dev FIGURE claim. Verified == 265.
Built directly from the released dataset (not feature_discovery) to stay self-contained.
`operation` and `label` are stored but verdict-blind: never sent to an annotator prompt."""
import json
from dataclasses import dataclass
from pathlib import Path
from sciclaimeval_scoring import config

@dataclass(frozen=True)
class Item:
    item_id: str
    source: str
    paperid: str
    claim_type: str
    vtype: str
    image_path: str   # absolute
    claim: str
    label: bool
    domain: str
    operation: str

def load_items():
    sd = config.sciclaimeval_dir()
    data = json.loads((sd / "dev_task1_release.json").read_text())
    items = []
    for x in data:
        if x.get("evi_type") != "figure":
            continue
        items.append(Item(
            item_id=f"scev_{x['claim_id']}",
            source="sciclaimeval",
            paperid=x.get("paper_id", ""),
            claim_type="",
            vtype="figure",
            image_path=str(sd / x["evi_path"]),
            claim=x["claim"],
            label=(x["label"] == "Supported"),
            domain=x.get("domain", ""),
            operation=x.get("operation", ""),
        ))
    return items
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest sciclaimeval_scoring/tests/test_items.py -v`
Expected: PASS (5 passed).

- [ ] **Step 5: Commit**

```bash
git add sciclaimeval_scoring/items.py sciclaimeval_scoring/tests/test_items.py
git commit -m "feat(sciclaimeval_scoring): item loader for 265 dev figures"
```

---

### Task 3: SQLite layer (schema + upserts with domain/operation)

**Files:**
- Create: `sciclaimeval_scoring/db.py`
- Test: `sciclaimeval_scoring/tests/test_db.py`

**Interfaces:**
- Consumes: `items.Item`.
- Produces: `db.connect(path)`, `db.init(conn)`, `db.register_pass(conn, pass_no, *, skill_version, model, active_dims, item_source, n_items, params, deviations, created=None)`, `db.upsert_item(conn, it)`, `db.upsert_annotation(conn, row: dict)`, `db.completed_cells(conn, pass_no) -> set[(item_id, dim_code)]`.

- [ ] **Step 1: Write the failing test `sciclaimeval_scoring/tests/test_db.py`**

```python
from sciclaimeval_scoring import db
from sciclaimeval_scoring.items import Item

def _item(i="scev_val_fig_0001"):
    return Item(i, "sciclaimeval", "2403.19137", "", "figure",
                "/x/figures/dev/val_fig_0001.png", "a claim", False, "ml", "Legend Swap")

def test_item_roundtrip_includes_domain_and_operation():
    conn = db.connect(":memory:"); db.init(conn)
    db.upsert_item(conn, _item())
    row = conn.execute("SELECT domain, operation, label FROM item WHERE item_id=?",
                       ("scev_val_fig_0001",)).fetchone()
    assert row == ("ml", "Legend Swap", 0)

def test_completed_cells_tracks_parse_ok():
    conn = db.connect(":memory:"); db.init(conn)
    db.register_pass(conn, 1, skill_version="0.1", model="m", active_dims="AS",
                     item_source="t", n_items=1, params="{}", deviations="")
    db.upsert_annotation(conn, {"pass": 1, "item_id": "scev_val_fig_0001",
                                "dim_code": "AS", "dim_name": "Attention and Scan",
                                "score": 3, "parse_ok": 1})
    assert ("scev_val_fig_0001", "AS") in db.completed_cells(conn, 1)

def test_upsert_annotation_is_idempotent():
    conn = db.connect(":memory:"); db.init(conn)
    for s in (2, 4):
        db.upsert_annotation(conn, {"pass": 1, "item_id": "i", "dim_code": "AS",
                                    "dim_name": "n", "score": s, "parse_ok": 1})
    n = conn.execute("SELECT COUNT(*) FROM annotation").fetchone()[0]
    last = conn.execute("SELECT score FROM annotation").fetchone()[0]
    assert (n, last) == (1, 4)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest sciclaimeval_scoring/tests/test_db.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sciclaimeval_scoring.db'`.

- [ ] **Step 3: Write `sciclaimeval_scoring/db.py`**

```python
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
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest sciclaimeval_scoring/tests/test_db.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add sciclaimeval_scoring/db.py sciclaimeval_scoring/tests/test_db.py
git commit -m "feat(sciclaimeval_scoring): sqlite schema with domain/operation columns"
```

---

### Task 4: Prompt builder (tag namespace, figure wording, verdict-blind) + score parser

**Files:**
- Create: `sciclaimeval_scoring/prompts.py`
- Create: `sciclaimeval_scoring/parse.py` (copy)
- Test: `sciclaimeval_scoring/tests/test_prompts.py`, `sciclaimeval_scoring/tests/test_parse.py`

**Interfaces:**
- Consumes: `extract_rubrics.ALL_DIMS`/`EMERGENT_DIMS`, `items.Item`.
- Produces: `prompts.DIMS` (dict code→name), `prompts.tag(pass_no, item_id, code) -> str`, `prompts.build_prompt(pass_no, item, code, rubric_text) -> str`; `parse.parse_score(text) -> int|None`.

- [ ] **Step 1: Copy the parser (byte-identical, no imports to repoint)**

```bash
cp rubric_scoring/parse.py sciclaimeval_scoring/parse.py
```

- [ ] **Step 2: Write `sciclaimeval_scoring/tests/test_parse.py`**

```python
from sciclaimeval_scoring.parse import parse_score

def test_parses_final_score_line():
    assert parse_score("reasoning... is: 4") == 4
    assert parse_score("the level ... is: **2**") == 2

def test_no_score_returns_none():
    assert parse_score("no number here") is None
    assert parse_score("") is None
```

- [ ] **Step 3: Write `sciclaimeval_scoring/prompts.py`** (tag prefix `rsc_p`; "chart image" → "an image")

```python
"""Filled DeLeAn prompts, verdict-blind. Rubric text is passed in verbatim.
build_prompt interpolates ONLY the claim and figure path from the item (never label/operation),
which is what makes it verdict-blind."""
from sciclaimeval_scoring.extract_rubrics import ALL_DIMS, EMERGENT_DIMS
DIMS = {c: v[0] for c, v in ALL_DIMS.items()}
DIMS.update(EMERGENT_DIMS)  # 18 DeLeAn + 3 authored emergent figure-grounding dims

def tag(pass_no, item_id, code):
    return f"rsc_p{pass_no:02d}__{item_id}__{code}"

_TEMPLATE = """ROLE: You are a demand-level annotator. Score ONE rubric on ONE task instance, then STOP.
Do not decide the claim's correctness. View the figure with the Read tool. Use no tool other than Read.

PILOT_TAG: {tag}

QUERY:
The following rubric describes six distinct levels of *{name}* required by different tasks:

{rubric}

TASK INSTANCE:
This is a scientific claim-verification task: the figure is the evidence for checking the claim below.
You ONLY assess the rubric demand of this task; you do NOT decide whether the claim is correct.

Claim: "{claim}"

Figure: an image for this task is provided. View it with the Read tool before scoring.
Figure path: {fig}

INSTRUCTION:
Score the level of *{name}* demanded by the given TASK INSTANCE using a discrete value from 0 to 5. Use
CHAIN-OF-THOUGHTS REASONING to reason step by step, then conclude with EXACTLY:

"Thus, the level of *{name}* demanded by the given TASK INSTANCE is: SCORE", where SCORE is your integer. Then STOP.
"""

def build_prompt(pass_no, item, code, rubric_text):
    return _TEMPLATE.format(tag=tag(pass_no, item.item_id, code), name=DIMS[code],
                            rubric=rubric_text, claim=item.claim, fig=item.image_path)
```

- [ ] **Step 4: Write `sciclaimeval_scoring/tests/test_prompts.py`** (tag namespace + verdict-blindness)

```python
from sciclaimeval_scoring.prompts import tag, build_prompt, DIMS
from sciclaimeval_scoring.items import Item

def _item():
    return Item("scev_val_fig_0001", "sciclaimeval", "2403.19137", "", "figure",
                "/data/figures/dev/val_fig_0001.png", "ConDA outperforms all baselines.",
                False, "nlp", "Legend Swap")

def test_tag_namespace_is_rsc_not_sciver():
    t = tag(1, "scev_val_fig_0001", "AS")
    assert t == "rsc_p01__scev_val_fig_0001__AS"
    assert not t.startswith("rs_p")  # disjoint from the SciVer DB namespace

def test_dims_cover_default_six():
    for d in ("AS", "QLq", "QLl", "MCr", "AT", "VO"):
        assert d in DIMS

def test_prompt_is_verdict_blind():
    p = build_prompt(1, _item(), "AS", "RUBRIC TEXT")
    assert "ConDA outperforms all baselines." in p          # claim present
    assert "/data/figures/dev/val_fig_0001.png" in p        # figure present
    for leak in ("Legend Swap", "Refuted", "Supported", "False"):
        assert leak not in p, leak                          # label/operation withheld
```

- [ ] **Step 5: Run the tests to verify they pass**

Run: `python -m pytest sciclaimeval_scoring/tests/test_prompts.py sciclaimeval_scoring/tests/test_parse.py -v`
Expected: PASS (5 passed).

- [ ] **Step 6: Commit**

```bash
git add sciclaimeval_scoring/prompts.py sciclaimeval_scoring/parse.py \
        sciclaimeval_scoring/tests/test_prompts.py sciclaimeval_scoring/tests/test_parse.py
git commit -m "feat(sciclaimeval_scoring): verdict-blind prompt builder + score parser"
```

---

### Task 5: Transcript scraper, collector CLI, aggregate, QA (pure copies)

**Files:**
- Create: `sciclaimeval_scoring/scrape.py`, `collect_cli.py`, `aggregate.py`, `qa.py` (copies, imports repointed)
- Test: `sciclaimeval_scoring/tests/test_aggregate.py`, `sciclaimeval_scoring/tests/test_qa.py`

**Interfaces:**
- Produces: `scrape.collect(conn, pass_no, manifest, projects_dir=None, session_id="unknown") -> int`; `aggregate.export_csv(conn, dims, path)`, `aggregate.aggregate(conn, dims)`; `qa.run_qa(conn, pass_no) -> dict`.

- [ ] **Step 1: Copy the four modules and repoint imports**

```bash
for f in scrape.py collect_cli.py aggregate.py qa.py; do
  cp rubric_scoring/$f sciclaimeval_scoring/$f
  sed -i 's/rubric_scoring/sciclaimeval_scoring/g' sciclaimeval_scoring/$f
done
grep -l rubric_scoring sciclaimeval_scoring/scrape.py sciclaimeval_scoring/collect_cli.py \
  sciclaimeval_scoring/aggregate.py sciclaimeval_scoring/qa.py || echo "no stray rubric_scoring refs (good)"
```

- [ ] **Step 2: Write `sciclaimeval_scoring/tests/test_aggregate.py`**

```python
from sciclaimeval_scoring import db, aggregate

def test_mode_and_mean_across_passes():
    conn = db.connect(":memory:"); db.init(conn)
    for p, s in ((1, 3), (2, 3), (3, 5)):
        db.upsert_annotation(conn, {"pass": p, "item_id": "i", "dim_code": "AS",
                                    "dim_name": "n", "score": s, "parse_ok": 1})
    rows = aggregate.aggregate(conn, ["AS"])
    assert rows[0]["mode"] == 3 and rows[0]["n_passes"] == 3
    assert rows[0]["min"] == 3 and rows[0]["max"] == 5
```

- [ ] **Step 3: Write `sciclaimeval_scoring/tests/test_qa.py`**

```python
from sciclaimeval_scoring import db, qa

def test_qa_parse_rate_and_thinking_clean():
    conn = db.connect(":memory:"); db.init(conn)
    db.upsert_annotation(conn, {"pass": 1, "item_id": "i", "dim_code": "AS", "dim_name": "n",
                                "score": 3, "parse_ok": 1, "output_thinking_tokens": 0})
    db.upsert_annotation(conn, {"pass": 1, "item_id": "j", "dim_code": "AS", "dim_name": "n",
                                "score": 4, "parse_ok": 1, "output_thinking_tokens": 0})
    r = qa.run_qa(conn, 1)
    assert r["parse_rate"] == 1.0
    assert r["thinking_clean"] is True
    assert r["per_dim"]["AS"]["flat"] is False
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest sciclaimeval_scoring/tests/test_aggregate.py sciclaimeval_scoring/tests/test_qa.py -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add sciclaimeval_scoring/scrape.py sciclaimeval_scoring/collect_cli.py \
        sciclaimeval_scoring/aggregate.py sciclaimeval_scoring/qa.py \
        sciclaimeval_scoring/tests/test_aggregate.py sciclaimeval_scoring/tests/test_qa.py
git commit -m "feat(sciclaimeval_scoring): scraper, collector CLI, aggregate, QA (copies)"
```

---

### Task 6: Resume-aware pass preparation

**Files:**
- Create: `sciclaimeval_scoring/prepare.py`
- Test: `sciclaimeval_scoring/tests/test_prepare.py`

**Interfaces:**
- Consumes: `config`, `db`, `items.load_items`, `prompts.build_prompt`, `prompts.tag`, `prompts.DIMS`.
- Produces: `prepare.prepare(pass_no, dims, out_dir, limit=None) -> manifest_path(str)`; `prepare.plan_missing(conn, pass_no, items, dims) -> list[{item_id, dim_code}]`.

- [ ] **Step 1: Write the failing test `sciclaimeval_scoring/tests/test_prepare.py`**

```python
import json
from sciclaimeval_scoring import config, db, prepare

def test_prepare_writes_manifest_and_resumes(tmp_path, monkeypatch):
    dbfile = tmp_path / "t.db"
    monkeypatch.setenv("SCICLAIMEVAL_SCORING_DB", str(dbfile))
    out = tmp_path / "prompts"
    mpath = prepare.prepare(1, ["AS", "VO"], str(out), limit=3)
    manifest = json.loads(open(mpath).read())
    assert len(manifest) == 3 * 2                      # 3 items x 2 dims, all missing
    assert all(t.startswith("rsc_p01__") for t in manifest)
    # mark one cell done, re-prepare -> one fewer prompt
    conn = db.connect(str(dbfile))
    first = next(iter(manifest.values()))
    db.upsert_annotation(conn, {"pass": 1, "item_id": first["item_id"],
                                "dim_code": first["dim_code"], "dim_name": first["dim_name"],
                                "score": 3, "parse_ok": 1})
    mpath2 = prepare.prepare(1, ["AS", "VO"], str(out), limit=3)
    assert len(json.loads(open(mpath2).read())) == 3 * 2 - 1
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest sciclaimeval_scoring/tests/test_prepare.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sciclaimeval_scoring.prepare'`.

- [ ] **Step 3: Write `sciclaimeval_scoring/prepare.py`**

```python
"""Resume-aware pass preparation: register the pass, write prompts for MISSING cells only,
emit a manifest the orchestrator and collector both use."""
import argparse, json, os
from sciclaimeval_scoring import config, db, items as items_mod
from sciclaimeval_scoring.prompts import build_prompt, tag, DIMS

SKILL_VERSION = "0.1"
MODEL = "claude-sonnet-4-6"
DEVIATIONS = ("temp!=0 (subagent path; SciVer-measured ~82% modal/89% within-1); "
              "output cap instructed not enforced; reasoning off; captions omitted; "
              "DeLeAn v1.0 textual-only anchors on multimodal figure items")

def plan_missing(conn, pass_no, items, dims):
    done = db.completed_cells(conn, pass_no)
    return [{"item_id": it.item_id, "dim_code": d} for it in items for d in dims
            if (it.item_id, d) not in done]

def prepare(pass_no, dims, out_dir, limit=None):
    conn = db.connect(config.db_path()); db.init(conn)
    items = items_mod.load_items()
    if limit is not None:
        items = items[:limit]
    for it in items:
        db.upsert_item(conn, it)
    db.register_pass(conn, pass_no, skill_version=SKILL_VERSION, model=MODEL,
                     active_dims=",".join(dims), item_source="SciClaimEval dev figures (265)",
                     n_items=len(items), params=json.dumps({"limit": limit}), deviations=DEVIATIONS)
    by_id = {it.item_id: it for it in items}
    rubrics = {d: (config.rubrics_dir() / f"{d}.txt").read_text(encoding="utf-8") for d in dims}
    os.makedirs(out_dir, exist_ok=True)
    manifest = {}
    for cell in plan_missing(conn, pass_no, items, dims):
        it = by_id[cell["item_id"]]; d = cell["dim_code"]; t = tag(pass_no, it.item_id, d)
        path = os.path.join(out_dir, f"{t}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(build_prompt(pass_no, it, d, rubrics[d]))
        manifest[t] = {"item_id": it.item_id, "dim_code": d, "dim_name": DIMS[d],
                       "prompt_file": path, "figure": it.image_path}
    mpath = os.path.join(out_dir, f"pass_{pass_no:02d}_manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"pass {pass_no}: {len(items)} items x {len(dims)} dims; "
          f"{len(manifest)} MISSING cells written to {out_dir}; manifest {mpath}")
    return mpath

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="pass_no", type=int, required=True)
    ap.add_argument("--dims", default="AS,QLq,QLl,MCr,AT,VO")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    out = a.out or str(config.repo_root() / "sciclaimeval_scoring" / "prompts" / f"pass_{a.pass_no:02d}")
    prepare(a.pass_no, a.dims.split(","), out, a.limit)
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `python -m pytest sciclaimeval_scoring/tests/test_prepare.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Full Part-A suite + commit**

Run: `python -m pytest sciclaimeval_scoring/ -v`
Expected: PASS (all green).

```bash
git add sciclaimeval_scoring/prepare.py sciclaimeval_scoring/tests/test_prepare.py
git commit -m "feat(sciclaimeval_scoring): resume-aware pass preparation"
```

---

### Task 7: Module README + skill

**Files:**
- Create: `sciclaimeval_scoring/README.md`
- Create: `.claude/skills/sciclaimeval-rubric-scoring/SKILL.md`

**Interfaces:** none (docs).

- [ ] **Step 1: Write `sciclaimeval_scoring/README.md`**

Document: purpose (DeLeAn demand annotation over 265 SciClaimEval dev figures), the prepare→dispatch→collect loop, verdict-blind + operation-leak note, the `scev_`/`rsc_` id+tag schemes, the join to `sciclaimeval_eval/predictions_sciclaimeval.db` on `item_id`, and the run commands:

```bash
source .venv/bin/activate
python -m sciclaimeval_scoring.prepare --pass 1 --dims AS,QLq,QLl,MCr,AT,VO   # resume-aware
# dispatch each MISSING cell to an isolated Sonnet subagent (Agent tool)
python -m sciclaimeval_scoring.collect_cli --pass 1 --manifest sciclaimeval_scoring/prompts/pass_01/pass_01_manifest.json
```

- [ ] **Step 2: Create the skill by cloning the SciVer one**

```bash
mkdir -p .claude/skills/sciclaimeval-rubric-scoring
cp .claude/skills/rubric-scoring/SKILL.md .claude/skills/sciclaimeval-rubric-scoring/SKILL.md
```

- [ ] **Step 3: Edit `.claude/skills/sciclaimeval-rubric-scoring/SKILL.md`**

Apply exactly these substitutions and additions:
- Frontmatter `name: rubric-scoring` → `name: sciclaimeval-rubric-scoring`; description "SciVer chart items" → "SciClaimEval dev figure items (265)".
- Every occurrence of `rubric_scoring` → `sciclaimeval_scoring` (covers both `rubric_scoring.<mod>` and `from rubric_scoring import ...` forms).
- Every standalone "chart"/"charts" → "figure"/"figures" (e.g. "SciVer chart items" → "SciClaimEval figure items").
- Item count "817" → "265"; the per-pass agent count line "817 × 6" → "265 × 6 = 1,590".
- `item_source`/title mentions of SciVer → "SciClaimEval dev figures".
- In the **Invariants** section, change the verdict-blind list to also name `operation`:
  > **Verdict-blind:** never put `label`/`operation`/`caption`/`context`/`claim_id_pair` into a prompt. `operation` leaks the verdict (`Supported_claim_only` ⇒ Supported). `prepare` already guarantees this (build_prompt reads only the claim + figure path).
- Tag-collision note: scoring tags are `rsc_p…` (disjoint from the SciVer `rs_p…`).

- [ ] **Step 4: Verify the skill has no stale SciVer references**

Run: `grep -niE "rubric_scoring|817|SciVer|chart" .claude/skills/sciclaimeval-rubric-scoring/SKILL.md || echo "clean"`
Expected: `clean` (or only an intentional "ported from SciVer" provenance line).

- [ ] **Step 5: Commit**

```bash
git add sciclaimeval_scoring/README.md .claude/skills/sciclaimeval-rubric-scoring/SKILL.md
git commit -m "docs(sciclaimeval_scoring): module README + sciclaimeval-rubric-scoring skill"
```

---

# Part B — `sciclaimeval_eval` (Haiku Supported/Refuted prediction)

### Task 8: Scaffold + predictions SQLite layer

**Files:**
- Create: `sciclaimeval_eval/__init__.py`, `sciclaimeval_eval/tests/__init__.py`
- Create: `sciclaimeval_eval/db.py`
- Test: `sciclaimeval_eval/tests/test_db.py`

**Interfaces:**
- Produces: `db.connect`, `db.init`, `db.default_db_path()` (→ `predictions_sciclaimeval.db`), `db.upsert_item(conn, it)`, `db.register_run(...)`, `db.upsert_prediction(conn, row)`, `db.completed_cells(conn, run_no, model) -> set[(item_id, trial)]`, `db.TEMPERATURE_UNCONTROLLED_NOTE`.

- [ ] **Step 1: Copy `sciver_eval/db.py`, repoint filename, add domain/operation to `item`**

```bash
mkdir -p sciclaimeval_eval/tests
touch sciclaimeval_eval/__init__.py sciclaimeval_eval/tests/__init__.py
cp sciver_eval/db.py sciclaimeval_eval/db.py
```

Then edit `sciclaimeval_eval/db.py`:
- `default_db_path()` return filename `predictions.db` → `predictions_sciclaimeval.db`.
- In `SCHEMA`, the `item` table line becomes (adds two columns for stratified analysis, kept consistent with `sciclaimeval_scoring`):

```sql
CREATE TABLE IF NOT EXISTS item(
  item_id TEXT PRIMARY KEY, source TEXT, paperid TEXT, claim_type TEXT,
  vtype TEXT, image_path TEXT, claim TEXT, label INTEGER, domain TEXT, operation TEXT);
```

- `upsert_item` writes 10 values:

```python
def upsert_item(conn, it):
    """`it` is a sciclaimeval_scoring.items.Item."""
    conn.execute("INSERT OR REPLACE INTO item VALUES(?,?,?,?,?,?,?,?,?,?)",
                 (it.item_id, it.source, it.paperid, it.claim_type, it.vtype,
                  it.image_path, it.claim, 1 if it.label else 0, it.domain, it.operation))
```

(Leave `run`/`prediction` tables, `register_run`, `upsert_prediction`, `completed_cells` exactly as copied.)

- [ ] **Step 2: Write `sciclaimeval_eval/tests/test_db.py`**

```python
from sciclaimeval_eval import db
from sciclaimeval_scoring.items import Item

def _item():
    return Item("scev_val_fig_0001", "sciclaimeval", "p", "", "figure",
                "/x.png", "claim", True, "nlp", "Supported_claim_only")

def test_default_db_path_is_sciclaimeval_named():
    assert db.default_db_path().name == "predictions_sciclaimeval.db"

def test_item_roundtrip_with_domain_operation():
    conn = db.connect(":memory:"); db.init(conn)
    db.upsert_item(conn, _item()); conn.commit()
    assert conn.execute("SELECT domain, operation FROM item").fetchone() == ("nlp", "Supported_claim_only")

def test_prediction_completed_cells_resume_key():
    conn = db.connect(":memory:"); db.init(conn)
    db.upsert_prediction(conn, {"run": 1, "item_id": "i", "model": "claude-haiku-4-5",
                                "trial": 1, "predicted": 1, "parse_ok": 1, "correct": 1})
    assert ("i", 1) in db.completed_cells(conn, 1, "claude-haiku-4-5")
```

- [ ] **Step 3: Run the tests**

Run: `python -m pytest sciclaimeval_eval/tests/test_db.py -v`
Expected: PASS (3 passed).

- [ ] **Step 4: Commit**

```bash
git add sciclaimeval_eval/__init__.py sciclaimeval_eval/tests/__init__.py \
        sciclaimeval_eval/db.py sciclaimeval_eval/tests/test_db.py
git commit -m "feat(sciclaimeval_eval): predictions sqlite layer with own DB"
```

---

### Task 9: Prediction prompt builder (single balanced template, inline caption/context)

**Files:**
- Create: `sciclaimeval_eval/prompts.py`
- Test: `sciclaimeval_eval/tests/test_prompts.py`

**Interfaces:**
- Consumes: `sciclaimeval_scoring.config`, `sciclaimeval_scoring.items.Item`.
- Produces: `prompts.tag(run_no, item_id, trial) -> str`, `prompts.load_records() -> dict[item_id, rec]`, `prompts.build_prompt(run_no, item, rec, trial, claim=None) -> str`.

- [ ] **Step 1: Write the failing test `sciclaimeval_eval/tests/test_prompts.py`**

```python
from sciclaimeval_eval.prompts import tag, build_prompt, load_records
from sciclaimeval_scoring.items import Item

def _item():
    return Item("scev_val_fig_0001", "sciclaimeval", "p", "", "figure",
                "/data/figures/dev/val_fig_0001.png", "ConDA outperforms all baselines.",
                False, "nlp", "Legend Swap")

def _rec():
    return {"claim": "ConDA outperforms all baselines.", "caption": "Table of AUROC.",
            "context": "We compare ConDA with baselines.", "label": "Refuted",
            "operation": "Legend Swap"}

def test_tag_namespace_is_sce():
    assert tag(1, "scev_val_fig_0001", 3) == "sce_r01__scev_val_fig_0001__t03"

def test_prompt_has_answer_line_and_evidence():
    p = build_prompt(1, _item(), _rec(), 1)
    assert "ConDA outperforms all baselines." in p
    assert "Table of AUROC." in p                 # caption shown (real task input)
    assert "/data/figures/dev/val_fig_0001.png" in p
    assert "Answer: yes" in p and "Answer: no" in p  # output format instruction

def test_prompt_is_verdict_clean():
    p = build_prompt(1, _item(), _rec(), 1)
    for leak in ("Legend Swap", "Refuted"):
        assert leak not in p, leak                  # label/operation never shown

def test_load_records_returns_265_figures():
    recs = load_records()
    assert len(recs) == 265
    assert all(k.startswith("scev_") for k in recs)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `python -m pytest sciclaimeval_eval/tests/test_prompts.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sciclaimeval_eval.prompts'`.

- [ ] **Step 3: Write `sciclaimeval_eval/prompts.py`**

```python
"""Build the SciClaimEval CoT Supported/Refuted prompt for a figure item.

Single balanced template (SciClaimEval has no claim_type), built from claim +
the inline caption + context fields + the figure path, plus a per-item TASK_TAG
so the collector can match the subagent transcript. Verdict-clean: never
interpolates label or operation.
"""
import json
from string import Template
from pathlib import Path

from sciclaimeval_scoring import config

_BALANCED = Template("""
Claim: $claim
Context: $context
Caption: $caption

Your task is to evaluate whether the claim is supported by the figure evidence and its caption. Carefully examine whether the figure and caption truly support the claim. Apply any relevant scientific principles, statistical logic, or domain knowledge needed to link the evidence to the claim. Be balanced: look for confirming details as well as inconsistencies, missing evidence, or ambiguities.

Start by explaining your reasoning step by step — describe what you observe in the figure and caption and how you test whether each key part of the claim is supported. If every essential component of the claim is clearly and completely supported by the figure and caption, respond 'yes'. If any critical point is contradicted, unsupported, or unclear, respond 'no'.

Conclude your analysis by stating: 'Therefore, the final answer is: Answer: $$ANSWER' (without quotes), where $$ANSWER is your final answer. Think step by step before answering.
""")

_WRAP = """TASK_TAG: {tag}

You are completing ONE item of the SciClaimEval multimodal scientific claim-verification
benchmark. Use ONLY the Read tool: first view the figure image, then reason, then answer.
{cot}
Figure -- view this path with the Read tool BEFORE answering:
{figs}

Your FINAL line must be exactly "Answer: yes" or "Answer: no" (lowercase).
"""


def tag(run_no, item_id, trial):
    return f"sce_r{run_no:02d}__{item_id}__t{trial:02d}"


def load_records():
    """item_id -> raw SciClaimEval figure record, SAME filter as items.load_items()."""
    sd = config.sciclaimeval_dir()
    data = json.loads((sd / "dev_task1_release.json").read_text())
    return {f"scev_{x['claim_id']}": x for x in data if x.get("evi_type") == "figure"}


def build_prompt(run_no, item, rec, trial, claim=None):
    """Verdict-clean CoT prompt. `claim` overrides the shown statement; defaults to rec['claim']."""
    claim_text = rec["claim"] if claim is None else claim
    cot = _BALANCED.substitute(claim=claim_text, context=rec.get("context", ""),
                               caption=rec.get("caption", ""))
    return _WRAP.format(tag=tag(run_no, item.item_id, trial), cot=cot, figs=item.image_path)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `python -m pytest sciclaimeval_eval/tests/test_prompts.py -v`
Expected: PASS (4 passed).

- [ ] **Step 5: Commit**

```bash
git add sciclaimeval_eval/prompts.py sciclaimeval_eval/tests/test_prompts.py
git commit -m "feat(sciclaimeval_eval): balanced Supported/Refuted CoT prompt builder"
```

---

### Task 10: Answer parser, verdict-aware scraper, collector CLI (copies)

**Files:**
- Create: `sciclaimeval_eval/parse.py` (byte copy), `scrape.py`, `collect_cli.py` (repointed)
- Test: `sciclaimeval_eval/tests/test_parse.py`, `sciclaimeval_eval/tests/test_scrape.py`

**Interfaces:**
- Produces: `parse.parse_answer(text) -> 'yes'|'no'|None`; `scrape.collect(conn, run_no, model, manifest, projects_dir=None, session_id="unknown") -> int`.

- [ ] **Step 1: Copy parse.py (no repoint needed), scrape.py + collect_cli.py (repoint)**

```bash
cp sciver_eval/parse.py sciclaimeval_eval/parse.py
cp sciver_eval/scrape.py sciclaimeval_eval/scrape.py
cp sciver_eval/collect_cli.py sciclaimeval_eval/collect_cli.py
# scrape.py imports `from rubric_scoring import config` and `from sciver_eval import ...`;
# repoint config to sciclaimeval_scoring and the package to sciclaimeval_eval.
sed -i 's/from rubric_scoring import config/from sciclaimeval_scoring import config/' sciclaimeval_eval/scrape.py
sed -i 's/sciver_eval/sciclaimeval_eval/g' sciclaimeval_eval/scrape.py sciclaimeval_eval/collect_cli.py
grep -nE "rubric_scoring|sciver_eval" sciclaimeval_eval/scrape.py sciclaimeval_eval/collect_cli.py || echo "clean"
```

- [ ] **Step 2: Write `sciclaimeval_eval/tests/test_parse.py`**

```python
from sciclaimeval_eval.parse import parse_answer

def test_parses_last_answer():
    assert parse_answer("reasoning ... Answer: yes") == "yes"
    assert parse_answer("Answer: no\n") == "no"

def test_none_when_absent():
    assert parse_answer("no verdict line") is None
    assert parse_answer("") is None
```

- [ ] **Step 3: Write `sciclaimeval_eval/tests/test_scrape.py`** (correctness scoring against the manifest label, no transcript needed)

```python
import json, os
from sciclaimeval_eval import db, scrape

def test_collect_scores_correct_against_manifest_label(tmp_path, monkeypatch):
    # one fake transcript that answers "yes" for a tag whose gold label is 1 (Supported)
    proj = tmp_path / "projects" / "sess" / "subagents"
    proj.mkdir(parents=True)
    tagname = "sce_r01__scev_val_fig_0001__t01"
    line = {"agentId": "a1", "timestamp": "2026-06-16T00:00:00Z",
            "message": {"role": "assistant", "usage": {"input_tokens": 10, "output_tokens": 5},
                        "content": [{"type": "text", "text": f"{tagname}\nAnswer: yes"}]}}
    (proj / "agent-a1.jsonl").write_text(json.dumps(line) + "\n")
    conn = db.connect(":memory:"); db.init(conn)
    manifest = {tagname: {"item_id": "scev_val_fig_0001", "trial": 1, "label": 1}}
    n = scrape.collect(conn, 1, "claude-haiku-4-5", manifest,
                       projects_dir=str(tmp_path / "projects"), session_id="sess")
    assert n == 1
    row = conn.execute("SELECT answer, predicted, correct, parse_ok FROM prediction").fetchone()
    assert row == ("yes", 1, 1, 1)
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest sciclaimeval_eval/tests/test_parse.py sciclaimeval_eval/tests/test_scrape.py -v`
Expected: PASS (3 passed).

- [ ] **Step 5: Commit**

```bash
git add sciclaimeval_eval/parse.py sciclaimeval_eval/scrape.py sciclaimeval_eval/collect_cli.py \
        sciclaimeval_eval/tests/test_parse.py sciclaimeval_eval/tests/test_scrape.py
git commit -m "feat(sciclaimeval_eval): answer parser + verdict-aware scraper (copies)"
```

---

### Task 11: build_db + resume-aware run preparation

**Files:**
- Create: `sciclaimeval_eval/build_db.py`, `sciclaimeval_eval/prepare.py`
- Test: `sciclaimeval_eval/tests/test_prepare.py`

**Interfaces:**
- Consumes: `sciclaimeval_scoring.config`, `sciclaimeval_scoring.items.load_items`, `sciclaimeval_eval.{db, prompts}`.
- Produces: `build_db.main()`; `prepare.prepare(run_no, out_dir, limit=None, trials=1, model=None) -> manifest_path`.

- [ ] **Step 1: Write `sciclaimeval_eval/build_db.py`**

```python
"""Create predictions_sciclaimeval.db, populate the 265 shared figure items, register run 1.

Items come from sciclaimeval_scoring.items.load_items() so item_ids match
annotations_sciclaimeval.db (the join key). Idempotent.

    python -m sciclaimeval_eval.build_db
"""
import argparse
from pathlib import Path

from sciclaimeval_scoring import items as scev_items
from sciclaimeval_eval import db

DEFAULT_DB = Path(__file__).resolve().parent / "predictions_sciclaimeval.db"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--prompt-version", default="sciclaimeval-cot-v1")
    args = ap.parse_args()

    conn = db.connect(args.db)
    db.init(conn)

    its = scev_items.load_items()
    for it in its:
        db.upsert_item(conn, it)
    conn.commit()

    db.register_run(
        conn, args.run, skill_version="sciclaimeval_eval-0.1", model=args.model,
        prompt_version=args.prompt_version, temperature=args.temperature,
        item_source="SciClaimEval dev figures (265)", n_items=len(its),
        params="harness-subagent; CoT; max_tokens=10240",
        deviations="" if args.temperature is not None else db.TEMPERATURE_UNCONTROLLED_NOTE,
    )
    n_item = conn.execute("SELECT COUNT(*) FROM item").fetchone()[0]
    print(f"DB: {args.db}")
    print(f"items: {n_item}  |  run {args.run} registered for model={args.model}")
    print("by domain:", conn.execute(
        "SELECT domain, COUNT(*) FROM item GROUP BY domain").fetchall())


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Write `sciclaimeval_eval/prepare.py`** (native run only; no conditions)

```python
"""Resume-aware run preparation: ensure items + run exist, write prompts for
MISSING (item, trial) cells only, emit a manifest carrying the gold label.

    python -m sciclaimeval_eval.prepare --run 1 --trials 5
"""
import argparse
import json
import os

from sciclaimeval_scoring import config, items as items_mod
from sciclaimeval_eval import db, prompts

DEFAULT_MODEL = "claude-haiku-4-5"


def prepare(run_no, out_dir, limit=None, trials=1, model=None):
    conn = db.connect(db.default_db_path())
    db.init(conn)

    items = items_mod.load_items()
    if limit is not None:
        items = items[:limit]
    for it in items:
        db.upsert_item(conn, it)
    conn.commit()

    model_id = model or DEFAULT_MODEL
    run = conn.execute("SELECT model FROM run WHERE run_id=?", (run_no,)).fetchone()
    if run is None:
        db.register_run(conn, run_no, skill_version="sciclaimeval_eval-0.1", model=model_id,
                        prompt_version="sciclaimeval-cot-v1", temperature=None,
                        item_source="SciClaimEval dev figures (265)",
                        n_items=len(items_mod.load_items()),
                        params="harness-subagent; CoT; max_tokens=10240",
                        deviations=db.TEMPERATURE_UNCONTROLLED_NOTE)
        model = model_id
    else:
        model = run[0]

    records = prompts.load_records()
    done = db.completed_cells(conn, run_no, model)  # {(item_id, trial)}
    os.makedirs(out_dir, exist_ok=True)

    manifest = {}
    for it in items:
        for tr in range(1, trials + 1):
            if (it.item_id, tr) in done:
                continue
            rec = records[it.item_id]
            t = prompts.tag(run_no, it.item_id, tr)
            path = os.path.join(out_dir, f"{t}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(prompts.build_prompt(run_no, it, rec, tr))
            manifest[t] = {"item_id": it.item_id, "trial": tr, "prompt_file": path,
                           "images": [it.image_path], "label": (1 if it.label else 0)}

    mpath = os.path.join(out_dir, f"run_{run_no:02d}_manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"run {run_no} ({model}): {len(items)} items x {trials} trials; "
          f"{len(manifest)} MISSING cells -> {out_dir}; manifest {mpath}")
    return mpath


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", dest="run_no", type=int, default=1)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    out = a.out or str(config.repo_root() / "sciclaimeval_eval" / "prompts" / f"run_{a.run_no:02d}")
    prepare(a.run_no, out, a.limit, a.trials, a.model)
```

- [ ] **Step 3: Write the failing test `sciclaimeval_eval/tests/test_prepare.py`**

```python
import json
from sciclaimeval_eval import prepare, db

def test_prepare_writes_manifest_with_labels_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "default_db_path", lambda: tmp_path / "p.db")
    out = tmp_path / "prompts"
    mpath = prepare.prepare(1, str(out), limit=3, trials=2)
    manifest = json.loads(open(mpath).read())
    assert len(manifest) == 3 * 2
    assert all(t.startswith("sce_r01__") for t in manifest)
    assert all(set(m) >= {"item_id", "trial", "label", "images"} for m in manifest.values())
    # mark one (item, trial) done -> one fewer prompt next prepare
    conn = db.connect(str(tmp_path / "p.db"))
    first = next(iter(manifest.values()))
    db.upsert_prediction(conn, {"run": 1, "item_id": first["item_id"],
                                "model": "claude-haiku-4-5", "trial": first["trial"],
                                "predicted": 1, "parse_ok": 1, "correct": 1})
    mpath2 = prepare.prepare(1, str(out), limit=3, trials=2)
    assert len(json.loads(open(mpath2).read())) == 3 * 2 - 1
```

- [ ] **Step 4: Run the test**

Run: `python -m pytest sciclaimeval_eval/tests/test_prepare.py -v`
Expected: PASS (1 passed).

- [ ] **Step 5: Smoke-run build_db against the real data**

Run: `python -m sciclaimeval_eval.build_db --db /tmp/pred_smoke.db`
Expected output includes `items: 265` and a `by domain:` line listing ml/nlp/peerj. Then clean up: `rm /tmp/pred_smoke.db`.

- [ ] **Step 6: Commit**

```bash
git add sciclaimeval_eval/build_db.py sciclaimeval_eval/prepare.py sciclaimeval_eval/tests/test_prepare.py
git commit -m "feat(sciclaimeval_eval): build_db + resume-aware native-run preparation"
```

---

### Task 12: README, full-suite gate, and the cross-DB join smoke test

**Files:**
- Create: `sciclaimeval_eval/README.md`
- Test: `sciclaimeval_eval/tests/test_join.py`

**Interfaces:** none new.

- [ ] **Step 1: Write the join smoke test `sciclaimeval_eval/tests/test_join.py`**

```python
"""The whole point: predictions ⋈ demand scores on item_id. Verify the keys align."""
from sciclaimeval_scoring import db as sdb
from sciclaimeval_eval import db as edb
from sciclaimeval_scoring.items import Item

def test_item_ids_join_across_dbs():
    it = Item("scev_val_fig_0001", "sciclaimeval", "p", "", "figure",
              "/x.png", "c", True, "nlp", "Legend Swap")
    s = sdb.connect(":memory:"); sdb.init(s); sdb.upsert_item(s, it)
    sdb.upsert_annotation(s, {"pass": 1, "item_id": it.item_id, "dim_code": "AS",
                              "dim_name": "n", "score": 3, "parse_ok": 1})
    e = edb.connect(":memory:"); edb.init(e); edb.upsert_item(e, it); e.commit()
    edb.upsert_prediction(e, {"run": 1, "item_id": it.item_id, "model": "claude-haiku-4-5",
                              "trial": 1, "predicted": 1, "parse_ok": 1, "correct": 0})
    score = s.execute("SELECT score FROM annotation WHERE item_id=?", (it.item_id,)).fetchone()[0]
    correct = e.execute("SELECT correct FROM prediction WHERE item_id=?", (it.item_id,)).fetchone()[0]
    assert (score, correct) == (3, 0)   # same item_id keys both rows
```

- [ ] **Step 2: Write `sciclaimeval_eval/README.md`**

Document: purpose (Haiku Supported/Refuted prediction over the 265 figures, trials → per-item `p(correct)`), the build_db → prepare → dispatch → collect loop, the `sce_` tag namespace, the model default `claude-haiku-4-5`, the deferred entailed/refuted conditions, and the join to `sciclaimeval_scoring/annotations_sciclaimeval.db` on `item_id` for `r(demand, p_correct)`. Run commands:

```bash
source .venv/bin/activate
python -m sciclaimeval_eval.build_db
python -m sciclaimeval_eval.prepare --run 1 --trials 5
# dispatch each MISSING cell to a Haiku subagent
python -m sciclaimeval_eval.collect_cli --run 1 --manifest sciclaimeval_eval/prompts/run_01/run_01_manifest.json
```

- [ ] **Step 3: Run the entire test suite (both modules)**

Run: `python -m pytest sciclaimeval_scoring/ sciclaimeval_eval/ -v`
Expected: PASS (all green, ~25+ tests).

- [ ] **Step 4: Final isolation check (no SciVer coupling, no SciVer DB writes)**

Run:
```bash
grep -rnE "rubric_scoring|sciver_eval" sciclaimeval_eval/ sciclaimeval_scoring/ \
  | grep -vE "sciclaimeval" | grep -v "from sciclaimeval_scoring" || echo "no SciVer imports (good)"
ls sciclaimeval_scoring/annotations_sciclaimeval.db sciclaimeval_eval/predictions_sciclaimeval.db 2>/dev/null
```
Expected: the grep prints `no SciVer imports (good)` (the only allowed cross refs are `from sciclaimeval_scoring import ...` inside `sciclaimeval_eval`). Note: `rubric_scoring/annotations_prod.db` and `sciver_eval/predictions.db` are never opened by these modules.

- [ ] **Step 5: Commit**

```bash
git add sciclaimeval_eval/README.md sciclaimeval_eval/tests/test_join.py
git commit -m "feat(sciclaimeval_eval): README + cross-DB item_id join smoke test"
```

---

## Notes for the executor

- **Running real annotation/prediction passes** (dispatching subagents) is operational, not part of this build. After the suite is green, follow each module's README and the `sciclaimeval-rubric-scoring` skill: `prepare` → dispatch MISSING cells to isolated subagents (Sonnet for scoring, Haiku for prediction) → `collect_cli`. Resume across sessions; never re-dispatch completed cells.
- **First-batch measurement:** capture per-agent tokens/seconds from the first ~18 scoring agents and the first ~25 prediction agents and record them in the module READMEs, rather than reusing SciVer's figures.
- **`.gitignore`:** the two `*.db` files are created at runtime; if the repo ignores SciVer's DBs, add `sciclaimeval_scoring/annotations_sciclaimeval.db` and `sciclaimeval_eval/predictions_sciclaimeval.db` to match (check `git status` before the first DB-creating run).
