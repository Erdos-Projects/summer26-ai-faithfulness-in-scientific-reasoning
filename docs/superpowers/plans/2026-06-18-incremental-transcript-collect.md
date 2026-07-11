# Incremental Transcript Collection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make `rubric_scoring` collection skip transcripts already ingested in prior runs and resolve manifest tags by O(1) index lookup, instead of re-reading and substring-scanning the entire transcript corpus every collect.

**Architecture:** Add a persistent ingestion ledger (`ingested_transcript` table keyed on `path`, validated by `(size, mtime)`). Rewrite `collect()` to stat each candidate transcript, skip unchanged ones via the ledger, scan only new/changed files once to build a `{tag: scan_result}` index, then resolve each manifest tag by dict lookup. Preserve today's "prefer scored transcript" semantics with a never-regress guard.

**Tech Stack:** Python 3, stdlib `sqlite3`/`glob`/`os`/`re`, pytest.

## Global Constraints

- **Byte-identical annotation output:** the `annotation` rows produced for a given corpus must match today's collector. `aggregate.csv` must not change.
- **No schema changes to `annotation`, `item`, `pass`.** Only add the new `ingested_transcript` table.
- **Ledger key is `(path, size, mtime)`, stat-only — no content hashing** (Design Decision A).
- **`collect()` keeps its current signature**, adding only a trailing `full=False` keyword; existing call sites and tests must keep working.
- Follow existing test style in `rubric_scoring/tests/` (terse, `tmp_path`, helper builders).
- Spec: `docs/superpowers/specs/2026-06-18-incremental-transcript-collect-design.md`.

---

### Task 1: Ingestion ledger in db.py

**Files:**
- Modify: `rubric_scoring/db.py` (extend `SCHEMA`; add `ledger_seen`, `mark_ingested`)
- Test: `rubric_scoring/tests/test_db.py`

**Interfaces:**
- Consumes: existing `db.connect`, `db.init`, `db._now`.
- Produces:
  - `db.ledger_seen(conn) -> dict[str, tuple[int, float]]` — maps `path -> (size, mtime)` for every ingested transcript.
  - `db.mark_ingested(conn, path: str, size: int, mtime: float) -> None` — INSERT OR REPLACE one ledger row, stamping `ingested` with `_now()`.

- [ ] **Step 1: Write the failing test**

Add to `rubric_scoring/tests/test_db.py`:

```python
from rubric_scoring import db

def test_ledger_seen_empty_then_roundtrip(tmp_path):
    conn = db.connect(tmp_path / "t.db"); db.init(conn)
    assert db.ledger_seen(conn) == {}
    db.mark_ingested(conn, "/x/agent-a.jsonl", 123, 1000.5)
    assert db.ledger_seen(conn) == {"/x/agent-a.jsonl": (123, 1000.5)}

def test_mark_ingested_replaces_on_change(tmp_path):
    conn = db.connect(tmp_path / "t.db"); db.init(conn)
    db.mark_ingested(conn, "/x/agent-a.jsonl", 123, 1000.5)
    db.mark_ingested(conn, "/x/agent-a.jsonl", 456, 2000.0)  # same path, new size/mtime
    assert db.ledger_seen(conn) == {"/x/agent-a.jsonl": (456, 2000.0)}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest rubric_scoring/tests/test_db.py -k ledger -v`
Expected: FAIL — `AttributeError: module 'rubric_scoring.db' has no attribute 'ledger_seen'`.

- [ ] **Step 3: Write minimal implementation**

In `rubric_scoring/db.py`, append a table to the `SCHEMA` string (after the `annotation` table, before the closing `"""`):

```sql
CREATE TABLE IF NOT EXISTS ingested_transcript(
  path TEXT PRIMARY KEY, size INTEGER, mtime REAL, ingested TEXT);
```

Then add these functions at module level:

```python
def ledger_seen(conn) -> dict:
    return {p: (s, m) for p, s, m in
            conn.execute("SELECT path,size,mtime FROM ingested_transcript")}


def mark_ingested(conn, path, size, mtime):
    conn.execute("INSERT OR REPLACE INTO ingested_transcript VALUES(?,?,?,?)",
                 (str(path), size, mtime, _now()))
    conn.commit()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest rubric_scoring/tests/test_db.py -k ledger -v`
Expected: PASS (2 passed).

- [ ] **Step 5: Commit**

```bash
git add rubric_scoring/db.py rubric_scoring/tests/test_db.py
git commit -m "feat(rubric_scoring): ingestion ledger table + helpers"
```

---

### Task 2: Tag extraction + scan_transcript returns tags

**Files:**
- Modify: `rubric_scoring/scrape.py` (add `extract_tags`; have `scan_transcript` return `tags`)
- Test: `rubric_scoring/tests/test_scrape.py`

**Interfaces:**
- Consumes: existing `scan_transcript` internals (it already reads the full file text into `raw`).
- Produces:
  - `scrape.extract_tags(text: str) -> list[str]` — sorted unique rubric tags found in `text`, matching `rs_p\d+__<item_id>__<code>`.
  - `scan_transcript(path)` return dict gains key `"tags": list[str]` (all rubric tags appearing in that transcript).

- [ ] **Step 1: Write the failing test**

Add to `rubric_scoring/tests/test_scrape.py`:

```python
from rubric_scoring.scrape import extract_tags

def test_extract_tags_finds_rubric_tag():
    txt = "PILOT_TAG: rs_p01__sciver_val_0__AS\nmore text rs_p01__sciver_val_0__AS again"
    assert extract_tags(txt) == ["rs_p01__sciver_val_0__AS"]
    assert extract_tags("no tags here") == []
    # underscores inside item_id must not break the split; lowercase codes allowed
    assert extract_tags("x rs_p02__sciver_test_137__QLq y") == ["rs_p02__sciver_test_137__QLq"]

def test_scan_transcript_returns_tags(tmp_path):
    f = tmp_path / "agent-agentX.jsonl"
    _make_transcript(f, "rs_p01__sciver_val_0__MCr", score=4)
    assert scan_transcript(f)["tags"] == ["rs_p01__sciver_val_0__MCr"]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python -m pytest rubric_scoring/tests/test_scrape.py -k "extract_tags or returns_tags" -v`
Expected: FAIL — `ImportError: cannot import name 'extract_tags'`.

- [ ] **Step 3: Write minimal implementation**

In `rubric_scoring/scrape.py`, add near the top (after imports):

```python
import re
_TAG_RE = re.compile(r"rs_p\d+__\S+?__[A-Za-z]+")

def extract_tags(text):
    return sorted(set(_TAG_RE.findall(text or "")))
```

In `scan_transcript`, the function already reads the whole file into `raw`. Before the `return`, compute tags and add to the returned dict:

```python
    tags = extract_tags(raw)
    return {"agent_id": agent_id, "input_tokens": inp, "cache_creation_tokens": cc,
            "cache_read_tokens": cr, "output_tokens": out, "total_input_tokens": inp + cc + cr,
            "output_thinking_tokens": think, "wall_clock_s": dur,
            "score": parse_score(final), "raw_output": final, "tags": tags}
```

- [ ] **Step 4: Run test to verify it passes**

Run: `python -m pytest rubric_scoring/tests/test_scrape.py -v`
Expected: PASS (all, including the 3 pre-existing scrape tests — the new `tags` key is additive).

- [ ] **Step 5: Commit**

```bash
git add rubric_scoring/scrape.py rubric_scoring/tests/test_scrape.py
git commit -m "feat(rubric_scoring): extract rubric tags from transcript text"
```

---

### Task 3: Incremental collect() with ledger, tag index, never-regress

**Files:**
- Modify: `rubric_scoring/scrape.py` (rewrite `collect`)
- Test: `rubric_scoring/tests/test_scrape.py`

**Interfaces:**
- Consumes: `db.ledger_seen`, `db.mark_ingested`, `db.completed_cells`, `db.upsert_annotation`, `scrape.scan_transcript` (now returns `tags`).
- Produces: `collect(conn, pass_no, manifest, projects_dir=None, session_id="unknown", full=False) -> int` — returns the number of annotation rows upserted this run. Ledger-incremental unless `full=True`.

- [ ] **Step 1: Write the failing tests**

Add to `rubric_scoring/tests/test_scrape.py`. First a helper for a no-score transcript, then the four behaviors:

```python
def _make_unscored(path, tag):
    lines = [
        {"agentId":"agentX","timestamp":"2026-06-14T00:00:00.000Z",
         "message":{"role":"user","content":f"do task {tag}"}},
        {"agentId":"agentX","timestamp":"2026-06-14T00:00:05.000Z",
         "message":{"role":"assistant",
                    "content":[{"type":"text","text":"I could not complete this"}],
                    "usage":{"input_tokens":5,"output_tokens":10,
                             "cache_creation_input_tokens":0,"cache_read_input_tokens":0}}},
    ]
    path.write_text("\n".join(json.dumps(o) for o in lines))

def _sub(tmp_path):
    p = tmp_path/"projects"/"p"/"sess"/"subagents"; p.mkdir(parents=True, exist_ok=True); return p

def test_collect_incremental_skips_unchanged(tmp_path, monkeypatch):
    sub = _sub(tmp_path)
    _make_transcript(sub/"agent-a1.jsonl", "rs_p01__sciver_val_0__AS", score=2)
    conn = db.connect(tmp_path/"t.db"); db.init(conn)
    man = {"rs_p01__sciver_val_0__AS": {"item_id":"sciver_val_0","dim_code":"AS","dim_name":"AS"}}
    assert collect(conn, 1, man, projects_dir=tmp_path/"projects", session_id="sess") == 1
    import rubric_scoring.scrape as S
    calls = []
    orig = S.scan_transcript
    monkeypatch.setattr(S, "scan_transcript", lambda p: calls.append(p) or orig(p))
    # nothing changed on disk -> second collect reads zero files, writes zero rows
    assert collect(conn, 1, man, projects_dir=tmp_path/"projects", session_id="sess") == 0
    assert calls == []

def test_collect_retry_supersedes_failure(tmp_path):
    sub = _sub(tmp_path)
    _make_unscored(sub/"agent-fail.jsonl", "rs_p01__sciver_val_0__AS")
    conn = db.connect(tmp_path/"t.db"); db.init(conn)
    man = {"rs_p01__sciver_val_0__AS": {"item_id":"sciver_val_0","dim_code":"AS","dim_name":"AS"}}
    collect(conn, 1, man, projects_dir=tmp_path/"projects", session_id="sess")
    assert conn.execute("SELECT score,parse_ok FROM annotation").fetchone() == (None, 0)
    _make_transcript(sub/"agent-ok.jsonl", "rs_p01__sciver_val_0__AS", score=4)  # new retry file
    collect(conn, 1, man, projects_dir=tmp_path/"projects", session_id="sess")
    assert conn.execute("SELECT score,parse_ok FROM annotation").fetchone() == (4, 1)

def test_collect_never_regresses_scored_cell(tmp_path):
    sub = _sub(tmp_path)
    _make_transcript(sub/"agent-ok.jsonl", "rs_p01__sciver_val_0__AS", score=4)
    conn = db.connect(tmp_path/"t.db"); db.init(conn)
    man = {"rs_p01__sciver_val_0__AS": {"item_id":"sciver_val_0","dim_code":"AS","dim_name":"AS"}}
    collect(conn, 1, man, projects_dir=tmp_path/"projects", session_id="sess")
    _make_unscored(sub/"agent-latefail.jsonl", "rs_p01__sciver_val_0__AS")  # late failed retry
    collect(conn, 1, man, projects_dir=tmp_path/"projects", session_id="sess")
    assert conn.execute("SELECT score,parse_ok FROM annotation").fetchone() == (4, 1)

def test_collect_full_rebuild_ignores_ledger(tmp_path, monkeypatch):
    sub = _sub(tmp_path)
    _make_transcript(sub/"agent-a1.jsonl", "rs_p01__sciver_val_0__AS", score=2)
    conn = db.connect(tmp_path/"t.db"); db.init(conn)
    man = {"rs_p01__sciver_val_0__AS": {"item_id":"sciver_val_0","dim_code":"AS","dim_name":"AS"}}
    collect(conn, 1, man, projects_dir=tmp_path/"projects", session_id="sess")
    import rubric_scoring.scrape as S
    calls = []
    orig = S.scan_transcript
    monkeypatch.setattr(S, "scan_transcript", lambda p: calls.append(p) or orig(p))
    assert collect(conn, 1, man, projects_dir=tmp_path/"projects", session_id="sess", full=True) == 1
    assert len(calls) == 1  # re-read despite ledger
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `python -m pytest rubric_scoring/tests/test_scrape.py -k "incremental or supersedes or never_regress or full_rebuild" -v`
Expected: FAIL — `collect()` has no `full` kwarg / re-reads files / regresses scored cell.

- [ ] **Step 3: Write the implementation**

Replace the body of `collect` in `rubric_scoring/scrape.py` with:

```python
def collect(conn, pass_no, manifest, projects_dir=None, session_id="unknown", full=False):
    projects_dir = projects_dir or config.claude_projects_dir()
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
```

Delete the now-unused `contents = {}` block and the old per-tag substring matching loop (they are fully replaced above).

- [ ] **Step 4: Run the full scrape suite**

Run: `python -m pytest rubric_scoring/tests/test_scrape.py -v`
Expected: PASS — the 4 new tests plus all 5 pre-existing scrape tests (`scan_transcript_extracts_usage`, `upserts_only_tagged_cells`, `scopes_to_session`, and Task 2's two).

- [ ] **Step 5: Commit**

```bash
git add rubric_scoring/scrape.py rubric_scoring/tests/test_scrape.py
git commit -m "feat(rubric_scoring): incremental ledger-based collect with tag index + never-regress"
```

---

### Task 4: --full flag on collect_cli

**Files:**
- Modify: `rubric_scoring/collect_cli.py`
- Test: none (thin CLI wrapper; behavior covered by Task 3). Verified by smoke run.

**Interfaces:**
- Consumes: `collect(..., full=...)` from Task 3.
- Produces: `python -m rubric_scoring.collect_cli --pass N --manifest M [--full]`.

- [ ] **Step 1: Add the flag and thread it through**

In `rubric_scoring/collect_cli.py`, add the argument and pass it:

```python
    ap.add_argument("--full", action="store_true",
                    help="ignore the ingestion ledger and rescan every transcript")
    a = ap.parse_args()
    conn = db.connect(config.db_path()); db.init(conn)
    with open(a.manifest) as f:
        manifest = json.load(f)
    session_id = os.environ.get("CLAUDE_CODE_SESSION_ID", "unknown")
    n = collect(conn, a.pass_no, manifest, session_id=session_id, full=a.full)
```

- [ ] **Step 2: Smoke-test the CLI parses and runs**

Run:
```bash
python - <<'PY'
import json, tempfile, os
from rubric_scoring import db, config
d = tempfile.mkdtemp()
man = os.path.join(d, "m.json"); json.dump({}, open(man, "w"))
os.environ["RUBRIC_SCORING_DB"] = os.path.join(d, "t.db")
import subprocess, sys
print(subprocess.run([sys.executable, "-m", "rubric_scoring.collect_cli",
      "--pass", "1", "--manifest", man, "--full"], capture_output=True, text=True).stdout)
PY
```
Expected: prints `collected 0 cells this session; pass 1 now has 0 completed cells` (empty manifest, no transcripts) with no traceback.

- [ ] **Step 3: Commit**

```bash
git add rubric_scoring/collect_cli.py
git commit -m "feat(rubric_scoring): --full rescan flag on collect_cli"
```

---

### Task 5: Regression verification against the real corpus

**Files:**
- Create: `rubric_scoring/tests/regression_collect.py` (a verification script, not a pytest unit — it depends on the live 4.8 GB corpus + prod DB).

**Interfaces:**
- Consumes: `collect`, `db`, `aggregate.export_csv`, the committed `annotations_prod.db`.

This task proves the Global Constraint "byte-identical annotation output". It copies the prod DB, runs the NEW collector in `full=True` mode over the real corpus into a fresh DB built from the same items/passes, and diffs the `annotation` score columns and the exported `aggregate.csv` against the committed prod outputs.

- [ ] **Step 1: Write the verification script**

Create `rubric_scoring/tests/regression_collect.py`:

```python
"""Manual regression: new collector over the real corpus must reproduce prod annotation rows.
Run from repo root: python -m rubric_scoring.tests.regression_collect
Requires the live ~/.claude/projects corpus and rubric_scoring/annotations_prod.db."""
import hashlib, shutil, tempfile, os
from pathlib import Path
from rubric_scoring import db, config, aggregate
from rubric_scoring.extract_rubrics import ALL_DIMS, EMERGENT_DIMS

DIMS = list(ALL_DIMS) + list(EMERGENT_DIMS)

def _agg_digest(conn):
    d = tempfile.mkdtemp()
    out = Path(d) / "agg.csv"
    aggregate.export_csv(conn, DIMS, out)
    return hashlib.sha256(out.read_bytes()).hexdigest()

def main():
    prod = config.db_path()
    prod_conn = db.connect(prod)
    prod_digest = _agg_digest(prod_conn)
    prod_rows = dict(prod_conn.execute(
        "SELECT pass||'|'||item_id||'|'||dim_code, score FROM annotation WHERE parse_ok=1"))
    print(f"prod: {len(prod_rows)} scored cells, aggregate sha256={prod_digest[:12]}")
    print("Now rebuild a fresh DB by running prepare.py + collect_cli --full for each "
          "pass/manifest used in prod, then re-run this script pointing RUBRIC_SCORING_DB "
          "at the rebuilt DB and compare the two printed digests. They MUST match.")

if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run it against prod to capture the baseline digest**

Run: `python -m rubric_scoring.tests.regression_collect`
Expected: prints the prod scored-cell count and an `aggregate sha256` baseline with no error.

- [ ] **Step 3: Rebuild with the new collector and compare**

Run (rebuild into a scratch DB, then diff digests):
```bash
SCRATCH=$(mktemp -d)/rebuild.db
for M in $(ls rubric_scoring/prompts/pass_*/pass_*_manifest.json); do
  P=$(echo "$M" | grep -oP 'pass_\K\d+' | head -1 | sed 's/^0*//')
  RUBRIC_SCORING_DB=$SCRATCH python -m rubric_scoring.collect_cli --pass "$P" --manifest "$M" --full
done
echo "baseline (prod):"; python -m rubric_scoring.tests.regression_collect
echo "rebuilt:"; RUBRIC_SCORING_DB=$SCRATCH python -m rubric_scoring.tests.regression_collect
```
Expected: the `aggregate sha256=` line is **identical** for prod and the rebuilt DB. If it differs, STOP — the new collector changed output; do not proceed.

- [ ] **Step 4: Commit the verification script**

```bash
git add rubric_scoring/tests/regression_collect.py
git commit -m "test(rubric_scoring): regression script proving byte-identical collect output"
```

---

## Self-Review

**Spec coverage:**
- Ledger table + `(path,size,mtime)` key → Task 1 ✓ (Decision A)
- Inverted tag index, O(files+tags) → Tasks 2–3 ✓
- No whole-corpus RAM dict (one scan per new file, discard) → Task 3 ✓
- Never-regress scored cell (Decision C) → Task 3 test + guard ✓
- `--full` escape hatch (Decision B) → Tasks 3–4 ✓
- Byte-identical output regression → Task 5 ✓
- Incremental skip / new-file pickup / retry-supersedes → Task 3 tests ✓

**Placeholder scan:** No TBD/TODO; every code step shows complete code. Task 5 is intentionally a manual verification script (depends on the live corpus), with exact commands and a hard STOP condition.

**Type consistency:** `ledger_seen -> dict[path,(size,mtime)]` produced in Task 1, consumed in Task 3 with `seen.get(f) == (st.st_size, st.st_mtime)` ✓. `scan_transcript(...)["tags"]: list[str]` produced in Task 2, consumed in Task 3 loop ✓. `collect(..., full=False) -> int` consistent across Tasks 3–4 ✓. `extract_tags` name consistent ✓.
