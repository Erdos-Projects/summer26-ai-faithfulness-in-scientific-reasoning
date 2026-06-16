# Paired Entailed-vs-Refuted SciVer Eval — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run Haiku on two new homogeneous SciVer conditions over the same 817 chart items — run 2 = all-entailed (`origin_statement`, label 1), run 3 = all-refuted (`perturbed_statement`, label 0) — and produce within-item paired accuracy data.

**Architecture:** Approach A (run-as-condition) from the spec. A new `conditions.py` maps a condition to (which statement to show, grade label). `build_prompt` takes the shown claim explicitly; `prepare` gains `--condition` and writes the condition's statement + grade-label into the manifest. The collector is unchanged because it already grades against `meta["label"]` (`sciver_eval/scrape.py:88`). A reproducible `shard.py` replaces the lost ad-hoc sharding step, `wf_dispatch.js` is parameterized by run number, and `analyze_paired.py` joins run 2 ⋈ run 3 on `item_id`.

**Tech Stack:** Python 3.12, sqlite3, pytest 9.0.3, stdlib `statistics.NormalDist` (no scipy needed), the Claude Code Workflow harness for dispatch.

**Spec:** `docs/superpowers/specs/2026-06-16-paired-entailed-refuted-eval-design.md`

**Pre-flight:** `source .venv/bin/activate` (all `python`/`pytest` commands assume the venv is active). The SciVer dataset must be present at `config.sciver_dir()` (project invariant; `prepare`/integration tests skip if absent).

---

### Task 1: `conditions.py` — verdict conditions as a first-class concept

**Files:**
- Create: `sciver_eval/conditions.py`
- Test: `tests/test_sciver_eval_conditions.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sciver_eval_conditions.py
import pytest

from sciver_eval import conditions


def _rec():
    return {
        "claim_type": "direct",
        "claim": "OFFICIAL CLAIM",
        "origin_statement": "TRUE STATEMENT",
        "perturbed_statement": "FALSE STATEMENT",
        "perturbed_explanation": "SECRET REASON",
    }


def test_entailed_resolves_to_origin_statement_and_label_1():
    claim, label = conditions.resolve("entailed", _rec())
    assert claim == "TRUE STATEMENT"
    assert label == 1


def test_refuted_resolves_to_perturbed_statement_and_label_0():
    claim, label = conditions.resolve("refuted", _rec())
    assert claim == "FALSE STATEMENT"
    assert label == 0


def test_unknown_condition_raises():
    with pytest.raises(KeyError):
        conditions.resolve("bogus", _rec())


def test_empty_statement_raises():
    rec = _rec()
    rec["origin_statement"] = "   "
    with pytest.raises(ValueError):
        conditions.resolve("entailed", rec)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sciver_eval_conditions.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sciver_eval.conditions'`

- [ ] **Step 3: Write minimal implementation**

```python
# sciver_eval/conditions.py
"""First-class verdict conditions for the paired entailed-vs-refuted eval.

Each condition selects WHICH statement becomes the shown claim and the grade
label the collector scores `correct` against. The native run (official
`claim`/`label`) is intentionally NOT here -- it is run 1 and already complete;
this module only drives the two new homogeneous runs.
"""

# condition -> (record field holding the statement to show, grade label)
CONDITIONS = {
    "entailed": ("origin_statement", 1),    # show the true statement;  correct answer = yes
    "refuted": ("perturbed_statement", 0),  # show the false statement; correct answer = no
}


def resolve(condition, rec):
    """Return (claim_text, grade_label) for `condition` over a raw SciVer record.

    Raises KeyError on an unknown condition, ValueError on an empty statement.
    """
    field, label = CONDITIONS[condition]
    claim_text = rec[field]
    if not claim_text or not claim_text.strip():
        raise ValueError(f"empty {field} for condition {condition!r}")
    return claim_text, label
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sciver_eval_conditions.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add sciver_eval/conditions.py tests/test_sciver_eval_conditions.py
git commit -m "feat(sciver_eval): conditions module (entailed/refuted -> statement + grade label)"
```

---

### Task 2: `build_prompt` takes the shown claim explicitly (verdict-clean)

**Files:**
- Modify: `sciver_eval/prompts.py:94-99` (the `build_prompt` function)
- Test: `tests/test_sciver_eval_prompts.py`

The current signature is `build_prompt(run_no, item, rec, trial)` and it hardcodes `rec["claim"]`. We add an optional `claim` override so the conditions can swap in `origin_statement` / `perturbed_statement` while the native run keeps `rec["claim"]`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sciver_eval_prompts.py
import json

from rubric_scoring.items import Item
from sciver_eval import prompts


def _item():
    return Item(item_id="sciver_val_0", source="sciver", paperid="p", claim_type="direct",
                vtype="chart", image_path="/tmp/fig.png", claim="OFFICIAL", label=True)


def _rec():
    return {
        "claim_type": "direct",
        "claim": "OFFICIAL CLAIM",
        "origin_statement": "THE TRUE STATEMENT XYZ",
        "perturbed_statement": "THE FALSE STATEMENT QRS",
        "perturbed_explanation": "SECRET LEAK REASON",
        "section": [],
        "item": "4",
        "paper_path": "anything.json",
    }


def _patch_paper(monkeypatch, tmp_path):
    paper = {"sections": [], "image_paths": {"4": {"caption": "CAPTION TEXT"}}}
    p = tmp_path / "paper.json"
    p.write_text(json.dumps(paper))
    monkeypatch.setattr(prompts, "_paper_path", lambda rec: p)


def test_default_shows_official_claim(monkeypatch, tmp_path):
    _patch_paper(monkeypatch, tmp_path)
    out = prompts.build_prompt(2, _item(), _rec(), 1)
    assert "OFFICIAL CLAIM" in out


def test_claim_override_swaps_statement(monkeypatch, tmp_path):
    _patch_paper(monkeypatch, tmp_path)
    out = prompts.build_prompt(3, _item(), _rec(), 1, claim="THE FALSE STATEMENT QRS")
    assert "THE FALSE STATEMENT QRS" in out
    assert "THE TRUE STATEMENT XYZ" not in out      # only the chosen statement appears
    assert "OFFICIAL CLAIM" not in out


def test_verdict_clean_never_leaks_explanation(monkeypatch, tmp_path):
    _patch_paper(monkeypatch, tmp_path)
    out = prompts.build_prompt(2, _item(), _rec(), 1, claim="THE TRUE STATEMENT XYZ")
    assert "SECRET LEAK REASON" not in out
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sciver_eval_prompts.py -v`
Expected: FAIL — `test_claim_override_swaps_statement` raises `TypeError: build_prompt() got an unexpected keyword argument 'claim'`

- [ ] **Step 3: Edit `build_prompt`**

Replace `sciver_eval/prompts.py:94-99` with:

```python
def build_prompt(run_no, item, rec, trial, claim=None):
    """Build the CoT prompt. `claim` overrides which statement is shown (used by
    the entailed/refuted conditions); defaults to the official `rec['claim']`
    (native run). Verdict-clean: never interpolates label or perturbed_explanation.
    """
    claim_text = rec["claim"] if claim is None else claim
    paper = json.loads(_paper_path(rec).read_text())
    cot = _TPL[rec["claim_type"]].substitute(
        claim=claim_text, context=_context(paper, rec["section"]),
        caption=_caption(paper, rec))
    return _WRAP.format(tag=tag(run_no, item.item_id, trial), cot=cot, figs=item.image_path)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sciver_eval_prompts.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add sciver_eval/prompts.py tests/test_sciver_eval_prompts.py
git commit -m "feat(sciver_eval): build_prompt accepts explicit claim (condition statement swap)"
```

---

### Task 3: `prepare --condition` writes condition-aware manifests

**Files:**
- Modify: `sciver_eval/prepare.py` (the `prepare` function + argparse)
- Test: `tests/test_sciver_eval_prepare.py`

`prepare` must: for each cell, when `condition` is set, resolve `(claim_text, grade_label)` via `conditions.resolve`, pass `claim_text` to `build_prompt`, and write `label=grade_label` plus `condition`/`claim` provenance into the manifest. The run registration records the condition. This test uses the real dataset (skips if absent) and a temp DB so it never touches `predictions.db`.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sciver_eval_prepare.py
import json

import pytest

from rubric_scoring import config
from sciver_eval import db, prepare


def _data_present():
    try:
        return (config.sciver_dir() / "valset.json").exists()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _data_present(), reason="SciVer dataset not present")


def test_refuted_manifest_has_label_0_and_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "default_db_path", lambda: tmp_path / "t.db")
    out = tmp_path / "run_03"
    mpath = prepare.prepare(3, str(out), limit=2, trials=1, condition="refuted")
    manifest = json.loads(open(mpath).read())
    assert len(manifest) == 2
    for cell in manifest.values():
        assert cell["label"] == 0
        assert cell["condition"] == "refuted"
        assert cell["claim"]            # the shown (perturbed) statement, non-empty


def test_entailed_manifest_has_label_1(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "default_db_path", lambda: tmp_path / "t.db")
    out = tmp_path / "run_02"
    mpath = prepare.prepare(2, str(out), limit=2, trials=1, condition="entailed")
    manifest = json.loads(open(mpath).read())
    assert all(c["label"] == 1 and c["condition"] == "entailed" for c in manifest.values())
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sciver_eval_prepare.py -v`
Expected: FAIL — `prepare()` has no `condition` parameter → `TypeError: prepare() got an unexpected keyword argument 'condition'`

- [ ] **Step 3: Edit `prepare.py`**

Add the import near the top (with the other `sciver_eval` import):

```python
from sciver_eval import db, prompts, conditions
```

Replace the `prepare(...)` signature and body (`sciver_eval/prepare.py:18-62`) with:

```python
def prepare(run_no, out_dir, limit=None, trials=1, condition=None):
    conn = db.connect(db.default_db_path())
    db.init(conn)

    items = items_mod.load_items()
    if limit is not None:
        items = items[:limit]
    for it in items:
        db.upsert_item(conn, it)
    conn.commit()

    cond_tag = condition or "native"
    run = conn.execute("SELECT model FROM run WHERE run_id=?", (run_no,)).fetchone()
    if run is None:
        db.register_run(conn, run_no, skill_version="sciver_eval-0.1", model=DEFAULT_MODEL,
                        prompt_version="sciver-cot-v1", temperature=1.0,
                        item_source=f"SciVer charts (val+test) [= rubric 817]; condition={cond_tag}",
                        n_items=len(items_mod.load_items()),
                        params=f"harness-subagent; CoT; max_tokens=10240; condition={cond_tag}")
        model = DEFAULT_MODEL
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
            if condition is None:
                claim_text, grade_label = rec["claim"], (1 if it.label else 0)
            else:
                claim_text, grade_label = conditions.resolve(condition, rec)
            t = prompts.tag(run_no, it.item_id, tr)
            path = os.path.join(out_dir, f"{t}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(prompts.build_prompt(run_no, it, rec, tr, claim=claim_text))
            manifest[t] = {"item_id": it.item_id, "trial": tr, "prompt_file": path,
                           "images": [it.image_path], "label": grade_label,
                           "condition": cond_tag, "claim": claim_text}

    mpath = os.path.join(out_dir, f"run_{run_no:02d}_manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"run {run_no} ({model}, condition={cond_tag}): {len(items)} items x {trials} "
          f"trials; {len(manifest)} MISSING cells -> {out_dir}; manifest {mpath}")
    return mpath
```

Replace the argparse block (`sciver_eval/prepare.py:65-73`) with:

```python
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", dest="run_no", type=int, default=1)
    ap.add_argument("--trials", type=int, default=1, help="replicates per item (resume-aware)")
    ap.add_argument("--condition", choices=["entailed", "refuted"], default=None,
                    help="swap claim to origin/perturbed statement; omit for the native run")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    out = a.out or str(config.repo_root() / "sciver_eval" / "prompts" / f"run_{a.run_no:02d}")
    prepare(a.run_no, out, a.limit, a.trials, a.condition)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sciver_eval_prepare.py -v`
Expected: PASS (2 passed) — or SKIPPED if the dataset is not present.

- [ ] **Step 5: Commit**

```bash
git add sciver_eval/prepare.py tests/test_sciver_eval_prepare.py
git commit -m "feat(sciver_eval): prepare --condition writes condition-aware manifests"
```

---

### Task 4: `shard.py` — reproducible manifest sharding

**Files:**
- Create: `sciver_eval/shard.py`
- Test: `tests/test_sciver_eval_shard.py`

Run-1's `cells_shardNN.json` + `wf_groups.json` were produced ad-hoc with no committed script. This makes that step reproducible and deterministic for runs 2/3.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sciver_eval_shard.py
import json

from sciver_eval import shard


def _manifest(n):
    return {f"tag{i}": {"item_id": f"sciver_val_{i}", "trial": 1} for i in range(n)}


def test_cells_from_manifest_sorted():
    cells = shard.cells_from_manifest(_manifest(3))
    assert cells == [
        {"id": "sciver_val_0", "trial": 1},
        {"id": "sciver_val_1", "trial": 1},
        {"id": "sciver_val_2", "trial": 1},
    ]


def test_shard_round_robin_balanced():
    cells = [{"id": f"x{i}", "trial": 1} for i in range(5)]
    shards = shard.shard(cells, 2)
    assert [len(s) for s in shards] == [3, 2]


def test_group_round_robin():
    assert shard.group(["a", "b", "c", "d"], 4) == [["a"], ["b"], ["c"], ["d"]]
    assert shard.group(["a", "b", "c"], 2) == [["a", "c"], ["b"]]


def test_write_layout_drops_empty_shards_and_preserves_cells(tmp_path):
    gpath, paths = shard.write_layout(str(tmp_path), _manifest(3), n_shards=16, n_groups=4)
    assert len(paths) == 3                       # 16 requested, only 3 non-empty
    total = sum(len(json.load(open(p))["cells"]) for p in paths)
    assert total == 3
    groups = json.load(open(gpath))
    assert sorted(p for g in groups for p in g) == sorted(paths)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sciver_eval_shard.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sciver_eval.shard'`

- [ ] **Step 3: Write minimal implementation**

```python
# sciver_eval/shard.py
"""Split a run manifest into cell-shards + workflow groups for wf_dispatch.js.

Reproduces the run-1 dispatch layout deterministically (the original step was
ad-hoc and uncommitted): reads a prepare manifest, emits cells_shardKK.json
({"cells":[{id,trial}]}) and wf_groups.json (list of groups, each a list of
shard paths). Empty shards are dropped so no agent is spawned for zero cells.

    python -m sciver_eval.shard --manifest <run_NN_manifest.json> [--shards 16 --groups 4]
"""
import argparse
import json
import os
from pathlib import Path


def cells_from_manifest(manifest):
    """[{'id':item_id,'trial':trial}] from a prepare manifest, sorted for determinism."""
    cells = [{"id": v["item_id"], "trial": int(v["trial"])} for v in manifest.values()]
    cells.sort(key=lambda c: (c["id"], c["trial"]))
    return cells


def shard(cells, n_shards):
    """Round-robin cells into n_shards lists (balanced, deterministic)."""
    shards = [[] for _ in range(n_shards)]
    for i, c in enumerate(cells):
        shards[i % n_shards].append(c)
    return shards


def group(shard_paths, n_groups):
    """Round-robin shard paths into n_groups non-empty groups (parallel workflows)."""
    groups = [[] for _ in range(n_groups)]
    for i, p in enumerate(shard_paths):
        groups[i % n_groups].append(p)
    return [g for g in groups if g]


def write_layout(out_dir, manifest, n_shards=16, n_groups=4):
    cells = cells_from_manifest(manifest)
    non_empty = [s for s in shard(cells, n_shards) if s]
    paths = []
    for k, s in enumerate(non_empty):
        p = os.path.join(out_dir, f"cells_shard{k:02d}.json")
        with open(p, "w") as f:
            json.dump({"cells": s}, f)
        paths.append(p)
    groups = group(paths, n_groups)
    gpath = os.path.join(out_dir, "wf_groups.json")
    with open(gpath, "w") as f:
        json.dump(groups, f)
    return gpath, paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--shards", type=int, default=16)
    ap.add_argument("--groups", type=int, default=4)
    a = ap.parse_args()
    manifest = json.load(open(a.manifest))
    out_dir = str(Path(a.manifest).parent)
    gpath, paths = write_layout(out_dir, manifest, a.shards, a.groups)
    total = sum(len(json.load(open(p))["cells"]) for p in paths)
    print(f"{total} cells -> {len(paths)} shards; groups -> {gpath}")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sciver_eval_shard.py -v`
Expected: PASS (4 passed)

- [ ] **Step 5: Commit**

```bash
git add sciver_eval/shard.py tests/test_sciver_eval_shard.py
git commit -m "feat(sciver_eval): reproducible manifest sharding (shard.py)"
```

---

### Task 5: parameterize `wf_dispatch.js` by run number

**Files:**
- Modify: `sciver_eval/wf_dispatch.js` (the hardcoded `se_r01__` and a new `RUN` binding)

The dispatcher hardcodes `se_r01__${c.id}` (run-1 only). Runs 2/3 write `se_r02__`/`se_r03__` prompt files (via `prompts.tag`), so the dispatcher must read the run number from `args`.

- [ ] **Step 1: Add a `RUN` binding**

In `sciver_eval/wf_dispatch.js`, immediately after the line `const DIR = A.dir`, add:

```javascript
const RUN = A.run || 1
```

- [ ] **Step 2: Use it in the prompt-file path**

In the `phase('Verify')` block, change the file path in the agent prompt from:

```javascript
  `Read the file ${DIR}/se_r01__${c.id}__t${t2(c.trial)}.txt and follow it EXACTLY. ` +
```

to:

```javascript
  `Read the file ${DIR}/se_r${t2(RUN)}__${c.id}__t${t2(c.trial)}.txt and follow it EXACTLY. ` +
```

- [ ] **Step 3: Verify the edit (no JS test harness in this repo)**

Run: `grep -n "se_r" sciver_eval/wf_dispatch.js`
Expected: exactly one match, reading `se_r${t2(RUN)}__${c.id}__t${t2(c.trial)}.txt` — and **no** remaining `se_r01__` literal.

Run: `grep -n "const RUN" sciver_eval/wf_dispatch.js`
Expected: one match `const RUN = A.run || 1` (back-compat: omitting `run` still targets run 1).

- [ ] **Step 4: Commit**

```bash
git add sciver_eval/wf_dispatch.js
git commit -m "fix(sciver_eval): wf_dispatch.js reads run number from args (was hardcoded r01)"
```

---

### Task 6: `analyze_paired.py` — paired metrics (gap, McNemar, d′/criterion)

**Files:**
- Create: `sciver_eval/analyze_paired.py`
- Test: `tests/test_sciver_eval_analyze.py`

Pure stat functions are unit-tested; the DB driver is tested against a seeded in-memory sqlite (hermetic). Uses `statistics.NormalDist` — no scipy dependency.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_sciver_eval_analyze.py
import math
import sqlite3

from sciver_eval import analyze_paired as ap


def test_dprime_criterion_known_values():
    # hit=0.69, fa=0.11  ->  d' = z(.69) - z(.11);  c = -0.5*(z(.69)+z(.11))
    d, c = ap.dprime_criterion(0.69, 0.11)
    assert math.isclose(d, 1.7227, abs_tol=1e-3)
    assert math.isclose(c, 0.3655, abs_tol=1e-3)


def test_dprime_clamps_perfect_rates():
    # hit=1.0, fa=0.0 must not blow up to infinity
    d, c = ap.dprime_criterion(1.0, 0.0, n_signal=100, n_noise=100)
    assert math.isfinite(d) and d > 0


def test_mcnemar_symmetric_is_zero():
    stat, _ = ap.mcnemar(10, 10)
    assert math.isclose(stat, 0.0, abs_tol=1e-9)


def test_mcnemar_continuity_correction():
    # (|b-c|-1)^2 / (b+c) = (|20-10|-1)^2 / 30 = 81/30 = 2.7
    stat, _ = ap.mcnemar(20, 10)
    assert math.isclose(stat, 2.7, abs_tol=1e-9)


def _seed(conn):
    conn.executescript("""
      CREATE TABLE prediction(run INTEGER, item_id TEXT, model TEXT, trial INTEGER,
                              predicted INTEGER, correct INTEGER, parse_ok INTEGER);
    """)
    rows = [
        # entailed run (2): item A correct, item B wrong
        (2, "A", "m", 1, 1, 1, 1), (2, "B", "m", 1, 0, 0, 1),
        # refuted run (3): item A correct, item B correct
        (3, "A", "m", 1, 0, 1, 1), (3, "B", "m", 1, 0, 1, 1),
    ]
    conn.executemany("INSERT INTO prediction VALUES (?,?,?,?,?,?,?)", rows)
    conn.commit()


def test_paired_metrics_driver():
    conn = sqlite3.connect(":memory:")
    _seed(conn)
    m = ap.paired_metrics(conn, run_ent=2, run_ref=3, model="m")
    assert m["n_items"] == 2
    assert math.isclose(m["acc_entailed"], 0.5, abs_tol=1e-9)   # A right, B wrong
    assert math.isclose(m["acc_refuted"], 1.0, abs_tol=1e-9)    # both right
    assert math.isclose(m["gap"], -0.5, abs_tol=1e-9)           # ent - ref
    # discordant pairs: B is refuted-only-correct -> c=1, b=0
    assert m["mcnemar_b"] == 0 and m["mcnemar_c"] == 1
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_sciver_eval_analyze.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'sciver_eval.analyze_paired'`

- [ ] **Step 3: Write minimal implementation**

```python
# sciver_eval/analyze_paired.py
"""Paired entailed-vs-refuted analysis: join run 2 (entailed) and run 3 (refuted)
on item_id and report the accuracy gap, McNemar on discordant pairs, and pooled
signal-detection d'/criterion. Pure stats use stdlib statistics.NormalDist.

    python -m sciver_eval.analyze_paired --ent 2 --ref 3
"""
import argparse
from statistics import NormalDist

from sciver_eval import db

_Z = NormalDist().inv_cdf


def dprime_criterion(hit, fa, n_signal=None, n_noise=None):
    """Signal-detection d' and criterion c. Clamps 0/1 rates (log-linear style)
    using the per-class N when given, else a small epsilon, so z stays finite."""
    def clamp(p, n):
        eps = (1.0 / (2 * n)) if n else 1e-3
        return min(1 - eps, max(eps, p))
    zh, zf = _Z(clamp(hit, n_signal)), _Z(clamp(fa, n_noise))
    return zh - zf, -0.5 * (zh + zf)


def mcnemar(b, c):
    """McNemar chi-square with continuity correction: max(0,|b-c|-1)^2/(b+c).
    Clamping the corrected difference at 0 makes a symmetric table (b==c) score
    exactly 0. Returns (statistic, dof); b,c are the discordant-pair counts."""
    n = b + c
    if n == 0:
        return 0.0, 1
    corr = max(0.0, abs(b - c) - 1)
    return corr * corr / n, 1


def _per_item_correct(conn, run, model):
    """item_id -> mean(correct) over parsed trials for one run."""
    rows = conn.execute(
        "SELECT item_id, AVG(correct) FROM prediction "
        "WHERE run=? AND model=? AND parse_ok=1 GROUP BY item_id", (run, model)).fetchall()
    return {i: a for i, a in rows}


def paired_metrics(conn, run_ent, run_ref, model):
    ent = _per_item_correct(conn, run_ent, model)
    ref = _per_item_correct(conn, run_ref, model)
    shared = sorted(set(ent) & set(ref))
    acc_e = sum(ent[i] for i in shared) / len(shared) if shared else 0.0
    acc_r = sum(ref[i] for i in shared) / len(shared) if shared else 0.0
    # per-item binary (majority-correct over trials) for discordant pairs
    be = {i: ent[i] >= 0.5 for i in shared}
    br = {i: ref[i] >= 0.5 for i in shared}
    b = sum(1 for i in shared if be[i] and not br[i])   # entailed-only correct
    c = sum(1 for i in shared if br[i] and not be[i])   # refuted-only correct
    stat, dof = mcnemar(b, c)
    hit = acc_e                                         # P(yes | entailed) = entailed accuracy
    fa = 1 - acc_r                                      # P(yes | refuted)  = 1 - refuted accuracy
    dprime, crit = dprime_criterion(hit, fa, n_signal=len(shared), n_noise=len(shared))
    return {"n_items": len(shared), "acc_entailed": acc_e, "acc_refuted": acc_r,
            "gap": acc_e - acc_r, "mcnemar_b": b, "mcnemar_c": c,
            "mcnemar_stat": stat, "mcnemar_dof": dof, "dprime": dprime, "criterion": crit}


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--ent", type=int, default=2)
    ap.add_argument("--ref", type=int, default=3)
    ap.add_argument("--model", default="claude-haiku-4-5")
    a = ap.parse_args()
    conn = db.connect(db.default_db_path())
    m = paired_metrics(conn, a.ent, a.ref, a.model)
    print(f"n={m['n_items']}  acc_entailed={m['acc_entailed']:.3f}  "
          f"acc_refuted={m['acc_refuted']:.3f}  gap={m['gap']:+.3f}")
    print(f"McNemar b={m['mcnemar_b']} c={m['mcnemar_c']} chi2={m['mcnemar_stat']:.3f}")
    print(f"SDT  d'={m['dprime']:.3f}  criterion c={m['criterion']:+.3f}  "
          f"(c>0 = skeptic bias toward 'no')")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_sciver_eval_analyze.py -v`
Expected: PASS (5 passed)

- [ ] **Step 5: Commit**

```bash
git add sciver_eval/analyze_paired.py tests/test_sciver_eval_analyze.py
git commit -m "feat(sciver_eval): paired analysis (gap, McNemar, d'/criterion)"
```

---

### Task 7: Pilot — full 817-item pass at 1 trial, both conditions (subagent-driven Workflow dispatch)

**Files:** none created — this is an **execution checklist** (a sequence of commands + Workflow dispatches to *run*, not code to author). It runs the new code end-to-end on **all 817 items at 1 trial per condition**, dispatched the same way run-1 was. "Pilot" = full breadth, shallow depth (1 trial) — Task 8 then deepens to 5 trials. A zero-cost pre-flight (Step 3) catches any label-wiring error *before* any subagent is spent.

- [ ] **Step 1: Full test sweep is green**

Run: `pytest tests/test_sciver_eval_conditions.py tests/test_sciver_eval_prompts.py tests/test_sciver_eval_prepare.py tests/test_sciver_eval_shard.py tests/test_sciver_eval_analyze.py -v`
Expected: all PASS (prepare tests may SKIP if dataset absent).

- [ ] **Step 2: Prepare the full pass for both conditions at 1 trial**

```bash
python -m sciver_eval.prepare --run 2 --condition entailed --trials 1
python -m sciver_eval.prepare --run 3 --condition refuted  --trials 1
```
Expected: each prints `~817 MISSING cells`; writes `sciver_eval/prompts/run_02/` and `run_03/` with `se_r02__*.txt` / `se_r03__*.txt` and a manifest.

- [ ] **Step 3: Pre-flight — inspect manifest labels + one prompt (NO dispatch, grading-direction gate)**

Open `sciver_eval/prompts/run_02/run_02_manifest.json`: every cell must have `"label": 1`, `"condition": "entailed"`, and `"claim"` equal to that item's `origin_statement`. Open `run_03`'s manifest: every cell `"label": 0`, `"condition": "refuted"`, `"claim"` equal to `perturbed_statement`. Read one `.txt` prompt per run and confirm it contains the chosen statement and **not** the `perturbed_explanation`. If any of these is wrong, STOP — fix the condition/label wiring before spending dispatch.

- [ ] **Step 4: Shard each run (16 shards, 4 groups — match run-1 layout)**

```bash
python -m sciver_eval.shard --manifest sciver_eval/prompts/run_02/run_02_manifest.json --shards 16 --groups 4
python -m sciver_eval.shard --manifest sciver_eval/prompts/run_03/run_03_manifest.json --shards 16 --groups 4
```
Expected: each prints `~817 cells -> 16 shards`.

- [ ] **Step 5: Dispatch via background Workflows (the subagent-driven step)**

For each run, read its `wf_groups.json` (4 groups). Launch **each group** as a background Workflow with `scriptPath: "sciver_eval/wf_dispatch.js"` and `args` `{run: <2 or 3>, dir: "sciver_eval/prompts/run_0X", shardFiles: [<the shard paths in that group>]}`. `wf_dispatch.js` fans one Haiku subagent per `(item, trial)` cell: it reads the `.txt` prompt, views the figure with the Read tool, reasons, and answers `Answer: yes`/`Answer: no`. Run 2 and run 3 can dispatch concurrently (8 background workflows total, as in run-1).

- [ ] **Step 6: Collect both runs**

```bash
python -m sciver_eval.collect_cli --run 2 --manifest sciver_eval/prompts/run_02/run_02_manifest.json
python -m sciver_eval.collect_cli --run 3 --manifest sciver_eval/prompts/run_03/run_03_manifest.json
```
Expected: `collected N predictions ...` for each (some cells may need a dispatch retry if a subagent didn't emit a parseable `Answer:` line).

- [ ] **Step 7: Confirm grading direction in the DB (safety gate)**

```bash
python -c "import sqlite3; c=sqlite3.connect('sciver_eval/predictions.db'); \
print('run2 entailed:', c.execute('SELECT predicted,correct,COUNT(*) FROM prediction WHERE run=2 GROUP BY predicted,correct').fetchall()); \
print('run3 refuted:', c.execute('SELECT predicted,correct,COUNT(*) FROM prediction WHERE run=3 GROUP BY predicted,correct').fetchall())"
```
Expected: in run 2, `predicted=1` rows have `correct=1` (a "yes" on an entailed claim is right); in run 3, `predicted=0` rows have `correct=1` (a "no" on a refuted claim is right). If inverted, STOP — the condition/label wiring is wrong.

- [ ] **Step 8: Resume loop until complete**

```bash
# re-prepare emits ONLY cells still missing a parse_ok=1 prediction
python -m sciver_eval.prepare --run 2 --condition entailed --trials 1
python -m sciver_eval.prepare --run 3 --condition refuted  --trials 1
```
If either reports `>0 MISSING cells`, re-shard (Step 4) + re-dispatch (Step 5) + collect (Step 6) those stragglers. Repeat until both report `0 MISSING cells`.

- [ ] **Step 9: Directional read + commit**

```bash
python -m sciver_eval.analyze_paired --ent 2 --ref 3 --model claude-haiku-4-5
git add sciver_eval/prompts/run_02 sciver_eval/prompts/run_03 sciver_eval/predictions.db
git commit -m "feat(sciver_eval): full entailed (run2) + refuted (run3) pilot, 817x1 trial"
```
The 1-trial gap/criterion is directional only (no per-item flip signal yet) — that arrives with Task 8's 5 trials.

---

### Task 8: Scale both conditions to 5 trials + paired report

**Files:** none created — execution checklist. Resume-aware, so re-running `prepare` at a higher `--trials` only emits the *new* trials (Task 7's trial-1 cells are kept).

- [ ] **Step 1: Re-prepare at 5 trials (resume-aware)**

```bash
python -m sciver_eval.prepare --run 2 --condition entailed --trials 5
python -m sciver_eval.prepare --run 3 --condition refuted  --trials 5
```
Expected: up to `~3268 MISSING cells` each (817 × 4 new trials; trial 1 already done in Task 7).

- [ ] **Step 2: Shard, dispatch, collect, resume loop**

Repeat Task 7 Steps 4→8 (shard 16/4 → launch 4 background Workflows per run → `collect_cli` → re-`prepare --trials 5`) until both runs report `0 MISSING cells`.

- [ ] **Step 3: Run the paired analysis**

```bash
python -m sciver_eval.analyze_paired --ent 2 --ref 3 --model claude-haiku-4-5
```
Expected: prints `acc_entailed`, `acc_refuted`, `gap`, McNemar `b/c/chi2`, and `d'`/`criterion`. Cross-check against run-1's recall split (refuted 0.89 / entailed 0.46) — a positive criterion `c` confirms the skeptic bias under the controlled paired design.

- [ ] **Step 4: Commit the completed runs + results**

```bash
git add sciver_eval/prompts/run_02 sciver_eval/prompts/run_03 sciver_eval/predictions.db
git commit -m "feat(sciver_eval): full entailed (run2) + refuted (run3) sweeps, 817x5; paired metrics"
```

---

## Notes for the implementer

- **DB safety:** `predictions.db` is a real artifact under version control. Unit tests monkeypatch `db.default_db_path` to a temp file (Task 3) or use `:memory:` (Task 6) and never touch it. Only the Task 7/8 runbooks write the real DB.
- **Grading invariant:** never changed `scrape.collect` — it already grades against `meta["label"]` (`sciver_eval/scrape.py:88`). Correctness of the new runs depends entirely on `prepare` writing the right per-condition `label`, which Task 3's test and Task 7 Step 7 both verify.
- **Trial depth is free:** every `prepare` is resume-aware; "pilot now, 5 later, 10 later" needs no code change — just re-run `prepare`/`shard`/dispatch/`collect`.
- **Demand-join extension (deferred):** `predictions.db ⋈ rubric_scoring/annotations_prod.db` on `item_id` to correlate the per-item entailed/refuted gap with DeLeAn demand scores. Out of scope for this plan; the `item_id`s remain byte-identical, so it is a pure follow-on query.
