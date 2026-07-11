# Incremental transcript collection for rubric scoring

**Date:** 2026-06-18
**Status:** Approved (design)
**Scope:** `rubric_scoring/scrape.py`, `rubric_scoring/db.py` (collection step only)

## Problem

After a rubric-scoring pass, `collect()` (`rubric_scoring/scrape.py`) recovers
per-cell scores and token usage from subagent transcripts. Its current algorithm
re-screens the **entire transcript corpus** on every run:

1. Globs every `subagents/**/agent-*.jsonl` transcript.
2. Reads each matched file **fully into a RAM dict** (`contents[f] = fh.read()`).
3. For **every manifest tag**, substring-matches it against **every file's full
   text** — `O(tags × files)`.

Measured corpus at design time: **48,112 transcripts, 4.8 GB**. When
`CLAUDE_CODE_SESSION_ID` is unset (`session_id == "unknown"`) the glob is
unscoped and the whole 4.8 GB is loaded into memory and cross-scanned. Even when
session-scoped, all of that session's transcripts are re-read on every collect,
and cells already ingested in prior runs are re-scanned.

The manifest tag *set* is already incremental (`prepare.py:plan_missing` emits
only MISSING cells). The waste is entirely on the **file side**: re-reading and
re-scanning transcripts already ingested.

## Goals

- Stop re-reading transcripts already ingested in prior collect runs.
- Replace the `O(tags × files)` substring scan with `O(files + tags)`.
- Stop holding the whole corpus in RAM.
- **Byte-identical output**: same `annotation` rows, so `aggregate.csv` and all
  downstream analysis are unchanged.

Non-goals: changing the scoring/dispatch path, the manifest format, the
`annotation` schema, or aggregate output.

## Design

### Ingestion ledger (db.py)

New table:

```sql
CREATE TABLE IF NOT EXISTS ingested_transcript(
  path TEXT PRIMARY KEY, size INTEGER, mtime REAL, ingested TEXT);
```

Helpers:

- `ledger_seen(conn) -> dict[str, tuple[int, float]]` — `{path: (size, mtime)}`.
- `mark_ingested(conn, path, size, mtime)` — INSERT OR REPLACE with `_now()`.

**Decision A — ledger key is `(path, size, mtime)`, stat-only, no hashing.**
Each subagent attempt is its own immutable `agent-*.jsonl`; retries create *new*
files rather than appending. So `os.stat` is sufficient to detect "already
ingested and unchanged". Worst case (clock/mtime weirdness) is a harmless
re-read, because upserts are idempotent. Content-hashing was rejected: it
requires reading every file, defeating the purpose.

### Incremental collector (scrape.py)

```
seen = ledger_seen(conn)              # unless full=True
index = {}                            # tag -> scan_result
for entry in scandir(candidate transcripts):   # stat only, no read
    st = entry.stat()
    if not full and seen.get(entry.path) == (st.st_size, st.st_mtime):
        continue                      # incremental skip — the whole point
    scan = scan_transcript(entry.path)   # streaming, line-by-line
    for tag in scan.tags:             # tag(s) embedded in transcript text
        index[tag] = scan
    mark_ingested(conn, entry.path, st.st_size, st.st_mtime)
for tag, meta in manifest.items():
    scan = index.get(tag)             # O(1), was O(files)
    if scan: upsert_annotation(...)   # never-regress (Decision C)
```

- Candidate listing uses the same session-scoped vs unscoped glob logic as today.
- `scan_transcript` becomes streaming (it already accumulates usage per JSONL
  line) and additionally returns the tag(s) it found, via regex
  `rs_p\d+__\S+?__\S+` matched against the `PILOT_TAG:` line / prompt text.
- A tag appearing in multiple transcripts is still resolved
  "prefer the transcript that parses to a score" — within a run via the existing
  pick logic, across runs via Decision C.

**Decision B — escape hatch.** `collect(..., full=False)` plus a `--full` CLI
flag on `collect_cli.py`. When set, the ledger is ignored and every candidate is
rescanned (full rebuild). Cheap insurance if the ledger ever drifts.

**Decision C — never regress a scored cell.** Today `collect()` prefers a scored
transcript over an unscored one for the same tag regardless of order. With
incremental ingestion, files arrive across runs, so the upsert must refuse to
overwrite a `parse_ok=1` row with a `parse_ok=0` one. Guard added at the
collector (skip the upsert when the incoming score is `None` and a `parse_ok=1`
row already exists for that `(pass, item_id, dim_code)`). This preserves current
semantics; without it a late-arriving failed retry could clobber a good score.

## Cost

- **First collect after deploy:** cold ledger → ingests everything once, same
  cost as today, plus it populates the ledger.
- **Every subsequent collect:** stat-only over candidates, read only new/changed
  transcripts, O(1) tag resolution. The full re-screen is gone.

## Testing (TDD)

Fixture corpus = temp dir with hand-written `agent-*.jsonl` + a fake manifest.
Cases:

1. **Cold ingest** — empty ledger, all tags resolved, rows written, ledger
   populated.
2. **Incremental skip** — second collect with no new files reads zero files
   (assert via a read counter / monkeypatch) yet DB unchanged.
3. **New-file pickup** — adding a transcript for a previously-missing tag is
   ingested on next collect.
4. **Retry supersedes failure** — failed (no-score) transcript ingested first,
   then a later scored transcript for the same tag upgrades the row to
   `parse_ok=1`.
5. **Never-regress** — scored row already present; a later no-score transcript
   for the same tag does NOT clobber it.
6. **`--full` rebuild** — ignores ledger, rescans all.

**Regression:** run the new collector over the current corpus and diff the
resulting `annotation` rows / `aggregate.csv` against the committed
`annotations_prod.db` output — must be identical.
