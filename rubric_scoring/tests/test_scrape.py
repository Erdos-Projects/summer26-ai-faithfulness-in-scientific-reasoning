import json
from rubric_scoring import db
from rubric_scoring.scrape import scan_transcript, collect, extract_tags

def _make_transcript(path, tag, score=3, thinking=False):
    lines = []
    lines.append({"agentId":"agentX","timestamp":"2026-06-14T00:00:00.000Z",
                  "message":{"role":"user","content":f"do task {tag}"}})
    content = [{"type":"text","text":f"reasoning ... is: {score}"}]
    if thinking:
        content = [{"type":"thinking","thinking":"secret"}] + content
    lines.append({"agentId":"agentX","timestamp":"2026-06-14T00:00:05.000Z",
                  "message":{"role":"assistant","content":content,
                             "usage":{"input_tokens":5,"output_tokens":120,
                                      "cache_creation_input_tokens":10000,"cache_read_input_tokens":30000}}})
    path.write_text("\n".join(json.dumps(o) for o in lines))

def test_scan_transcript_extracts_usage_score_thinking(tmp_path):
    f = tmp_path / "agent-agentX.jsonl"
    _make_transcript(f, "rs_p01__sciver_val_0__AS", score=4)
    info = scan_transcript(f)
    assert info["score"] == 4
    assert info["output_tokens"] == 120
    assert info["total_input_tokens"] == 5 + 10000 + 30000
    assert info["output_thinking_tokens"] == 0
    assert info["agent_id"] == "agentX"

def test_collect_upserts_only_tagged_cells(tmp_path):
    proj = tmp_path / "projects" / "p" / "sess" / "subagents"
    proj.mkdir(parents=True)
    _make_transcript(proj / "agent-a1.jsonl", "rs_p01__sciver_val_0__AS", score=2)
    _make_transcript(proj / "agent-a2.jsonl", "rs_p01__sciver_val_0__QLq", score=5)
    conn = db.connect(tmp_path / "t.db"); db.init(conn)
    manifest = {"rs_p01__sciver_val_0__AS": {"item_id":"sciver_val_0","dim_code":"AS","dim_name":"Attention and Scan"},
                "rs_p01__sciver_val_0__QLq":{"item_id":"sciver_val_0","dim_code":"QLq","dim_name":"Quantitative Reasoning"}}
    n = collect(conn, 1, manifest, projects_dir=tmp_path / "projects", session_id="sess")
    assert n == 2
    assert conn.execute("SELECT score FROM annotation WHERE dim_code='QLq'").fetchone()[0] == 5
    assert db.completed_cells(conn, 1) == {("sciver_val_0","AS"),("sciver_val_0","QLq")}

def test_collect_scopes_to_session(tmp_path):
    a = tmp_path/"projects"/"p"/"sessA"/"subagents"; a.mkdir(parents=True)
    b = tmp_path/"projects"/"p"/"sessB"/"subagents"; b.mkdir(parents=True)
    _make_transcript(a/"agent-a.jsonl", "rs_p01__sciver_val_0__AS", score=2)
    _make_transcript(b/"agent-b.jsonl", "rs_p01__sciver_val_0__AS", score=5)  # same tag, WRONG session
    conn = db.connect(tmp_path/"t.db"); db.init(conn)
    manifest = {"rs_p01__sciver_val_0__AS": {"item_id":"sciver_val_0","dim_code":"AS","dim_name":"Attention and Scan"}}
    collect(conn, 1, manifest, projects_dir=tmp_path/"projects", session_id="sessA")
    row = conn.execute("SELECT score FROM annotation").fetchone()
    assert row is not None and row[0] == 2  # picked sessA's transcript, not sessB's

def test_extract_tags_finds_rubric_tag():
    txt = "PILOT_TAG: rs_p01__sciver_val_0__AS\nmore text rs_p01__sciver_val_0__AS again"
    assert extract_tags(txt) == ["rs_p01__sciver_val_0__AS"]
    assert extract_tags("no tags here") == []
    # underscores inside item_id must not break the split; lowercase codes allowed
    assert extract_tags("x rs_p02__sciver_test_137__QLq y") == ["rs_p02__sciver_test_137__QLq"]
    assert extract_tags(None) == []

def test_scan_transcript_returns_tags(tmp_path):
    f = tmp_path / "agent-agentX.jsonl"
    _make_transcript(f, "rs_p01__sciver_val_0__MCr", score=4)
    assert scan_transcript(f)["tags"] == ["rs_p01__sciver_val_0__MCr"]

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

def test_collect_same_run_prefers_scored_over_unscored(tmp_path):
    sub = _sub(tmp_path)
    # both files are NEW in the same collect run; the scored one must win irrespective of order
    _make_unscored(sub/"agent-a0_fail.jsonl", "rs_p01__sciver_val_0__AS")
    _make_transcript(sub/"agent-z9_ok.jsonl", "rs_p01__sciver_val_0__AS", score=3)
    conn = db.connect(tmp_path/"t.db"); db.init(conn)
    man = {"rs_p01__sciver_val_0__AS": {"item_id":"sciver_val_0","dim_code":"AS","dim_name":"AS"}}
    collect(conn, 1, man, projects_dir=tmp_path/"projects", session_id="sess")
    assert conn.execute("SELECT score,parse_ok FROM annotation").fetchone() == (3, 1)
