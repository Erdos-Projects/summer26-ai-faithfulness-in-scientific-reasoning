import json
from rubric_scoring import db
from rubric_scoring.scrape import scan_transcript, collect

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
