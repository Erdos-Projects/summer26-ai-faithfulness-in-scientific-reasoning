from rubric_scoring import db
from rubric_scoring.qa import run_qa

def _ann(conn, p, item, dim, score, thinking=0, parse_ok=1):
    db.upsert_annotation(conn, {"pass":p, "session_id":"s", "item_id":item, "dim_code":dim,
        "dim_name":dim, "agent_id":"a", "raw_output":"r", "score":score, "parse_ok":parse_ok,
        "input_tokens":0,"cache_creation_tokens":0,"cache_read_tokens":0,"output_tokens":0,
        "total_input_tokens":0,"output_thinking_tokens":thinking,"wall_clock_s":1.0})

def test_qa_reports_parse_rate_and_thinking(tmp_path):
    conn = db.connect(tmp_path/"t.db"); db.init(conn)
    _ann(conn,1,"i0","AS",2); _ann(conn,1,"i1","AS",None,parse_ok=0)
    rep = run_qa(conn, 1)
    assert rep["n"] == 2 and rep["parse_rate"] == 0.5
    assert rep["thinking_tokens_total"] == 0 and rep["thinking_clean"] is True

def test_qa_flags_thinking(tmp_path):
    conn = db.connect(tmp_path/"t.db"); db.init(conn)
    _ann(conn,1,"i0","AS",2,thinking=500)
    assert run_qa(conn,1)["thinking_clean"] is False

def test_qa_per_dim_spread(tmp_path):
    conn = db.connect(tmp_path/"t.db"); db.init(conn)
    for i,s in enumerate([0,2,4,5]): _ann(conn,1,f"i{i}","QLq",s)
    rep = run_qa(conn,1)
    assert rep["per_dim"]["QLq"]["distinct_scores"] == 4
