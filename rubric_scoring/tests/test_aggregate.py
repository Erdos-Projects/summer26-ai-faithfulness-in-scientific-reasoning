import csv
from rubric_scoring import db
from rubric_scoring.aggregate import aggregate, export_csv

def _ann(conn, p, item, dim, score):
    db.upsert_annotation(conn, {"pass":p, "session_id":"s", "item_id":item, "dim_code":dim,
        "dim_name":dim, "agent_id":"a", "raw_output":f"is: {score}", "score":score, "parse_ok":1,
        "input_tokens":0,"cache_creation_tokens":0,"cache_read_tokens":0,"output_tokens":0,
        "total_input_tokens":0,"output_thinking_tokens":0,"wall_clock_s":1.0})

def test_mode_and_mean_across_passes(tmp_path):
    conn = db.connect(tmp_path/"t.db"); db.init(conn)
    for p, s in [(1,2),(2,2),(3,1)]:
        _ann(conn, p, "sciver_val_0", "AS", s)
    rows = {(r["item_id"], r["dim_code"]): r for r in aggregate(conn, ["AS"])}
    r = rows[("sciver_val_0","AS")]
    assert r["mode"] == 2 and abs(r["mean"] - (5/3)) < 1e-6 and r["n_passes"] == 3

def test_export_csv(tmp_path):
    conn = db.connect(tmp_path/"t.db"); db.init(conn)
    _ann(conn, 1, "sciver_val_0", "AS", 3)
    out = tmp_path/"agg.csv"; export_csv(conn, ["AS"], out)
    rows = list(csv.DictReader(open(out)))
    assert rows[0]["item_id"] == "sciver_val_0" and rows[0]["mode"] == "3"
