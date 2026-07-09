import json
from rubric_scoring import db
from rubric_scoring.prepare import plan_missing
from rubric_scoring.items import Item

ITEMS = [Item(f"sciver_val_{i}","sciver","p","analytical","chart","/abs/x.png","claim",True) for i in range(3)]

def test_plan_missing_skips_completed(tmp_path):
    conn = db.connect(tmp_path / "t.db"); db.init(conn)
    db.upsert_annotation(conn, {"pass":1, "session_id":"s", "item_id":"sciver_val_0", "dim_code":"AS",
        "dim_name":"Attention and Scan", "agent_id":"a", "raw_output":"is: 2", "score":2, "parse_ok":1,
        "input_tokens":0,"cache_creation_tokens":0,"cache_read_tokens":0,"output_tokens":0,
        "total_input_tokens":0,"output_thinking_tokens":0,"wall_clock_s":1.0})
    missing = plan_missing(conn, 1, ITEMS, ["AS","QLq"])
    # 3 items x 2 dims = 6 cells, minus the 1 done = 5
    assert len(missing) == 5
    assert ("sciver_val_0","AS") not in [(m["item_id"], m["dim_code"]) for m in missing]
