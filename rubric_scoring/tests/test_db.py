from rubric_scoring import db

def _row(pass_no=1, item="sciver_val_0", dim="AS", score=2, parse_ok=1):
    return {
        "pass": pass_no, "session_id": "sess", "item_id": item, "dim_code": dim,
        "dim_name": "Attention and Scan", "agent_id": "a1", "raw_output": "...is: 2",
        "score": score, "parse_ok": parse_ok, "input_tokens": 5,
        "cache_creation_tokens": 10, "cache_read_tokens": 20, "output_tokens": 100,
        "total_input_tokens": 35, "output_thinking_tokens": 0, "wall_clock_s": 19.5,
    }

def test_init_and_upsert_then_unique(tmp_path):
    conn = db.connect(tmp_path / "t.db"); db.init(conn)
    db.upsert_annotation(conn, _row(score=2))
    db.upsert_annotation(conn, _row(score=4))  # same (pass,item,dim) -> overwrite, not duplicate
    assert conn.execute("SELECT COUNT(*) FROM annotation").fetchone()[0] == 1
    assert conn.execute("SELECT score FROM annotation").fetchone()[0] == 4

def test_completed_cells_only_parse_ok(tmp_path):
    conn = db.connect(tmp_path / "t.db"); db.init(conn)
    db.upsert_annotation(conn, _row(item="sciver_val_0", dim="AS", parse_ok=1))
    db.upsert_annotation(conn, _row(item="sciver_val_0", dim="QLq", parse_ok=0))
    done = db.completed_cells(conn, 1)
    assert ("sciver_val_0", "AS") in done
    assert ("sciver_val_0", "QLq") not in done

def test_register_pass(tmp_path):
    conn = db.connect(tmp_path / "t.db"); db.init(conn)
    db.register_pass(conn, 1, skill_version="0.1", model="claude-sonnet-4-6",
                     active_dims="AS,QLq", item_source="sciver charts", n_items=817,
                     params="{}", deviations="temp!=0")
    assert conn.execute("SELECT n_items FROM pass WHERE pass=1").fetchone()[0] == 817
