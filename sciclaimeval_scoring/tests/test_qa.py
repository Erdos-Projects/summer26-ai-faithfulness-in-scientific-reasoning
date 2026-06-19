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
