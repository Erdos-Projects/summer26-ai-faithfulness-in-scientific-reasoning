from sciclaimeval_scoring import db, aggregate

def test_mode_and_mean_across_passes():
    conn = db.connect(":memory:"); db.init(conn)
    for p, s in ((1, 3), (2, 3), (3, 5)):
        db.upsert_annotation(conn, {"pass": p, "item_id": "i", "dim_code": "AS",
                                    "dim_name": "n", "score": s, "parse_ok": 1})
    rows = aggregate.aggregate(conn, ["AS"])
    assert rows[0]["mode"] == 3 and rows[0]["n_passes"] == 3
    assert rows[0]["min"] == 3 and rows[0]["max"] == 5
