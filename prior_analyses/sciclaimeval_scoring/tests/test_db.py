from sciclaimeval_scoring import db
from sciclaimeval_scoring.items import Item

def _item(i="scev_val_fig_0001"):
    return Item(i, "sciclaimeval", "2403.19137", "", "figure",
                "/x/figures/dev/val_fig_0001.png", "a claim", False, "ml", "Legend Swap")

def test_item_roundtrip_includes_domain_and_operation():
    conn = db.connect(":memory:"); db.init(conn)
    db.upsert_item(conn, _item())
    row = conn.execute("SELECT domain, operation, label FROM item WHERE item_id=?",
                       ("scev_val_fig_0001",)).fetchone()
    assert row == ("ml", "Legend Swap", 0)

def test_completed_cells_tracks_parse_ok():
    conn = db.connect(":memory:"); db.init(conn)
    db.register_pass(conn, 1, skill_version="0.1", model="m", active_dims="AS",
                     item_source="t", n_items=1, params="{}", deviations="")
    db.upsert_annotation(conn, {"pass": 1, "item_id": "scev_val_fig_0001",
                                "dim_code": "AS", "dim_name": "Attention and Scan",
                                "score": 3, "parse_ok": 1})
    assert ("scev_val_fig_0001", "AS") in db.completed_cells(conn, 1)

def test_upsert_annotation_is_idempotent():
    conn = db.connect(":memory:"); db.init(conn)
    for s in (2, 4):
        db.upsert_annotation(conn, {"pass": 1, "item_id": "i", "dim_code": "AS",
                                    "dim_name": "n", "score": s, "parse_ok": 1})
    n = conn.execute("SELECT COUNT(*) FROM annotation").fetchone()[0]
    last = conn.execute("SELECT score FROM annotation").fetchone()[0]
    assert (n, last) == (1, 4)
