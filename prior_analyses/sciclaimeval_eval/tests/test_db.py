from sciclaimeval_eval import db
from sciclaimeval_scoring.items import Item

def _item():
    return Item("scev_val_fig_0001", "sciclaimeval", "p", "", "figure",
                "/x.png", "claim", True, "nlp", "Supported_claim_only")

def test_default_db_path_is_sciclaimeval_named():
    assert db.default_db_path().name == "predictions_sciclaimeval.db"

def test_item_roundtrip_with_domain_operation():
    conn = db.connect(":memory:"); db.init(conn)
    db.upsert_item(conn, _item()); conn.commit()
    assert conn.execute("SELECT domain, operation FROM item").fetchone() == ("nlp", "Supported_claim_only")

def test_prediction_completed_cells_resume_key():
    conn = db.connect(":memory:"); db.init(conn)
    db.upsert_prediction(conn, {"run": 1, "item_id": "i", "model": "claude-haiku-4-5",
                                "trial": 1, "predicted": 1, "parse_ok": 1, "correct": 1})
    assert ("i", 1) in db.completed_cells(conn, 1, "claude-haiku-4-5")
