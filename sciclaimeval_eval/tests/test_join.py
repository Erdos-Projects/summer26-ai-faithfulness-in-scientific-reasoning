"""The whole point: predictions ⋈ demand scores on item_id. Verify the keys align."""
from sciclaimeval_scoring import db as sdb
from sciclaimeval_eval import db as edb
from sciclaimeval_scoring.items import Item

def test_item_ids_join_across_dbs():
    it = Item("scev_val_fig_0001", "sciclaimeval", "p", "", "figure",
              "/x.png", "c", True, "nlp", "Legend Swap")
    s = sdb.connect(":memory:"); sdb.init(s); sdb.upsert_item(s, it)
    sdb.upsert_annotation(s, {"pass": 1, "item_id": it.item_id, "dim_code": "AS",
                              "dim_name": "n", "score": 3, "parse_ok": 1})
    e = edb.connect(":memory:"); edb.init(e); edb.upsert_item(e, it); e.commit()
    edb.upsert_prediction(e, {"run": 1, "item_id": it.item_id, "model": "claude-haiku-4-5",
                              "trial": 1, "predicted": 1, "parse_ok": 1, "correct": 0})
    score = s.execute("SELECT score FROM annotation WHERE item_id=?", (it.item_id,)).fetchone()[0]
    correct = e.execute("SELECT correct FROM prediction WHERE item_id=?", (it.item_id,)).fetchone()[0]
    assert (score, correct) == (3, 0)   # same item_id keys both rows
