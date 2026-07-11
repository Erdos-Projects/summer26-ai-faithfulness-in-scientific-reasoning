import json
from sciclaimeval_eval import prepare, db

def test_prepare_writes_manifest_with_labels_and_resumes(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "default_db_path", lambda: tmp_path / "p.db")
    out = tmp_path / "prompts"
    mpath = prepare.prepare(1, str(out), limit=3, trials=2)
    manifest = json.loads(open(mpath).read())
    assert len(manifest) == 3 * 2
    assert all(t.startswith("sce_r01__") for t in manifest)
    assert all(set(m) >= {"item_id", "trial", "label", "images"} for m in manifest.values())
    # mark one (item, trial) done -> one fewer prompt next prepare
    conn = db.connect(str(tmp_path / "p.db"))
    first = next(iter(manifest.values()))
    db.upsert_prediction(conn, {"run": 1, "item_id": first["item_id"],
                                "model": "claude-haiku-4-5", "trial": first["trial"],
                                "predicted": 1, "parse_ok": 1, "correct": 1})
    mpath2 = prepare.prepare(1, str(out), limit=3, trials=2)
    assert len(json.loads(open(mpath2).read())) == 3 * 2 - 1
