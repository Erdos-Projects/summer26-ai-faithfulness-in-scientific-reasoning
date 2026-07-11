import json
from sciclaimeval_scoring import config, db, prepare

def test_prepare_writes_manifest_and_resumes(tmp_path, monkeypatch):
    dbfile = tmp_path / "t.db"
    monkeypatch.setenv("SCICLAIMEVAL_SCORING_DB", str(dbfile))
    out = tmp_path / "prompts"
    mpath = prepare.prepare(1, ["AS", "VO"], str(out), limit=3)
    manifest = json.loads(open(mpath).read())
    assert len(manifest) == 3 * 2                      # 3 items x 2 dims, all missing
    assert all(t.startswith("rsc_p01__") for t in manifest)
    # mark one cell done, re-prepare -> one fewer prompt
    conn = db.connect(str(dbfile))
    first = next(iter(manifest.values()))
    db.upsert_annotation(conn, {"pass": 1, "item_id": first["item_id"],
                                "dim_code": first["dim_code"], "dim_name": first["dim_name"],
                                "score": 3, "parse_ok": 1})
    mpath2 = prepare.prepare(1, ["AS", "VO"], str(out), limit=3)
    assert len(json.loads(open(mpath2).read())) == 3 * 2 - 1
