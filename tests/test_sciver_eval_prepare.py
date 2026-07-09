import json

import pytest

from rubric_scoring import config
from sciver_eval import db, prepare


def _data_present():
    try:
        return (config.sciver_dir() / "valset.json").exists()
    except Exception:
        return False


pytestmark = pytest.mark.skipif(not _data_present(), reason="SciVer dataset not present")


def test_refuted_manifest_has_label_0_and_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "default_db_path", lambda: tmp_path / "t.db")
    out = tmp_path / "run_03"
    mpath = prepare.prepare(3, str(out), limit=2, trials=1, condition="refuted")
    manifest = json.loads(open(mpath).read())
    assert len(manifest) == 2
    for cell in manifest.values():
        assert cell["label"] == 0
        assert cell["condition"] == "refuted"
        assert cell["claim"]            # the shown (perturbed) statement, non-empty


def test_entailed_manifest_has_label_1(tmp_path, monkeypatch):
    monkeypatch.setattr(db, "default_db_path", lambda: tmp_path / "t.db")
    out = tmp_path / "run_02"
    mpath = prepare.prepare(2, str(out), limit=2, trials=1, condition="entailed")
    manifest = json.loads(open(mpath).read())
    assert all(c["label"] == 1 and c["condition"] == "entailed" for c in manifest.values())
