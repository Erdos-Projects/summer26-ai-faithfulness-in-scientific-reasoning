import json

from rubric_scoring.items import Item
from sciver_eval import prompts


def _item():
    return Item(item_id="sciver_val_0", source="sciver", paperid="p", claim_type="direct",
                vtype="chart", image_path="/tmp/fig.png", claim="OFFICIAL", label=True)


def _rec():
    return {
        "claim_type": "direct",
        "claim": "OFFICIAL CLAIM",
        "origin_statement": "THE TRUE STATEMENT XYZ",
        "perturbed_statement": "THE FALSE STATEMENT QRS",
        "perturbed_explanation": "SECRET LEAK REASON",
        "section": [],
        "item": "4",
        "paper_path": "anything.json",
    }


def _patch_paper(monkeypatch, tmp_path):
    paper = {"sections": [], "image_paths": {"4": {"caption": "CAPTION TEXT"}}}
    p = tmp_path / "paper.json"
    p.write_text(json.dumps(paper))
    monkeypatch.setattr(prompts, "_paper_path", lambda rec: p)


def test_default_shows_official_claim(monkeypatch, tmp_path):
    _patch_paper(monkeypatch, tmp_path)
    out = prompts.build_prompt(2, _item(), _rec(), 1)
    assert "OFFICIAL CLAIM" in out


def test_claim_override_swaps_statement(monkeypatch, tmp_path):
    _patch_paper(monkeypatch, tmp_path)
    out = prompts.build_prompt(3, _item(), _rec(), 1, claim="THE FALSE STATEMENT QRS")
    assert "THE FALSE STATEMENT QRS" in out
    assert "THE TRUE STATEMENT XYZ" not in out      # only the chosen statement appears
    assert "OFFICIAL CLAIM" not in out


def test_verdict_clean_never_leaks_explanation(monkeypatch, tmp_path):
    _patch_paper(monkeypatch, tmp_path)
    out = prompts.build_prompt(2, _item(), _rec(), 1, claim="THE TRUE STATEMENT XYZ")
    assert "SECRET LEAK REASON" not in out
