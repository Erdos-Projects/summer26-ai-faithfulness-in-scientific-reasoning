from rubric_scoring.prompts import tag, build_prompt, DIMS
from rubric_scoring.items import Item

IT = Item("sciver_val_7","sciver","p1","analytical","chart","/abs/x.png",
          "Accuracy rose after 2020.", True)

def test_tag_format():
    assert tag(1, IT.item_id, "QLq") == "rs_p01__sciver_val_7__QLq"

def test_prompt_contains_rubric_claim_figure_and_tag():
    p = build_prompt(1, IT, "QLq", "QUANT RUBRIC TEXT")
    assert "QUANT RUBRIC TEXT" in p
    assert IT.claim in p and IT.image_path in p
    assert "rs_p01__sciver_val_7__QLq" in p
    assert "Quantitative Reasoning" in p

def test_prompt_is_verdict_blind():
    p = build_prompt(1, IT, "QLq", "R").lower()
    for leak in ("label", "entailed", "refuted", "origin_statement", "perturbed"):
        assert leak not in p

def test_prompt_independent_of_label():
    # the strongest verdict-blind guarantee: label never affects the prompt
    it_true = Item("sciver_val_7","sciver","p1","analytical","chart","/abs/x.png","C", True)
    it_false = Item("sciver_val_7","sciver","p1","analytical","chart","/abs/x.png","C", False)
    assert build_prompt(1, it_true, "QLq", "R") == build_prompt(1, it_false, "QLq", "R")

def test_all_dims_known():
    # 18 DeLeAn + 3 authored emergent figure-grounding dims
    assert len(DIMS) == 21 and DIMS["MCr"] == "Identifying Relevant Information"
    assert DIMS["VL"] == "Visual Localization and Grounding"
    assert DIMS["GS"] == "Gestalt and Shape Judgment"
    assert DIMS["MA"] == "Multi-Element Visual Aggregation"


def test_prompt_builds_for_emergent_dim():
    p = build_prompt(2, IT, "VL", "VL RUBRIC TEXT")
    assert "VL RUBRIC TEXT" in p
    assert "Visual Localization and Grounding" in p
    assert "rs_p02__sciver_val_7__VL" in p
