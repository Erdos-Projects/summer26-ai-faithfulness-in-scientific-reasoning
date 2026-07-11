from sciclaimeval_scoring.prompts import tag, build_prompt, DIMS
from sciclaimeval_scoring.items import Item

def _item():
    return Item("scev_val_fig_0001", "sciclaimeval", "2403.19137", "", "figure",
                "/data/figures/dev/val_fig_0001.png", "ConDA outperforms all baselines.",
                False, "nlp", "Legend Swap")

def test_tag_namespace_is_rsc_not_sciver():
    t = tag(1, "scev_val_fig_0001", "AS")
    assert t == "rsc_p01__scev_val_fig_0001__AS"
    assert not t.startswith("rs_p")  # disjoint from the SciVer DB namespace

def test_dims_cover_default_six():
    for d in ("AS", "QLq", "QLl", "MCr", "AT", "VO"):
        assert d in DIMS

def test_prompt_is_verdict_blind():
    p = build_prompt(1, _item(), "AS", "RUBRIC TEXT")
    assert "ConDA outperforms all baselines." in p          # claim present
    assert "/data/figures/dev/val_fig_0001.png" in p        # figure present
    for leak in ("Legend Swap", "Refuted", "Supported", "False"):
        assert leak not in p, leak                          # label/operation withheld
