from sciclaimeval_eval.prompts import tag, build_prompt, load_records
from sciclaimeval_scoring.items import Item

def _item():
    return Item("scev_val_fig_0001", "sciclaimeval", "p", "", "figure",
                "/data/figures/dev/val_fig_0001.png", "ConDA outperforms all baselines.",
                False, "nlp", "Legend Swap")

def _rec(use_context="yes", context="We compare ConDA with baselines."):
    return {"claim": "ConDA outperforms all baselines.", "caption": "Table of AUROC.",
            "context": context, "use_context": use_context, "label": "Refuted",
            "operation": "Legend Swap"}

def test_tag_namespace_is_sce():
    assert tag(1, "scev_val_fig_0001", 3) == "sce_r01__scev_val_fig_0001__t03"

def test_prompt_has_answer_line_and_evidence():
    p = build_prompt(1, _item(), _rec(), 1)
    assert "ConDA outperforms all baselines." in p
    assert "Table of AUROC." in p                 # caption shown (real task input)
    assert "/data/figures/dev/val_fig_0001.png" in p
    assert "Answer: yes" in p and "Answer: no" in p  # output format instruction

def test_prompt_is_verdict_clean():
    p = build_prompt(1, _item(), _rec(), 1)
    for leak in ("Legend Swap", "Refuted"):
        assert leak not in p, leak                  # label/operation never shown

def test_context_included_when_use_context_not_no():
    for uc in ("yes", "other sources"):
        p = build_prompt(1, _item(), _rec(use_context=uc, context="UNIQUE_CTX_TOKEN"), 1)
        assert "Context:" in p and "UNIQUE_CTX_TOKEN" in p, uc

def test_context_omitted_when_use_context_no():
    p = build_prompt(1, _item(), _rec(use_context="no", context="UNIQUE_CTX_TOKEN"), 1)
    assert "UNIQUE_CTX_TOKEN" not in p          # context text withheld
    assert "Context:" not in p                  # the Context block is dropped entirely
    assert "Table of AUROC." in p               # caption still shown (it is evidence, not context)
    assert "ConDA outperforms all baselines." in p

def test_context_block_dropped_when_context_empty_even_if_flagged():
    p = build_prompt(1, _item(), _rec(use_context="yes", context="   "), 1)
    assert "Context:" not in p                  # no dangling empty Context line

def test_load_records_returns_265_figures():
    recs = load_records()
    assert len(recs) == 265
    assert all(k.startswith("scev_") for k in recs)
