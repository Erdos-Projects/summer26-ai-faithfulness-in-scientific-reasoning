import pytest

from sciver_eval import conditions


def _rec():
    return {
        "claim_type": "direct",
        "claim": "OFFICIAL CLAIM",
        "origin_statement": "TRUE STATEMENT",
        "perturbed_statement": "FALSE STATEMENT",
        "perturbed_explanation": "SECRET REASON",
    }


def test_entailed_resolves_to_origin_statement_and_label_1():
    claim, label = conditions.resolve("entailed", _rec())
    assert claim == "TRUE STATEMENT"
    assert label == 1


def test_refuted_resolves_to_perturbed_statement_and_label_0():
    claim, label = conditions.resolve("refuted", _rec())
    assert claim == "FALSE STATEMENT"
    assert label == 0


def test_unknown_condition_raises():
    with pytest.raises(KeyError):
        conditions.resolve("bogus", _rec())


def test_empty_statement_raises():
    rec = _rec()
    rec["origin_statement"] = "   "
    with pytest.raises(ValueError):
        conditions.resolve("entailed", rec)
