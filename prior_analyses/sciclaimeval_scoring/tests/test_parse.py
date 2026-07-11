from sciclaimeval_scoring.parse import parse_score

def test_parses_final_score_line():
    assert parse_score("reasoning... is: 4") == 4
    assert parse_score("the level ... is: **2**") == 2

def test_no_score_returns_none():
    assert parse_score("no number here") is None
    assert parse_score("") is None
