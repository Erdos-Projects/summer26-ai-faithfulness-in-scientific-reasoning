from sciclaimeval_eval.parse import parse_answer

def test_parses_last_answer():
    assert parse_answer("reasoning ... Answer: yes") == "yes"
    assert parse_answer("Answer: no\n") == "no"

def test_none_when_absent():
    assert parse_answer("no verdict line") is None
    assert parse_answer("") is None
