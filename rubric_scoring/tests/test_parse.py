from rubric_scoring.parse import parse_score

def test_canonical_sentence():
    assert parse_score("...\nThus, the level of *Atypicality* demanded by the given TASK INSTANCE is: 3") == 3

def test_markdown_bold_variant():
    assert parse_score("the level ... is: **4**") == 4

def test_alt_phrasings():
    assert parse_score("Score: 2") == 2
    assert parse_score("the level is 5") == 5

def test_takes_last_match():
    assert parse_score("is: 1 ... finally is: 4") == 4

def test_none_when_absent_or_out_of_range():
    assert parse_score("no number here") is None
    assert parse_score("is: 9") is None
