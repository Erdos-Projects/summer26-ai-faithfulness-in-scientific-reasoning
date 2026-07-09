from rubric_scoring import config
from rubric_scoring.extract_rubrics import ALL_DIMS, EMERGENT_DIMS

def test_all_18_dims_defined():
    assert len(ALL_DIMS) == 18
    assert set(ALL_DIMS) >= {"AS","QLq","QLl","MCr","AT","VO"}

def test_all_18_rubric_files_present_and_nonempty():
    for code in ALL_DIMS:
        p = config.rubrics_dir() / f"{code}.txt"
        assert p.is_file(), f"missing rubric {code}"
        text = p.read_text(encoding="utf-8")
        assert len(text) > 400, f"rubric {code} too short"
        name = ALL_DIMS[code][2].split("(")[0].strip()  # marker text before "("
        assert name in text.splitlines()[0], f"{code}: first line missing dim name"

def test_rubric_has_six_levels():
    txt = (config.rubrics_dir() / "QLq.txt").read_text()
    assert txt.count("Level ") >= 6


def test_three_emergent_dims_defined():
    assert set(EMERGENT_DIMS) == {"VL", "GS", "MA"}
    assert not (set(EMERGENT_DIMS) & set(ALL_DIMS))  # disjoint from the 18 DeLeAn dims


def test_emergent_rubric_files_present_and_well_formed():
    for code, name in EMERGENT_DIMS.items():
        p = config.rubrics_dir() / f"{code}.txt"
        assert p.is_file(), f"missing emergent rubric {code}"
        text = p.read_text(encoding="utf-8")
        assert len(text) > 400, f"emergent rubric {code} too short"
        assert text.splitlines()[0] == f"{name} ({code})", f"{code}: header mismatch"
        assert text.count("Level ") >= 6, f"{code}: fewer than six levels"
