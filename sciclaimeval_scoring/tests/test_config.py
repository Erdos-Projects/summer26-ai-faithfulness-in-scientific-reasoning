from sciclaimeval_scoring import config

def test_data_root_resolves_and_has_release_file():
    sd = config.sciclaimeval_dir()
    assert (sd / "dev_task1_release.json").is_file(), sd

def test_rubric_library_has_21_files():
    txts = list(config.rubrics_dir().glob("*.txt"))
    assert len(txts) == 21, [p.name for p in txts]

def test_db_path_is_module_local_and_sciclaimeval_named():
    p = config.db_path()
    assert p.name == "annotations_sciclaimeval.db"
    assert "sciclaimeval_scoring" in str(p)
