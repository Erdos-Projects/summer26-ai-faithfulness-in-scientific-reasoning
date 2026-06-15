from pathlib import Path
from rubric_scoring import config

def test_repo_root_is_summer26():
    assert (config.repo_root() / "rubric_scoring").is_dir()

def test_sciver_dir_has_images_and_valset():
    sd = config.sciver_dir()
    assert (sd / "images").is_dir()
    assert (sd / "valset.json").is_file()

def test_rubrics_dir_points_into_package():
    assert config.rubrics_dir().name == "rubrics"

def test_db_path_default_and_env_override(tmp_path, monkeypatch):
    monkeypatch.delenv("RUBRIC_SCORING_DB", raising=False)
    assert config.db_path().name == "annotations_prod.db"
    monkeypatch.setenv("RUBRIC_SCORING_DB", str(tmp_path / "x.db"))
    assert config.db_path() == tmp_path / "x.db"

def test_claude_projects_dir_default_and_env_override(tmp_path, monkeypatch):
    monkeypatch.delenv("RUBRIC_SCORING_PROJECTS", raising=False)
    assert config.db_path  # sanity: module imported
    assert config.claude_projects_dir().parts[-2:] == (".claude", "projects")
    monkeypatch.setenv("RUBRIC_SCORING_PROJECTS", str(tmp_path / "proj"))
    assert config.claude_projects_dir() == tmp_path / "proj"
