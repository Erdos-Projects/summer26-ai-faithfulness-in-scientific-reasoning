"""Path resolution. Everything is relative or env-overridable — no hardcoded absolutes."""
import os
from pathlib import Path

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def sciver_dir() -> Path:
    env = os.environ.get("RUBRIC_SCORING_SCIVER")
    candidates = [Path(env)] if env else []
    candidates += [repo_root() / "data" / "raw" / "SciVer", repo_root().parent / "SciVer"]
    for c in candidates:
        if (c / "valset.json").is_file():
            return c
    return candidates[-1]  # best guess; tests/CLIs will surface a clear error if missing

def rubrics_dir() -> Path:
    return Path(__file__).resolve().parent / "rubrics"

def db_path() -> Path:
    env = os.environ.get("RUBRIC_SCORING_DB")
    return Path(env) if env else repo_root() / "rubric_scoring" / "annotations_prod.db"

def claude_projects_dir() -> Path:
    env = os.environ.get("RUBRIC_SCORING_PROJECTS")
    return Path(env) if env else Path.home() / ".claude" / "projects"
