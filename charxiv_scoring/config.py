"""Path resolution for the CharXiv rubric-scoring module.
Everything is relative or env-overridable — no hardcoded absolutes."""
import os
from pathlib import Path

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def charxiv_dir() -> Path:
    """CharXiv data root holding data/reasoning_val.json + images/ + existing_evaluations/."""
    env = os.environ.get("CHARXIV_DIR")
    candidates = [Path(env)] if env else []
    candidates += [
        repo_root() / "data" / "raw" / "CharXiv",
        repo_root().parent / "CharXiv",
    ]
    for c in candidates:
        if (c / "data" / "reasoning_val.json").is_file():
            return c
    return candidates[-1]  # best guess; CLIs/tests surface a clear error if missing

def rubrics_dir() -> Path:
    return Path(__file__).resolve().parent / "rubrics"

def db_path() -> Path:
    env = os.environ.get("CHARXIV_SCORING_DB")
    return Path(env) if env else repo_root() / "charxiv_scoring" / "annotations_charxiv.db"

def claude_projects_dir() -> Path:
    env = os.environ.get("RUBRIC_SCORING_PROJECTS")
    return Path(env) if env else Path.home() / ".claude" / "projects"
