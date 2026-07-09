"""Path resolution for the SciClaimEval rubric-scoring module.
Everything is relative or env-overridable — no hardcoded absolutes."""
import os
from pathlib import Path

def repo_root() -> Path:
    return Path(__file__).resolve().parents[1]

def sciclaimeval_dir() -> Path:
    """SciClaimEval data root holding dev_task1_release.json + figures/."""
    env = os.environ.get("SCICLAIMEVAL_DIR")
    candidates = [Path(env)] if env else []
    candidates += [
        repo_root() / "data" / "raw" / "SciClaimEval" / "sciclaimeval-shared-task" / "data",
        repo_root().parent / "SciClaimEval" / "sciclaimeval-shared-task" / "data",
    ]
    for c in candidates:
        if (c / "dev_task1_release.json").is_file():
            return c
    return candidates[-1]  # best guess; CLIs/tests surface a clear error if missing

def rubrics_dir() -> Path:
    return Path(__file__).resolve().parent / "rubrics"

def db_path() -> Path:
    env = os.environ.get("SCICLAIMEVAL_SCORING_DB")
    return Path(env) if env else repo_root() / "sciclaimeval_scoring" / "annotations_sciclaimeval.db"

def claude_projects_dir() -> Path:
    env = os.environ.get("RUBRIC_SCORING_PROJECTS")
    return Path(env) if env else Path.home() / ".claude" / "projects"
