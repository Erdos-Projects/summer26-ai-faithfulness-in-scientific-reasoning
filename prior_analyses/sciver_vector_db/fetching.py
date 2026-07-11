from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable


DEFAULT_DATASET_ID = "chengyewang/SciVer"
DEFAULT_RAW_DIR = Path("data/raw")
DEFAULT_RAW_DATASET_DIR = DEFAULT_RAW_DIR / "SciVer"
DEFAULT_PROCESSED_DIR = Path("data/processed")
DEFAULT_QDRANT_PATH = DEFAULT_PROCESSED_DIR / "qdrant_sciver"


@dataclass(frozen=True)
class FetchResult:
    dataset_id: str
    revision: str
    local_path: Path
    manifest_path: Path
    files_found: list[str]


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, default=str)


def list_relative_files(path: Path) -> list[str]:
    return sorted(str(item.relative_to(path)) for item in path.rglob("*") if item.is_file())


def validate_sciver_snapshot(path: Path) -> dict[str, str]:
    candidates = {
        "val": sorted(path.rglob("valset.json")),
        "test": sorted(path.rglob("testset.json")),
    }
    missing = [split for split, matches in candidates.items() if not matches]
    if missing:
        raise FileNotFoundError(
            f"Downloaded SciVer snapshot at {path} is missing required file(s): "
            f"{', '.join(f'{split}set.json' for split in missing)}"
        )
    return {split: str(matches[0]) for split, matches in candidates.items()}


def fetch_sciver_dataset(
    *,
    raw_dir: Path = DEFAULT_RAW_DIR,
    dataset_id: str = DEFAULT_DATASET_ID,
    revision: str = "main",
    force: bool = False,
    token: str | None = None,
    downloader: Callable[..., str] | None = None,
) -> FetchResult:
    target_dir = raw_dir / "SciVer"
    if force and target_dir.exists():
        shutil.rmtree(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    if downloader is None:
        from huggingface_hub import snapshot_download

        downloader = snapshot_download

    local_path = Path(
        downloader(
            repo_id=dataset_id,
            repo_type="dataset",
            revision=revision,
            local_dir=str(target_dir),
            token=token,
        )
    )
    split_files = validate_sciver_snapshot(local_path)
    files_found = list_relative_files(local_path)
    manifest_path = local_path / "sciver_download_manifest.json"
    write_json(
        manifest_path,
        {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "dataset_id": dataset_id,
            "revision": revision,
            "local_path": str(local_path.resolve()),
            "split_files": split_files,
            "file_count": len(files_found),
            "files_found": files_found,
        },
    )
    return FetchResult(
        dataset_id=dataset_id,
        revision=revision,
        local_path=local_path,
        manifest_path=manifest_path,
        files_found=files_found,
    )
