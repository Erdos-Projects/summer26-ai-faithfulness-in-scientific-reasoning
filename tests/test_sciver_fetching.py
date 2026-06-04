from __future__ import annotations

import json
from pathlib import Path

import pytest

from sciver_vector_db.fetching import DEFAULT_QDRANT_PATH, DEFAULT_RAW_DATASET_DIR, fetch_sciver_dataset


def test_fetch_sciver_dataset_uses_huggingface_dataset_repo_type(tmp_path) -> None:
    calls = []

    def fake_downloader(**kwargs):
        calls.append(kwargs)
        target = Path(kwargs["local_dir"])
        target.mkdir(parents=True, exist_ok=True)
        (target / "valset.json").write_text("[]", encoding="utf-8")
        (target / "testset.json").write_text("[]", encoding="utf-8")
        (target / "image.png").write_bytes(b"fake")
        return str(target)

    result = fetch_sciver_dataset(
        raw_dir=tmp_path / "raw",
        dataset_id="chengyewang/SciVer",
        revision="main",
        downloader=fake_downloader,
    )

    assert calls == [
        {
            "repo_id": "chengyewang/SciVer",
            "repo_type": "dataset",
            "revision": "main",
            "local_dir": str(tmp_path / "raw" / "SciVer"),
            "token": None,
        }
    ]
    assert result.local_path == tmp_path / "raw" / "SciVer"
    assert result.manifest_path.exists()
    manifest = json.loads(result.manifest_path.read_text(encoding="utf-8"))
    assert manifest["dataset_id"] == "chengyewang/SciVer"
    assert manifest["revision"] == "main"
    assert manifest["file_count"] == 3
    assert "valset.json" in manifest["files_found"]
    assert "testset.json" in manifest["files_found"]


def test_fetch_sciver_dataset_fails_without_required_splits(tmp_path) -> None:
    def fake_downloader(**kwargs):
        target = Path(kwargs["local_dir"])
        target.mkdir(parents=True, exist_ok=True)
        (target / "valset.json").write_text("[]", encoding="utf-8")
        return str(target)

    with pytest.raises(FileNotFoundError, match="testset.json"):
        fetch_sciver_dataset(raw_dir=tmp_path / "raw", downloader=fake_downloader)


def test_default_fetch_and_qdrant_paths() -> None:
    assert DEFAULT_RAW_DATASET_DIR == Path("data/raw/SciVer")
    assert DEFAULT_QDRANT_PATH == Path("data/processed/qdrant_sciver")
