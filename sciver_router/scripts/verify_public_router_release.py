"""Verify the public SciVer router release artifacts.

This check is intentionally lightweight: it validates the sanitized modeling
dataset, public comparison tables, run inventory, headline metrics, and optional
checksums. It does not load generated model pickle/joblib artifacts.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pandas as pd


EXPECTED_ROWS = 811
EXPECTED_COMPARISON_ROWS = 12
EXPECTED_METRICS_FILES = 24
EXPECTED_MODELS = {"qwen3vl", "pixtral12b", "kimi_vl_a3b"}
EXPECTED_HEADLINE_RUN = "xgboost_chart_tda_regret_first"
EXPECTED_HEADLINE_TOP1 = 0.491667
EXPECTED_HEADLINE_REGRET = 0.243333
ROOT_COPY_FILENAMES = (
    "router_dataset_public.csv",
    "router_dataset_public_audit.json",
    "chart_tda_features_public.csv",
    "claim_nlp_features_public.csv",
)
FORBIDDEN_EXACT_COLUMNS = {
    "claim",
    "caption_1",
    "caption_2",
    "label",
    "image_paths",
    "annotation_image_path",
    "annotation_image_file",
    "raw_output",
    "prompt",
    "response",
    "completion",
    "image_path",
}
FORBIDDEN_PREFIXES = (
    "majority_correct_",
    "parse_fail_count_",
    "model_id_",
    "run_started_at_utc_",
)
FORBIDDEN_FILENAME_SUFFIXES = (".db", ".sqlite", ".sqlite3", ".jsonl")
EXPECTED_RELEASE_EXACT_FILES = {
    ".gitignore",
    ".github/workflows/verify-public-release.yml",
    "DATA_LICENSE.md",
    "LICENSE",
    "PUBLIC_RELEASE_MANIFEST.md",
    "README.md",
    "chart_tda_features_public.csv",
    "claim_nlp_features_public.csv",
    "docs/public_data_statement.md",
    "docs/public_review_iterations.md",
    "docs/reproduce_router_experiments.md",
    "docs/sciver_router_presentation.html",
    "outputs/sciver_router_models_presentation.pptx",
    "outputs/sciver_router_models_presentation_montage.png",
    "requirements-lock.txt",
    "requirements.txt",
    "router_dataset_public.csv",
    "router_dataset_public_audit.json",
    "scripts/build_sciver_router_presentation_report.py",
    "scripts/compare_router_models.py",
    "scripts/router_model_lib.py",
    "scripts/run_public_router_experiments.py",
    "scripts/torch_mlp_router_lib.py",
    "scripts/train_model_router_ridge.py",
    "scripts/train_model_router_torch_mlp.py",
    "scripts/train_model_router_xgboost.py",
    "scripts/verify_public_router_release.py",
}
EXPECTED_RELEASE_PREFIXES = ("Data/derived/model_router/public_router/",)
TEXT_SCAN_SUFFIXES = {"", ".csv", ".gitignore", ".json", ".md", ".py", ".txt", ".yaml", ".yml"}
FORBIDDEN_TEXT_LITERALS = {
    "non" + "_gemma": "internal experiment name",
    "gemma" + "4": "excluded model reference",
    "/Users/" + "conglongxu": "local absolute path",
    "annotations_prod" + ".db": "private annotation database",
}
SECRET_PATTERNS = {
    "GitHub token": re.compile(r"(?:github_pat_|gh[pousr]_)[A-Za-z0-9_]{20,}"),
    "Hugging Face token": re.compile(r"hf_[A-Za-z0-9]{20,}"),
    "OpenAI-style key": re.compile(r"sk-[A-Za-z0-9]{20,}"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".", help="Repository root or dist/public_router_release root.")
    parser.add_argument("--json-out", default="", help="Optional path to write the verification JSON summary.")
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_public_dir(root: Path) -> Path:
    nested = root / "Data/derived/model_router/public_router"
    if nested.exists():
        return nested
    if (root / "router_dataset_public.csv").exists():
        return root
    raise FileNotFoundError(f"Could not find public router dataset under {root}")


def forbidden_columns(columns: list[str]) -> list[str]:
    found = []
    for column in columns:
        lower = column.lower()
        if column in FORBIDDEN_EXACT_COLUMNS:
            found.append(column)
        elif any(column.startswith(prefix) for prefix in FORBIDDEN_PREFIXES):
            found.append(column)
        elif any(fragment in lower for fragment in ["raw", "prompt", "response", "completion"]):
            if column not in {"best_models"}:
                found.append(column)
    return sorted(set(found))


def check_no_private_files(root: Path) -> list[str]:
    release_like_root = (
        (root / "README.md").exists()
        and (root / "router_dataset_public.csv").exists()
        and (root / "Data/derived/model_router/public_router").exists()
    )
    if release_like_root:
        scan_roots = [root]
    else:
        public_dir = resolve_public_dir(root)
        scan_roots = [public_dir]
        for relative in ["docs", "outputs", "scripts"]:
            path = root / relative
            if path.exists():
                scan_roots.append(path)

    bad: list[str] = []
    for scan_root in scan_roots:
        for path in scan_root.rglob("*"):
            if path.is_file() and path.suffix.lower() in FORBIDDEN_FILENAME_SUFFIXES:
                bad.append(path.relative_to(root).as_posix())
    return sorted(set(bad))


def is_release_root(root: Path) -> bool:
    return (
        (root / "README.md").exists()
        and (root / "router_dataset_public.csv").exists()
        and (root / "Data/derived/model_router/public_router").exists()
    )


def check_release_inventory(root: Path) -> list[str]:
    if not is_release_root(root):
        return []
    unexpected: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        if relative in EXPECTED_RELEASE_EXACT_FILES:
            continue
        if any(relative.startswith(prefix) for prefix in EXPECTED_RELEASE_PREFIXES):
            continue
        unexpected.append(relative)
    return sorted(unexpected)


def check_manifest_coverage(root: Path) -> list[str]:
    if not is_release_root(root):
        return []
    manifest = (root / "PUBLIC_RELEASE_MANIFEST.md").read_text(encoding="utf-8")
    entries = set(re.findall(r"`([^`]+)`", manifest))
    missing: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(root).as_posix()
        if relative in entries:
            continue
        if any(entry.endswith("/") and relative.startswith(entry) for entry in entries):
            continue
        missing.append(relative)
    return sorted(missing)


def check_text_safety(root: Path) -> list[str]:
    if not is_release_root(root):
        return []
    findings: list[str] = []
    for path in root.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        if path.suffix.lower() not in TEXT_SCAN_SUFFIXES and path.name != ".gitignore":
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        relative = path.relative_to(root).as_posix()
        lower = text.lower()
        for literal, label in FORBIDDEN_TEXT_LITERALS.items():
            if literal.lower() in lower:
                findings.append(f"{relative}: {label} ({literal})")
        for label, pattern in SECRET_PATTERNS.items():
            if pattern.search(text):
                findings.append(f"{relative}: possible {label}")
    return sorted(set(findings))


def verify(root: Path) -> dict[str, Any]:
    public_dir = resolve_public_dir(root)
    dataset_path = public_dir / "router_dataset_public.csv"
    audit_path = public_dir / "router_dataset_public_audit.json"
    checksum_path = public_dir / "public_router_checksums.json"

    df = pd.read_csv(dataset_path)
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    forbidden = forbidden_columns(df.columns.tolist())
    missing_accuracy = [f"acc_{model}" for model in EXPECTED_MODELS if f"acc_{model}" not in df.columns]
    unexpected_accuracy = [
        column
        for column in df.columns
        if column.startswith("acc_") and column.removeprefix("acc_") not in EXPECTED_MODELS
    ]

    comparison_counts: dict[str, int] = {}
    headline: dict[str, float | str] = {}
    for policy in ["regret_first", "top1_first"]:
        metrics_path = public_dir / "comparisons" / policy / "model_comparison_metrics.csv"
        metrics = pd.read_csv(metrics_path)
        comparison_counts[policy] = int(len(metrics))
        if policy == "regret_first":
            row = metrics.loc[metrics["run_label"] == EXPECTED_HEADLINE_RUN]
            if not row.empty:
                headline = {
                    "run_label": EXPECTED_HEADLINE_RUN,
                    "test_top1_hit_rate": float(row.iloc[0]["test_top1_hit_rate"]),
                    "test_mean_regret": float(row.iloc[0]["test_mean_regret"]),
                }

    metrics_files = sorted(public_dir.glob("runs/*/*/metrics.json"))
    bad_private_files = check_no_private_files(root)
    unexpected_release_files = check_release_inventory(root)
    manifest_coverage_missing = check_manifest_coverage(root)
    text_safety_findings = check_text_safety(root)
    root_copy_mismatches: list[str] = []
    if public_dir != root:
        for filename in ROOT_COPY_FILENAMES:
            root_copy = root / filename
            canonical = public_dir / filename
            if root_copy.exists() and canonical.exists() and sha256_file(root_copy) != sha256_file(canonical):
                root_copy_mismatches.append(filename)

    checksum_mismatches: list[str] = []
    checksums_checked = 0
    if checksum_path.exists():
        checksums = json.loads(checksum_path.read_text(encoding="utf-8"))
        for relative, expected in checksums.items():
            path = public_dir / relative
            if path.exists():
                checksums_checked += 1
                actual = sha256_file(path)
                if actual != expected:
                    checksum_mismatches.append(relative)

    failures = []
    if len(df) != EXPECTED_ROWS:
        failures.append(f"Expected {EXPECTED_ROWS} rows, found {len(df)}")
    if audit.get("row_count") != EXPECTED_ROWS:
        failures.append(f"Audit row_count should be {EXPECTED_ROWS}, found {audit.get('row_count')}")
    if forbidden:
        failures.append(f"Forbidden columns present: {forbidden}")
    if missing_accuracy:
        failures.append(f"Missing accuracy columns: {missing_accuracy}")
    if unexpected_accuracy:
        failures.append(f"Unexpected accuracy columns: {unexpected_accuracy}")
    for policy, count in comparison_counts.items():
        if count != EXPECTED_COMPARISON_ROWS:
            failures.append(f"{policy} comparison has {count} rows, expected {EXPECTED_COMPARISON_ROWS}")
    if len(metrics_files) != EXPECTED_METRICS_FILES:
        failures.append(f"Found {len(metrics_files)} metrics.json files, expected {EXPECTED_METRICS_FILES}")
    if headline:
        if abs(float(headline["test_top1_hit_rate"]) - EXPECTED_HEADLINE_TOP1) > 1e-6:
            failures.append("Headline top-1 hit rate changed")
        if abs(float(headline["test_mean_regret"]) - EXPECTED_HEADLINE_REGRET) > 1e-6:
            failures.append("Headline mean regret changed")
    else:
        failures.append(f"Missing headline run {EXPECTED_HEADLINE_RUN}")
    if bad_private_files:
        failures.append(f"Private/raw file suffixes found: {bad_private_files}")
    if unexpected_release_files:
        failures.append(f"Unexpected files outside the public allowlist: {unexpected_release_files}")
    if manifest_coverage_missing:
        failures.append(f"Files missing from PUBLIC_RELEASE_MANIFEST.md: {manifest_coverage_missing}")
    if text_safety_findings:
        failures.append(f"Unsafe text findings: {text_safety_findings}")
    if checksum_mismatches:
        failures.append(f"Checksum mismatches: {checksum_mismatches}")
    if root_copy_mismatches:
        failures.append(f"Root-level convenience copies differ from canonical public files: {root_copy_mismatches}")

    summary: dict[str, Any] = {
        "ok": not failures,
        "root": root.as_posix(),
        "public_dir": public_dir.as_posix(),
        "row_count": int(len(df)),
        "column_count": int(len(df.columns)),
        "comparison_counts": comparison_counts,
        "metrics_file_count": len(metrics_files),
        "headline": headline,
        "checksums_checked": checksums_checked,
        "checksum_mismatches": checksum_mismatches,
        "root_copy_mismatches": root_copy_mismatches,
        "forbidden_columns_present": forbidden,
        "private_file_suffixes_present": bad_private_files,
        "unexpected_release_files": unexpected_release_files,
        "manifest_coverage_missing": manifest_coverage_missing,
        "text_safety_findings": text_safety_findings,
        "failures": failures,
    }
    return summary


def main() -> None:
    args = parse_args()
    summary = verify(Path(args.root))
    text = json.dumps(summary, indent=2, sort_keys=True)
    if args.json_out:
        Path(args.json_out).write_text(text + "\n", encoding="utf-8")
    print(text)
    if not summary["ok"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
