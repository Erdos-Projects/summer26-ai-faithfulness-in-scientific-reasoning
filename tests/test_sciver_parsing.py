from __future__ import annotations

import json

import pytest

from sciver_vector_db.parsing import (
    build_pair_records,
    find_split_files,
    label_to_name_id,
    load_split_examples,
    normalize_modality,
)


def write_split(root, name, rows):
    path = root / name
    path.write_text(json.dumps(rows), encoding="utf-8")
    return path


def touch_image(root, name="chart.png"):
    path = root / "images" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"not-a-real-image-but-path-exists")
    return path


def test_label_mapping() -> None:
    assert label_to_name_id(True) == ("entailed", 1)
    assert label_to_name_id(False) == ("refuted", 0)
    assert label_to_name_id("entailed") == ("entailed", 1)
    assert label_to_name_id("refuted") == ("refuted", 0)
    with pytest.raises(ValueError):
        label_to_name_id("maybe")


def test_single_direct_visual_evidence(tmp_path) -> None:
    image = touch_image(tmp_path)
    row = {
        "claim": "The chart shows increasing accuracy.",
        "label": True,
        "paperid": "p1",
        "request_id": "r1",
        "claim_type": "trend",
        "type": "figure",
        "image_path": str(image.relative_to(tmp_path)),
        "caption": "Accuracy by epoch.",
    }
    write_split(tmp_path, "valset.json", [row])
    write_split(tmp_path, "testset.json", [])

    examples, parser_skips = load_split_examples(tmp_path, find_split_files(tmp_path))
    pairs, pair_skips = build_pair_records(examples, modalities={"chart"})

    assert parser_skips == {}
    assert pair_skips == {}
    assert len(pairs) == 1
    assert pairs[0].label == "entailed"
    assert pairs[0].label_id == 1
    assert pairs[0].modality == "chart"
    assert pairs[0].image_path == image.resolve()


def test_item1_item2_multi_evidence_detection(tmp_path) -> None:
    chart = touch_image(tmp_path, "chart.png")
    table = touch_image(tmp_path, "table.png")
    row = {
        "claim": "The table and figure agree.",
        "label": False,
        "paperid": "p2",
        "request_id": "r2",
        "item1_type": "fig",
        "item1_path": str(chart.relative_to(tmp_path)),
        "item1_caption": "A figure.",
        "item2_type": "table",
        "item2_path": str(table.relative_to(tmp_path)),
        "item2_caption": "A table.",
    }
    write_split(tmp_path, "valset.json", [row])
    write_split(tmp_path, "testset.json", [])

    examples, _ = load_split_examples(tmp_path, find_split_files(tmp_path))

    assert len(examples[0].visual_items) == 2
    assert {item.modality for item in examples[0].visual_items} == {"chart", "table"}
    pairs, skips = build_pair_records(examples, modalities={"chart", "table"})
    assert len(pairs) == 0
    assert skips["multiple_matching_visual_evidence"] == 1

    chart_pairs, chart_skips = build_pair_records(examples, modalities={"chart"})
    assert len(chart_pairs) == 1
    assert chart_skips == {}


def test_skip_reasons(tmp_path) -> None:
    touch_image(tmp_path, "chart.png")
    rows = [
        {"claim": "", "label": True, "type": "chart", "image_path": "images/chart.png"},
        {"claim": "No visual.", "label": True},
        {"claim": "Missing image.", "label": False, "type": "chart", "image_path": "missing.png"},
        {
            "claim": "Only table.",
            "label": True,
            "type": "table",
            "image_path": "images/chart.png",
        },
    ]
    write_split(tmp_path, "valset.json", rows)
    write_split(tmp_path, "testset.json", [])

    examples, parser_skips = load_split_examples(tmp_path, find_split_files(tmp_path))
    pairs, pair_skips = build_pair_records(examples, modalities={"chart"})

    assert len(pairs) == 0
    assert parser_skips["missing_claim"] == 1
    assert pair_skips["no_visual_evidence"] == 1
    assert pair_skips["image_not_found"] == 1
    assert pair_skips["no_matching_visual_evidence"] == 1


def test_deterministic_pair_ids(tmp_path) -> None:
    touch_image(tmp_path, "chart.png")
    row = {
        "claim": "Bars are higher in group A.",
        "label": True,
        "paperid": "p3",
        "request_id": "r3",
        "type": "graph",
        "image_path": "images/chart.png",
    }
    write_split(tmp_path, "valset.json", [row])
    write_split(tmp_path, "testset.json", [])

    examples_a, _ = load_split_examples(tmp_path, find_split_files(tmp_path))
    examples_b, _ = load_split_examples(tmp_path, find_split_files(tmp_path))
    pairs_a, _ = build_pair_records(examples_a, modalities={"chart"})
    pairs_b, _ = build_pair_records(examples_b, modalities={"chart"})

    assert pairs_a[0].claim_id == pairs_b[0].claim_id
    assert pairs_a[0].evidence_item_id == pairs_b[0].evidence_item_id
    assert pairs_a[0].pair_id == pairs_b[0].pair_id


def test_no_leakage_fields_in_embedded_text(tmp_path) -> None:
    touch_image(tmp_path, "chart.png")
    row = {
        "claim": "The chart reports a higher mean for treatment.",
        "label": False,
        "paperid": "p4",
        "request_id": "r4",
        "type": "chart",
        "image_path": "images/chart.png",
        "caption": "Treatment and control means.",
        "rationale": "LEAK_RATIONALE",
        "explanation": "LEAK_EXPLANATION",
        "perturbed_explanation": "LEAK_PERTURBED",
        "answer": "LEAK_ANSWER",
        "origin_statement": "LEAK_ORIGIN",
        "perturbed_statement": "LEAK_STATEMENT",
    }
    write_split(tmp_path, "valset.json", [row])
    write_split(tmp_path, "testset.json", [])

    examples, _ = load_split_examples(tmp_path, find_split_files(tmp_path))
    pairs, _ = build_pair_records(examples, modalities={"chart"})
    embedded = "\n".join([pairs[0].claim_text(), pairs[0].evidence_text(), pairs[0].pair_text()])

    assert "LEAK_" not in embedded
    assert "refuted" not in embedded.lower()
    assert "entailed" not in embedded.lower()


def test_default_modalities_cover_chart_and_table_names() -> None:
    assert normalize_modality("figure") == "chart"
    assert normalize_modality("plot") == "chart"
    assert normalize_modality("table") == "table"


def test_fetched_snapshot_style_path_is_accepted(tmp_path) -> None:
    snapshot = tmp_path / "data" / "raw" / "SciVer"
    image = touch_image(snapshot, "table.png")
    row = {
        "claim": "The table reports the highest score for method A.",
        "label": True,
        "paperid": "p5",
        "request_id": "r5",
        "type": "table",
        "image_path": str(image.relative_to(snapshot)),
    }
    write_split(snapshot, "valset.json", [row])
    write_split(snapshot, "testset.json", [])

    examples, _ = load_split_examples(snapshot, find_split_files(snapshot))
    pairs, skips = build_pair_records(examples, modalities={"chart", "table"})

    assert skips == {}
    assert len(pairs) == 1
    assert pairs[0].modality == "table"
