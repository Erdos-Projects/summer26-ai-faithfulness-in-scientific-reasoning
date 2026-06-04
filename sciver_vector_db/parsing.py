from __future__ import annotations

import hashlib
import json
import re
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff", ".webp"}
LEAKAGE_KEY_TERMS = (
    "rationale",
    "explanation",
    "answer",
    "label",
    "perturbed",
    "origin_statement",
    "perturbed_statement",
)


@dataclass(frozen=True)
class VisualEvidenceItem:
    item_key: str
    modality: str
    image_path_raw: str
    image_path: Path | None
    caption: str = ""
    ocr_text: str = ""
    context_text: str = ""
    section: Any = None
    paper_path: str = ""

    @property
    def section_json(self) -> str:
        return stable_json(self.section)

    @property
    def section_title(self) -> str:
        return extract_section_title(self.section)


@dataclass(frozen=True)
class ParsedExample:
    split: str
    source_file: Path
    row_index: int
    raw: dict[str, Any]
    claim: str
    label: str
    label_id: int
    paperid: str
    request_id: str
    claim_type: str
    paper_path: str
    section: Any
    visual_items: tuple[VisualEvidenceItem, ...]


@dataclass(frozen=True)
class PairRecord:
    pair_id: str
    claim_id: str
    evidence_item_id: str
    split: str
    source_file: Path
    paperid: str
    request_id: str
    claim: str
    label: str
    label_id: int
    claim_type: str
    modality: str
    image_path: Path
    caption: str
    ocr_text: str
    context_text: str
    section_json: str
    paper_path: str
    item_key: str
    n_evidence_items: int
    pair_mode: str
    gold_pair: bool = True

    def claim_text(self) -> str:
        return self.claim

    def evidence_text(self) -> str:
        return join_nonempty(
            [
                self.caption,
                self.ocr_text,
                self.context_text,
                extract_section_title_from_json(self.section_json),
            ]
        )

    def pair_text(self) -> str:
        return f"CLAIM:\n{self.claim}\n\nEVIDENCE:\n{self.evidence_text()}".strip()


class PathResolver:
    def __init__(self, data_root: Path):
        self.data_root = data_root.resolve()
        self._basename_index: dict[str, list[Path]] | None = None

    def resolve(self, raw_path: str | Path | None, json_dir: Path | None = None) -> Path | None:
        if raw_path is None:
            return None
        raw = str(raw_path).strip()
        if not raw or raw.startswith(("http://", "https://")):
            return None
        raw = raw.replace("\\", "/")
        path = Path(raw).expanduser()
        candidates: list[Path] = []
        if path.is_absolute():
            candidates.append(path)
        else:
            candidates.append(self.data_root / path)
            if json_dir is not None:
                candidates.append(json_dir / path)
        for candidate in candidates:
            if candidate.exists() and candidate.is_file():
                return candidate.resolve()
        basename = path.name
        if basename:
            matches = self.basename_index().get(basename, [])
            if matches:
                return sorted(matches, key=lambda p: (len(str(p)), str(p)))[0].resolve()
        return None

    def basename_index(self) -> dict[str, list[Path]]:
        if self._basename_index is None:
            index: dict[str, list[Path]] = {}
            for path in self.data_root.rglob("*"):
                if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS:
                    index.setdefault(path.name, []).append(path)
            self._basename_index = index
        return self._basename_index


def stable_hash(value: str, n: int = 16) -> str:
    return hashlib.sha1(value.encode("utf-8", errors="ignore")).hexdigest()[:n]


def stable_json(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=False)
    except TypeError:
        return str(value)


def safe_id_part(value: Any, fallback: str = "unknown") -> str:
    text = str(value).strip() if value is not None else ""
    if not text:
        text = fallback
    text = re.sub(r"\s+", "_", text)
    text = re.sub(r"[^A-Za-z0-9_.:-]+", "_", text)
    return text[:96]


def is_leakage_key(key: str) -> bool:
    lowered = key.lower()
    return any(term in lowered for term in LEAKAGE_KEY_TERMS)


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return re.sub(r"\s+", " ", value).strip()
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        return " ".join(clean_text(v) for v in value if clean_text(v)).strip()
    if isinstance(value, dict):
        safe_values: list[str] = []
        for key, item in value.items():
            if not is_leakage_key(str(key)):
                text = clean_text(item)
                if text:
                    safe_values.append(text)
        return " ".join(safe_values).strip()
    return str(value).strip()


def first_text(row: dict[str, Any], keys: Iterable[str]) -> str:
    for key in keys:
        if key in row and not is_leakage_key(key):
            text = clean_text(row.get(key))
            if text:
                return text
    return ""


def first_raw(row: dict[str, Any], keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row and row.get(key) not in (None, ""):
            return row.get(key)
    return None


def join_nonempty(parts: Iterable[str]) -> str:
    return "\n".join(part.strip() for part in parts if part and part.strip())


def label_to_name_id(value: Any) -> tuple[str, int]:
    if isinstance(value, bool):
        return ("entailed", 1) if value else ("refuted", 0)
    if isinstance(value, (int, float)) and value in (0, 1):
        return ("entailed", 1) if int(value) == 1 else ("refuted", 0)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"true", "entail", "entailed", "supports", "support", "1", "yes"}:
            return "entailed", 1
        if lowered in {"false", "refute", "refuted", "contradicts", "contradiction", "0", "no"}:
            return "refuted", 0
    raise ValueError(f"Unsupported SciVer label value: {value!r}")


def normalize_modality(value: Any, hint: str = "") -> str:
    text = f"{value or ''} {hint or ''}".lower()
    text = re.sub(r"[_-]+", " ", text)
    if "table" in text or re.search(r"\btab\b", text):
        return "table"
    if any(token in text for token in ("figure", "fig", "chart", "plot", "graph")):
        return "chart"
    return text.strip() or "unknown"


def looks_like_image_path(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    lowered = value.strip().lower()
    if not lowered or lowered.startswith(("http://", "https://")):
        return False
    return Path(lowered).suffix in IMAGE_EXTENSIONS


def image_path_from_dict(item: dict[str, Any]) -> str:
    path_keys = [
        "image_path",
        "img_path",
        "figure_path",
        "table_path",
        "path",
        "file_path",
        "filename",
        "image_file",
        "img_file",
        "image",
        "img",
    ]
    for key in path_keys:
        value = item.get(key)
        if isinstance(value, str) and value.strip() and (looks_like_image_path(value) or "path" in key or "file" in key):
            return value.strip()
    for key, value in item.items():
        if re.match(r"^item\d+[_-]", str(key), flags=re.IGNORECASE):
            continue
        if is_leakage_key(str(key)):
            continue
        if isinstance(value, str) and looks_like_image_path(value):
            return value.strip()
    return ""


def evidence_from_dict(
    item: dict[str, Any],
    *,
    item_key: str,
    resolver: PathResolver,
    json_dir: Path,
    inherited_section: Any,
    inherited_paper_path: str,
) -> VisualEvidenceItem | None:
    image_path_raw = image_path_from_dict(item)
    modality_raw = first_raw(item, ["modality", "type", "visual_type", "evidence_type", "item_type", "kind"])
    modality = normalize_modality(modality_raw, hint=f"{item_key} {image_path_raw}")
    if not image_path_raw and modality not in {"chart", "table"}:
        return None
    caption = first_text(item, ["caption", "figure_caption", "table_caption", "title", "name"])
    ocr_text = first_text(item, ["ocr_text", "ocr", "extracted_text", "text_in_image"])
    context_text = first_text(item, ["context_text", "context", "nearby_context", "paragraph", "paragraph_text"])
    section = item.get("section", inherited_section)
    paper_path = clean_text(item.get("paper_path", inherited_paper_path))
    return VisualEvidenceItem(
        item_key=item_key,
        modality=modality,
        image_path_raw=image_path_raw,
        image_path=resolver.resolve(image_path_raw, json_dir),
        caption=caption,
        ocr_text=ocr_text,
        context_text=context_text,
        section=section,
        paper_path=paper_path,
    )


def extract_visual_items(
    row: dict[str, Any],
    *,
    resolver: PathResolver,
    json_dir: Path,
    section: Any,
    paper_path: str,
) -> tuple[VisualEvidenceItem, ...]:
    items: list[VisualEvidenceItem] = []

    direct = evidence_from_dict(
        row,
        item_key="direct",
        resolver=resolver,
        json_dir=json_dir,
        inherited_section=section,
        inherited_paper_path=paper_path,
    )
    if direct is not None:
        items.append(direct)

    groups: dict[str, dict[str, Any]] = {}
    for key, value in row.items():
        match = re.match(r"^(item\d+)[_-](.+)$", str(key), flags=re.IGNORECASE)
        if match:
            groups.setdefault(match.group(1).lower(), {})[match.group(2).lower()] = value
    for item_key in sorted(groups, key=lambda k: int(re.sub(r"\D", "", k) or "0")):
        evidence = evidence_from_dict(
            groups[item_key],
            item_key=item_key,
            resolver=resolver,
            json_dir=json_dir,
            inherited_section=section,
            inherited_paper_path=paper_path,
        )
        if evidence is not None:
            items.append(evidence)

    for list_key in ("evidence_items", "visual_evidence", "evidence", "items", "figures", "tables"):
        value = row.get(list_key)
        if isinstance(value, list):
            for idx, item in enumerate(value, start=1):
                if isinstance(item, dict):
                    evidence = evidence_from_dict(
                        item,
                        item_key=f"{list_key}{idx}",
                        resolver=resolver,
                        json_dir=json_dir,
                        inherited_section=section,
                        inherited_paper_path=paper_path,
                    )
                    if evidence is not None:
                        items.append(evidence)

    deduped: dict[tuple[str, str], VisualEvidenceItem] = {}
    for item in items:
        key = (item.item_key, str(item.image_path or item.image_path_raw))
        deduped.setdefault(key, item)
    return tuple(deduped.values())


def parse_json_rows(value: Any) -> list[dict[str, Any]]:
    if isinstance(value, list):
        return [row for row in value if isinstance(row, dict)]
    if isinstance(value, dict):
        for key in ("data", "examples", "rows", "records", "requests", "instances"):
            if isinstance(value.get(key), list):
                return [row for row in value[key] if isinstance(row, dict)]
        if all(isinstance(v, dict) for v in value.values()):
            rows = []
            for key, row in value.items():
                copied = dict(row)
                copied.setdefault("request_id", key)
                rows.append(copied)
            return rows
    raise ValueError("JSON file does not contain a recognizable list of examples")


def find_split_files(data_root: Path) -> dict[str, Path]:
    data_root = data_root.expanduser().resolve()
    if not data_root.exists():
        raise FileNotFoundError(f"SciVer data root does not exist: {data_root}")
    result: dict[str, Path] = {}
    for split, exact in (("val", "valset.json"), ("test", "testset.json")):
        direct = data_root / exact
        if direct.exists():
            result[split] = direct
            continue
        candidates = sorted(data_root.rglob(exact))
        if not candidates:
            candidates = sorted(p for p in data_root.rglob("*.json") if split in p.name.lower())
        if candidates:
            result[split] = candidates[0]
    missing = [split for split in ("val", "test") if split not in result]
    if missing:
        raise FileNotFoundError(
            f"Could not locate required split JSON file(s) under {data_root}: {', '.join(missing)}"
        )
    return result


def load_split_examples(data_root: Path, split_files: dict[str, Path] | None = None) -> tuple[list[ParsedExample], Counter[str]]:
    data_root = data_root.expanduser().resolve()
    resolver = PathResolver(data_root)
    split_files = split_files or find_split_files(data_root)
    examples: list[ParsedExample] = []
    skip_counts: Counter[str] = Counter()
    for split, path in split_files.items():
        with path.open("r", encoding="utf-8") as handle:
            rows = parse_json_rows(json.load(handle))
        for idx, row in enumerate(rows):
            claim = first_text(row, ["claim", "claim_text", "statement"])
            if not claim:
                skip_counts["missing_claim"] += 1
                continue
            if "label" not in row:
                skip_counts["missing_label"] += 1
                continue
            try:
                label, label_id = label_to_name_id(row.get("label"))
            except ValueError:
                skip_counts["invalid_label"] += 1
                continue
            paperid = clean_text(first_raw(row, ["paperid", "paper_id", "paperId", "doc_id"]) or "unknown")
            request_id = clean_text(first_raw(row, ["request_id", "requestid", "id", "uid", "example_id"]) or str(idx))
            claim_type = first_text(row, ["claim_type", "reasoning_type", "reasoning", "category"])
            paper_path = clean_text(first_raw(row, ["paper_path", "paper", "paper_file"]) or "")
            section = row.get("section", row.get("section_title", ""))
            visual_items = extract_visual_items(
                row,
                resolver=resolver,
                json_dir=path.parent,
                section=section,
                paper_path=paper_path,
            )
            examples.append(
                ParsedExample(
                    split=split,
                    source_file=path,
                    row_index=idx,
                    raw=row,
                    claim=claim,
                    label=label,
                    label_id=label_id,
                    paperid=paperid,
                    request_id=request_id,
                    claim_type=claim_type,
                    paper_path=paper_path,
                    section=section,
                    visual_items=visual_items,
                )
            )
    return examples, skip_counts


def evidence_item_id(example: ParsedExample, item: VisualEvidenceItem) -> str:
    image_token = stable_hash(str(item.image_path) if item.image_path else Path(item.image_path_raw).name, 16)
    return "evidence:{paperid}:{modality}:{item_key}:{image_hash}".format(
        paperid=safe_id_part(example.paperid),
        modality=safe_id_part(item.modality),
        item_key=safe_id_part(item.item_key),
        image_hash=image_token,
    )


def claim_id(example: ParsedExample) -> str:
    return "claim:{split}:{paperid}:{request_id}:{claim_hash}".format(
        split=safe_id_part(example.split),
        paperid=safe_id_part(example.paperid),
        request_id=safe_id_part(example.request_id),
        claim_hash=stable_hash(example.claim, 16),
    )


def pair_id(example: ParsedExample, evidence_id: str) -> str:
    return "pair:{split}:{paperid}:{request_id}:{claim_hash}:{evidence_hash}".format(
        split=safe_id_part(example.split),
        paperid=safe_id_part(example.paperid),
        request_id=safe_id_part(example.request_id),
        claim_hash=stable_hash(example.claim, 16),
        evidence_hash=stable_hash(evidence_id, 16),
    )


def build_pair_records(
    examples: Iterable[ParsedExample],
    *,
    modalities: set[str],
    pair_mode: str = "single_visual_only",
) -> tuple[list[PairRecord], Counter[str]]:
    if pair_mode != "single_visual_only":
        raise NotImplementedError(
            f"pair_mode={pair_mode!r} is not implemented yet; use single_visual_only for this converter version"
        )
    records: list[PairRecord] = []
    skip_counts: Counter[str] = Counter()
    normalized_modalities = {normalize_modality(modality) for modality in modalities}
    for example in examples:
        if not example.claim:
            skip_counts["missing_claim"] += 1
            continue
        visual_items = [item for item in example.visual_items if item.modality in {"chart", "table"}]
        if not visual_items:
            skip_counts["no_visual_evidence"] += 1
            continue
        matching = [item for item in visual_items if item.modality in normalized_modalities]
        if not matching:
            skip_counts["no_matching_visual_evidence"] += 1
            continue
        if len(matching) > 1:
            skip_counts["multiple_matching_visual_evidence"] += 1
            continue
        item = matching[0]
        if item.image_path is None:
            skip_counts["image_not_found"] += 1
            continue
        evid_id = evidence_item_id(example, item)
        records.append(
            PairRecord(
                pair_id=pair_id(example, evid_id),
                claim_id=claim_id(example),
                evidence_item_id=evid_id,
                split=example.split,
                source_file=example.source_file,
                paperid=example.paperid,
                request_id=example.request_id,
                claim=example.claim,
                label=example.label,
                label_id=example.label_id,
                claim_type=example.claim_type,
                modality=item.modality,
                image_path=item.image_path,
                caption=item.caption,
                ocr_text=item.ocr_text,
                context_text=item.context_text,
                section_json=item.section_json,
                paper_path=item.paper_path or example.paper_path,
                item_key=item.item_key,
                n_evidence_items=len(matching),
                pair_mode=pair_mode,
            )
        )
    return records, skip_counts


def extract_section_title(section: Any) -> str:
    if isinstance(section, str):
        return clean_text(section)
    if isinstance(section, dict):
        for key in ("title", "section_title", "heading", "name"):
            if key in section and not is_leakage_key(key):
                text = clean_text(section.get(key))
                if text:
                    return text
    return ""


def extract_section_title_from_json(section_json: str) -> str:
    if not section_json:
        return ""
    try:
        return extract_section_title(json.loads(section_json))
    except json.JSONDecodeError:
        return clean_text(section_json)


def counts_for_examples(examples: Iterable[ParsedExample]) -> dict[str, Counter[str]]:
    examples = list(examples)
    modality_counts: Counter[str] = Counter()
    for example in examples:
        if not example.visual_items:
            modality_counts["none"] += 1
        for item in example.visual_items:
            modality_counts[item.modality] += 1
    return {
        "split": Counter(example.split for example in examples),
        "label": Counter(example.label for example in examples),
        "claim_type": Counter(example.claim_type or "unknown" for example in examples),
        "modality": modality_counts,
    }


def pair_manifest_row(record: PairRecord) -> dict[str, Any]:
    return {
        "pair_id": record.pair_id,
        "claim_id": record.claim_id,
        "evidence_item_id": record.evidence_item_id,
        "paperid": record.paperid,
        "request_id": record.request_id,
        "split": record.split,
        "claim": record.claim,
        "label": record.label,
        "label_id": record.label_id,
        "claim_type": record.claim_type,
        "modality": record.modality,
        "image_path": str(record.image_path),
        "section_json": record.section_json,
        "n_evidence_items": record.n_evidence_items,
        "gold_pair": record.gold_pair,
        "pair_mode": record.pair_mode,
        "caption": record.caption,
        "ocr_text": record.ocr_text,
        "context_text": record.context_text,
        "paper_path": record.paper_path,
        "item_key": record.item_key,
    }
