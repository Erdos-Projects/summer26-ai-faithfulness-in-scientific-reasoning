"""Canonical item set: every SciVer chart claim (val + test). Verified == 817."""
import json
from dataclasses import dataclass
from pathlib import Path
from rubric_scoring import config

@dataclass(frozen=True)
class Item:
    item_id: str
    source: str
    paperid: str
    claim_type: str
    vtype: str
    image_path: str   # absolute
    claim: str
    label: bool

def _img_abs(image_path: str) -> str:
    return str(config.sciver_dir() / "images" / Path(image_path).name)

def load_items():
    sd = config.sciver_dir()
    items = []
    for split, fname in (("val", "valset.json"), ("test", "testset.json")):
        data = json.loads((sd / fname).read_text())
        for idx, x in enumerate(data):
            if x.get("type") != "chart":
                continue
            items.append(Item(
                item_id=f"sciver_{split}_{idx}",
                source="sciver",
                paperid=x.get("paperid", ""),
                claim_type=x.get("claim_type", ""),
                vtype="chart",
                image_path=_img_abs(x["image_path"]),
                claim=x["claim"],
                label=bool(x["label"]),
            ))
    return items
