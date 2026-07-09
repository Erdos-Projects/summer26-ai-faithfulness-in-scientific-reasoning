"""Canonical item set: every CharXiv VALIDATION reasoning question. Verified == 1000.
One item == one (chart, reasoning question), keyed by figure_id. Built directly from the
released val JSONs (data/reasoning_val.json + data/image_metadata_val.json).

`answer` and `inst_category` are stored for analysis but are verdict-blind: never sent to an
annotator prompt (build_prompt reads only `query` + `image_path`)."""
import json
from dataclasses import dataclass
from charxiv_scoring import config

@dataclass(frozen=True)
class Item:
    item_id: str        # charxiv_val_<figure_id>
    source: str         # "charxiv"
    paperid: str        # arXiv id
    figure_id: int
    image_path: str     # absolute path to <figure_id>.jpg
    query: str          # the reasoning question (the task)
    answer: str         # ground truth — VERDICT-BLIND, stored not prompted
    inst_category: int  # 1=TC 2=TG 3=NC 4=NG (answer-format reasoning category)
    category: str       # arXiv subject (cs, stat, ...)
    year: str
    title: str

def load_items():
    cd = config.charxiv_dir()
    reason = json.loads((cd / "data" / "reasoning_val.json").read_text())
    meta = json.loads((cd / "data" / "image_metadata_val.json").read_text())
    items = []
    for key, r in reason.items():
        fid = r["figure_id"]
        m = meta.get(str(fid), {})
        items.append(Item(
            item_id=f"charxiv_val_{fid}",
            source="charxiv",
            paperid=m.get("paper_id", ""),
            figure_id=fid,
            image_path=str(cd / "images" / f"{fid}.jpg"),
            query=r["query"],
            answer=str(r.get("answer", "")),
            inst_category=int(r.get("inst_category", 0)),
            category=m.get("category", ""),
            year=m.get("year", ""),
            title=m.get("title", ""),
        ))
    items.sort(key=lambda it: it.figure_id)
    return items
