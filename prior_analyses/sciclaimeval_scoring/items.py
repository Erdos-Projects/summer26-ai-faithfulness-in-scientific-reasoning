"""Canonical item set: every SciClaimEval dev FIGURE claim. Verified == 265.
Built directly from the released dataset (not feature_discovery) to stay self-contained.
`operation` and `label` are stored but verdict-blind: never sent to an annotator prompt."""
import json
from dataclasses import dataclass
from sciclaimeval_scoring import config

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
    domain: str
    operation: str

def load_items():
    sd = config.sciclaimeval_dir()
    data = json.loads((sd / "dev_task1_release.json").read_text())
    items = []
    for x in data:
        if x.get("evi_type") != "figure":
            continue
        items.append(Item(
            item_id=f"scev_{x['claim_id']}",
            source="sciclaimeval",
            paperid=x.get("paper_id", ""),
            claim_type="",
            vtype="figure",
            image_path=str(sd / x["evi_path"]),
            claim=x["claim"],
            label=(x["label"] == "Supported"),
            domain=x.get("domain", ""),
            operation=x.get("operation", ""),
        ))
    return items
