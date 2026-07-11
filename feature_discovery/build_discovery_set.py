"""Build the feature-discovery sample, splits, manifest, and batch plan.

Run from the Erdos root. Deterministic: SEED fixed below.

Outputs (all under feature_discovery/):
  sciver_singlefig_split.json   - authoritative paper-grouped 80/20 split over
                                  the SciVer single-figure (type=='chart') pool
  sciclaimeval_split.json       - authoritative paper-grouped 80/20 split over
                                  the SciClaimEval figure-evidence dev pool
  discovery_items.csv           - item_id, source, paper_id, figure_path
  manifest.csv                  - full per-item metadata for the discovery set
  figures/<item_id>.png         - copied evidence figures
  batches.json                  - 30 batches of 10 item_ids (sources mixed,
                                  twin pairs kept within one batch)
"""
import csv
import json
import os
import random
import shutil
from collections import Counter, defaultdict

SEED = 20260612
ROOT = os.path.dirname(os.path.abspath(__file__))
ERDOS = os.path.dirname(ROOT)
FIGDIR = os.path.join(ROOT, "figures")
os.makedirs(FIGDIR, exist_ok=True)

rng = random.Random(SEED)

# ---------------------------------------------------------------- SciVer ----
sciver_items = []
for fname in ("valset.json", "testset.json"):
    data = json.load(open(os.path.join(ERDOS, "SciVer", fname)))
    src = fname.replace(".json", "")
    for idx, x in enumerate(data):
        if x.get("type") != "chart":
            continue  # tables and multi-evidence items are out of scope
        sciver_items.append({
            "item_id": f"sciver_{src.replace('set','')}_{idx:04d}",
            "source_file": f"SciVer/{fname}",
            "source_index": idx,
            "paper_id": x["paperid"],
            "request_id": x["request_id"],
            "item": x["item"],
            "claim_type": x["claim_type"],
            "label": bool(x["label"]),
            "image_path": x["image_path"],
            "claim": x["claim"],
        })

papers = sorted({x["paper_id"] for x in sciver_items})
rng.shuffle(papers)
items_per_paper = Counter(x["paper_id"] for x in sciver_items)
total = len(sciver_items)
test_papers, test_n = set(), 0
for p in papers:
    if test_n >= round(0.2 * total):
        break
    test_papers.add(p)
    test_n += items_per_paper[p]
sciver_train = [x for x in sciver_items if x["paper_id"] not in test_papers]
sciver_test = [x for x in sciver_items if x["paper_id"] in test_papers]

# captions from paper JSONs
def sciver_caption(x):
    try:
        paper = json.load(open(os.path.join(ERDOS, "SciVer", "papers",
                                            x["paper_id"] + ".json")))
        info = paper.get("image_paths", {}).get(str(x["item"]))
        if info is None:  # subfigure key like '3(a)' -> try base number
            base = str(x["item"]).split("(")[0]
            info = paper.get("image_paths", {}).get(base)
        return (info or {}).get("caption", "")
    except Exception:
        return ""

# sample 200 from train: stratified over claim_type (only direct/analytical
# exist in the single-figure pool), label-balanced within each stratum
sciver_sample = []
for ctype in ("direct", "analytical"):
    for lab in (True, False):
        pool = [x for x in sciver_train
                if x["claim_type"] == ctype and x["label"] == lab]
        rng.shuffle(pool)
        sciver_sample.extend(pool[:50])
assert len(sciver_sample) == 200, len(sciver_sample)

# ---------------------------------------------------------- SciClaimEval ----
sce_all = json.load(open(os.path.join(
    ERDOS, "SciClaimEval", "sciclaimeval-shared-task", "data",
    "dev_task1_release.json")))
sce_figs = [x for x in sce_all if x["evi_type"] == "figure"]
sce_papers = sorted({x["paper_id"] for x in sce_figs})
rng.shuffle(sce_papers)
sce_per_paper = Counter(x["paper_id"] for x in sce_figs)
sce_total = len(sce_figs)
sce_test_papers, n = set(), 0
for p in sce_papers:
    if n >= round(0.2 * sce_total):
        break
    sce_test_papers.add(p)
    n += sce_per_paper[p]
sce_train = [x for x in sce_figs if x["paper_id"] not in sce_test_papers]
sce_test = [x for x in sce_figs if x["paper_id"] in sce_test_papers]

# twin pairs: claim_id_pair groups of exactly 2 (Supported original vs
# Refuted modified). The supported-only items share one large group id.
pair_groups = defaultdict(list)
for x in sce_train:
    pair_groups[x["claim_id_pair"]].append(x)
twin_pairs = [g for g in pair_groups.values() if len(g) == 2]
rng.shuffle(twin_pairs)

# pick 10 twin pairs spread across modification operations
pairs_by_op = defaultdict(list)
for g in twin_pairs:
    pairs_by_op[g[0]["operation"]].append(g)
chosen_pairs = []
ops_cycle = sorted(pairs_by_op, key=lambda o: -len(pairs_by_op[o]))
while len(chosen_pairs) < 10:
    progressed = False
    for op in ops_cycle:
        if pairs_by_op[op] and len(chosen_pairs) < 10:
            chosen_pairs.append(pairs_by_op[op].pop())
            progressed = True
    if not progressed:
        break
chosen_pair_items = [x for g in chosen_pairs for x in g]
chosen_ids = {x["claim_id"] for x in chosen_pair_items}

# proportional stratification over operation for the remaining quota
remaining_quota = 100 - len(chosen_pair_items)
train_ops = Counter(x["operation"] for x in sce_train)
already = Counter(x["operation"] for x in chosen_pair_items)
targets = {}
for op, cnt in train_ops.items():
    targets[op] = max(0, round(100 * cnt / len(sce_train)) - already[op])
# adjust rounding drift
drift = remaining_quota - sum(targets.values())
for op in sorted(targets, key=lambda o: -train_ops[o]):
    if drift == 0:
        break
    step = 1 if drift > 0 else -1
    targets[op] += step
    drift -= step

sce_sample = list(chosen_pair_items)
for op, t in targets.items():
    pool = [x for x in sce_train
            if x["operation"] == op and x["claim_id"] not in chosen_ids]
    rng.shuffle(pool)
    sce_sample.extend(pool[:t])
assert len(sce_sample) == 100, len(sce_sample)

twin_id_of = {}
for i, g in enumerate(chosen_pairs):
    for x in g:
        twin_id_of[x["claim_id"]] = f"twin_{i:02d}_{g[0]['claim_id_pair']}"

# ------------------------------------------------------------- figures ------
def copy_fig(src_abs, item_id):
    dst = os.path.join(FIGDIR, item_id + ".png")
    if not os.path.exists(dst):
        shutil.copyfile(src_abs, dst)
    return os.path.join("feature_discovery", "figures", item_id + ".png")

manifest_rows = []
for x in sciver_sample:
    src_abs = os.path.join(ERDOS, x["image_path"].lstrip("./"))
    figpath = copy_fig(src_abs, x["item_id"])
    manifest_rows.append({
        "item_id": x["item_id"], "source": "sciver",
        "claim_text": x["claim"], "label": x["label"],
        "figure_path": figpath, "paper_id": x["paper_id"],
        "subset_or_operation": x["claim_type"], "twin_id": "",
        "needs_context_dataset": "",  # SciVer has no such field
        "needs_context_judged": "tbd",
        "caption": sciver_caption(x),
    })
for x in sce_sample:
    item_id = "scev_" + x["claim_id"]
    src_abs = os.path.join(ERDOS, "SciClaimEval", "sciclaimeval-shared-task",
                           "data", x["evi_path"])
    figpath = copy_fig(src_abs, item_id)
    manifest_rows.append({
        "item_id": item_id, "source": "sciclaimeval",
        "claim_text": x["claim"], "label": x["label"],
        "figure_path": figpath, "paper_id": x["paper_id"],
        "subset_or_operation": x["operation"],
        "twin_id": twin_id_of.get(x["claim_id"], ""),
        "needs_context_dataset": x["use_context"],
        "needs_context_judged": "tbd",
        "caption": x["caption"],
    })

with open(os.path.join(ROOT, "manifest.csv"), "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(manifest_rows[0]))
    w.writeheader()
    w.writerows(manifest_rows)

with open(os.path.join(ROOT, "discovery_items.csv"), "w", newline="") as f:
    w = csv.writer(f)
    w.writerow(["item_id", "source", "paper_id", "figure_path"])
    for r in manifest_rows:
        w.writerow([r["item_id"], r["source"], r["paper_id"],
                    r["figure_path"]])

# ------------------------------------------------------------- splits -------
discovery_sciver = {x["item_id"] for x in sciver_sample}
def sciver_record(x):
    r = {k: x[k] for k in ("item_id", "source_file", "source_index",
                           "paper_id", "request_id", "item", "claim_type",
                           "label", "image_path")}
    r["used_in_feature_discovery"] = x["item_id"] in discovery_sciver
    return r

json.dump({
    "meta": {
        "description": "Authoritative paper-grouped 80/20 split over the "
                       "SciVer single-figure-evidence pool (type=='chart', "
                       "valset+testset). Multi-evidence items excluded.",
        "created": "2026-06-12", "seed": SEED,
        "pool_size": total, "n_train": len(sciver_train),
        "n_test": len(sciver_test),
        "n_train_papers": len(papers) - len(test_papers),
        "n_test_papers": len(test_papers),
        "note": "Single-figure pool contains only claim_type direct and "
                "analytical; parallel and sequential are multi-evidence "
                "by construction and out of scope.",
    },
    "train": [sciver_record(x) for x in sciver_train],
    "test": [sciver_record(x) for x in sciver_test],
}, open(os.path.join(ROOT, "sciver_singlefig_split.json"), "w"), indent=1)

discovery_sce = {"scev_" + x["claim_id"] for x in sce_sample}
def sce_record(x):
    return {
        "item_id": "scev_" + x["claim_id"], "claim_id": x["claim_id"],
        "paper_id": x["paper_id"], "label": x["label"],
        "operation": x["operation"], "evi_path": x["evi_path"],
        "claim_id_pair": x["claim_id_pair"], "domain": x["domain"],
        "use_context": x["use_context"],
        "used_in_feature_discovery": ("scev_" + x["claim_id"]) in discovery_sce,
    }

json.dump({
    "meta": {
        "description": "Authoritative paper-grouped 80/20 split over the "
                       "SciClaimEval dev figure-evidence pool "
                       "(dev_task1_release.json, evi_type=='figure').",
        "created": "2026-06-12", "seed": SEED,
        "pool_size": sce_total, "n_train": len(sce_train),
        "n_test": len(sce_test),
        "n_train_papers": len(sce_papers) - len(sce_test_papers),
        "n_test_papers": len(sce_test_papers),
    },
    "train": [sce_record(x) for x in sce_train],
    "test": [sce_record(x) for x in sce_test],
}, open(os.path.join(ROOT, "sciclaimeval_split.json"), "w"), indent=1)

# ------------------------------------------------------------- batches ------
# 30 batches of 10, mixed sources; twin-pair members stay in one batch.
singles = [r["item_id"] for r in manifest_rows if not r["twin_id"]]
pairs = defaultdict(list)
for r in manifest_rows:
    if r["twin_id"]:
        pairs[r["twin_id"]].append(r["item_id"])
units = [[s] for s in singles] + list(pairs.values())
rng.shuffle(units)
batches, cur = [], []
for u in units:
    if len(cur) + len(u) > 10:
        batches.append(cur)
        cur = []
    cur += u
if cur:
    batches.append(cur)
assert sum(len(b) for b in batches) == 300
json.dump({"seed": SEED, "batches": batches},
          open(os.path.join(ROOT, "batches.json"), "w"), indent=1)

# ------------------------------------------------------------- report -------
print("SciVer pool:", total, "train:", len(sciver_train),
      "test:", len(sciver_test))
print("SciVer sample claim_type:",
      Counter(x["claim_type"] for x in sciver_sample),
      "labels:", Counter(x["label"] for x in sciver_sample))
print("SCE pool:", sce_total, "train:", len(sce_train),
      "test:", len(sce_test))
print("SCE sample ops:", Counter(x["operation"] for x in sce_sample))
print("twin pairs:", len(chosen_pairs),
      "ops:", Counter(g[0]["operation"] for g in chosen_pairs))
print("batches:", len(batches),
      "sizes:", Counter(len(b) for b in batches))
print("captions missing:",
      sum(1 for r in manifest_rows if not r["caption"]))
