"""Validate and merge batch_results/batch_NN.json into the discovery state:
- append emergent dimensions to codebook.md (## Emergent dimensions)
- append rows to saturation_log.csv (item_index = global 1-based index)
- update manifest.csv needs_context_judged
- append twin notes to codebook.md (## Leakage-suspect flags)
- append a line to codebook.md (## Batch log)
- print the saturation-criterion status

Usage: python3 feature_discovery/merge_batch.py N
Idempotence guard: refuses to merge a batch already in the Batch log.
"""
import csv
import json
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
N = int(sys.argv[1])

batches = json.load(open(os.path.join(ROOT, "batches.json")))["batches"]
batch_ids = batches[N]
start_index = sum(len(b) for b in batches[:N]) + 1
index_of = {iid: start_index + k for k, iid in enumerate(batch_ids)}
end_index = start_index + len(batch_ids) - 1

res = json.load(open(os.path.join(ROOT, "batch_results", f"batch_{N:02d}.json")))

cb_path = os.path.join(ROOT, "codebook.md")
codebook = open(cb_path).read()
if f"\n- batch {N:02d}:" in codebook:
    sys.exit(f"batch {N:02d} already merged; aborting")

# --- validate ---------------------------------------------------------------
dim_names = re.findall(r"^### (\w+)", codebook, flags=re.M)
got_items = {it["item_id"] for it in res["items"]}
assert got_items == set(batch_ids), f"item mismatch: {got_items ^ set(batch_ids)}"
seeded_dims = set(dim_names[:23])  # first 23 are the seeded set
items_missing_seeded = 0
for it in res["items"]:
    if [d for d in seeded_dims if d not in it["tallies"]]:
        items_missing_seeded += 1
if items_missing_seeded:
    print(f"WARN {items_missing_seeded} items missing some SEEDED tallies "
          f"(emergent-dim gaps are expected for parallel-wave batches)")
existing_lower = {d.lower() for d in dim_names}
new_dims = []
for d in res.get("new_dimensions", []):
    if d["name"].lower() in existing_lower:
        print(f"NOTE: '{d['name']}' duplicates an existing dimension name; skipped")
        continue
    new_dims.append(d)

# --- codebook: emergent dimensions -------------------------------------------
emergent_blocks = []
for d in new_dims:
    fs = index_of.get(d.get("first_seen_item_id"), start_index)
    anchors = "; ".join(f"{k} = {v}" for k, v in sorted(d["anchors"].items()))
    emergent_blocks.append(
        f"### {d['name']}  [{d.get('taxonomy', 'unspecified')}]\n"
        f"- origin: emergent\n"
        f"- first_seen: item {fs} ({d.get('first_seen_item_id', '?')})\n"
        f"- definition: {d['definition']}\n"
        f"- anchors: {anchors}\n"
        f"- exemplars: high = {', '.join(d.get('exemplars_high', []))}; "
        f"low = {', '.join(d.get('exemplars_low', []))}\n"
        f"- rationale: {d.get('rationale', '')}\n"
    )
if emergent_blocks:
    codebook = codebook.replace(
        "---\n\n## Leakage-suspect flags",
        "\n".join(emergent_blocks) + "\n---\n\n## Leakage-suspect flags")

# --- codebook: twin notes -----------------------------------------------------
twin_lines = []
for t in res.get("twin_notes", []):
    dims = t.get("dimensions_scoring_differently", [])
    if dims:
        twin_lines.append(f"- {t['twin_group']} (batch {N:02d}): "
                          f"{', '.join(dims)} — {t.get('why', '')}")
if twin_lines:
    codebook = codebook.replace(
        "---\n\n## Batch log",
        "\n".join(twin_lines) + "\n\n---\n\n## Batch log")

# --- codebook: batch log -------------------------------------------------------
codebook = codebook.rstrip() + (
    f"\n- batch {N:02d}: items {start_index}-{end_index}, "
    f"{len(new_dims)} new dims"
    f" ({', '.join(d['name'] for d in new_dims) if new_dims else 'none'})\n")
open(cb_path, "w").write(codebook)

# --- saturation log -----------------------------------------------------------
with open(os.path.join(ROOT, "saturation_log.csv"), "a", newline="") as f:
    w = csv.writer(f)
    for d in new_dims:
        w.writerow([index_of.get(d.get("first_seen_item_id"), start_index),
                    d["name"]])

# --- manifest needs_context ----------------------------------------------------
man_path = os.path.join(ROOT, "manifest.csv")
rows = list(csv.DictReader(open(man_path)))
nc = {it["item_id"]: it.get("needs_context", "tbd") for it in res["items"]}
for r in rows:
    if r["item_id"] in nc:
        r["needs_context_judged"] = nc[r["item_id"]]
with open(man_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

# --- saturation criterion -------------------------------------------------------
log = list(csv.DictReader(open(os.path.join(ROOT, "saturation_log.csv"))))
recent = [r for r in log if int(r["item_index"]) > end_index - 50]
print(f"merged batch {N:02d}: {len(new_dims)} new dims; "
      f"total emergent so far: {len(log)}")
print(f"new dims in last 50 items (window {max(1, end_index - 49)}-{end_index}): "
      f"{len(recent)}")
if end_index >= 50 and len(recent) < 2:
    print("SATURATION CRITERION MET: fewer than 2 new dimensions in the "
          "last 50 items — stop early and write final outputs.")
else:
    print("Continue to next batch.")
