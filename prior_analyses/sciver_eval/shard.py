"""Split a run manifest into cell-shards + workflow groups for wf_dispatch.js.

Reproduces the run-1 dispatch layout deterministically (the original step was
ad-hoc and uncommitted): reads a prepare manifest, emits cells_shardKK.json
({"cells":[{id,trial}]}) and wf_groups.json (list of groups, each a list of
shard paths). Empty shards are dropped so no agent is spawned for zero cells.

    python -m sciver_eval.shard --manifest <run_NN_manifest.json> [--shards 16 --groups 4]
"""
import argparse
import json
import os
from pathlib import Path


def cells_from_manifest(manifest):
    """[{'id':item_id,'trial':trial}] from a prepare manifest, sorted for determinism."""
    cells = [{"id": v["item_id"], "trial": int(v["trial"])} for v in manifest.values()]
    cells.sort(key=lambda c: (c["id"], c["trial"]))
    return cells


def shard(cells, n_shards):
    """Round-robin cells into n_shards lists (balanced, deterministic)."""
    shards = [[] for _ in range(n_shards)]
    for i, c in enumerate(cells):
        shards[i % n_shards].append(c)
    return shards


def group(shard_paths, n_groups):
    """Round-robin shard paths into n_groups non-empty groups (parallel workflows)."""
    groups = [[] for _ in range(n_groups)]
    for i, p in enumerate(shard_paths):
        groups[i % n_groups].append(p)
    return [g for g in groups if g]


def write_layout(out_dir, manifest, n_shards=16, n_groups=4):
    cells = cells_from_manifest(manifest)
    non_empty = [s for s in shard(cells, n_shards) if s]
    paths = []
    for k, s in enumerate(non_empty):
        p = os.path.join(out_dir, f"cells_shard{k:02d}.json")
        with open(p, "w") as f:
            json.dump({"cells": s}, f)
        paths.append(p)
    groups = group(paths, n_groups)
    gpath = os.path.join(out_dir, "wf_groups.json")
    with open(gpath, "w") as f:
        json.dump(groups, f)
    return gpath, paths


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--shards", type=int, default=16)
    ap.add_argument("--groups", type=int, default=4)
    a = ap.parse_args()
    manifest = json.load(open(a.manifest))
    out_dir = str(Path(a.manifest).parent)
    gpath, paths = write_layout(out_dir, manifest, a.shards, a.groups)
    total = sum(len(json.load(open(p))["cells"]) for p in paths)
    print(f"{total} cells -> {len(paths)} shards; groups -> {gpath}")
