import json

from sciver_eval import shard


def _manifest(n):
    return {f"tag{i}": {"item_id": f"sciver_val_{i}", "trial": 1} for i in range(n)}


def test_cells_from_manifest_sorted():
    cells = shard.cells_from_manifest(_manifest(3))
    assert cells == [
        {"id": "sciver_val_0", "trial": 1},
        {"id": "sciver_val_1", "trial": 1},
        {"id": "sciver_val_2", "trial": 1},
    ]


def test_shard_round_robin_balanced():
    cells = [{"id": f"x{i}", "trial": 1} for i in range(5)]
    shards = shard.shard(cells, 2)
    assert [len(s) for s in shards] == [3, 2]


def test_group_round_robin():
    assert shard.group(["a", "b", "c", "d"], 4) == [["a"], ["b"], ["c"], ["d"]]
    assert shard.group(["a", "b", "c"], 2) == [["a", "c"], ["b"]]


def test_write_layout_drops_empty_shards_and_preserves_cells(tmp_path):
    gpath, paths = shard.write_layout(str(tmp_path), _manifest(3), n_shards=16, n_groups=4)
    assert len(paths) == 3                       # 16 requested, only 3 non-empty
    total = sum(len(json.load(open(p))["cells"]) for p in paths)
    assert total == 3
    groups = json.load(open(gpath))
    assert sorted(p for g in groups for p in g) == sorted(paths)
