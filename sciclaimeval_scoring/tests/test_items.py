import os
from sciclaimeval_scoring.items import load_items, Item

def test_loads_265_figures():
    items = load_items()
    assert len(items) == 265
    assert all(isinstance(i, Item) for i in items)

def test_fields_and_id_scheme():
    i = load_items()[0]
    assert i.vtype == "figure"
    assert i.source == "sciclaimeval"
    assert i.claim and i.image_path.endswith(".png")
    assert i.item_id.startswith("scev_")
    assert i.domain in {"ml", "nlp", "peerj"}

def test_label_mapping_counts():
    items = load_items()
    supported = sum(1 for i in items if i.label is True)
    refuted = sum(1 for i in items if i.label is False)
    assert (supported, refuted) == (149, 116)

def test_item_ids_unique():
    ids = [i.item_id for i in load_items()]
    assert len(ids) == len(set(ids))

def test_image_files_exist_for_sample():
    for i in load_items()[:25]:
        assert os.path.isfile(i.image_path), i.image_path
