from rubric_scoring.items import load_items, Item

def test_loads_817_charts():
    items = load_items()
    assert len(items) == 817            # build-report chart count; reconcile if different
    assert all(isinstance(i, Item) for i in items)

def test_items_are_charts_with_required_fields():
    i = load_items()[0]
    assert i.vtype == "chart"
    assert i.claim and i.image_path.endswith(".png")
    assert i.item_id.startswith("sciver_")

def test_item_ids_unique():
    ids = [i.item_id for i in load_items()]
    assert len(ids) == len(set(ids))

def test_image_files_exist_for_sample():
    import os
    for i in load_items()[:25]:
        assert os.path.isfile(i.image_path), i.image_path
