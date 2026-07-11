"""Resume-aware run preparation: ensure items + run exist, write prompts for
MISSING (item, trial) cells only, emit a manifest carrying the gold label.

    python -m sciclaimeval_eval.prepare --run 1 --trials 5
"""
import argparse
import json
import os

from sciclaimeval_scoring import config, items as items_mod
from sciclaimeval_eval import db, prompts

DEFAULT_MODEL = "claude-haiku-4-5"


def prepare(run_no, out_dir, limit=None, trials=1, model=None):
    conn = db.connect(db.default_db_path())
    db.init(conn)

    items = items_mod.load_items()
    if limit is not None:
        items = items[:limit]
    for it in items:
        db.upsert_item(conn, it)
    conn.commit()

    model_id = model or DEFAULT_MODEL
    run = conn.execute("SELECT model FROM run WHERE run_id=?", (run_no,)).fetchone()
    if run is None:
        db.register_run(conn, run_no, skill_version="sciclaimeval_eval-0.1", model=model_id,
                        prompt_version="sciclaimeval-cot-v2-usecontext", temperature=None,
                        item_source="SciClaimEval dev figures (265)",
                        n_items=len(items),
                        params="harness-subagent; CoT; max_tokens=10240",
                        deviations=db.TEMPERATURE_UNCONTROLLED_NOTE)
        model = model_id
    else:
        model = run[0]

    records = prompts.load_records()
    done = db.completed_cells(conn, run_no, model)  # {(item_id, trial)}
    os.makedirs(out_dir, exist_ok=True)

    manifest = {}
    for it in items:
        for tr in range(1, trials + 1):
            if (it.item_id, tr) in done:
                continue
            rec = records[it.item_id]
            t = prompts.tag(run_no, it.item_id, tr)
            path = os.path.join(out_dir, f"{t}.txt")
            with open(path, "w", encoding="utf-8") as f:
                f.write(prompts.build_prompt(run_no, it, rec, tr))
            manifest[t] = {"item_id": it.item_id, "trial": tr, "prompt_file": path,
                           "images": [it.image_path], "label": (1 if it.label else 0)}

    mpath = os.path.join(out_dir, f"run_{run_no:02d}_manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"run {run_no} ({model}): {len(items)} items x {trials} trials; "
          f"{len(manifest)} MISSING cells -> {out_dir}; manifest {mpath}")
    return mpath


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", dest="run_no", type=int, default=1)
    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    out = a.out or str(config.repo_root() / "sciclaimeval_eval" / "prompts" / f"run_{a.run_no:02d}")
    prepare(a.run_no, out, a.limit, a.trials, a.model)
