"""Resume-aware pass preparation: register the pass, write prompts for MISSING cells only,
emit a manifest the orchestrator and collector both use."""
import argparse, json, os
from sciclaimeval_scoring import config, db, items as items_mod
from sciclaimeval_scoring.prompts import build_prompt, tag, DIMS

SKILL_VERSION = "0.1"
MODEL = "claude-sonnet-4-6"
DEVIATIONS = ("temp!=0 (subagent path; SciVer-measured ~82% modal/89% within-1); "
              "output cap instructed not enforced; reasoning off; captions omitted; "
              "DeLeAn v1.0 textual-only anchors on multimodal figure items")

def plan_missing(conn, pass_no, items, dims):
    done = db.completed_cells(conn, pass_no)
    return [{"item_id": it.item_id, "dim_code": d} for it in items for d in dims
            if (it.item_id, d) not in done]

def prepare(pass_no, dims, out_dir, limit=None):
    conn = db.connect(config.db_path()); db.init(conn)
    items = items_mod.load_items()
    if limit is not None:
        items = items[:limit]
    for it in items:
        db.upsert_item(conn, it)
    db.register_pass(conn, pass_no, skill_version=SKILL_VERSION, model=MODEL,
                     active_dims=",".join(dims), item_source="SciClaimEval dev figures (265)",
                     n_items=len(items), params=json.dumps({"limit": limit}), deviations=DEVIATIONS)
    by_id = {it.item_id: it for it in items}
    rubrics = {d: (config.rubrics_dir() / f"{d}.txt").read_text(encoding="utf-8") for d in dims}
    os.makedirs(out_dir, exist_ok=True)
    manifest = {}
    for cell in plan_missing(conn, pass_no, items, dims):
        it = by_id[cell["item_id"]]; d = cell["dim_code"]; t = tag(pass_no, it.item_id, d)
        path = os.path.join(out_dir, f"{t}.txt")
        with open(path, "w", encoding="utf-8") as f:
            f.write(build_prompt(pass_no, it, d, rubrics[d]))
        manifest[t] = {"item_id": it.item_id, "dim_code": d, "dim_name": DIMS[d],
                       "prompt_file": path, "figure": it.image_path}
    mpath = os.path.join(out_dir, f"pass_{pass_no:02d}_manifest.json")
    with open(mpath, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"pass {pass_no}: {len(items)} items x {len(dims)} dims; "
          f"{len(manifest)} MISSING cells written to {out_dir}; manifest {mpath}")
    return mpath

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--pass", dest="pass_no", type=int, required=True)
    ap.add_argument("--dims", default="AS,QLq,QLl,MCr,AT,VO")
    ap.add_argument("--out", default=None)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    out = a.out or str(config.repo_root() / "sciclaimeval_scoring" / "prompts" / f"pass_{a.pass_no:02d}")
    prepare(a.pass_no, a.dims.split(","), out, a.limit)
