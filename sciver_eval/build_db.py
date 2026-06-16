"""Create predictions.db, populate the 817 shared chart items, register run 1.

Items come from `rubric_scoring.items.load_items()` so the `item_id`s are
guaranteed identical to `annotations_prod.db` (the join key). Idempotent.

    python -m sciver_eval.build_db
"""
import argparse
from pathlib import Path

from rubric_scoring import items as rubric_items
from sciver_eval import db

DEFAULT_DB = Path(__file__).resolve().parent / "predictions.db"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--temperature", type=float, default=1.0)
    ap.add_argument("--prompt-version", default="sciver-cot-v1")
    args = ap.parse_args()

    conn = db.connect(args.db)
    db.init(conn)

    its = rubric_items.load_items()
    for it in its:
        db.upsert_item(conn, it)
    conn.commit()

    db.register_run(
        conn, args.run,
        skill_version="sciver_eval-0.1",
        model=args.model,
        prompt_version=args.prompt_version,
        temperature=args.temperature,
        item_source="SciVer charts (val+test) [= rubric 817]",
        n_items=len(its),
        params=f"harness-subagent; CoT; max_tokens=10240",
    )

    n_item = conn.execute("SELECT COUNT(*) FROM item").fetchone()[0]
    print(f"DB: {args.db}")
    print(f"items: {n_item}  |  run {args.run} registered for model={args.model}")
    print("by claim_type:", conn.execute(
        "SELECT claim_type, COUNT(*) FROM item GROUP BY claim_type").fetchall())


if __name__ == "__main__":
    main()
