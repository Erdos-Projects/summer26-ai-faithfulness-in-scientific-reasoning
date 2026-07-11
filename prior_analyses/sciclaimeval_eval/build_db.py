"""Create predictions_sciclaimeval.db, populate the 265 shared figure items, register run 1.

Items come from sciclaimeval_scoring.items.load_items() so item_ids match
annotations_sciclaimeval.db (the join key). Idempotent.

    python -m sciclaimeval_eval.build_db
"""
import argparse
from pathlib import Path

from sciclaimeval_scoring import items as scev_items
from sciclaimeval_eval import db

DEFAULT_DB = Path(__file__).resolve().parent / "predictions_sciclaimeval.db"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--run", type=int, default=1)
    ap.add_argument("--model", default="claude-haiku-4-5")
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--prompt-version", default="sciclaimeval-cot-v2-usecontext")
    args = ap.parse_args()

    conn = db.connect(args.db)
    db.init(conn)

    its = scev_items.load_items()
    for it in its:
        db.upsert_item(conn, it)
    conn.commit()

    db.register_run(
        conn, args.run, skill_version="sciclaimeval_eval-0.1", model=args.model,
        prompt_version=args.prompt_version, temperature=args.temperature,
        item_source="SciClaimEval dev figures (265)", n_items=len(its),
        params="harness-subagent; CoT; max_tokens=10240",
        deviations="" if args.temperature is not None else db.TEMPERATURE_UNCONTROLLED_NOTE,
    )
    n_item = conn.execute("SELECT COUNT(*) FROM item").fetchone()[0]
    print(f"DB: {args.db}")
    print(f"items: {n_item}  |  run {args.run} registered for model={args.model}")
    print("by domain:", conn.execute(
        "SELECT domain, COUNT(*) FROM item GROUP BY domain").fetchall())


if __name__ == "__main__":
    main()
