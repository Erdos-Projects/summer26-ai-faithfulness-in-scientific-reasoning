"""Draft the CharXiv DB: init schema, load the 1000 val items, load all released model scores.
Idempotent — safe to re-run. Does NOT run the demand pass (that's prepare + dispatch + collect).

    python -m charxiv_scoring.build_db
"""
from charxiv_scoring import config, db, items as items_mod, scores as scores_mod

def main():
    conn = db.connect(config.db_path()); db.init(conn)
    items = items_mod.load_items()
    for it in items:
        db.upsert_item(conn, it)
    conn.commit()
    info = scores_mod.load_scores(conn)

    n_items = conn.execute("SELECT COUNT(*) FROM item").fetchone()[0]
    # reasoning correctness coverage per model (1 row/item), for a quick sanity readout
    rows = conn.execute(
        "SELECT model, COUNT(*), SUM(score) FROM model_score "
        "WHERE task='reasoning' AND score IS NOT NULL GROUP BY model ORDER BY model").fetchall()
    print(f"DB: {config.db_path()}")
    print(f"items: {n_items} | score files: {info['files']} | score rows: {info['rows']} "
          f"| models: {len(info['models'])}")
    print("\nreasoning accuracy by model (released CharXiv scores):")
    for m, n, c in rows:
        print(f"  {m:28s} {c:4d}/{n:<4d}  acc={c/n:.3f}")
    conn.close()

if __name__ == "__main__":
    main()
