"""Load CharXiv's RELEASED per-model correctness scores into the model_score table.

Source: <CharXiv>/existing_evaluations/scores-<model>-<task>_val.json (downloaded from the HF
dataset repo). Two shapes:
  * reasoning  : keyed by figure_id   -> {figure_id, extracted_answer, score}   (1 per chart)
  * descriptive: keyed by '<fid>_<q>' -> {resp_id, extracted_answer, score}     (4 per chart)
Each row maps to item_id = charxiv_val_<figure_id>, joinable to item / annotation.
"""
import json
from charxiv_scoring import config

def _parse_name(fname):
    # scores-<model>-<task>_val.json  ->  (model, task)
    stem = fname[len("scores-"):]
    for task in ("reasoning", "descriptive"):
        suf = f"-{task}_val.json"
        if stem.endswith(suf):
            return stem[: -len(suf)], task
    return None, None

def load_scores(conn):
    eval_dir = config.charxiv_dir() / "existing_evaluations"
    files = sorted(eval_dir.glob("scores-*_val.json"))
    n_rows = 0; n_files = 0; models = set()
    for fp in files:
        model, task = _parse_name(fp.name)
        if model is None:
            continue
        data = json.loads(fp.read_text())
        for key, rec in data.items():
            if task == "reasoning":
                fid = int(rec.get("figure_id", key)); sub_q = 0
            else:  # descriptive resp_id like '3_1'
                fid = int(str(rec.get("resp_id", key)).split("_")[0]); sub_q = int(str(key).split("_")[1])
            sc = rec.get("score")
            conn.execute(
                "INSERT OR REPLACE INTO model_score VALUES(?,?,?,?,?,?,?)",
                (model, task, f"charxiv_val_{fid}", sub_q, fid,
                 rec.get("extracted_answer"), int(sc) if sc is not None else None))
            n_rows += 1
        n_files += 1; models.add(model)
    conn.commit()
    return {"files": n_files, "rows": n_rows, "models": sorted(models)}
