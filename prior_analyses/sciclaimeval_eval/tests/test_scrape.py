import json, os
from sciclaimeval_eval import db, scrape

def test_collect_scores_correct_against_manifest_label(tmp_path, monkeypatch):
    # one fake transcript that answers "yes" for a tag whose gold label is 1 (Supported)
    proj = tmp_path / "projects" / "sess" / "subagents"
    proj.mkdir(parents=True)
    tagname = "sce_r01__scev_val_fig_0001__t01"
    line = {"agentId": "a1", "timestamp": "2026-06-16T00:00:00Z",
            "message": {"role": "assistant", "usage": {"input_tokens": 10, "output_tokens": 5},
                        "content": [{"type": "text", "text": f"{tagname}\nAnswer: yes"}]}}
    (proj / "agent-a1.jsonl").write_text(json.dumps(line) + "\n")
    conn = db.connect(":memory:"); db.init(conn)
    manifest = {tagname: {"item_id": "scev_val_fig_0001", "trial": 1, "label": 1}}
    n = scrape.collect(conn, 1, "claude-haiku-4-5", manifest,
                       projects_dir=str(tmp_path / "projects"), session_id="sess")
    assert n == 1
    row = conn.execute("SELECT answer, predicted, correct, parse_ok FROM prediction").fetchone()
    assert row == ("yes", 1, 1, 1)
