"""Build the full SciVer CoT verification prompt for a chart item, harness-flavored.

Faithful to the paper's prompts (SciVer utils/constant.py): direct -> skeptical
template, analytical -> balanced template (the only two claim_types in the 817-item
chart set). We include claim + section context + caption (the real benchmark task;
the rubric pipeline used claim+figure only) plus a per-item TASK_TAG so the collector
can match the subagent transcript. Verdict-clean: never interpolates label/perturbed_*.
"""
import json
from string import Template
from pathlib import Path

from rubric_scoring import config

# --- paper-faithful CoT templates (ported verbatim from SciVer utils/constant.py) ---
_DIRECT = Template("""
Claim: $claim
Context: $context
Caption: $caption

Your task is to critically evaluate the claim based on the image and the caption. Carefully examine whether the information in the caption truly supports the claim. Be skeptical and cautious: if there is any inconsistency, missing evidence, or ambiguity, consider the claim incorrect.

Start by explaining your reasoning process clearly, focusing on identifying potential contradictions, lack of support, or misleading interpretations. If the claim is unsupported or contradicted by the caption and image, respond with 'no'. Only respond with 'yes' if the claim is fully and clearly supported.

Conclude your analysis by stating: 'Therefore, the final answer is: Answer: $$ANSWER' (without quotes), where $$ANSWER is your final answer. Think step by step before answering.
""")

_ANALYTICAL = Template("""
Claim: $claim
Context: $context
Caption: $caption

Your task is to evaluate the claim based on the image and the caption. Carefully examine whether the information in the caption truly supports the claim. Apply any relevant scientific principles, statistical logic, or domain knowledge necessary to link the evidence to the claim. Be balanced: actively look for confirming details as well as inconsistencies, missing evidence, or ambiguities.

Start by explaining your reasoning process step by step—describe what you observe in the image and caption, what background knowledge you use, and how you test whether each key part of the claim is supported. If every essential component of the claim is clearly and completely backed by the caption and image, respond with 'yes'. If any critical point is contradicted, unsupported, or unclear, respond with 'no'.

Conclude your analysis by stating: 'Therefore, the final answer is: Answer: $$ANSWER' (without quotes), where $$ANSWER is your final answer. Think step by step before answering.
""")

_TPL = {"direct": _DIRECT, "analytical": _ANALYTICAL}

_WRAP = """TASK_TAG: {tag}

You are completing ONE item of the SciVer multimodal scientific claim-verification
benchmark. Use ONLY the Read tool: first view the figure image(s), then reason, then answer.
{cot}
Figure(s) -- view EACH path with the Read tool BEFORE answering:
{figs}

Your FINAL line must be exactly "Answer: yes" or "Answer: no" (lowercase).
"""


def tag(run_no, item_id, trial):
    # item_id is followed by the "__t.." token so a shorter id (sciver_val_1) is
    # never a substring of a longer one (sciver_val_10) during transcript matching.
    return f"se_r{run_no:02d}__{item_id}__t{trial:02d}"


def _paper_path(rec):
    return config.sciver_dir() / "papers" / Path(rec["paper_path"]).name


def _context(paper, section_list):
    tops = []
    for s in section_list:
        t = s.split(".")[0]
        if t not in tops:
            tops.append(t)
    out = ""
    for sec in paper["sections"]:
        if sec["section_id"].split(".")[0] in tops:
            out += sec["section_name"] + ":\n" + sec["text"] + "\n"
    return out


def _caption(paper, rec):
    return paper["image_paths"][rec["item"]]["caption"]  # chart-only set


def load_records():
    """item_id -> raw SciVer record, using the SAME filter as items.load_items()."""
    sd = config.sciver_dir()
    recs = {}
    for split, fname in (("val", "valset.json"), ("test", "testset.json")):
        data = json.loads((sd / fname).read_text())
        for idx, x in enumerate(data):
            if x.get("type") != "chart":
                continue
            recs[f"sciver_{split}_{idx}"] = x
    return recs


def build_prompt(run_no, item, rec, trial, claim=None):
    """Build the CoT prompt. `claim` overrides which statement is shown (used by
    the entailed/refuted conditions); defaults to the official `rec['claim']`
    (native run). Verdict-clean: never interpolates label or perturbed_explanation.
    """
    claim_text = rec["claim"] if claim is None else claim
    paper = json.loads(_paper_path(rec).read_text())
    cot = _TPL[rec["claim_type"]].substitute(
        claim=claim_text, context=_context(paper, rec["section"]),
        caption=_caption(paper, rec))
    return _WRAP.format(tag=tag(run_no, item.item_id, trial), cot=cot, figs=item.image_path)
