"""Filled DeLeAn prompts, verdict-blind. Rubric text is passed in verbatim.
build_prompt interpolates ONLY the reasoning question and figure path from the item (never the
answer or inst_category) — that is what makes it verdict-blind. The DeLeAn task instance here is
'answer this reasoning question using the chart' (vs the claim-verification framing in SciVer/
SciClaimEval); demand is scored for that task, not for deciding the answer."""
from charxiv_scoring.extract_rubrics import ALL_DIMS, EMERGENT_DIMS

DIMS = {c: v[0] for c, v in ALL_DIMS.items()}
DIMS.update(EMERGENT_DIMS)  # 18 DeLeAn + 3 authored emergent figure-grounding dims

# Distinct tag prefix (cxr = CharXiv Rubric) so transcript matching never collides with the
# SciVer (rs_p…) or SciClaimEval (rsc_p…) scoring tags.
def tag(pass_no, item_id, code):
    return f"cxr_p{pass_no:02d}__{item_id}__{code}"

_TEMPLATE = """ROLE: You are a demand-level annotator. Score ONE rubric on ONE task instance, then STOP.
Do not answer the question. View the figure with the Read tool. Use no tool other than Read.

PILOT_TAG: {tag}

QUERY:
The following rubric describes six distinct levels of *{name}* required by different tasks:

{rubric}

TASK INSTANCE:
This is a scientific chart question-answering task: the figure is a chart from a scientific paper,
and the task is to answer the question below using the chart.
You ONLY assess the rubric demand of this task; you do NOT answer the question.

Question: "{query}"

Figure: an image for this task is provided. View it with the Read tool before scoring.
Figure path: {fig}

INSTRUCTION:
Score the level of *{name}* demanded by the given TASK INSTANCE using a discrete value from 0 to 5. Use
CHAIN-OF-THOUGHTS REASONING to reason step by step, then conclude with EXACTLY:

"Thus, the level of *{name}* demanded by the given TASK INSTANCE is: SCORE", where SCORE is your integer. Then STOP.
"""

def build_prompt(pass_no, item, code, rubric_text):
    return _TEMPLATE.format(tag=tag(pass_no, item.item_id, code), name=DIMS[code],
                            rubric=rubric_text, query=item.query, fig=item.image_path)
