"""Filled DeLeAn prompts, verdict-blind. Rubric text is passed in verbatim.
build_prompt interpolates ONLY the claim and figure path from the item (never the label),
which is what makes it verdict-blind."""
from rubric_scoring.extract_rubrics import ALL_DIMS, EMERGENT_DIMS
DIMS = {c: v[0] for c, v in ALL_DIMS.items()}
DIMS.update(EMERGENT_DIMS)  # 18 DeLeAn + 3 authored emergent figure-grounding dims

def tag(pass_no, item_id, code):
    return f"rs_p{pass_no:02d}__{item_id}__{code}"

_TEMPLATE = """ROLE: You are a demand-level annotator. Score ONE rubric on ONE task instance, then STOP.
Do not decide the claim's correctness. View the figure with the Read tool. Use no tool other than Read.

PILOT_TAG: {tag}

QUERY:
The following rubric describes six distinct levels of *{name}* required by different tasks:

{rubric}

TASK INSTANCE:
This is a scientific claim-verification task: the figure is the evidence for checking the claim below.
You ONLY assess the rubric demand of this task; you do NOT decide whether the claim is correct.

Claim: "{claim}"

Figure: a chart image for this task is provided. View it with the Read tool before scoring.
Figure path: {fig}

INSTRUCTION:
Score the level of *{name}* demanded by the given TASK INSTANCE using a discrete value from 0 to 5. Use
CHAIN-OF-THOUGHTS REASONING to reason step by step, then conclude with EXACTLY:

"Thus, the level of *{name}* demanded by the given TASK INSTANCE is: SCORE", where SCORE is your integer. Then STOP.
"""

def build_prompt(pass_no, item, code, rubric_text):
    return _TEMPLATE.format(tag=tag(pass_no, item.item_id, code), name=DIMS[code],
                            rubric=rubric_text, claim=item.claim, fig=item.image_path)
