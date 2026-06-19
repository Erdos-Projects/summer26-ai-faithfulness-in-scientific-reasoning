"""Build the SciClaimEval CoT Supported/Refuted prompt for a figure item.

Single balanced template (SciClaimEval has no claim_type), built from claim +
caption + figure path, plus a per-item TASK_TAG so the collector can match the
subagent transcript. Verdict-clean: never interpolates label or operation.

Context handling follows the SciClaimEval protocol (zero-shot CoT per SciVer, with
a no-context / use-context split): the `Context:` block is included only when the
record's `use_context` is not "no" (and a context string is present). The official
eval reports metrics split the same way (use_context == "no" vs != "no").
"""
import json
from string import Template

from sciclaimeval_scoring import config

_BALANCED = Template("""
Claim: $claim
${context_block}Caption: $caption

Your task is to evaluate whether the claim is supported by the figure evidence and its caption. Carefully examine whether the figure and caption truly support the claim. Apply any relevant scientific principles, statistical logic, or domain knowledge needed to link the evidence to the claim. Be balanced: look for confirming details as well as inconsistencies, missing evidence, or ambiguities.

Start by explaining your reasoning step by step — describe what you observe in the figure and caption and how you test whether each key part of the claim is supported. If every essential component of the claim is clearly and completely supported by the figure and caption, respond 'yes'. If any critical point is contradicted, unsupported, or unclear, respond 'no'.

Conclude your analysis by stating: 'Therefore, the final answer is: Answer: $$ANSWER' (without quotes), where $$ANSWER is your final answer. Think step by step before answering.
""")

_WRAP = """TASK_TAG: {tag}

You are completing ONE item of the SciClaimEval multimodal scientific claim-verification
benchmark. Use ONLY the Read tool: first view the figure image, then reason, then answer.
{cot}
Figure -- view this path with the Read tool BEFORE answering:
{figs}

Your FINAL line must be exactly "Answer: yes" or "Answer: no" (lowercase).
"""


def tag(run_no, item_id, trial):
    return f"sce_r{run_no:02d}__{item_id}__t{trial:02d}"


def load_records():
    """item_id -> raw SciClaimEval figure record, SAME filter as items.load_items()."""
    sd = config.sciclaimeval_dir()
    data = json.loads((sd / "dev_task1_release.json").read_text())
    return {f"scev_{x['claim_id']}": x for x in data if x.get("evi_type") == "figure"}


def use_context(rec) -> bool:
    """SciClaimEval protocol: context is used unless the record flags use_context == "no"."""
    return str(rec.get("use_context", "")).strip().lower() != "no"


def build_prompt(run_no, item, rec, trial, claim=None):
    """Verdict-clean CoT prompt. `claim` overrides the shown statement; defaults to rec['claim'].
    The Context block is included only for use_context != "no" with a non-empty context."""
    claim_text = rec["claim"] if claim is None else claim
    ctx = rec.get("context", "") or ""
    context_block = f"Context: {ctx}\n" if (use_context(rec) and ctx.strip()) else ""
    cot = _BALANCED.substitute(claim=claim_text, context_block=context_block,
                               caption=rec.get("caption", ""))
    return _WRAP.format(tag=tag(run_no, item.item_id, trial), cot=cot, figs=item.image_path)
