"""SciVer claim-verification evaluation via the Claude Code harness.

Sibling of `rubric_scoring`: same 817 chart-item universe and the SAME `item_id`
keys (reused from `rubric_scoring.items.load_items`), so predictions join 1:1 to
the DeLeAn demand scores in `annotations_prod.db`.

Verdict-AWARE (the model judges the claim, scored against `label`) -- the inverse
of the verdict-blind rubric pipeline -- but it still never leaks `label` or
`perturbed_explanation` into a prompt.
"""
