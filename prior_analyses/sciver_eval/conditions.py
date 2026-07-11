"""First-class verdict conditions for the paired entailed-vs-refuted eval.

Each condition selects WHICH statement becomes the shown claim and the grade
label the collector scores `correct` against. The native run (official
`claim`/`label`) is intentionally NOT here -- it is run 1 and already complete;
this module only drives the two new homogeneous runs.
"""

# condition -> (record field holding the statement to show, grade label)
CONDITIONS = {
    "entailed": ("origin_statement", 1),    # show the true statement;  correct answer = yes
    "refuted": ("perturbed_statement", 0),  # show the false statement; correct answer = no
}


def resolve(condition, rec):
    """Return (claim_text, grade_label) for `condition` over a raw SciVer record.

    Raises KeyError on an unknown condition, ValueError on an empty statement.
    """
    field, label = CONDITIONS[condition]
    claim_text = rec[field]
    if not claim_text or not claim_text.strip():
        raise ValueError(f"empty {field} for condition {condition!r}")
    return claim_text, label
