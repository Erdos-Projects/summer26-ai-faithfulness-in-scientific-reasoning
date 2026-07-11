"""Feature definitions for CharXiv.

The **12 DeLeAn demand dimensions are the only engineered features** in this project. They are used
as-is (raw ordinal 0-5), optionally standardized inside the preprocessing pipeline
(see ``preprocessing.build_preprocessor``); there are **no custom derived transformers**.

This module therefore just exports the canonical dim ordering shared with ``charxiv_analysis``.
"""
from __future__ import annotations

# Canonical demand-dim order (shared with charxiv_analysis).
DEMAND_DIMS = ["VL", "AS", "MCr", "MCu", "MA", "VO", "AT", "GS", "QLl", "KNf", "QLq", "CL"]
