"""Cross-validation splitter for the CharXiv evaluation — the ``item_holdout`` regime.

The single deployment regime we evaluate: **predict failure on a new chart** (an unseen paper) for
a given target model. The unit of analysis is one item (one row); grouping by ``paperid`` guarantees
no paper (hence no item) straddles a train/test fold. The same folds are reused for each target's
classifier, which share the item feature block and differ only in the failure label.

(The earlier model-holdout and block regimes were dropped — this project only evaluates item_holdout.)
"""
from __future__ import annotations

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


class ItemHoldoutSplit:
    """Paper-grouped, label-stratified folds — generalization to **new charts**.

    Wraps :class:`sklearn.model_selection.StratifiedGroupKFold` with groups = ``paperid`` so no
    paper (hence no item) straddles a fold, while keeping the failure-rate balanced across folds.
    Repeats give a mean ± SD across folds.

    API: ``split(df)`` yields ``(train_idx, test_idx)`` positional arrays (``df`` must carry ``y`` and
    ``paperid`` columns); ``get_n_splits()`` returns ``n_splits * n_repeats``.
    """

    def __init__(self, n_splits=5, n_repeats=3, random_state=0):
        self.n_splits = n_splits
        self.n_repeats = n_repeats
        self.random_state = random_state

    def split(self, df):
        y = df["y"].to_numpy()
        groups = df["paperid"].to_numpy()
        for r in range(self.n_repeats):
            sgk = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True,
                                       random_state=self.random_state + r)
            for tr, te in sgk.split(np.zeros(len(df)), y, groups):
                yield tr, te

    def get_n_splits(self, df=None):
        return self.n_splits * self.n_repeats
