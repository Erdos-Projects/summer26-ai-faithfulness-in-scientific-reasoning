"""Reusable modeling code for the CharXiv failure-prediction checkpoint.

The modules here plug into the shared evaluation contract in :mod:`src.eval.core`: a model is just a
``clf_factory() -> estimator`` fitted inside ``Pipeline([("pre", make_pre()), ("clf", ...)])`` per
cross-validation fold. Nothing in this package reads the sealed 200-item test set.
"""
