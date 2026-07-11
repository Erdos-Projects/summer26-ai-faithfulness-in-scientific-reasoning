#!/usr/bin/env bash
# Reproduce the CharXiv final results by executing notebooks/20_charxiv_final_results.ipynb end to end.
#
# The notebook goes from the released database, through feature selection and the train/validation
# setup, to tuning and serializing the three final per-target logistic-regression models and writing
# every table and figure under results/final/. It runs entirely in the project .venv (no XGBoost or
# conda env needed).
#
# The held-out 200-item test set is scored only when CHARXIV_EVAL_TEST=1 is set, so by default this
# script regenerates the train-side results and the serialized model without ever reading the test set:
#
#   bash run.sh                    # train-side final results only (test set stays sealed)
#   CHARXIV_EVAL_TEST=1 bash run.sh  # also run the one-time sealed test evaluation
#
# The database charxiv_scoring/annotations_charxiv.db is assumed present. To rebuild it first (it is
# idempotent), uncomment the build_db line below.
set -euo pipefail
cd "$(dirname "$0")"

# .venv/bin/python -m charxiv_scoring.build_db     # optional: rebuild the database from released files

if [ "${CHARXIV_EVAL_TEST:-0}" = "1" ]; then
  echo "CHARXIV_EVAL_TEST=1 : the sealed 200-item test set WILL be scored."
else
  echo "CHARXIV_EVAL_TEST is off : the sealed test set stays sealed. Set CHARXIV_EVAL_TEST=1 to score it."
fi

# The "charxiv" kernel is the project .venv (see ~/Library/Jupyter/kernels/charxiv).
.venv/bin/python -m jupyter nbconvert --to notebook --execute --inplace \
  --ExecutePreprocessor.kernel_name=charxiv --ExecutePreprocessor.timeout=1200 \
  notebooks/20_charxiv_final_results.ipynb

echo "Done. Tables and figures are in results/final/ ; the three per-target models are in artifacts/final_model.joblib"
