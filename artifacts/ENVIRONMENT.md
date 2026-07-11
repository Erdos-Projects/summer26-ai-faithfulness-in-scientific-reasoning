# Serialized models — environment and reload notes

This directory holds the final CharXiv failure-prediction models, serialized with joblib. The chosen
family is Logistic Regression for every target.

- final_model.joblib — a dictionary keyed by target model. Each value is a fitted scikit-learn
  Pipeline of the Checkpoint 2 preprocessor and the tuned classifier, refit on all 800 training items.
  The three keys are the three targets the project predicts separately: `GPT-4o`,
  `Claude-3-5-Sonnet`, and `GPT-4o-Random`. There is a separate classifier per target, and no model's
  correctness (and no model-identity feature) is ever used as an input. The sealed 200-item test set
  is not used to fit these models.

## Environment

Notebook 20 writes final_model.joblib from the project virtual environment `.venv`, so `.venv` is the
authoritative environment for reloading it. The versions that matter for reloading are:

- python 3.12
- scikit-learn 1.9
- numpy 2.x
- joblib 1.5

The earlier modeling notebooks (17 through 19) run in the separate charxiv-model conda environment
(scikit-learn 1.5.2), whose full package list is in requirements_charxiv-model.txt. A model pickled in
one environment may print compatibility warnings when reloaded in the other because the scikit-learn
versions differ, so reload final_model.joblib in `.venv` to avoid them.

## Reload example

```python
from src.models import train
from src.features.preprocessing import connect, get_split, make_design_matrix

con = connect()
train_ids, _ = get_split(con)
df = make_design_matrix(con, train_ids)
con.close()

models = train.load("artifacts/final_model.joblib")   # {target: fitted pipeline}
pipe = models["GPT-4o"]                                # pick a target's classifier
p_failure = pipe.predict_proba(df)[:, 1]               # probability the target model fails the item
```
