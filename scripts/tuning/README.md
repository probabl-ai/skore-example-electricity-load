# Hyperparameter tuning

- `search.py` runs a randomized Optuna search on the default pipeline and writes
  `results/tuning/search_<timestamp>/`.

Use `--n_trials` to control the search budget and `--max_train_rows` to study
the effect of limiting each training set to its most recent rows.
