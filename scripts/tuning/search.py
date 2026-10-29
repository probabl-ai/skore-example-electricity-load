"""Randomized hyperparameter search with Optuna.

Writes its outputs under ``results/tuning/search_<timestamp>/``.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

from sklearn.metrics import mean_absolute_percentage_error

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import electricity_load_forecasting as elf

parser = argparse.ArgumentParser()
parser.add_argument("--n_trials", type=int, default=100)
parser.add_argument(
    "--max_train_rows",
    type=int,
    default=None,
    help="limit each training set to its last this many rows",
)
args = parser.parse_args()

output_dir = elf.get_output_dir("tuning/search_")
env = elf.get_env()
pred = elf.make_data_op(max_train_rows=args.max_train_rows)

storage = f"sqlite:///{output_dir / 'optuna'}"
study_name = "randomized_search"
split = pred.skb.train_test_split(environment=env, split_func=elf.train_test_split)
search = pred.skb.make_randomized_search(
    backend="optuna",
    n_iter=args.n_trials,
    n_jobs=1,
    refit=True,
    storage=storage,
    study_name=study_name,
)

search.fit(split["train"])
with open(output_dir / "search.pickle", "wb") as f:
    pickle.dump(search, f)

search.results_.to_csv(output_dir / "search_results.csv", index=False)
predictions = search.predict(split["test"])
results = elf.concat_X_y_predictions(split["X_test"], split["y_test"], predictions)
results.write_parquet(output_dir / "predictions.parquet")
test_score = mean_absolute_percentage_error(split["y_test"], predictions)
print(f"MAPE: {test_score}")
(output_dir / "score.json").write_text(json.dumps({"mape": test_score}), "utf-8")
fig = elf.plot_predictions(results)
fig.write_html(output_dir / "cv_predictions_plot.html")

search.plot_results().show(renderer="browser")
