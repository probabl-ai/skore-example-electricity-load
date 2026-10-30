"""Randomized hyperparameter search with Optuna.

Searches on the full dataset (no manual train/test split): the DataOp's
walk-forward CV splitter is used, so every trial is evaluated on all folds.

Writes its outputs under ``results/tuning/search_<timestamp>/``, including a
persistent ``optuna.db`` SQLite database that can be reopened with
``optuna.load_study(study_name=..., storage="sqlite:///.../optuna.db")``.
"""

import argparse
import json
import pickle
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import electricity_load_forecasting as elf

parser = argparse.ArgumentParser()
parser.add_argument("--n_trials", type=int, default=100)
parser.add_argument(
    "--output_dir",
    type=Path,
    default=None,
    help="output directory (default: results/tuning/search_<timestamp>)",
)
parser.add_argument(
    "--max_train_rows",
    type=int,
    default=None,
    help="limit each training set to its last this many rows",
)
parser.add_argument(
    "--model",
    default="hgb_mean",
    choices=["hgb_mean", "hgb_quantile", "binning", "tabicl"],
    help="model passed to make_data_op (default: hgb_mean point forecast)",
)
parser.add_argument(
    "--quantiles",
    type=float,
    nargs="+",
    default=None,
    help="quantile levels, e.g. --quantiles 0.5",
)
parser.add_argument("--max_iter", type=int, default=3000)
parser.add_argument(
    "--refit",
    default="neg_mape__average",
    help="metric that drives the Optuna optimization (a key of the CV scores)",
)
parser.add_argument("--n_jobs", type=int, default=1)
args = parser.parse_args()

if args.output_dir is None:
    output_dir = elf.get_output_dir("tuning/search_")
else:
    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
db_path = output_dir / "optuna.db"
storage = f"sqlite:///{db_path}"
study_name = "randomized_search"
print(f"output_dir: {output_dir}", flush=True)
print(f"optuna storage: {storage}", flush=True)

env = elf.get_env()
op_kwargs = dict(
    model=args.model, max_train_rows=args.max_train_rows, max_iter=args.max_iter
)
if args.model != "hgb_mean":
    op_kwargs["quantiles"] = tuple(args.quantiles or (0.05, 0.5, 0.95))
pred = elf.make_data_op(**op_kwargs)

search = pred.skb.make_randomized_search(
    backend="optuna",
    n_iter=args.n_trials,
    n_jobs=args.n_jobs,
    refit=args.refit,
    random_state=0,
    storage=storage,
    study_name=study_name,
)

search.fit(env)

search.results_.to_csv(output_dir / "search_results.csv", index=False)
search.study_.trials_dataframe().to_csv(
    output_dir / "optuna_trials.csv", index=False
)
best = {
    "refit": args.refit,
    "best_score": search.best_score_,
    "best_params": search.best_params_,
    "n_trials": args.n_trials,
    "storage": storage,
    "study_name": study_name,
}
(output_dir / "best.json").write_text(
    json.dumps(best, indent=2, default=str), "utf-8"
)
print(f"best {args.refit}: {search.best_score_}", flush=True)
print(f"best params: {search.best_params_}", flush=True)

search.plot_results().write_html(output_dir / "search_results.html")
try:
    with open(output_dir / "search.pickle", "wb") as f:
        pickle.dump(search, f)
except Exception as exc:
    print(f"could not pickle search: {exc}", flush=True)
