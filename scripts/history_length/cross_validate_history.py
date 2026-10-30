"""Walk-forward CV with different training-history sizes, for HGB mean or TabICL.

Each configuration is simply ``elf.make_data_op(max_train_rows=...)`` run
through the regular walk-forward CV. ``max_train_rows`` caps the tail of each
fold's training window (inside the pipeline, during fit); it does not change
the CV splits or the test rows, so all configurations are directly comparable.

Run from the repository root:

    python scripts/history_length/cross_validate_history.py \
        --model hgb_mean \
        --max_train_rows 2160 8760 17520 26280 0 \
        --output_dir results/history_length/hgb_mean_cv
    python scripts/history_length/cross_validate_history.py \
        --model tabicl \
        --max_train_rows 2000 4000 8000 12000 0 \
        --output_dir results/history_length/tabicl_cv

``0`` means no cap (all available training history). Defaults are given per
model; TabICL configurations stay smaller because its inference cost grows with
the in-context training size. For each configuration the script writes
``<model>_rows<N>_cv_predictions.parquet`` and ``<model>_rows<N>_scores.csv``
(``rows_all`` for no cap); ``summary.csv`` pools all folds per configuration.
Fit and score times are recorded per fold by ``elf.cross_val_predict``; feature
engineering cost is independent of the cap, so compare those times when
studying compute.
"""

import argparse
import sys
import time
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_comparison"))
from compare_predictions import compute_scores

import electricity_load_forecasting as elf

MAX_TRAIN_ROWS = {
    "hgb_mean": [2160, 8760, 17520, 26280, 0],
    "tabicl": [2000, 4000, 8000, 12000, 0],
}

parser = argparse.ArgumentParser()
parser.add_argument("--model", choices=list(MAX_TRAIN_ROWS), default="hgb_mean")
parser.add_argument(
    "--max_train_rows",
    type=int,
    nargs="+",
    default=None,
    help="training-row caps (0 = all available history); defaults depend on --model",
)
parser.add_argument("--output_dir", type=Path, default=None)
parser.add_argument("--max_iter", type=int, default=3000, help="HGB only")
parser.add_argument("--tabicl_n_estimators", type=int, default=1)
parser.add_argument("--quantiles", type=float, nargs="+", default=[0.05, 0.5, 0.95])
parser.add_argument("--horizons", type=int, nargs="+", default=[1, 12, 24])
args = parser.parse_args()

if args.max_train_rows is None:
    args.max_train_rows = MAX_TRAIN_ROWS[args.model]
if args.output_dir is None:
    args.output_dir = Path(f"results/history_length/{args.model}_cv")
args.output_dir.mkdir(parents=True, exist_ok=True)
env = elf.get_env()


def build_data_op(max_train_rows):
    kwargs = dict(
        horizons=tuple(args.horizons), max_train_rows=max_train_rows or None
    )
    if args.model == "hgb_mean":
        return elf.make_data_op(
            quantile_strategy=None, max_iter=args.max_iter, **kwargs
        )
    return elf.make_data_op(
        quantile_strategy="tabicl",
        quantiles=tuple(args.quantiles),
        tabicl_n_estimators=args.tabicl_n_estimators,
        **kwargs,
    )


def tag(rows):
    return "all" if rows == 0 else str(rows)


summary_metrics = [
    "mae__average",
    "mape__average",
    "neg_mape__average",
    "d2_pinball_score__average",
]
summary = []
for rows in args.max_train_rows:
    name = f"{args.model}_rows{tag(rows)}"
    print(f"running {name}")
    t0 = time.perf_counter()
    predictions, scores = elf.cross_val_predict(
        build_data_op(max_train_rows=rows), env
    )
    total_time = time.perf_counter() - t0
    predictions.write_parquet(args.output_dir / f"{name}_cv_predictions.parquet")
    scores.write_csv(args.output_dir / f"{name}_scores.csv")
    metrics = compute_scores(predictions)
    summary.append(
        {
            "name": name,
            "model": args.model,
            "max_train_rows": rows,
            "n_folds": scores.height,
            "fit_time_mean": scores["fit_time"].mean(),
            "fit_time_total": scores["fit_time"].sum(),
            "score_time_total": scores["score_time"].sum(),
            "total_time": total_time,
            **{k: metrics[k] for k in summary_metrics if k in metrics},
        }
    )
    pl.DataFrame(summary).write_csv(args.output_dir / "summary.csv")
    print(
        f"{name}: {total_time:.1f}s, pooled mae={metrics['mae__average']:.1f} MW, "
        f"mape={metrics['mape__average']:.4f}"
    )

print(pl.DataFrame(summary))
