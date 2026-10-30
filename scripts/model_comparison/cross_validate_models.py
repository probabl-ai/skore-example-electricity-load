"""
Walk-forward cross-validation for one pipeline (hgb_mean, hgb_quantile,
hgb_quantile_recal, tabicl, kumo_tabular, chronos2 or kumo), using the same
strategy as
`cross_validate.py` (`elf.cross_val_predict`, no skore): iterate over all
`electricity_load_forecasting.splitter.TimeSeriesSplitter` folds, fit on train,
predict and score on test, concatenate across folds.

Stores the full predictions (actual + predicted, one row per prediction_time x
split) as parquet and the per-fold scores as csv, so MAE / pinball / any other
metric can be recomputed later from `compare_predictions.py` without re-running
the models.

Run from the repository root:

    python scripts/model_comparison/cross_validate_models.py \
        --model hgb_mean --output_dir results/model_comparison/my_study
    python scripts/model_comparison/cross_validate_models.py \
        --model hgb_quantile --output_dir results/model_comparison/my_study
    python scripts/model_comparison/cross_validate_models.py \
        --model hgb_quantile_recal --max_iter 2000 \
        --output_dir results/model_comparison/my_study
    python scripts/model_comparison/cross_validate_models.py \
        --model tabicl --max_train_rows 0 \
        --output_dir results/model_comparison/my_study
    python scripts/model_comparison/cross_validate_models.py \
        --model kumo_tabular --max_train_rows 0 \
        --output_dir results/model_comparison/my_study
    python scripts/model_comparison/cross_validate_models.py \
        --model chronos2 --device cuda \
        --output_dir results/model_comparison/my_study
    python scripts/model_comparison/cross_validate_models.py \
        --model kumo --device cuda \
        --output_dir results/model_comparison/my_study

Use `--reports` to also write the skrub fit/predict/single-date reports.

`hgb_quantile` is `elf.make_data_op(model="hgb_quantile")`:
one independent `HistGradientBoostingRegressor` per quantile (see
`electricity_load_forecasting.quantile_regressor.HGBQuantileRegressor`), as
opposed to `hgb_mean`'s single mean-output regressor. `hgb_quantile_recal` is
the same wrapped in `calibration.Recalibrator` (post-hoc correction of the
quantiles), with its default settings. `--max_iter` applies to the HGB models
only (ignored by tabicl/chronos2).

All pipelines share the horizons (1, 12, 24), the environment
(`elf.get_env()`), and
`electricity_load_forecasting.splitter.TimeSeriesSplitter` (attached to X the
same way in `elf.make_data_op` and `cf.make_chronos_data_op`), so folds are
identical across models.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import electricity_load_forecasting as elf

HORIZONS = (1, 12, 24)
QUANTILES = (0.05, 0.5, 0.95)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    choices=[
        "hgb_mean",
        "hgb_quantile",
        "hgb_quantile_recal",
        "tabicl",
        "kumo_tabular",
        "chronos2",
        "kumo",
    ],
    required=True,
)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument(
    "--tag",
    default="",
    help="suffix added to the output file names, e.g. '--tag n8' writes "
    "'<model>_n8_cv_predictions.parquet' (use to avoid overwriting a previous run)",
)
parser.add_argument(
    "--reports",
    action="store_true",
    help="also write the skrub fit/predict/single-date reports",
)
# HGB-specific
parser.add_argument("--max_iter", type=int, default=3000)
# chronos2-specific
parser.add_argument(
    "--forecaster",
    choices=["chronos", "stub"],
    default="chronos",
    help="forecaster used by --model chronos2",
)
parser.add_argument("--chronos_model", default="amazon/chronos-2")
parser.add_argument("--device", default=None, help="default: cuda if available")
parser.add_argument("--dtype", default=None, help="e.g. bfloat16")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--context_hours", type=int, default=1024)
parser.add_argument("--no_weather", action="store_true")
parser.add_argument("--no_calendar", action="store_true")
# kumo-specific
parser.add_argument("--kumo_context_hours", type=int, default=512)
parser.add_argument("--kumo_num_estimators", type=int, default=1)
parser.add_argument("--kumo_chunk_size", type=int, default=8)
parser.add_argument("--kumo_no_cross_channel", action="store_true")
parser.add_argument(
    "--quantiles",
    type=float,
    nargs="+",
    default=QUANTILES,
    help="quantile levels to predict (default: 0.05 0.5 0.95)",
)
parser.add_argument(
    "--max_train_rows",
    type=int,
    default=9000,
    help="cap on training rows for tabicl/kumo_tabular (see "
    "elf.make_data_op's docstring); 0 disables the cap",
)
args = parser.parse_args()

args.output_dir.mkdir(parents=True, exist_ok=True)
name = args.model + (f"_{args.tag}" if args.tag else "")

if args.model == "hgb_mean":
    pred = elf.make_data_op(
        horizons=HORIZONS, model="hgb_mean", max_iter=args.max_iter
    )
elif args.model == "hgb_quantile":
    pred = elf.make_data_op(
        horizons=HORIZONS,
        model="hgb_quantile",
        quantiles=args.quantiles,
        max_iter=args.max_iter,
    )
elif args.model == "hgb_quantile_recal":
    pred = elf.make_data_op(
        horizons=HORIZONS,
        model="hgb_quantile",
        quantiles=args.quantiles,
        max_iter=args.max_iter,
        recalibrate=True,
    )
elif args.model == "tabicl":
    pred = elf.make_data_op(
        horizons=HORIZONS,
        model="tabicl",
        quantiles=args.quantiles,
        max_train_rows=args.max_train_rows or None,
    )
elif args.model == "kumo_tabular":
    pred = elf.make_data_op(
        horizons=HORIZONS,
        model="kumo_tabular",
        quantiles=args.quantiles,
        max_train_rows=args.max_train_rows or None,
    )
elif args.model == "chronos2":
    from electricity_load_forecasting import chronos as cf

    if args.forecaster == "stub":
        forecaster = cf.SeasonalNaiveForecaster()
    else:
        forecaster = cf.Chronos2Forecaster(
            model=args.chronos_model,
            device_map=args.device,
            dtype=args.dtype,
            batch_size=args.batch_size,
        )
    pred = cf.make_chronos_data_op(
        horizons=HORIZONS,
        forecaster=forecaster,
        quantiles=args.quantiles,
        context_hours=args.context_hours,
        cities=() if args.no_weather else "all",
        temperature_only=True,
        calendar=not args.no_calendar,
        verbose=True,
    )
else:
    from electricity_load_forecasting import kumo as kf

    forecaster = kf.KumoForecastForecaster(
        device=args.device,
        cross_channel=not args.kumo_no_cross_channel,
        num_estimators=args.kumo_num_estimators,
    )
    pred = kf.make_kumo_data_op(
        horizons=HORIZONS,
        forecaster=forecaster,
        context_hours=args.kumo_context_hours,
        cities=() if args.no_weather else "all",
        temperature_only=True,
        calendar=not args.no_calendar,
        chunk_size=args.kumo_chunk_size,
        verbose=True,
    )

env = elf.get_env()
t0 = time.time()
cv_predictions, scores = elf.cross_val_predict(pred, env)
dt = time.time() - t0

cv_predictions.write_parquet(args.output_dir / f"{name}_cv_predictions.parquet")
scores.write_csv(args.output_dir / f"{name}_scores.csv")

if args.reports:
    report_dir = args.output_dir / "reports" / name
    split = pred.skb.train_test_split(environment=env)
    learner = pred.skb.make_learner()
    learner.report(
        environment=split["train"],
        mode="fit",
        title="fit",
        output_dir=report_dir / "fit_report",
    )
    learner.report(
        environment=split["test"],
        mode="predict",
        title="predict",
        output_dir=report_dir / "predict_report",
    )
    learner.report(
        environment={"start": elf.get_new_date()},
        mode="predict",
        title="predict_single_date",
        output_dir=report_dir / "predict_single_date_report",
    )

print(f"{name}: {dt:.1f}s total, {cv_predictions.height} predicted rows")
print(scores)
