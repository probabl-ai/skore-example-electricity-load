"""
Walk-forward cross-validation for one pipeline (hgb_mean, hgb_quantile,
hgb_quantile_recal, hgb_quantile_conformal, tabicl, tabicl_recal,
tabicl_conformal, kumo_tabular, kumo_tabular_recal, chronos2, kumo_forecast,
tabicl_forecast or t0_gamma), using the same strategy as `cross_validate.py` (`elf.cross_val_predict`, no skore): iterate over all
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
        --model hgb_mean --weather forecast \
        --output_dir results/model_comparison/my_study
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
        --model tabicl_recal --max_train_rows 0 \
        --output_dir results/model_comparison/my_study
    python scripts/model_comparison/cross_validate_models.py \
        --model kumo_tabular_recal --max_train_rows 0 \
        --output_dir results/model_comparison/my_study
    python scripts/model_comparison/cross_validate_models.py \
        --model chronos2 --device cuda \
        --output_dir results/model_comparison/my_study
    python scripts/model_comparison/cross_validate_models.py \
        --model kumo_forecast --device cuda \
        --output_dir results/model_comparison/my_study
    python scripts/model_comparison/cross_validate_models.py \
        --model tabicl_forecast --device cuda \
        --output_dir results/model_comparison/my_study
    python scripts/model_comparison/cross_validate_models.py \
        --model t0_gamma --device cuda \
        --output_dir results/model_comparison/my_study

Use `--reports` to also write the skrub fit/predict/single-date reports.
Weather defaults to the safe-lead historical forecasts (`--weather forecast`):
day1 for horizons <=18h and day2 above that. Use `--weather oracle` for the
oracle weather files.

`hgb_quantile` is `elf.make_data_op(model="hgb_quantile")`:
one independent `HistGradientBoostingRegressor` per quantile (see
`electricity_load_forecasting.quantile_regressor.HGBQuantileRegressor`), as
opposed to `hgb_mean`'s single mean-output regressor. `hgb_quantile_recal` is
the same wrapped in `calibration.Recalibrator` (post-hoc correction of the
quantiles), with its default settings. `hgb_quantile_conformal` uses instead
the simpler `calibration.ConformalRecalibrator` (same inner CV, but one global
shift per quantile level, no binning or spline); `tabicl_conformal` is the same
for TabICL. `--max_iter` applies to the
HGB models only (ignored by tabicl/chronos2).

`t0_gamma` is the `theforecastingcompany/t0-gamma` foundation model from The
Forecasting Company (`tfc-t0` package), a zero-shot quantile forecaster that
conditions on known-future weather/calendar covariates, like the Chronos
pipeline. It uses the recommended inference-time adjustments
(`trim_leading_missing`, `symmetric_averaging`, `clip_nonnegative`,
`snap_integers`); `symmetric_averaging` runs the model twice. `--device` and
`--dtype` apply to it, as does `--context_hours`.

All pipelines share the horizons (1, 12, 24), the environment
(`elf.get_env()`), and
`electricity_load_forecasting.splitter.TimeSeriesSplitter` (attached to X the
same way in `elf.make_data_op` and `cf.make_chronos_data_op`), so folds are
identical across models.
"""

import argparse
import datetime
import functools
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

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
        "hgb_quantile_conformal",
        "tabicl",
        "tabicl_recal",
        "tabicl_conformal",
        "kumo_tabular",
        "kumo_tabular_recal",
        "chronos2",
        "kumo_forecast",
        "tabicl_forecast",
        "t0_gamma",
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
parser.add_argument(
    "--feature_groups",
    nargs="*",
    default=[],
    choices=elf.FEATURE_GROUPS,
    help="extra feature groups for the lagged-feature models (hgb_*, tabicl*, "
    "kumo_tabular*); see elf.FEATURE_GROUPS",
)
# chronos2-specific
parser.add_argument(
    "--forecaster",
    choices=["chronos", "stub"],
    default="chronos",
    help="forecaster used by --model chronos2",
)
parser.add_argument("--chronos_model", default="amazon/chronos-2")
parser.add_argument(
    "--weather",
    choices=["oracle", "forecast"],
    default="forecast",
    help="weather features: safe-lead historical forecasts (default) or oracle files",
)
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
# tabicl_forecast-specific
parser.add_argument("--tabicl_context_hours", type=int, default=4096)
parser.add_argument("--tabicl_chunk_size", type=int, default=8)
parser.add_argument("--tabicl_n_estimators", type=int, default=8)
# t0-specific
parser.add_argument("--t0_chunk_size", type=int, default=64)
parser.add_argument(
    "--tabicl_point_estimate",
    choices=["mean", "median"],
    default="mean",
)
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

uses_lagged_features = not args.model.endswith("_forecast") and args.model not in (
    "chronos2",
    "t0_gamma",
)
if args.feature_groups and not uses_lagged_features:
    parser.error(f"--feature_groups is not supported by --model {args.model}")
make_data_op = functools.partial(
    elf.make_data_op, feature_groups=tuple(args.feature_groups)
)
args.output_dir.mkdir(parents=True, exist_ok=True)
name = args.model + (f"_{args.tag}" if args.tag else "")

if args.model == "hgb_mean":
    pred = make_data_op(horizons=HORIZONS, model="hgb_mean", max_iter=args.max_iter)
elif args.model == "hgb_quantile":
    pred = make_data_op(
        horizons=HORIZONS,
        model="hgb_quantile",
        quantiles=args.quantiles,
        max_iter=args.max_iter,
    )
elif args.model == "hgb_quantile_recal":
    pred = make_data_op(
        horizons=HORIZONS,
        model="hgb_quantile",
        quantiles=args.quantiles,
        max_iter=args.max_iter,
        recalibrate=True,
    )
elif args.model == "hgb_quantile_conformal":
    pred = make_data_op(
        horizons=HORIZONS,
        model="hgb_quantile",
        quantiles=args.quantiles,
        max_iter=args.max_iter,
        recalibrate="global",
    )
elif args.model == "tabicl":
    pred = make_data_op(
        horizons=HORIZONS,
        model="tabicl",
        quantiles=args.quantiles,
        max_train_rows=args.max_train_rows or None,
    )
elif args.model == "tabicl_recal":
    pred = make_data_op(
        horizons=HORIZONS,
        model="tabicl",
        quantiles=args.quantiles,
        max_train_rows=args.max_train_rows or None,
        recalibrate=True,
    )
elif args.model == "tabicl_conformal":
    pred = make_data_op(
        horizons=HORIZONS,
        model="tabicl",
        quantiles=args.quantiles,
        max_train_rows=args.max_train_rows or None,
        recalibrate="global",
    )
elif args.model == "kumo_tabular":
    pred = make_data_op(
        horizons=HORIZONS,
        model="kumo_tabular",
        quantiles=args.quantiles,
        max_train_rows=args.max_train_rows or None,
    )
elif args.model == "kumo_tabular_recal":
    pred = make_data_op(
        horizons=HORIZONS,
        model="kumo_tabular",
        quantiles=args.quantiles,
        max_train_rows=args.max_train_rows or None,
        recalibrate=True,
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
elif args.model == "kumo_forecast":
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
elif args.model == "tabicl_forecast":
    from electricity_load_forecasting import tabicl_forecast as tf

    tabicl_config = {"n_estimators": args.tabicl_n_estimators}
    if args.device is not None:
        tabicl_config["device"] = args.device
    pred = tf.make_tabicl_forecast_data_op(
        horizons=HORIZONS,
        quantiles=args.quantiles,
        max_context_length=args.tabicl_context_hours,
        chunk_size=args.tabicl_chunk_size,
        cities=() if args.no_weather else "all",
        temperature_only=True,
        calendar=not args.no_calendar,
        tabicl_config=tabicl_config,
        point_estimate=args.tabicl_point_estimate,
    )
elif args.model == "t0_gamma":
    from electricity_load_forecasting import t0 as t0mod

    forecaster = t0mod.T0Forecaster(device=args.device, dtype=args.dtype)
    pred = t0mod.make_t0_data_op(
        horizons=HORIZONS,
        forecaster=forecaster,
        quantiles=args.quantiles,
        context_hours=args.context_hours,
        cities=() if args.no_weather else "all",
        temperature_only=True,
        calendar=not args.no_calendar,
        chunk_size=args.t0_chunk_size,
        verbose=True,
    )

env = elf.get_env()
if args.weather == "forecast":
    env = env | {"city_weather_fetcher": elf.fetch_city_weather_forecast}
t0 = time.time()
cv_predictions, scores = elf.cross_val_predict(pred, env)
dt = time.time() - t0

cv_predictions.write_parquet(args.output_dir / f"{name}_cv_predictions.parquet")
scores.write_csv(args.output_dir / f"{name}_scores.csv")


def git(*cmd):
    return subprocess.run(
        ["git", *cmd], cwd=elf.pipeline.project_dir(), capture_output=True, text=True
    ).stdout.strip()


def version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


metadata = {
    "command": " ".join(sys.argv),
    "date": datetime.datetime.now().isoformat(timespec="seconds"),
    "duration_s": round(dt, 1),
    "git_commit": git("rev-parse", "HEAD"),
    "git_modified_files": git(
        "status", "--porcelain", "--untracked-files=no"
    ).splitlines(),
    "host": {
        "node": platform.node(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
        "OMP_NUM_THREADS": os.environ.get("OMP_NUM_THREADS"),
    },
    "versions": {
        pkg: version(pkg)
        for pkg in (
            "polars",
            "scikit-learn",
            "skrub",
            "holidays",
            "tabicl",
            "torch",
            "structured-data-models",
            "chronos-forecasting",
        )
    }
    | {"python": platform.python_version()},
    "weather_fetcher": env.get(
        "city_weather_fetcher", elf.fetch_city_weather_forecast
    ).__name__,
    "env": {k: str(v) for k, v in env.items()},
    "horizons": HORIZONS,
    "quantiles": args.quantiles,
    "max_iter": args.max_iter,
    "max_train_rows": args.max_train_rows,
    "splitter": vars(elf.splitter.TimeSeriesSplitter())
    | {"gap_days": elf.splitter._TRAIN_TEST_GAP_DAYS},
}
if uses_lagged_features:
    metadata["feature_groups"] = args.feature_groups
    metadata["pipeline_defaults"] = {
        str(k): str(v) for k, v in pred.skb.describe_defaults().items()
    }
    # feature names as resolved by the pipeline, per horizon
    one_time = elf.pipeline.time_range("2024-06-01")
    metadata["feature_names"] = {
        f"{h}h": [
            c
            for c in elf.pipeline.add_features(
                one_time.rename({"time": "prediction_time"}),
                horizon=h,
                temperature_only=True,
                city_names="all",
                load_mw_history=elf.resample(elf.fetch_load_mw_history()),
                city_weather_fetcher=env.get(
                    "city_weather_fetcher", elf.fetch_city_weather_forecast
                ),
                holidays_fetcher=elf.pipeline.fetch_holidays,
            ).columns
            if c not in ("prediction_time", "target_time")
        ]
        for h in HORIZONS
    }
(args.output_dir / f"{name}_metadata.json").write_text(json.dumps(metadata, indent=2))

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
