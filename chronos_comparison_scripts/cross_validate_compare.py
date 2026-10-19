"""
Walk-forward cross-validation for one pipeline (hgb_mean, hgb_quantile, tabicl
or chronos2), using the same strategy as `cross_validate.py`
(`elf.cross_val_predict`, no skore): iterate over all `elf.TimeSeriesSplitter`
folds, fit on train, predict and score on test, concatenate across folds.

Stores the full predictions (actual + predicted, one row per prediction_time x
split) as parquet and the per-fold scores as csv, so MAE / pinball / any other
metric can be recomputed later from `compare_predictions.py` without re-running
the models.

Run from the repository root (see `chronos_comparison_scripts/README.md`):

    python chronos_comparison_scripts/cross_validate_compare.py \
        --model hgb_mean --output_dir results/cv_compare_xxx
    python chronos_comparison_scripts/cross_validate_compare.py \
        --model hgb_quantile --output_dir results/cv_compare_xxx
    python chronos_comparison_scripts/cross_validate_compare.py \
        --model tabicl --output_dir results/cv_compare_xxx
    python chronos_comparison_scripts/cross_validate_compare.py \
        --model tabicl --output_dir results/cv_compare_xxx \
        --tabicl_n_estimators 8 --tabicl_binning_train_rows_cap 0 --tag n8_unlimited
    python chronos_comparison_scripts/cross_validate_compare.py \
        --model chronos2 --output_dir results/cv_compare_xxx --device cuda

`hgb_quantile` is `elf.make_data_op(quantile_strategy="multiple_regressors")`:
one independent `HistGradientBoostingRegressor` per quantile (see
`quantile_regressor.HGBQuantileRegressor`), as opposed to `hgb_mean`'s single
mean-output regressor.

All pipelines share the horizons (1, 12, 24), the environment
(`elf.get_env()`), and `elf.TimeSeriesSplitter` (attached to X the same way in
`elf.make_data_op` and `cf.make_chronos_data_op`), so folds are identical
across models.
"""

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import electricity_load_forecasting as elf

HORIZONS = (1, 12, 24)
QUANTILES = (0.05, 0.5, 0.95)

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    choices=["hgb_mean", "hgb_quantile", "tabicl", "chronos2"],
    required=True,
)
parser.add_argument("--output_dir", type=Path, required=True)
parser.add_argument(
    "--tag",
    default="",
    help="suffix added to the output file names, e.g. '--tag n8' writes "
    "'<model>_n8_cv_predictions.parquet' (use to avoid overwriting a previous run)",
)
# chronos2-specific
parser.add_argument("--chronos_model", default="amazon/chronos-2")
parser.add_argument("--device", default=None, help="default: cuda if available")
parser.add_argument("--batch_size", type=int, default=256)
parser.add_argument("--context_hours", type=int, default=1024)
# tabicl-specific
parser.add_argument("--tabicl_n_estimators", type=int, default=1)
parser.add_argument(
    "--tabicl_binning_train_rows_cap",
    type=int,
    default=9000,
    help="cap on training rows for tabicl/binning (see "
    "elf.make_data_op's docstring for why only those two need it); 0 disables the cap",
)
args = parser.parse_args()

args.output_dir.mkdir(parents=True, exist_ok=True)
name = args.model + (f"_{args.tag}" if args.tag else "")

if args.model == "hgb_mean":
    pred = elf.make_data_op(horizons=HORIZONS, quantile_strategy=None)
elif args.model == "hgb_quantile":
    pred = elf.make_data_op(
        horizons=HORIZONS, quantile_strategy="multiple_regressors", quantiles=QUANTILES
    )
elif args.model == "tabicl":
    pred = elf.make_data_op(
        horizons=HORIZONS,
        quantile_strategy="tabicl",
        quantiles=QUANTILES,
        tabicl_n_estimators=args.tabicl_n_estimators,
        tabicl_binning_train_rows_cap=args.tabicl_binning_train_rows_cap or None,
    )
else:
    import chronos_forecasting as cf

    forecaster = cf.Chronos2Forecaster(
        model=args.chronos_model, device_map=args.device, batch_size=args.batch_size
    )
    pred = cf.make_chronos_data_op(
        horizons=HORIZONS,
        forecaster=forecaster,
        quantiles=QUANTILES,
        context_hours=args.context_hours,
        cities="all",
        temperature_only=True,
        calendar=True,
        verbose=True,
    )

env = elf.get_env()
t0 = time.time()
cv_predictions, scores = elf.cross_val_predict(pred, env)
dt = time.time() - t0

cv_predictions.write_parquet(args.output_dir / f"{name}_cv_predictions.parquet")
scores.write_csv(args.output_dir / f"{name}_scores.csv")

print(f"{name}: {dt:.1f}s total, {cv_predictions.height} predicted rows")
print(scores)
