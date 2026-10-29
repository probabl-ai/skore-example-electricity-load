"""
Compute MAE, MAPE and pinball (d2) scores from `*_cv_predictions.parquet`
files written by `cross_validate_compare.py`, without re-running any model.

Run from the repository root:

    python scripts/chronos_comparison_scripts/compare_predictions.py \
        results/cv_compare_2026.../*_cv_predictions.parquet

Each input file must have the layout produced by
`electricity_load_forecasting.concat_X_y_predictions` +
`electricity_load_forecasting.cross_val_predict`: columns
`prediction_time`, one actual-value column per horizon (`"{h}h"`), one or more
prediction columns per horizon (`"pred_{h}h"` for a point forecast, or
`"pred_{h}h__q_{alpha}"` per quantile), and `split` (CV fold index).

MAE and MAPE always use the median quantile (or the point forecast, if there
is no quantile). Pinball (d2) is only reported for models with quantile
predictions. All rows across folds are pooled before computing metrics
(a walk-forward CV acts as one big held-out set, split-by-split for fitting
but not for scoring).

The functions here are also imported by `build_cv_report.py` and
`plot_cv_comparison.py`.
"""

import argparse
import re
import sys
from pathlib import Path

import polars as pl
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from electricity_load_forecasting.metrics import pinball


def horizons_of(df):
    return sorted(
        int(m.group(1)) for c in df.columns if (m := re.match(r"^(\d+)h$", c))
    )


def quantile_levels_of(df, h):
    return sorted(
        float(m.group(1))
        for c in df.columns
        if (m := re.match(rf"^pred_{h}h__q_([\d.]+)$", c))
    )


def mae(y_true, y_pred_point):
    average = mean_absolute_error(y_true, y_pred_point)
    detail = mean_absolute_error(y_true, y_pred_point, multioutput="raw_values")
    return {"mae__average": average} | {
        f"mae__{c}": float(s) for c, s in zip(y_true.columns, detail)
    }


def mape(y_true, y_pred_point):
    average = mean_absolute_percentage_error(y_true, y_pred_point)
    detail = mean_absolute_percentage_error(
        y_true, y_pred_point, multioutput="raw_values"
    )
    return {"mape__average": average} | {
        f"mape__{c}": float(s) for c, s in zip(y_true.columns, detail)
    }


def interval_metrics(y_true, pred_wide, horizons, low, high):
    """
    Coverage (fraction of true values within [q_low, q_high]) and average
    interval width (q_high - q_low, in the target's units), per horizon and
    averaged. `low`/`high` must be quantile levels present in `pred_wide`
    (columns `"{h}h__q_{low}"` / `"{h}h__q_{high}"`), with `low < 0.5 < high`.
    The nominal interval level (e.g. 90 for low=0.05, high=0.95) is used as a
    suffix on the metric names so several intervals can coexist.
    """
    nominal = round((high - low) * 100)
    coverage_detail, width_detail = [], []
    for h in horizons:
        actual = y_true[f"{h}h"]
        lo = pred_wide[f"{h}h__q_{low}"]
        hi = pred_wide[f"{h}h__q_{high}"]
        coverage_detail.append(float(((actual >= lo) & (actual <= hi)).mean()))
        width_detail.append(float((hi - lo).mean()))
    prefix = f"coverage_{nominal}"
    return (
        {f"{prefix}__average": sum(coverage_detail) / len(coverage_detail)}
        | {f"{prefix}__{h}h": c for h, c in zip(horizons, coverage_detail)}
        | {f"width_{nominal}__average": sum(width_detail) / len(width_detail)}
        | {f"width_{nominal}__{h}h": w for h, w in zip(horizons, width_detail)}
    )


def compute_scores(df):
    """
    Compute MAE, MAPE and (if available) pinball-d2 scores for one
    `*_cv_predictions.parquet` dataframe. Returns a flat dict of metric name
    -> value, with the same naming convention as `metrics.neg_mape` / `metrics.pinball`.
    """
    horizons = horizons_of(df)
    y_true = df.select([f"{h}h" for h in horizons])
    scores = {}

    levels = quantile_levels_of(df, horizons[0])
    if levels:
        pred_wide = df.select(
            pl.col(f"pred_{h}h__q_{q}").alias(f"{h}h__q_{q}")
            for h in horizons
            for q in levels
        )
        scores |= pinball(y_true, pred_wide)
        median = min(levels, key=lambda q: abs(q - 0.5))
        y_pred_point = df.select(
            [pl.col(f"pred_{h}h__q_{median}").alias(f"{h}h") for h in horizons]
        )
        # coverage/width for each symmetric pair of quantiles around 0.5
        # (e.g. 0.05/0.95 -> nominal 90% interval)
        pairs = {
            (lo, hi)
            for lo in levels
            for hi in levels
            if lo < 0.5 < hi and abs((lo + hi) - 1.0) < 1e-9
        }
        for lo, hi in sorted(pairs):
            scores |= interval_metrics(y_true, pred_wide, horizons, lo, hi)
    else:
        y_pred_point = df.select([f"pred_{h}h" for h in horizons])
    scores |= mae(y_true, y_pred_point)
    scores |= mape(y_true, y_pred_point)
    return scores


def compare(paths):
    """Return (results, table): per-model score dicts, and a polars table."""
    results = {}
    for path in paths:
        name = path.stem.removesuffix("_cv_predictions")
        df = pl.read_parquet(path)
        results[name] = compute_scores(df)
        print(f"{name}: {df.height} pooled rows across {df['split'].n_unique()} folds")

    all_keys = sorted(
        {k for s in results.values() for k in s},
        key=lambda k: (not k.startswith(("mae", "mape", "coverage", "width")), k),
    )
    table = pl.DataFrame(
        {"metric": all_keys}
        | {name: [s.get(k) for k in all_keys] for name, s in results.items()}
    )
    return results, table


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("predictions", nargs="+", type=Path)
    args = parser.parse_args()

    _, table = compare(args.predictions)
    with pl.Config(tbl_rows=-1, tbl_width_chars=160):
        print(table)
