"""
Uncertainty of paired model differences, shared by the report scripts.

Two complementary views of "is model A better than model B" on the pooled CV
predictions of two models (identical rows):

- per-fold mean differences (one number per CV fold): shows how consistent the
  difference is across periods, with no distributional assumption;
- a 95% moving-block bootstrap interval of the pooled mean difference, built
  on daily mean differences so that autocorrelation within the test period is
  respected. It is conditional on the fitted models: it ignores training
  randomness and does not cover periods outside the test range.

The folds are not independent samples (nested training sets, adjacent test
blocks, 8 seasons), so a t-interval across folds is not used.
"""

import numpy as np
from compare_predictions import horizons_of, quantile_levels_of

PINBALL_LEVELS = (0.05, 0.5, 0.95)
BLOCK_DAYS = 14


def row_index(df):
    """Integer day and fold index of each row (rows must be time-sorted)."""
    days = df["prediction_time"].dt.truncate("1d").dt.epoch("d").to_numpy()
    return np.unique(days, return_inverse=True)[1], df["split"].to_numpy()


def row_loss(df, kind, h, q=None):
    """Per-row loss of one model at horizon h, or None if not available.

    kind: "mae" (MW), "mape" (%), "pinball" (MW, quantile level q), "width"
    (MW, 5%-95% interval).
    """
    y = df[f"{h}h"].to_numpy()
    if kind in ("mae", "mape"):
        column = f"pred_{h}h" if f"pred_{h}h" in df.columns else f"pred_{h}h__q_0.5"
        if column not in df.columns:
            return None
        error = np.abs(df[column].to_numpy() - y)
        return error if kind == "mae" else 100 * error / y
    if kind == "pinball":
        column = f"pred_{h}h__q_{q}"
        if column not in df.columns:
            return None
        e = y - df[column].to_numpy()
        return np.maximum(q * e, (q - 1) * e)
    if kind == "width":
        low, high = f"pred_{h}h__q_0.05", f"pred_{h}h__q_0.95"
        if low not in df.columns or high not in df.columns:
            return None
        return df[high].to_numpy() - df[low].to_numpy()
    raise ValueError(kind)


def average_row_loss(df, kind):
    """Per-row loss averaged over horizons (and, for pinball, over the
    5/50/95% quantiles, or all available levels if these are missing)."""
    hs = horizons_of(df)
    if kind == "mape":
        return np.mean([row_loss(df, "mape", h) for h in hs], axis=0)
    levels = quantile_levels_of(df, hs[0])
    chosen = [q for q in levels if any(abs(q - t) < 1e-9 for t in PINBALL_LEVELS)]
    chosen = chosen or levels
    if not chosen:
        return None
    return np.mean([row_loss(df, "pinball", h, q) for h in hs for q in chosen], axis=0)


def block_bootstrap_interval(daily, block_days=BLOCK_DAYS, n_boot=2000, seed=0):
    """95% percentile interval of the mean of ``daily`` (moving blocks)."""
    n = len(daily)
    block_days = min(block_days, n)
    rng = np.random.default_rng(seed)
    starts = rng.integers(0, n - block_days + 1, (n_boot, -(-n // block_days)))
    idx = (starts[:, :, None] + np.arange(block_days)).reshape(n_boot, -1)[:, :n]
    return np.percentile(daily[idx].mean(axis=1), [2.5, 97.5])


def paired_difference(loss, baseline_loss, day, fold):
    """Summary of the row-level difference ``loss - baseline_loss``.

    Returns a dict: ``mean`` (pooled), ``low`` / ``high`` (block bootstrap
    interval, shifted to be centred on the pooled mean), ``folds`` (mean
    difference in each fold) and ``n_lower`` (number of folds where the model
    has the lower value).
    """
    diff = loss - baseline_loss
    daily = np.bincount(day, diff) / np.bincount(day)
    low, high = block_bootstrap_interval(daily) + (diff.mean() - daily.mean())
    folds = np.array([diff[fold == f].mean() for f in np.unique(fold)])
    return {
        "mean": float(diff.mean()),
        "low": float(low),
        "high": float(high),
        "folds": folds,
        "n_lower": int((folds < 0).sum()),
    }


def error_bar(summary):
    """Plotly asymmetric error bar arguments for one summary."""
    return summary["high"] - summary["mean"], summary["mean"] - summary["low"]


INTERVAL_NOTE = (
    "Large markers: pooled mean difference over all test rows, with a 95% "
    f"moving-block bootstrap interval ({BLOCK_DAYS}-day blocks over daily "
    "differences; conditional on the fitted models, it ignores training "
    "randomness). Small dots: mean difference in each of the 8 CV folds (one "
    "dot per fold; hover shows the fold). Folds are not independent, so no "
    "t-interval across folds is shown."
)
