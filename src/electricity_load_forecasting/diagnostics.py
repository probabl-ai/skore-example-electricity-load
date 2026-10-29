import numpy as np


def pred_columns(predictions):
    columns = {}
    for name in predictions.columns:
        if not name.startswith("pred_"):
            continue
        rest = name.removeprefix("pred_")
        horizon, _, quantile = rest.partition("h__q_")
        if not quantile:
            continue
        columns.setdefault(int(horizon), {})[float(quantile)] = name
    return columns


def reliability(predictions, horizon_columns, horizon):
    y = predictions[f"{horizon}h"].to_numpy()
    return {
        q: float((y <= predictions[c].to_numpy()).mean())
        for q, c in horizon_columns[horizon].items()
    }


def binned_quantiles(y, prediction, tau, n_bins):
    prediction = np.asarray(prediction, dtype=float)
    y = np.asarray(y, dtype=float)
    edges = np.unique(np.quantile(prediction, np.linspace(0, 1, n_bins + 1)))
    bin_ids = np.searchsorted(edges[1:-1], prediction, side="right")
    xs, quantiles = [], []
    for b in np.unique(bin_ids):
        in_bin = bin_ids == b
        mean_prediction = float(prediction[in_bin].mean())
        xs.append(mean_prediction)
        quantiles.append(
            mean_prediction + float(np.quantile(y[in_bin] - prediction[in_bin], tau))
        )
    return xs, quantiles


def isotonic_quantiles(prediction, y, tau):
    order = np.argsort(prediction, kind="stable")
    prediction = np.asarray(prediction, dtype=float)[order]
    y = np.asarray(y, dtype=float)[order]
    starts = np.flatnonzero(np.r_[True, prediction[1:] != prediction[:-1]])
    ends = np.r_[starts[1:], len(prediction)]
    blocks = []
    for start, end in zip(starts, ends):
        blocks.append([int(start), int(end), float(np.quantile(y[start:end], tau))])
        while len(blocks) >= 2 and blocks[-2][2] > blocks[-1][2]:
            first = blocks[-2][0]
            last = blocks[-1][1]
            blocks[-2:] = [[first, last, float(np.quantile(y[first:last], tau))]]
    xs = [float(prediction[start:end].mean()) for start, end, _ in blocks]
    quantiles = [value for _, _, value in blocks]
    return xs, quantiles


def interval_coverage_by_level(y, low, high, center, n_bins):
    y = np.asarray(y, dtype=float)
    low = np.asarray(low, dtype=float)
    high = np.asarray(high, dtype=float)
    center = np.asarray(center, dtype=float)
    edges = np.unique(np.quantile(center, np.linspace(0, 1, n_bins + 1)))
    bin_ids = np.searchsorted(edges[1:-1], center, side="right")
    xs, coverage = [], []
    for b in np.unique(bin_ids):
        in_bin = bin_ids == b
        xs.append(float(center[in_bin].mean()))
        coverage.append(float(((y[in_bin] >= low[in_bin]) & (y[in_bin] <= high[in_bin])).mean()))
    return xs, coverage


def panel_range(values):
    low, high = float(np.min(values)), float(np.max(values))
    margin = 0.02 * (high - low)
    return [low - margin, high + margin]
