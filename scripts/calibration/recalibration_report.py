"""Report for the raw vs recalibrated walk-forward CV.

Reads the files written by ``run_recalibration.py`` and writes ``index.html``.
"""

import datetime
import html
import json
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

from electricity_load_forecasting import diagnostics as dg

pred_columns = dg.pred_columns
reliability = dg.reliability
binned_quantiles = dg.binned_quantiles
isotonic_quantiles = dg.isotonic_quantiles
interval_coverage_by_level = dg.interval_coverage_by_level
panel_range = dg.panel_range

QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
CORRECTION_LEVELS = (0.95, 0.5, 0.05)
COLORS = {
    "truth": "#000000",
    "raw": "#d62728",
    "recalibrated": "#1f77b4",
    "ideal": "gray",
}


def mean_scores(scores):
    numeric = scores.drop("split", "test_start")
    return {c: float(numeric[c].mean()) for c in numeric.columns}




# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------
def on_hourly_grid(df):
    """df with one row per hour (a few missing) -> exactly one row per hour, so
    that plots can use x0/dx instead of a (large) array of timestamps.

    The few missing hours are filled with the previous value (for display only):
    plotly's filled areas do not handle missing values well.
    """
    times = df["prediction_time"]
    grid = pl.DataFrame(
        {
            "prediction_time": pl.datetime_range(
                times.min(), times.max(), "1h", eager=True
            )
        }
    )
    return (
        grid.join(df, on="prediction_time", how="left")
        .sort("prediction_time")
        .fill_null(strategy="forward")
    )


def fig_time_series(raw, recal, horizon_columns, window_days=14):
    """Ground truth, median and quantile bands over time, raw vs recalibrated.

    One dropdown entry per horizon; x is the target time (prediction time +
    horizon). The range slider covers all the CV test folds; the initial window
    is ``window_days`` days from the last CV split.
    """
    horizons = list(horizon_columns)
    raw_split = raw
    raw, recal = on_hourly_grid(raw), on_hourly_grid(recal)
    quantiles = sorted(next(iter(horizon_columns.values())))
    outer = (quantiles[0], quantiles[-1])
    median = min(quantiles, key=lambda q: abs(q - 0.5))
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("Raw model", "Recalibrated model"),
    )
    traces_per_horizon = []
    for k, h in enumerate(horizons):
        n_before = len(fig.data)
        # regular hourly x axis: x0 / dx (ms) instead of one timestamp per point
        x_axis = dict(
            x0=(raw["prediction_time"].min() + datetime.timedelta(hours=h)).strftime(
                "%Y-%m-%d %H:%M:%S"
            ),
            dx=3_600_000,
        )
        truth = raw[f"{h}h"].round(0)
        for row, (name, preds) in enumerate([("raw", raw), ("recalibrated", recal)], 1):
            color = COLORS[name]
            r, g, b = (int(color[i : i + 2], 16) for i in (1, 3, 5))
            for (lo, hi), alpha in [(outer, 0.2)]:
                label = f"{lo}-{hi} quantiles"
                fig.add_trace(
                    go.Scatter(
                        **x_axis,
                        y=preds[horizon_columns[h][hi]].round(0),
                        line=dict(width=0),
                        hoverinfo="skip",
                        legendgroup=f"{h}h {label} {name}",
                        showlegend=False,
                        visible=k == 0,
                    ),
                    row=row,
                    col=1,
                )
                fig.add_trace(
                    go.Scatter(
                        **x_axis,
                        y=preds[horizon_columns[h][lo]].round(0),
                        fill="tonexty",
                        line=dict(width=0),
                        fillcolor=f"rgba({r},{g},{b},{alpha})",
                        hoverinfo="skip",
                        name=f"{name}: {label}",
                        legendgroup=f"{h}h {label} {name}",
                        showlegend=True,
                        visible=k == 0,
                    ),
                    row=row,
                    col=1,
                )
            fig.add_trace(
                go.Scatter(
                    **x_axis,
                    y=preds[horizon_columns[h][median]].round(0),
                    line=dict(color=color, width=1.5),
                    name=f"{name}: median",
                    visible=k == 0,
                ),
                row=row,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    **x_axis,
                    y=truth,
                    line=dict(color=COLORS["truth"], width=1),
                    name="ground truth",
                    legendgroup=f"{h}h truth",
                    showlegend=row == 1,
                    visible=k == 0,
                ),
                row=row,
                col=1,
            )
        traces_per_horizon.append(range(n_before, len(fig.data)))
    buttons = []
    for k, h in enumerate(horizons):
        visible = [False] * len(fig.data)
        for i in traces_per_horizon[k]:
            visible[i] = True
        buttons.append(
            dict(label=f"{h}h", method="update", args=[{"visible": visible}])
        )
    last_split_start = raw_split.filter(pl.col("split") == raw_split["split"].max())[
        "prediction_time"
    ].min() + datetime.timedelta(days=14)
    window = [
        last_split_start,
        last_split_start + datetime.timedelta(days=window_days),
    ]
    fig.update_xaxes(range=window, row=2, col=1)
    fig.update_yaxes(title_text="load (MW)")
    # Same y range on both panels (all the values of all the horizons), and linked
    # so that zooming y in one panel zooms the other. (A range slider, which we do
    # not use, prevents zooming y with a selected rectangle when the axes are linked.)
    values = [raw[f"{h}h"] for h in horizons] + [
        preds[c]
        for preds in (raw, recal)
        for qs in horizon_columns.values()
        for c in qs.values()
    ]
    y_min = min(float(v.min()) for v in values)
    y_max = max(float(v.max()) for v in values)
    margin = 0.03 * (y_max - y_min)
    fig.update_yaxes(range=[y_min - margin, y_max + margin])
    fig.update_yaxes(matches="y", row=2, col=1)
    fig.update_layout(
        height=1000,
        margin=dict(t=120),
        hovermode="x unified",
        dragmode="zoom",
        title="Ground truth, median and quantiles over time (target time on the x axis)",
        updatemenus=[
            dict(
                type="dropdown",
                buttons=buttons,
                x=0.0,
                xanchor="left",
                y=1.0,
                yanchor="bottom",
                active=0,
            )
        ],
    )
    return fig


def fig_reliability(raw, recal, horizon_columns):
    horizons = list(horizon_columns)
    fig = make_subplots(
        rows=1,
        cols=len(horizons),
        subplot_titles=[f"horizon {h}h" for h in horizons],
        shared_yaxes=True,
        horizontal_spacing=0.05,
    )
    for j, h in enumerate(horizons, 1):
        fig.add_trace(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode="lines",
                line=dict(color=COLORS["ideal"], dash="dash"),
                name="perfect calibration",
                showlegend=j == 1,
            ),
            row=1,
            col=j,
        )
        for name, preds in [("raw", raw), ("recalibrated", recal)]:
            rel = reliability(preds, horizon_columns, h)
            fig.add_trace(
                go.Scatter(
                    x=list(rel),
                    y=list(rel.values()),
                    mode="lines+markers",
                    line=dict(color=COLORS[name]),
                    name=name,
                    showlegend=j == 1,
                ),
                row=1,
                col=j,
            )
        fig.update_xaxes(title_text="nominal level", range=[0, 1], row=1, col=j)
    fig.update_yaxes(title_text="observed fraction below prediction", row=1, col=1)
    fig.update_layout(
        height=450,
        title="Coverage curve (marginal reliability), one panel per horizon, all CV test "
        "folds pooled",
    )
    return fig




def fig_interval_coverage(raw, recal, horizon_columns, n_bins):
    """Coverage of the [q_low, q_high] interval, by predicted load level.

    One panel per horizon. The test cases are split into ``n_bins`` equal-count
    bins by each model's predicted median; the curve is the observed coverage of
    the interval in each bin, against its mean predicted median. This shows
    whether the marginal coverage is uniform across predicted loads, or whether
    over- and under-confidence in different regions compensate each other.
    """
    horizons = list(horizon_columns)
    quantiles = sorted(next(iter(horizon_columns.values())))
    q_low, q_high = quantiles[0], quantiles[-1]
    median = min(quantiles, key=lambda q: abs(q - 0.5))
    nominal = q_high - q_low
    fig = make_subplots(
        rows=1,
        cols=len(horizons),
        subplot_titles=[f"horizon {h}h" for h in horizons],
        shared_yaxes=True,
        horizontal_spacing=0.05,
    )
    for j, h in enumerate(horizons, 1):
        fig.add_hline(
            y=nominal,
            line=dict(color=COLORS["ideal"], dash="dash"),
            row=1,
            col=j,
            exclude_empty_subplots=False,
        )
        for name, preds in (("raw", raw), ("recalibrated", recal)):
            y = preds[f"{h}h"].to_numpy()
            low = preds[horizon_columns[h][q_low]].to_numpy()
            high = preds[horizon_columns[h][q_high]].to_numpy()
            center = preds[horizon_columns[h][median]].to_numpy()
            xs, coverage = interval_coverage_by_level(y, low, high, center, n_bins)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=coverage,
                    mode="lines+markers",
                    line=dict(color=COLORS[name]),
                    name=name,
                    showlegend=j == 1,
                ),
                row=1,
                col=j,
            )
        fig.update_xaxes(title_text="predicted median (MW)", row=1, col=j)
    fig.update_yaxes(
        title_text=f"observed coverage of the {q_low}-{q_high} interval",
        range=[0, 1],
        row=1,
        col=1,
    )
    fig.update_layout(
        height=480,
        title=f"Conditional coverage of the {q_low}-{q_high} interval by predicted "
        "load, all CV test folds pooled (dashed: nominal level)",
    )
    return fig


def fig_reliability_by_level(raw, recal, horizon_columns, n_bins, method="bins"):
    """Conditional reliability curves: one panel per quantile.

    For each model, the conditional tau-quantile of y given its own prediction
    is estimated with either equal-count bins (``method="bins"``, ``n_bins``
    bins, y = mean prediction + tau-quantile of the residuals in the bin) or
    isotonic regression (``method="isotonic"``, PAV, a monotone step function
    whose blocks are placed by the data). A calibrated model lies on the dashed
    diagonal (actual = predicted); the axes are on the same scale (1:1). A
    dropdown selects the horizon.
    """
    horizons = list(horizon_columns)
    quantiles = sorted(next(iter(horizon_columns.values())))
    n_cols = 4
    n_rows = -(-len(quantiles) // n_cols)
    fig = make_subplots(
        rows=n_rows,
        cols=n_cols,
        subplot_titles=[f"q_{q}" for q in quantiles],
        vertical_spacing=0.14,
        horizontal_spacing=0.06,
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            line=dict(color=COLORS["ideal"], dash="dash"),
            name="perfect calibration (actual = predicted)",
        ),
        row=1,
        col=1,
    )
    # curves[(horizon, quantile, model)] = (mean predictions, quantiles)
    curves = {}
    for h in horizons:
        for q in quantiles:
            for name, preds in (("raw", raw), ("recalibrated", recal)):
                prediction = preds[horizon_columns[h][q]].to_numpy()
                y = preds[f"{h}h"].to_numpy()
                if method == "bins":
                    curves[(h, q, name)] = binned_quantiles(y, prediction, q, n_bins)
                else:
                    curves[(h, q, name)] = isotonic_quantiles(prediction, y, q)
    traces_per_horizon = []
    for k, h in enumerate(horizons):
        first = len(fig.data)
        for i, q in enumerate(quantiles):
            row, col = divmod(i, n_cols)
            row, col = row + 1, col + 1
            for name in ("raw", "recalibrated"):
                xs, ys = curves[(h, q, name)]
                if method == "bins":
                    trace = go.Scatter(
                        x=xs,
                        y=ys,
                        mode="markers",
                        marker=dict(color=COLORS[name], size=7),
                        name=name,
                        legendgroup=name,
                        showlegend=k == 0 and i == 0,
                        visible=k == 0,
                    )
                else:
                    trace = go.Scatter(
                        x=xs,
                        y=ys,
                        mode="lines",
                        line=dict(color=COLORS[name], width=2, shape="hv"),
                        name=name,
                        legendgroup=name,
                        showlegend=k == 0 and i == 0,
                        visible=k == 0,
                    )
                fig.add_trace(trace, row=row, col=col)
        traces_per_horizon.append(range(first, len(fig.data)))
    for i, q in enumerate(quantiles):
        row, col = divmod(i, n_cols)
        row, col = row + 1, col + 1
        bounds = panel_range(
            [
                v
                for h in horizons
                for name in ("raw", "recalibrated")
                for part in curves[(h, q, name)]
                for v in part
            ]
        )
        fig.add_shape(
            type="line",
            x0=bounds[0],
            x1=bounds[1],
            y0=bounds[0],
            y1=bounds[1],
            line=dict(color=COLORS["ideal"], dash="dash"),
            row=row,
            col=col,
        )
        # 1:1 scale, so that the diagonal is at 45 degrees
        x_axis = "x" if i == 0 else f"x{i + 1}"
        fig.update_xaxes(
            title_text="predicted value (MW)", range=bounds, row=row, col=col
        )
        fig.update_yaxes(
            range=bounds,
            scaleanchor=x_axis,
            scaleratio=1,
            row=row,
            col=col,
        )
        if col == 1:
            fig.update_yaxes(
                title_text="estimated &tau;-quantile of y (MW)", row=row, col=col
            )
    buttons = []
    for k, h in enumerate(horizons):
        visible = [False] * len(fig.data)
        for i in traces_per_horizon[k]:
            visible[i] = True
        buttons.append(
            dict(label=f"{h}h", method="update", args=[{"visible": visible}])
        )
    kind = "equal-count bins" if method == "bins" else "isotonic / PAV"
    fig.update_layout(
        height=360 * n_rows + 190,
        margin=dict(t=120),
        title=f"Conditional reliability ({kind}), one panel per quantile, "
        "all CV test folds pooled",
        updatemenus=[
            dict(
                type="dropdown",
                buttons=buttons,
                x=0.0,
                xanchor="left",
                y=1.0,
                yanchor="bottom",
                active=0,
            )
        ],
    )
    return fig


def test_bins(p, y, tau, n_bins):
    """Equal-count bins of the predictions p.

    Returns a dict of lists: mean p, tau-quantile of y - p (the quantity the
    recalibrator learns), true tau-quantile of y, and number of points.
    """
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    bin_ids = np.searchsorted(edges[1:-1], p, side="right")
    out = {"x": [], "true_quantile": [], "residual_shift": [], "count": []}
    for b in np.unique(bin_ids):
        m = bin_ids == b
        out["x"].append(float(p[m].mean()))
        out["true_quantile"].append(float(np.quantile(y[m], tau)))
        out["residual_shift"].append(float(np.quantile(y[m] - p[m], tau)))
        out["count"].append(int(m.sum()))
    return out


def fig_corrections(corrections, q, raw, n_bins):
    """Learned correction (line) vs correction measured on the test fold (points).

    One colour per CV split. Lines: the correction learned by the recalibrator
    on the training set of the split (spline + ridge fit of the shifts of the
    inner-CV bins). Points: the same measurement made on the outer test fold
    with the raw predictions (``n_bins`` equal-count bins of the predicted
    value; shift = tau-quantile of y - p in the bin), i.e. what the correction
    should have been on the test data. The line should go through the points
    if the calibration learned on the training set transfers to the test fold.

    Bottom row: the correction (shift in MW) as a function of the predicted
    value. Top row: the same as predicted quantile -> actual quantile
    (predicted value + shift).
    """
    horizons = sorted(corrections["horizon"].unique().to_list())
    splits = sorted(corrections["split"].unique().to_list())
    palette = [
        "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
        "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
    ]  # fmt: skip
    column = f"q_{q}"
    fig = make_subplots(
        rows=2,
        cols=len(horizons),
        subplot_titles=[f"q_{q}, {h}h" for h in horizons] + [""] * len(horizons),
        vertical_spacing=0.1,
        horizontal_spacing=0.06,
    )
    # legend entries explaining the line / point types
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="lines", line=dict(color="gray"),
            name="line: learned on the training set",
        ),
        row=1,
        col=1,
    )  # fmt: skip
    fig.add_trace(
        go.Scatter(
            x=[None], y=[None], mode="markers", marker=dict(color="gray", size=7),
            name="points: measured on the test fold",
        ),
        row=1,
        col=1,
    )  # fmt: skip
    x_all = corrections.filter((pl.col("column") == column))["x"]
    lims = [float(x_all.min()), float(x_all.max())]
    for j, h in enumerate(horizons, 1):
        fig.add_trace(
            go.Scatter(
                x=lims,
                y=lims,
                mode="lines",
                line=dict(color=COLORS["ideal"], dash="dash"),
                name="calibrated (actual = predicted)",
                showlegend=j == 1,
            ),
            row=1,
            col=j,
        )
        fig.add_hline(
            y=0,
            line=dict(color=COLORS["ideal"], dash="dash"),
            row=2,
            col=j,
            exclude_empty_subplots=False,
        )
        per_split = {}
        for s in splits:
            curve = corrections.filter(
                (pl.col("horizon") == h)
                & (pl.col("split") == s)
                & (pl.col("column") == column)
                & (pl.col("kind") == "curve")
            )
            test_part = raw.filter(pl.col("split") == s)
            tb = test_bins(
                test_part[f"pred_{h}h__q_{q}"].to_numpy(),
                test_part[f"{h}h"].to_numpy(),
                q,
                n_bins,
            )
            x_test = np.array(tb["x"])
            shift_test = np.array(tb["residual_shift"])
            customdata = np.column_stack([tb["count"], shift_test])
            per_split[s] = (curve, x_test, shift_test, customdata)
        all_x = np.concatenate(
            [curve["x"].to_numpy() for curve, _, _, _ in per_split.values()]
            + [x_test for _, x_test, _, _ in per_split.values()]
        )
        x_lo, x_hi = float(all_x.min()), float(all_x.max())
        pad = 0.05 * (x_hi - x_lo) if x_hi > x_lo else 1.0
        xlim = [x_lo - pad, x_hi + pad]
        for s in splits:
            color = palette[s % len(palette)]
            curve, x_test, shift_test, customdata = per_split[s]
            xs = curve["x"].to_numpy()
            shifts = curve["y"].to_numpy()
            # The spline uses constant extrapolation: show the flat extension
            # on both sides of the learned range.
            x_ext = np.concatenate([[xlim[0]], xs, [xlim[1]]])
            shift_ext = np.concatenate([[shifts[0]], shifts, [shifts[-1]]])
            for row in (1, 2):
                y_curve = x_ext + shift_ext if row == 1 else shift_ext
                y_points = x_test + shift_test if row == 1 else shift_test
                fig.add_trace(
                    go.Scatter(
                        x=x_ext,
                        y=y_curve,
                        mode="lines",
                        line=dict(color=color, width=2),
                        hoverinfo="skip",
                        legendgroup=f"split {s}",
                        name=f"split {s}",
                        showlegend=(row, j) == (1, 1),
                    ),
                    row=row,
                    col=j,
                )
                fig.add_trace(
                    go.Scatter(
                        x=x_test,
                        y=y_points,
                        mode="markers",
                        marker=dict(color=color, size=7),
                        customdata=customdata,
                        hovertemplate=f"split {s}, test fold<br>predicted %{{x:.0f}} MW"
                        "<br>shift %{customdata[1]:.0f} MW"
                        "<br>%{customdata[0]} points<extra></extra>",
                        legendgroup=f"split {s}",
                        showlegend=False,
                    ),
                    row=row,
                    col=j,
                )
        for row in (1, 2):
            fig.update_xaxes(range=xlim, row=row, col=j)
        fig.update_xaxes(title_text="predicted value (MW)", row=2, col=j)
    fig.update_yaxes(title_text=f"actual {column} of the bin (MW)", row=1, col=1)
    fig.update_yaxes(title_text="correction = actual - predicted (MW)", row=2, col=1)
    fig.update_layout(
        height=800,
        title=f"{column}: correction learned on the training set (lines) vs measured on "
        "the test fold (points)",
    )
    return fig


# ---------------------------------------------------------------------------
# Tables
# ---------------------------------------------------------------------------
def table(header, rows, fmt="{:.3f}"):
    head = "".join(f"<th>{h}</th>" for h in header)
    body = ""
    for row in rows:
        cells = "".join(
            f"<td>{fmt.format(c) if isinstance(c, float) else html.escape(str(c))}</td>"
            for c in row
        )
        body += f"<tr>{cells}</tr>"
    return f"<table><tr>{head}</tr>{body}</table>"


def tables(raw_scores, recal_scores, horizons, quantiles):
    q_low, q_high = min(quantiles), max(quantiles)
    r, c = mean_scores(raw_scores), mean_scores(recal_scores)
    interval_rows = []
    for h in horizons:
        cov = f"coverage_{q_low}_{q_high}__{h}h"
        wid = f"mean_width_{q_low}_{q_high}__{h}h"
        interval_rows.append(
            [f"{h}h", r[cov], c[cov], r[wid], c[wid], c[wid] / r[wid] - 1]
        )
    interval = table(
        [
            "horizon",
            "coverage<br>raw",
            "coverage<br>recalibrated",
            "mean width (MW)<br>raw",
            "mean width (MW)<br>recalibrated",
            "width change",
        ],
        [
            [row[0], row[1], row[2], f"{row[3]:.0f}", f"{row[4]:.0f}", f"{row[5]:+.1%}"]
            for row in interval_rows
        ],
    )
    mape_rows = []
    for h in horizons:
        key = f"neg_mape__{h}h"
        raw_mape = -r[key] * 100
        recal_mape = -c[key] * 100
        mape_rows.append([f"{h}h", raw_mape, recal_mape, recal_mape - raw_mape])
    mape = table(
        [
            "horizon",
            "MAPE of the median<br>raw (%)",
            "MAPE of the median<br>recalibrated (%)",
            "change (points)",
        ],
        mape_rows,
    )
    mape_split = table(
        ["test start"]
        + [f"{h}h {name}" for h in horizons for name in ("raw", "recal.")],
        [
            [str(raw_scores["test_start"][s])[:10]]
            + [
                value
                for h in horizons
                for value in (
                    -float(raw_scores[f"neg_mape__{h}h"][s]) * 100,
                    -float(recal_scores[f"neg_mape__{h}h"][s]) * 100,
                )
            ]
            for s in range(raw_scores.height)
        ],
        fmt="{:.2f}",
    )
    pinball_rows = []
    for q in list(quantiles):
        key = f"d2_pinball_score__average__q_{q}"
        pinball_rows.append([f"q_{q}", r[key], c[key], c[key] - r[key]])
    pinball_rows.append(
        [
            "mean over quantiles",
            float(np.mean([row[1] for row in pinball_rows])),
            float(np.mean([row[2] for row in pinball_rows])),
            float(np.mean([row[3] for row in pinball_rows])),
        ]
    )
    pinball = table(
        ["quantile", "d2 pinball<br>raw", "d2 pinball<br>recalibrated", "difference"],
        pinball_rows,
        fmt="{:.4f}",
    )
    per_split = []
    for s in range(raw_scores.height):
        row = [
            str(raw_scores["test_start"][s])[:10],
        ]
        for h in horizons:
            cov = f"coverage_{q_low}_{q_high}__{h}h"
            row += [float(raw_scores[cov][s]), float(recal_scores[cov][s])]
        per_split.append(row)
    split_header = ["test start"] + [
        f"{h}h {name}" for h in horizons for name in ("raw", "recal.")
    ]
    split = table(split_header, per_split)
    return interval, mape, mape_split, pinball, split


# ---------------------------------------------------------------------------
PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Recalibration on the electricity load data</title>
<style>
  body {{ font-family: sans-serif; max-width: 1800px; margin: 0.5em auto; line-height: 1.5; padding: 0; }}
  table {{ border-collapse: collapse; margin: 1em 0; }}
  th, td {{ border: 1px solid #ccc; padding: 0.3em 0.7em; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
  code {{ background: #f0f0f0; padding: 0 0.2em; }}
</style>
</head>
<body>
<h1>Recalibration of the quantile regressor on the real data</h1>
<p>Walk-forward CV of the data op with
<code>quantile_strategy="multiple_regressors"</code> (HGB, <code>max_iter={max_iter}</code>),
horizons {horizons}, quantiles {quantiles}, {n_splits} CV splits, with
(<code>recalibrate=True</code>, {n_bins} bins, calibrated on {calibration_window}) and without
<code>calibration.Recalibrator</code>. See <code>calibration_demo.html</code> for how the
correction works. Scores are those reported by the data op, averaged over the CV
splits. Coverage and width are for the {q_low}&ndash;{q_high} interval (nominal
coverage {nominal:.2f}).</p>

<h2>Scores</h2>
<h3>Interval coverage and width</h3>
{interval}
<h3>MAPE of the median prediction</h3>
<p><code>mean_absolute_percentage_error</code> of each model's median prediction,
averaged over the CV splits. The last column is the change in percentage points
(negative is better).</p>
{mape}
<h3>MAPE of the median for each CV split (%)</h3>
{mape_split}
<h3>d2 pinball score (higher is better), averaged over horizons</h3>
{pinball}
<h3>Coverage of the {q_low}&ndash;{q_high} interval for each CV split</h3>
{split}

{reading}

<h2>Predictions over time</h2>
<p>Ground truth (black), median (line) and the {q_low}&ndash;{q_high} quantile band, for the raw
(top) and recalibrated (bottom) models, concatenating the test folds of all the CV splits. The x
axis is the target time (prediction time plus horizon). Use the dropdown to change the horizon.
Drag a rectangle to zoom in both x and y; double-click to reset. The initial window is two weeks
of the last split. The two panels look nearly identical because the correction is small: over
these test folds the recalibrated median differs from the raw one by at most {median_diff:.0f}
MW, and the interval bounds by at most {band_diff:.0f} MW (about {relative_diff:.1%} of the
load). Zoom in to see it (for example around {max_diff_date}).</p>
{time_series}

<h2>Reliability diagrams</h2>
<p>Both plots below are reliability diagrams, but they check different properties. All the
CV test folds are pooled.</p>

<h3>Coverage curve (marginal reliability)</h3>
<p>For each nominal level &tau;, the empirical coverage: the fraction of observed values at
or below the &tau;-quantile forecast, plotted against &tau; (one panel per horizon, dashed
line = perfect calibration). It is the marginal check, averaged over all situations, so a
model can be on the diagonal on average while being miscalibrated in particular situations,
with errors cancelling out.</p>
{reliability}

<h3>Conditional coverage of the interval (by predicted load)</h3>
<p>The marginal coverage above can hide over-confidence in some regions and under-confidence
in others. Here the test cases are split into {n_bins} bins of equal size by each model's
predicted median, and in each bin we compute the observed coverage of the
{q_low}&ndash;{q_high} interval (the fraction of observations inside it). The dashed line is
the nominal level. A curve below it means the interval is too narrow at that predicted load,
above it too wide.</p>
{interval_coverage}

<h3>Conditional reliability curves, binned (one panel per quantile)</h3>
<p>One panel per quantile level &tau; (use the dropdown to change the horizon). The test cases
are split into {n_bins} bins of equal size by predicted value (each model's own predictions);
in each bin, x is the mean prediction and y is the mean prediction plus the &tau;-quantile of
the residuals <code>y &minus; p</code> in the bin. Using the residuals (rather than the plain
quantile of <code>y</code> in the bin) removes an artefact of the predictions varying inside a
bin, so a perfectly calibrated model lies exactly on the dashed 45&deg; diagonal (actual =
predicted); the axes are on the same scale (1:1). This is the conditional check: it asks
whether the claimed level holds for the cases with that prediction, and reveals
miscalibration that depends on the predicted value, which the coverage curve can hide.</p>
{reliability_bins}

<h3>Conditional reliability curves, isotonic / PAV (one panel per quantile)</h3>
<p>The same check, with the conditional &tau;-quantile estimated by isotonic regression: sort
the cases by prediction, then repeatedly pool adjacent blocks whenever their &tau;-quantile
would decrease. The result is a monotone step function whose blocks are placed by the data,
with no bin width to choose. Again, a perfectly calibrated model lies on the dashed 45&deg;
diagonal, and the axes are on the same scale (1:1).</p>
{reliability_isotonic}



<h2>Learned corrections</h2>
<p>For q_0.95, q_0.5 and q_0.05, one colour per CV split. Lines: the correction (shift in
MW added to the predicted quantile) learned by the recalibrator on the training set of the
split, as a function of the predicted value. Points: the same measured on the outer test fold
with the raw predictions (equal-count bins of the predicted value, shift = quantile of
y &minus; p in the bin), i.e. what the correction should have been on the test data. If the
calibration transfers from the training set to the test fold, the lines go through the points.
Bottom: the correction itself; top: the same as predicted quantile &rarr; actual quantile
(predicted + shift), where the dashed diagonal is a calibrated model (the corrections are small
compared to the range of loads, so the bottom row is easier to read).</p>
{corrections}
</body>
</html>
"""


def read_config(output_dir):
    """Settings of a run (``config.json``), or {} for runs that predate it."""
    path = Path(output_dir) / "config.json"
    return json.loads(path.read_text()) if path.exists() else {}


def describe_calibration_window(config):
    if config.get("n_last_folds") is not None:
        n = config["n_last_folds"]
        return f"the last {n} inner fold{'s' if n > 1 else ''} of each training set"
    # runs from earlier versions
    if config.get("last_rows") is not None:
        rows = config["last_rows"]
        return f"the last {rows} rows (about {rows / 24 / 30.4:.1f} months) of each training set"
    if config.get("recent_rows") is not None:
        return (
            "the most recent whole inner folds "
            f"(totalling at least {config['recent_rows']} rows)"
        )
    return "all the inner folds"


def build_report(output_dir):
    raw = pl.read_parquet(output_dir / "raw_cv_predictions.parquet")
    recal = pl.read_parquet(output_dir / "recalibrated_cv_predictions.parquet")
    raw_scores = pl.read_csv(output_dir / "raw_scores.csv", try_parse_dates=True)
    recal_scores = pl.read_csv(
        output_dir / "recalibrated_scores.csv", try_parse_dates=True
    )
    corrections = pl.read_parquet(output_dir / "corrections.parquet")
    assert raw["prediction_time"].equals(recal["prediction_time"])
    config = read_config(output_dir)

    horizon_columns = pred_columns(raw)
    horizons = list(horizon_columns)
    quantiles = tuple(next(iter(horizon_columns.values())))
    n_bins = (
        corrections.filter(pl.col("kind") == "bin")
        .group_by("split", "horizon", "column")
        .len()["len"]
        .max()
    )
    interval, mape, mape_split, pinball, split = tables(
        raw_scores, recal_scores, horizons, quantiles
    )
    # how much the recalibration moves the predictions (for the time-series section)
    median_diff, band_diff = 0.0, 0.0
    for h in horizons:
        median_diff = max(
            median_diff,
            float((recal[f"pred_{h}h__q_0.5"] - raw[f"pred_{h}h__q_0.5"]).abs().max()),
        )
        for q in (quantiles[0], quantiles[-1]):
            band_diff = max(
                band_diff,
                float(
                    (recal[f"pred_{h}h__q_{q}"] - raw[f"pred_{h}h__q_{q}"]).abs().max()
                ),
            )
    median_change = (
        recal[[f"pred_{h}h__q_0.5" for h in horizons]]
        - raw[[f"pred_{h}h__q_0.5" for h in horizons]]
    ).to_numpy()
    row, col = np.unravel_index(np.abs(median_change).argmax(), median_change.shape)
    max_diff_date = str(raw["prediction_time"][int(row)])[:16]
    load = float(raw.select([f"{h}h" for h in horizons]).to_numpy().mean())
    relative_diff = band_diff / load
    reading_path = output_dir / "reading.html"
    reading = reading_path.read_text() if reading_path.exists() else ""
    plots = {
        "time_series": fig_time_series(raw, recal, horizon_columns),
        "reliability": fig_reliability(raw, recal, horizon_columns),
        "interval_coverage": fig_interval_coverage(raw, recal, horizon_columns, n_bins),
        "reliability_bins": fig_reliability_by_level(
            raw, recal, horizon_columns, n_bins, method="bins"
        ),
        "reliability_isotonic": fig_reliability_by_level(
            raw, recal, horizon_columns, n_bins, method="isotonic"
        ),
    }
    for q in CORRECTION_LEVELS:
        plots[f"corrections_{q}"] = fig_corrections(corrections, q, raw, n_bins)
    parts = {
        k: f.to_html(full_html=False, include_plotlyjs=True if i == 0 else False)
        for i, (k, f) in enumerate(plots.items())
    }
    parts["corrections"] = "".join(
        parts.pop(f"corrections_{q}") for q in CORRECTION_LEVELS
    )
    (output_dir / "index.html").write_text(
        PAGE.format(
            max_iter=config.get("max_iter", "?"),
            calibration_window=describe_calibration_window(config),
            horizons=", ".join(f"{h}h" for h in horizons),
            quantiles=", ".join(map(str, quantiles)),
            median_diff=median_diff,
            band_diff=band_diff,
            relative_diff=relative_diff,
            max_diff_date=max_diff_date,
            n_splits=raw_scores.height,
            n_bins=n_bins,
            q_low=min(quantiles),
            q_high=max(quantiles),
            nominal=max(quantiles) - min(quantiles),
            interval=interval,
            mape=mape,
            mape_split=mape_split,
            pinball=pinball,
            split=split,
            reading=reading,
            **parts,
        ),
        "utf-8",
    )
    print(f"wrote {output_dir / 'index.html'}")
