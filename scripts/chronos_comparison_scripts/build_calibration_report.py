"""
Reliability / calibration diagrams for every `*_cv_predictions.parquet` in a
directory, in the same style as the report built by `compare_calibration.py`
(coverage curve, conditional interval coverage, conditional reliability curves
per quantile, binned and isotonic), but overlaying all the models found there
(e.g. hgb_quantile, hgb_quantile_recal, tabicl).

    python scripts/chronos_comparison_scripts/build_calibration_report.py results/cv_compare_tabicl

Writes `<dir>/calibration.html`.
"""

import argparse
import sys
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "calibration"))
import compare_calibration as cc

IDEAL = "gray"
PALETTE = ["#d62728", "#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e", "#8c564b"]


def color_for(name, i):
    if "recal" in name:
        return "#1f77b4"
    if "tabicl" in name:
        return "#2ca02c"
    if "hgb" in name:
        return "#d62728"
    return PALETTE[i % len(PALETTE)]


def load_models(directory):
    paths = sorted(directory.glob("*_cv_predictions.parquet"))
    if not paths:
        raise SystemExit(f"no *_cv_predictions.parquet found in {directory}")
    models = []
    for i, path in enumerate(paths):
        name = path.stem.removesuffix("_cv_predictions")
        models.append((name, pl.read_parquet(path), color_for(name, i)))
    return models


def check_layout(models):
    horizon_columns = cc.pred_columns(models[0][1])
    for name, df, _ in models[1:]:
        hc = cc.pred_columns(df)
        if list(hc) != list(horizon_columns) or {h: list(q) for h, q in hc.items()} != {
            h: list(q) for h, q in horizon_columns.items()
        }:
            raise SystemExit(f"{name}: horizons/quantiles differ from {models[0][0]}")
    return horizon_columns


def fig_marginal(models, horizon_columns):
    """Coverage curve (marginal reliability), one panel per horizon."""
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
                line=dict(color=IDEAL, dash="dash"),
                name="perfect calibration",
                showlegend=j == 1,
            ),
            row=1,
            col=j,
        )
        for name, df, color in models:
            rel = cc.reliability(df, horizon_columns, h)
            fig.add_trace(
                go.Scatter(
                    x=list(rel),
                    y=list(rel.values()),
                    mode="lines+markers",
                    line=dict(color=color),
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
        title="Coverage curve (marginal reliability), all CV test folds pooled",
    )
    return fig


def fig_interval(models, horizon_columns, n_bins):
    """Coverage of the outermost interval, in equal-count bins of the median."""
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
            line=dict(color=IDEAL, dash="dash"),
            row=1,
            col=j,
            exclude_empty_subplots=False,
        )
        for name, df, color in models:
            y = df[f"{h}h"].to_numpy()
            low = df[horizon_columns[h][q_low]].to_numpy()
            high = df[horizon_columns[h][q_high]].to_numpy()
            center = df[horizon_columns[h][median]].to_numpy()
            xs, coverage = cc.interval_coverage_by_level(y, low, high, center, n_bins)
            fig.add_trace(
                go.Scatter(
                    x=xs,
                    y=coverage,
                    mode="lines+markers",
                    line=dict(color=color),
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


def fig_by_level(models, horizon_columns, n_bins, method="bins"):
    """Conditional reliability curves, one panel per quantile.

    method="bins": equal-count bins; method="isotonic": PAV step function.
    A perfectly calibrated model lies on the dashed 45 degree diagonal.
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
            line=dict(color=IDEAL, dash="dash"),
            name="perfect calibration (actual = predicted)",
        ),
        row=1,
        col=1,
    )
    curves = {}
    for h in horizons:
        for q in quantiles:
            for name, df, _ in models:
                prediction = df[horizon_columns[h][q]].to_numpy()
                y = df[f"{h}h"].to_numpy()
                if method == "bins":
                    curves[(h, q, name)] = cc.binned_quantiles(y, prediction, q, n_bins)
                else:
                    curves[(h, q, name)] = cc.isotonic_quantiles(prediction, y, q)
    traces_per_horizon = []
    for k, h in enumerate(horizons):
        first = len(fig.data)
        for i, q in enumerate(quantiles):
            row, col = divmod(i, n_cols)
            row, col = row + 1, col + 1
            for name, _, color in models:
                xs, ys = curves[(h, q, name)]
                if method == "bins":
                    trace = go.Scatter(
                        x=xs,
                        y=ys,
                        mode="markers",
                        marker=dict(color=color, size=7),
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
                        line=dict(color=color, width=2, shape="hv"),
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
        bounds = cc.panel_range(
            [
                v
                for h in horizons
                for name, _, _ in models
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
            line=dict(color=IDEAL, dash="dash"),
            row=row,
            col=col,
        )
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


PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Reliability / calibration diagrams</title>
<style>
  body {{ font-family: sans-serif; max-width: 1800px; margin: 0.5em auto; line-height: 1.5; padding: 0; }}
  code {{ background: #f0f0f0; padding: 0 0.2em; }}
</style>
</head>
<body>
<h1>Reliability / calibration diagrams</h1>
<p>All the CV test folds are pooled. Models: {models}. Nominal interval
{q_low}&ndash;{q_high} (nominal coverage {nominal:.2f}).</p>

<h2>Coverage curve (marginal reliability)</h2>
<p>For each nominal level &tau;, the fraction of observations at or below the
&tau;-quantile forecast. Dashed line = perfect calibration. This is the
marginal check; a model can be on the diagonal on average while being
miscalibrated in particular situations.</p>
{marginal}

<h2>Conditional coverage of the interval (by predicted load)</h2>
<p>Test cases split into {n_bins} equal-count bins by each model's predicted
median; observed coverage of the {q_low}&ndash;{q_high} interval in each bin.
Below the dashed nominal line means the interval is too narrow at that load,
above means too wide.</p>
{interval}

<h2>Conditional reliability curves, binned (one panel per quantile)</h2>
<p>Equal-count bins of each model's own predictions; x = mean prediction,
y = mean prediction + &tau;-quantile of the residuals in the bin. A perfectly
calibrated model lies on the dashed 45 degree diagonal (1:1 axes). Use the
dropdown to change the horizon.</p>
{binned}

<h2>Conditional reliability curves, isotonic / PAV (one panel per quantile)</h2>
<p>Same check with the conditional &tau;-quantile estimated by isotonic
regression (no bin width to choose). A perfectly calibrated model lies on the
dashed 45 degree diagonal (1:1 axes).</p>
{isotonic}
</body>
</html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dir", type=Path)
    parser.add_argument("--n_bins", type=int, default=10)
    args = parser.parse_args()

    models = load_models(args.dir)
    horizon_columns = check_layout(models)
    quantiles = sorted(next(iter(horizon_columns.values())))
    n_bins = args.n_bins

    plots = {
        "marginal": fig_marginal(models, horizon_columns),
        "interval": fig_interval(models, horizon_columns, n_bins),
        "binned": fig_by_level(models, horizon_columns, n_bins, method="bins"),
        "isotonic": fig_by_level(models, horizon_columns, n_bins, method="isotonic"),
    }
    parts = {
        k: f.to_html(full_html=False, include_plotlyjs=i == 0)
        for i, (k, f) in enumerate(plots.items())
    }
    page = PAGE.format(
        models=", ".join(name for name, _, _ in models),
        q_low=min(quantiles),
        q_high=max(quantiles),
        nominal=max(quantiles) - min(quantiles),
        n_bins=n_bins,
        **parts,
    )
    out = args.dir / "calibration.html"
    out.write_text(page, "utf-8")
    print("wrote", out)


if __name__ == "__main__":
    main()
