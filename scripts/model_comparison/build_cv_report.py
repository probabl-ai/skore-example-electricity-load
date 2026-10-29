"""
Build an HTML report comparing hgb_mean, tabicl and chronos2 from the
`*_cv_predictions.parquet` files written by `cross_validate_models.py` (no
re-running any model): a metrics table (MAE, MAPE, pinball-d2, coverage/width,
from `compare_predictions.compute_scores`) with links to one true-vs-predicted
plot per model and horizon.

Plots are written as separate standalone HTML files under `<dir>/plots/` (kept
out of the main page so it stays small and fast to load). For models with
quantiles, the plot shows the true load, the median prediction as a line, and
the outermost quantile interval (e.g. 5%-95%) as a shaded band, rather than a
separate line per quantile.

    python scripts/model_comparison/build_cv_report.py results/model_comparison/my_study

Writes `<dir>/report.html` and `<dir>/plots/<model>_<horizon>h.html`.
"""

import argparse
import datetime
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import polars as pl
from compare_predictions import compare, horizons_of, quantile_levels_of

parser = argparse.ArgumentParser()
parser.add_argument("dir", type=Path)
args = parser.parse_args()

paths = sorted(args.dir.glob("*_cv_predictions.parquet"))
if not paths:
    raise SystemExit(f"no *_cv_predictions.parquet found in {args.dir}")
frames = {p.stem.removesuffix("_cv_predictions"): pl.read_parquet(p) for p in paths}

results, table = compare(paths)
metric_order = table["metric"].to_list()

table_rows = "\n".join(
    "<tr><td>{}</td>{}</tr>".format(
        metric,
        "".join(
            f"<td>{results[name].get(metric):.4g}</td>"
            if results[name].get(metric) is not None
            else "<td>-</td>"
            for name in frames
        ),
    )
    for metric in metric_order
)
table_html = f"""
<table border="1" cellpadding="4" cellspacing="0">
<thead><tr><th>metric</th>{"".join(f"<th>{n}</th>" for n in frames)}</tr></thead>
<tbody>
{table_rows}
</tbody>
</table>
"""


def add_model_traces(fig, df, name, h, *, row=None, col=None, showlegend=True, color=None):
    """Add true load plus the model prediction (median line and shaded interval
    between the outermost quantiles, if any) to ``fig``."""
    target_time = df["prediction_time"] + datetime.timedelta(hours=h)
    hover = "%{x|%Y-%m-%d} (%{x|%A}): %{y}<extra></extra>"

    def add(trace):
        if row is None:
            fig.add_trace(trace)
        else:
            fig.add_trace(trace, row=row, col=col)

    add(
        go.Scatter(
            x=target_time,
            y=df[f"{h}h"],
            mode="lines",
            line={"dash": "dash", "color": "gray"},
            name="true_load_mw",
            hovertemplate=hover,
            showlegend=showlegend,
        )
    )
    levels = quantile_levels_of(df, h)
    if levels:
        low, high = min(levels), max(levels)
        median = min(levels, key=lambda q: abs(q - 0.5))
        if high > low:
            add(
                go.Scatter(
                    x=target_time,
                    y=df[f"pred_{h}h__q_{high}"],
                    mode="lines",
                    line={"width": 0},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            add(
                go.Scatter(
                    x=target_time,
                    y=df[f"pred_{h}h__q_{low}"],
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor="rgba(0, 100, 250, 0.2)",
                    name=f"{round(low * 100)}% - {round(high * 100)}% interval",
                    hoverinfo="skip",
                    showlegend=showlegend,
                )
            )
        add(
            go.Scatter(
                x=target_time,
                y=df[f"pred_{h}h__q_{median}"],
                mode="lines",
                line={} if color is None else {"color": color},
                name="pred_load_mw (median)",
                hovertemplate=hover,
                showlegend=showlegend,
            )
        )
    else:
        add(
            go.Scatter(
                x=target_time,
                y=df[f"pred_{h}h"],
                mode="lines",
                line={} if color is None else {"color": color},
                name="pred_load_mw",
                hovertemplate=hover,
                showlegend=showlegend,
            )
        )


def make_figure(df, name, h):
    """True vs predicted for one model/horizon."""
    fig = go.Figure()
    add_model_traces(fig, df, name, h)
    fig.update_layout(height=600, title=f"{name} -- {h}h horizon: true vs predicted")
    return fig


def make_combined_figure(h):
    """All models for one horizon, stacked with shared (tied) axes."""
    fig = make_subplots(
        rows=len(frames),
        cols=1,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.03,
        subplot_titles=list(frames),
    )
    for row, (name, df) in enumerate(frames.items(), 1):
        add_model_traces(
            fig, df, name, h, row=row, col=1, showlegend=row == 1, color="#1f77b4"
        )
    fig.update_layout(
        height=350 * len(frames),
        title=f"All models -- {h}h horizon: true vs predicted",
    )
    fig.update_xaxes(title_text="target time", row=len(frames), col=1)
    fig.update_yaxes(title_text="load (MW)", row=1, col=1)
    for row in range(2, len(frames) + 1):
        fig.update_yaxes(matches="y", row=row, col=1)
    return fig


plots_dir = args.dir / "plots"
plots_dir.mkdir(exist_ok=True)

sections = []
for name, df in frames.items():
    links = []
    for h in horizons_of(df):
        fig = make_figure(df, name, h)
        out_name = f"{name}_{h}h.html"
        fig.write_html(plots_dir / out_name)
        links.append(f'<li><a href="plots/{out_name}">{h}h horizon</a></li>')
    sections.append(f"<h2>{name}</h2>\n<ul>\n" + "\n".join(links) + "\n</ul>")

combined_links = []
for h in horizons_of(next(iter(frames.values()))):
    out_name = f"all_models_{h}h.html"
    make_combined_figure(h).write_html(plots_dir / out_name)
    combined_links.append(f'<li><a href="plots/{out_name}">{h}h horizon</a></li>')
combined_section = (
    "<h2>All models per horizon</h2>\n<ul>\n"
    + "\n".join(combined_links)
    + "\n</ul>"
)

calibration_link = (
    '<p><a href="calibration.html">Reliability / calibration diagrams</a></p>'
    if (args.dir / "calibration.html").exists()
    else ""
)

page = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Walk-forward CV model comparison</title>
<style>
body {{ font-family: sans-serif; margin: 2em; }}
table {{ border-collapse: collapse; margin-bottom: 2em; }}
th, td {{ padding: 4px 10px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
tbody tr:hover {{ background-color: #eef3ff; }}
</style>
</head>
<body>
<h1>Walk-forward CV comparison</h1>
<p>MAE / MAPE use the median (point) prediction; pinball-d2, coverage and
width use all available quantiles ("-" = model has no quantiles).
All rows across the 5 CV folds are pooled.</p>
{table_html}
{calibration_link}
<h1>Plots</h1>
<p>True vs predicted load per model and horizon (separate pages, linked below
to keep this page small). For quantile models the plot shows the median as a
line and the 5%-95% interval as a shaded band. The combined pages stack all
models for a horizon with tied axes: zooming one subplot zooms the others.</p>
{combined_section}
{"".join(sections)}
</body>
</html>
"""

out = args.dir / "report.html"
out.write_text(page, "utf-8")
print("wrote", out)
