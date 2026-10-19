"""
Build an HTML report comparing hgb_mean, tabicl and chronos2 from the
`*_cv_predictions.parquet` files written by `cross_validate_compare.py` (no
re-running any model): a metrics table (MAE, MAPE, pinball-d2, coverage/width,
from `compare_predictions.compute_scores`) with links to one true-vs-predicted
plot per model and horizon.

Plots are written as separate standalone HTML files under `<dir>/plots/` (kept
out of the main page so it stays small and fast to load). For models with
quantiles, the plot shows the true load, the median prediction as a line, and
the outermost quantile interval (e.g. 5%-95%) as a shaded band, rather than a
separate line per quantile.

    python chronos_comparison_scripts/build_cv_report.py results/cv_compare_remote

Writes `<dir>/report.html` and `<dir>/plots/<model>_<horizon>h.html`.
"""

import argparse
import datetime
from pathlib import Path

import plotly.graph_objects as go
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


def make_figure(df, name, h):
    """True vs predicted for one model/horizon: median line, shaded CI band
    between the outermost quantiles (if any), instead of one line per
    quantile."""
    target_time = df["prediction_time"] + datetime.timedelta(hours=h)
    hover = "%{x|%Y-%m-%d} (%{x|%A}): %{y}<extra></extra>"
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=target_time,
            y=df[f"{h}h"],
            mode="lines",
            line={"dash": "dash", "color": "gray"},
            name="true_load_mw",
            hovertemplate=hover,
        )
    )
    levels = quantile_levels_of(df, h)
    if levels:
        low, high = min(levels), max(levels)
        median = min(levels, key=lambda q: abs(q - 0.5))
        if high > low:
            fig.add_trace(
                go.Scatter(
                    x=target_time,
                    y=df[f"pred_{h}h__q_{high}"],
                    mode="lines",
                    line={"width": 0},
                    showlegend=False,
                    hoverinfo="skip",
                )
            )
            fig.add_trace(
                go.Scatter(
                    x=target_time,
                    y=df[f"pred_{h}h__q_{low}"],
                    mode="lines",
                    line={"width": 0},
                    fill="tonexty",
                    fillcolor="rgba(0, 100, 250, 0.2)",
                    name=f"{round(low * 100)}% - {round(high * 100)}% interval",
                    hoverinfo="skip",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=target_time,
                y=df[f"pred_{h}h__q_{median}"],
                mode="lines",
                name="pred_load_mw (median)",
                hovertemplate=hover,
            )
        )
    else:
        fig.add_trace(
            go.Scatter(
                x=target_time,
                y=df[f"pred_{h}h"],
                mode="lines",
                name="pred_load_mw",
                hovertemplate=hover,
            )
        )
    fig.update_layout(height=600, title=f"{name} -- {h}h horizon: true vs predicted")
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

page = f"""
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>CV comparison: hgb_mean vs tabicl vs chronos2</title>
<style>
body {{ font-family: sans-serif; margin: 2em; }}
table {{ border-collapse: collapse; margin-bottom: 2em; }}
th, td {{ padding: 4px 10px; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
</style>
</head>
<body>
<h1>Walk-forward CV comparison</h1>
<p>MAE / MAPE use the median (point) prediction; pinball-d2, coverage and
width use all available quantiles ("-" = model has no quantiles).
All rows across the 5 CV folds are pooled.</p>
{table_html}
<h1>Plots</h1>
<p>True vs predicted load per model and horizon (separate pages, linked below
to keep this page small). For quantile models the plot shows the median as a
line and the 5%-95% interval as a shaded band.</p>
{"".join(sections)}
</body>
</html>
"""

out = args.dir / "report.html"
out.write_text(page, "utf-8")
print("wrote", out)
