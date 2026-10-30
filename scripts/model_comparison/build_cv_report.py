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
separate line per quantile. A separate `pareto.html` page compares average MAPE
and average pinball loss against total fit + predict time.

    python scripts/model_comparison/build_cv_report.py results/model_comparison/my_study

Writes `<dir>/report.html`, `<dir>/pareto.html` and
`<dir>/plots/<model>_<horizon>h.html`.
"""

import argparse
import datetime
import re
from html import escape
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
from compare_predictions import compare, horizons_of, quantile_levels_of
from sklearn.metrics import mean_pinball_loss

PRED_COLOR = "#1f77b4"

parser = argparse.ArgumentParser()
parser.add_argument("dir", type=Path)
args = parser.parse_args()

paths = sorted(args.dir.glob("*_cv_predictions.parquet"))
if not paths:
    raise SystemExit(f"no *_cv_predictions.parquet found in {args.dir}")
frames = {p.stem.removesuffix("_cv_predictions"): pl.read_parquet(p) for p in paths}

results, table = compare(paths)


def fit_predict_time_seconds(path):
    """Total fit + score time over all CV folds for one prediction file.

    Uses the per-fold timings in the sibling scores CSV when available, and
    falls back to the total printed in the run log for older runs.
    """
    source = path.resolve()
    stem = path.stem.removesuffix("_cv_predictions")
    candidates = []
    for directory in (path.parent, source.parent):
        candidates.append(directory / f"{stem}_scores.csv")
        candidates.extend(sorted(directory.glob("*_scores.csv")))
    for score_path in dict.fromkeys(candidates):
        if not score_path.exists():
            continue
        scores = pl.read_csv(score_path)
        if {"fit_time", "score_time"} <= set(scores.columns):
            return float(scores["fit_time"].sum() + scores["score_time"].sum())
    for directory in (source.parent, path.parent):
        log_path = directory / "run.log"
        if log_path.exists():
            matches = re.findall(
                r":\s*([\d.]+)s total", log_path.read_text(errors="replace")
            )
            if matches:
                return float(matches[-1])
    return None


def average_pinball_loss(df):
    """Mean pinball loss (MW) over 5/50/95% quantiles and all horizons."""
    hs = horizons_of(df)
    levels = quantile_levels_of(df, hs[0])
    chosen = [
        q for q in levels if any(abs(q - target) < 1e-9 for target in (0.05, 0.5, 0.95))
    ]
    if not chosen:
        chosen = levels
    if not chosen:
        return None
    losses = []
    for h in hs:
        y = df[f"{h}h"].to_numpy()
        for q in chosen:
            losses.append(
                mean_pinball_loss(y, df[f"pred_{h}h__q_{q}"].to_numpy(), alpha=q)
            )
    return float(np.mean(losses))


pareto_records = []
for path in paths:
    name = path.stem.removesuffix("_cv_predictions")
    records = {
        "name": name,
        "time": fit_predict_time_seconds(path),
        "mape": 100 * results[name]["mape__average"],
        "pinball": average_pinball_loss(frames[name]),
    }
    pareto_records.append(records)

missing_times = [r["name"] for r in pareto_records if r["time"] is None]
if missing_times:
    print("warning: no fit+predict time for", ", ".join(missing_times))


def pareto_frontier(records, metric):
    """Points not dominated on (lower time, lower metric)."""
    frontier, best = [], float("inf")
    for record in sorted(records, key=lambda r: (r["time"], r[metric])):
        if record[metric] < best:
            frontier.append(record)
            best = record[metric]
    return frontier


def pareto_figure(records, metric, title, y_title, hover_format):
    data = [r for r in records if r["time"] is not None and r[metric] is not None]
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[r["time"] for r in data],
            y=[r[metric] for r in data],
            mode="markers+text",
            text=[r["name"] for r in data],
            textposition="top center",
            marker={"size": 10, "color": "#1f77b4"},
            customdata=[[r["name"]] for r in data],
            hovertemplate=(
                "%{customdata[0]}<br>fit + predict: %{x:,.0f} s<br>"
                + hover_format
                + "<extra></extra>"
            ),
            name="models",
        )
    )
    frontier = pareto_frontier(data, metric)
    if len(frontier) > 1:
        fig.add_trace(
            go.Scatter(
                x=[r["time"] for r in frontier],
                y=[r[metric] for r in frontier],
                mode="lines+markers",
                line={"color": "#d62728", "dash": "dash"},
                marker={"size": 7, "color": "#d62728"},
                name="Pareto frontier",
                hoverinfo="skip",
            )
        )
    fig.update_xaxes(
        type="log",
        title_text="Fit + predict time (s, log scale)",
    )
    fig.update_yaxes(title_text=y_title)
    fig.update_layout(
        title=title,
        height=480,
        margin={"t": 55, "b": 55, "l": 70, "r": 25},
    )
    return fig


def write_pareto_page(path, records):
    figures = [
        pareto_figure(
            records,
            "mape",
            "Average MAPE vs fit + predict time",
            "Average MAPE (%, lower is better)",
            "MAPE: %{y:.3f}%",
        ),
        pareto_figure(
            records,
            "pinball",
            "Average pinball loss vs fit + predict time",
            "Average pinball loss (MW, lower is better)",
            "pinball loss: %{y:.2f} MW",
        ),
    ]
    parts = "".join(
        f'<div class="chart">{fig.to_html(full_html=False, include_plotlyjs=i == 0, config={"responsive": True, "displaylogo": False})}</div>'
        for i, fig in enumerate(figures)
    )
    table_rows = "".join(
        "<tr>"
        f"<td>{escape(r['name'])}</td>"
        f"<td>{r['time']:,.0f}</td>"
        f"<td>{r['mape']:.3f}</td>"
        f"<td>{r['pinball']:.2f}</td>"
        "</tr>"
        if r["pinball"] is not None
        else (
            "<tr>"
            f"<td>{escape(r['name'])}</td>"
            f"<td>{r['time']:,.0f}</td>"
            f"<td>{r['mape']:.3f}</td>"
            "<td>-</td>"
            "</tr>"
        )
        for r in sorted(
            (r for r in records if r["time"] is not None),
            key=lambda r: r["time"],
        )
    )
    page = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Pareto plots: accuracy vs fit + predict time</title>
<style>
body {{ font-family: sans-serif; max-width: 1100px; margin: 2em auto; padding: 0 1em; }}
table {{ border-collapse: collapse; margin-bottom: 1.5em; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 0.4em 0.8em; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
.chart {{ width: 100%; margin-bottom: 1.2em; }}
.note {{ color: #555; font-size: 0.9em; }}
</style>
</head>
<body>
<h1>Pareto plots: accuracy vs fit + predict time</h1>
<p class="note">Times are summed over the 8 CV folds (fit + predict/score).
They come from per-fold timings in the scores CSV when available, otherwise
from the run log. Hardware and parallel load differ across runs (CPU vs GPU), so
times are indicative rather than a controlled benchmark. The horizontal axis is
log-scaled. The average neg-MAPE score is shown as positive MAPE (%), so lower
is better; MAPE uses the median prediction. Pinball loss is averaged over the
5%, 50% and 95% quantiles and all horizons; models without quantiles are omitted
from the pinball figure. Lower is better on both axes, and the dashed line
marks the Pareto frontier.</p>
<table>
<thead><tr><th>model</th><th>fit + predict (s)</th><th>MAPE (%)</th><th>pinball loss (MW)</th></tr></thead>
<tbody>{table_rows}</tbody>
</table>
{parts}
</body>
</html>
"""
    path.write_text(page, "utf-8")


def keep_metric(metric):
    """Only show the 5/50/95% pinball scores and the 90% interval coverage."""
    if "__q_" in metric:
        return float(metric.rsplit("__q_", 1)[1]) in {0.05, 0.5, 0.95}
    if metric.startswith(("coverage_", "width_")):
        return metric.startswith(("coverage_90__", "width_90__"))
    return True


metric_order = [m for m in table["metric"].to_list() if keep_metric(m)]

table_rows = "\n".join(
    "<tr><td>{}</td>{}</tr>".format(
        metric,
        "".join(
            (
                f'<td data-col="{i}">{results[name].get(metric):.4g}</td>'
                if results[name].get(metric) is not None
                else f'<td data-col="{i}">-</td>'
            )
            for i, name in enumerate(frames)
        ),
    )
    for metric in metric_order
)
table_controls = (
    '<div class="controls" id="table-controls">'
    + "".join(
        f'<label><input type="checkbox" data-col="{i}" checked> {escape(name)}</label>'
        for i, name in enumerate(frames)
    )
    + "</div>"
)
table_html = f"""
<table border="1" cellpadding="4" cellspacing="0">
<thead><tr><th>metric</th>{"".join(f'<th data-col="{i}">{escape(n)}</th>' for i, n in enumerate(frames))}</tr></thead>
<tbody>
{table_rows}
</tbody>
</table>
"""


def add_model_traces(
    fig, df, name, h, *, row=None, col=None, showlegend=True, color=None
):
    """Add true load plus the model prediction (median line and shaded interval
    between the outermost quantiles, if any) to ``fig``."""
    target_time = (df["prediction_time"] + datetime.timedelta(hours=h)).to_list()
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
                line={"color": color or PRED_COLOR},
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
                line={"color": color or PRED_COLOR},
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


def write_combined_page(path, h):
    """One page per horizon with each model in its own stacked chart.

    The charts share the same initial x/y ranges and their zoom/pan is kept in
    sync with a small relayout listener. Checkboxes hide individual models.
    """
    names = list(frames)
    xs, ys = [], []
    for df in frames.values():
        target_time = df["prediction_time"] + datetime.timedelta(hours=h)
        xs.extend([target_time.min(), target_time.max()])
        columns = [f"{h}h"] + [c for c in df.columns if c.startswith(f"pred_{h}h")]
        for c in columns:
            ys.extend([df[c].min(), df[c].max()])
    x0, x1 = min(xs).isoformat(), max(xs).isoformat()
    y0, y1 = float(min(ys)), float(max(ys))
    pad = 0.02 * (y1 - y0)
    y0, y1 = y0 - pad, y1 + pad

    parts = []
    for i, (name, df) in enumerate(frames.items()):
        fig = make_figure(df, name, h)
        fig.update_layout(
            height=330,
            margin=dict(t=45, b=30, l=60, r=20),
            showlegend=False,
            title_font_size=14,
        )
        fig.update_xaxes(range=[x0, x1])
        fig.update_yaxes(range=[y0, y1])
        fig_html = fig.to_html(
            full_html=False,
            include_plotlyjs=i == 0,
            div_id=f"plot-{i}",
            config={"responsive": True, "displaylogo": False},
        )
        parts.append(
            f'<div class="model" data-model="{i}">'
            f'<div class="model-label">{escape(name)}</div>{fig_html}</div>'
        )

    boxes = "".join(
        f'<label><input type="checkbox" data-model="{i}" checked> {escape(name)}</label>'
        for i, name in enumerate(names)
    )
    js = """
<script>
const plots = Array.from({length: %d}, (_, i) => document.getElementById('plot-' + i));
let syncing = false;
function syncFrom(src, ev) {
  if (syncing) return;
  const upd = {};
  let has = false;
  for (const ax of ['xaxis', 'yaxis']) {
    const r0 = ev[ax + '.range[0]'];
    const r1 = ev[ax + '.range[1]'];
    if (r0 !== undefined && r1 !== undefined) {
      upd[ax + '.range'] = [r0, r1];
      has = true;
    } else if (ev[ax + '.autorange']) {
      upd[ax + '.autorange'] = true;
      has = true;
    }
  }
  if (!has) return;
  syncing = true;
  Promise.all(
    plots.filter(p => p && p !== src).map(p => Plotly.relayout(p, upd))
  ).finally(() => { syncing = false; });
}
plots.forEach(p => { if (p) p.on('plotly_relayout', ev => syncFrom(p, ev)); });
document.querySelectorAll('.controls input').forEach(cb => {
  cb.addEventListener('change', () => {
    const el = document.querySelector('.model[data-model="' + cb.dataset.model + '"]');
    el.style.display = cb.checked ? '' : 'none';
  });
});
</script>
""" % len(names)
    page = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>All models -- {h}h horizon</title>
<style>
body {{ font-family: sans-serif; margin: 1em 2em; }}
.controls {{ margin-bottom: 0.8em; }}
.controls label {{ margin-right: 1em; white-space: nowrap; }}
.model-label {{ font-weight: bold; margin-top: 0.6em; }}
.legend {{ margin-bottom: 0.8em; font-size: 0.9em; }}
.legend .swatch {{ display: inline-block; width: 1.5em; height: 0.9em; margin: 0 0.3em 0 1.2em; vertical-align: middle; }}
.legend .swatch.true {{ border-top: 2px dashed #888; height: 0; }}
.legend .swatch.pred {{ border-top: 2px solid #1f77b4; height: 0; }}
.legend .swatch.band {{ background: rgba(0, 100, 250, 0.2); }}
</style>
</head>
<body>
<h1>All models -- {h}h horizon: true vs predicted</h1>
<p>Zoom or pan any chart to apply the same range to all charts. Uncheck a
model to hide it.</p>
<div class="controls">{boxes}</div>
<div class="legend">
<span class="swatch true"></span>true load
<span class="swatch pred"></span>predicted median
<span class="swatch band"></span>5&ndash;95% interval
</div>
{"".join(parts)}
{js}
</body>
</html>
"""
    path.write_text(page, "utf-8")


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
    write_combined_page(plots_dir / out_name, h)
    combined_links.append(f'<li><a href="plots/{out_name}">{h}h horizon</a></li>')
combined_section = (
    "<h2>All models per horizon</h2>\n<ul>\n" + "\n".join(combined_links) + "\n</ul>"
)

calibration_link = (
    '<p><a href="calibration.html">Reliability / calibration diagrams</a></p>'
    if (args.dir / "calibration.html").exists()
    else ""
)

pareto_path = args.dir / "pareto.html"
write_pareto_page(pareto_path, pareto_records)
pareto_link = (
    '<p><a href="pareto.html">Pareto plots: average MAPE and pinball loss vs '
    "fit + predict time</a></p>"
)

table_js = """<script>
document.querySelectorAll('#table-controls input').forEach(cb => {
  cb.addEventListener('change', () => {
    const i = cb.dataset.col;
    document.querySelectorAll('th[data-col="' + i + '"], td[data-col="' + i + '"]').forEach(el => {
      el.style.display = cb.checked ? '' : 'none';
    });
  });
});
</script>"""

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
.controls {{ margin-bottom: 0.8em; }}
.controls label {{ margin-right: 1em; white-space: nowrap; }}
</style>
</head>
<body>
<h1>Walk-forward CV comparison</h1>
<p>MAE / MAPE use the median (point) prediction; pinball-d2, coverage and
width use all available quantiles ("-" = model has no quantiles).
All rows across the 8 CV folds are pooled.</p>
{table_controls}
{table_html}
{calibration_link}
{pareto_link}
<h1>Plots</h1>
<p>True vs predicted load per model and horizon (separate pages, linked below
to keep this page small). For quantile models the plot shows the median as a
line and the 5%-95% interval as a shaded band. The combined pages stack all
models for a horizon with tied axes: zooming one chart zooms the others.</p>
{combined_section}
{"".join(sections)}
{table_js}
</body>
</html>
"""

out = args.dir / "report.html"
out.write_text(page, "utf-8")
print("wrote", out)
print("wrote", pareto_path)
