"""
Build an HTML report comparing hgb_mean, tabicl and chronos2 from the
`*_cv_predictions.parquet` files written by `cross_validate_models.py` (no
re-running any model): a metrics table (MAE, MAPE, pinball-d2, coverage/width,
from `compare_predictions.compute_scores`) and links to true-vs-predicted
plots.

Plots are written as separate standalone HTML files under `<dir>/plots/` (kept
out of the main page so it stays small and fast to load). There is one page per
horizon (`all_models_<horizon>h.html`) that stacks every model with tied axes,
instead of one page per model and horizon: the combined pages carry the same
series and avoid duplicating the embedded data. For models with quantiles, the
plot shows the true load, the median prediction as a line, and the outermost
quantile interval (e.g. 5%-95%) as a shaded band, rather than a separate line
per quantile. A separate `pareto.html` page compares average MAPE and average
pinball loss against total fit + predict time.

    python scripts/model_comparison/build_cv_report.py results/model_comparison/my_study

Writes `<dir>/report.html`, `<dir>/pareto.html` and
`<dir>/plots/all_models_<horizon>h.html`.
"""

import argparse
import datetime
import re
from html import escape
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
from compare_predictions import (
    compare,
    compute_scores,
    horizons_of,
    quantile_levels_of,
)
from plotly.subplots import make_subplots
from sklearn.metrics import mean_pinball_loss
from uncertainty import (
    INTERVAL_NOTE,
    average_row_loss,
    error_bar,
    paired_difference,
    row_index,
    row_loss,
)

PRED_COLOR = "#1f77b4"

parser = argparse.ArgumentParser()
parser.add_argument("dir", type=Path)
args = parser.parse_args()

paths = sorted(args.dir.glob("*_cv_predictions.parquet"))
if not paths:
    raise SystemExit(f"no *_cv_predictions.parquet found in {args.dir}")
frames = {
    p.stem.removesuffix("_cv_predictions"): pl.read_parquet(p).sort("prediction_time")
    for p in paths
}
_first = next(iter(frames.values()))
for _name, _df in frames.items():
    if not _df.select("prediction_time", "split").equals(
        _first.select("prediction_time", "split")
    ):
        raise SystemExit(f"{_name} is not on the same rows (prediction_time, split)")
DAY, FOLD = row_index(_first)

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


fold_scores = {}
for name, df in frames.items():
    splits = sorted(df["split"].unique().to_list())
    per_fold = [compute_scores(df.filter(pl.col("split") == split)) for split in splits]
    fold_scores[name] = {
        key: np.array([fold[key] for fold in per_fold], dtype=float)
        for key in per_fold[0]
    }

fold_pinball = {}
for name, df in frames.items():
    splits = sorted(df["split"].unique().to_list())
    values = [
        average_pinball_loss(df.filter(pl.col("split") == split))
        for split in splits
    ]
    fold_pinball[name] = (
        None if any(value is None for value in values) else np.array(values)
    )

BAR_METRICS = [
    ("mae__{h}h", "MAE (MW)", 1.0),
    ("mape__{h}h", "MAPE (%)", 100.0),
    ("d2_pinball_score__{h}h__q_0.5", "median pinball-d2", 1.0),
    ("d2_pinball_score__{h}h__q_0.05", "q0.05 pinball-d2", 1.0),
    ("d2_pinball_score__{h}h__q_0.95", "q0.95 pinball-d2", 1.0),
    ("coverage_90__{h}h", "90% coverage", 1.0),
    ("width_90__{h}h", "90% width (MW)", 1.0),
]
# (kind, quantile level, label) of the row-level losses shown in the paired
# difference plot; all are lower-is-better means over rows
DIFF_METRICS = [
    ("mae", None, "MAE (MW)"),
    ("mape", None, "MAPE (%)"),
    ("pinball", 0.5, "median pinball loss (MW)"),
    ("pinball", 0.05, "q0.05 pinball loss (MW)"),
    ("pinball", 0.95, "q0.95 pinball loss (MW)"),
    ("width", None, "90% width (MW)"),
]
HORIZON_COLORS = {1: "#1f77b4", 12: "#ff7f0e", 24: "#2ca02c"}


def baseline_name(template):
    if template.startswith(("mae__", "mape__")):
        candidates = [
            name
            for name in frames
            if name.startswith("hgb_mean") and "fast" not in name
        ]
    else:
        candidates = [
            name
            for name in frames
            if name.startswith("hgb_quantile_q7") and "recal" not in name
        ]
    if candidates:
        return sorted(candidates)[0]
    prefix = "hgb_mean" if template.startswith(("mae__", "mape__")) else "hgb_quantile"
    fallback = sorted(name for name in frames if name.startswith(prefix))
    return fallback[0] if fallback else None


def bars_figure(fold_scores):
    horizons = horizons_of(next(iter(frames.values())))
    n_rows = (len(BAR_METRICS) + 1) // 2
    fig = make_subplots(
        rows=n_rows,
        cols=2,
        subplot_titles=[label for _, label, _ in BAR_METRICS],
        horizontal_spacing=0.14,
        vertical_spacing=0.08,
    )
    for i, (template, label, scale) in enumerate(BAR_METRICS):
        row, col = divmod(i, 2)
        row, col = row + 1, col + 1
        data = {}
        for name in frames:
            keys = [template.format(h=h) for h in horizons]
            if not all(key in fold_scores[name] for key in keys):
                continue
            data[name] = {
                h: scale * float(np.mean(fold_scores[name][template.format(h=h)]))
                for h in horizons
            }
        models = sorted(
            data,
            key=lambda name: float(np.mean(list(data[name].values()))),
        )
        for h in horizons:
            fig.add_trace(
                go.Bar(
                    x=[data[name][h] for name in models],
                    y=models,
                    orientation="h",
                    marker_color=HORIZON_COLORS.get(h, PRED_COLOR),
                    name=f"{h}h",
                    showlegend=i == 0,
                    hovertemplate=(
                        "%{y}<br>"
                        + f"{h}h " + label
                        + ": %{x:.3g}<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )
        if template.startswith("coverage_90__"):
            fig.add_vline(
                x=0.9,
                line={"color": "gray", "dash": "dash"},
                annotation_text="90%",
                annotation_position="top",
                row=row,
                col=col,
            )
        fig.update_yaxes(autorange="reversed", row=row, col=col)
        fig.update_xaxes(title_text=label, row=row, col=col)
    fig.update_layout(
        height=550 * n_rows,
        barmode="group",
        title="Absolute metrics by horizon (mean across folds)",
        margin={"t": 55, "b": 40, "l": 100, "r": 25},
    )
    return fig


def differences_figure():
    horizons = horizons_of(next(iter(frames.values())))
    n_rows = (len(DIFF_METRICS) + 1) // 2
    fig = make_subplots(
        rows=n_rows,
        cols=2,
        subplot_titles=[label for _, _, label in DIFF_METRICS],
        horizontal_spacing=0.14,
        vertical_spacing=0.08,
    )
    for i, (kind, q, label) in enumerate(DIFF_METRICS):
        row, col = divmod(i, 2)
        row, col = row + 1, col + 1
        baseline = baseline_name("mae__{h}h" if kind in ("mae", "mape") else "d2_pinball_score__{h}h__q_0.5")
        if baseline is None:
            continue
        data = {}
        for name in frames:
            if name == baseline:
                continue
            losses = {h: row_loss(frames[name], kind, h, q) for h in horizons}
            base = {h: row_loss(frames[baseline], kind, h, q) for h in horizons}
            if any(losses[h] is None or base[h] is None for h in horizons):
                continue
            data[name] = {
                h: paired_difference(losses[h], base[h], DAY, FOLD) for h in horizons
            }
        models = sorted(
            data,
            key=lambda name: float(np.mean([data[name][h]["mean"] for h in horizons])),
        )
        for j, h in enumerate(horizons):
            color = HORIZON_COLORS.get(h, PRED_COLOR)
            # models are on integer y positions; horizons and folds are offsets
            offset = (j - (len(horizons) - 1) / 2) * 0.27
            fold_x, fold_y, fold_text = [], [], []
            for k, name in enumerate(models):
                folds = data[name][h]["folds"]
                fold_x += list(folds)
                fold_y += [k + offset + (f - len(folds) / 2) * 0.012 for f in range(len(folds))]
                fold_text += [f"{name}, {h}h, fold {f}" for f in range(len(folds))]
            fig.add_trace(
                go.Scatter(
                    x=fold_x,
                    y=fold_y,
                    mode="markers",
                    marker={"size": 4, "color": color, "opacity": 0.55},
                    text=fold_text,
                    hovertemplate="%{text}: %{x:+.3g}<extra></extra>",
                    legendgroup=f"{h}h",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            above, below = zip(*(error_bar(data[name][h]) for name in models))
            fig.add_trace(
                go.Scatter(
                    x=[data[name][h]["mean"] for name in models],
                    y=[k + offset for k in range(len(models))],
                    mode="markers",
                    marker={
                        "size": 9,
                        "color": color,
                        "symbol": "diamond",
                        "line": {"color": "#333333", "width": 1},
                    },
                    error_x={
                        "type": "data",
                        "symmetric": False,
                        "array": above,
                        "arrayminus": below,
                        "color": "#333333",
                        "thickness": 1.5,
                        "width": 3,
                    },
                    customdata=[
                        [
                            name,
                            data[name][h]["low"],
                            data[name][h]["high"],
                            data[name][h]["n_lower"],
                            len(data[name][h]["folds"]),
                        ]
                        for name in models
                    ],
                    hovertemplate=(
                        "%{customdata[0]}<br>"
                        + f"{h}h "
                        + label
                        + " difference: %{x:+.3g}<br>"
                        "95% block bootstrap: [%{customdata[1]:+.3g}, %{customdata[2]:+.3g}]<br>"
                        "lower than baseline in %{customdata[3]}/%{customdata[4]} folds"
                        "<extra></extra>"
                    ),
                    name=f"{h}h",
                    legendgroup=f"{h}h",
                    showlegend=i == 0,
                ),
                row=row,
                col=col,
            )
        fig.add_vline(
            x=0,
            line={"color": "gray", "dash": "dash"},
            row=row,
            col=col,
        )
        fig.update_yaxes(
            tickvals=list(range(len(models))),
            ticktext=models,
            autorange="reversed",
            row=row,
            col=col,
        )
        fig.update_xaxes(title_text=f"{label} difference", row=row, col=col)
    fig.update_layout(
        height=600 * n_rows,
        title="Paired difference vs baseline (mean, 95% block bootstrap interval, per-fold dots)",
        margin={"t": 55, "b": 40, "l": 100, "r": 25},
    )
    return fig


def coverage_width_figure(fold_scores):
    horizons = horizons_of(next(iter(frames.values())))
    data = {}
    all_coverage = []
    for h in horizons:
        points = []
        for name in frames:
            coverage = fold_scores[name].get(f"coverage_90__{h}h")
            width = fold_scores[name].get(f"width_90__{h}h")
            if coverage is None or width is None:
                continue
            x = float(np.mean(coverage))
            y = float(np.mean(width))
            points.append((name, x, y))
            all_coverage.append(x)
        data[h] = points

    fig = make_subplots(
        rows=len(horizons),
        cols=1,
        subplot_titles=[f"{h}h horizon" for h in horizons],
        vertical_spacing=0.09,
    )
    for row, h in enumerate(horizons, start=1):
        points = data[h]
        fig.add_trace(
            go.Scatter(
                x=[point[1] for point in points],
                y=[point[2] for point in points],
                mode="markers+text",
                text=[point[0] for point in points],
                textposition="top center",
                textfont={"size": 11},
                cliponaxis=False,
                marker={"size": 10, "color": HORIZON_COLORS.get(h, PRED_COLOR)},
                customdata=[[point[0], h] for point in points],
                hovertemplate=(
                    "%{customdata[0]} (%{customdata[1]}h)"
                    "<br>coverage: %{x:.3f}"
                    "<br>width: %{y:,.0f} MW<extra></extra>"
                ),
                showlegend=False,
            ),
            row=row,
            col=1,
        )
        fig.add_vline(
            x=0.9,
            line={"color": "gray", "dash": "dash"},
            annotation_text="90%",
            annotation_position="top",
            row=row,
            col=1,
        )
    fig.update_xaxes(
        title_text="90% interval coverage",
        range=[min(all_coverage) - 0.02 if all_coverage else 0.7, 1.0],
    )
    fig.update_yaxes(title_text="90% interval width (MW)")
    fig.update_layout(
        height=1200,
        title="Coverage vs interval width (fold mean per horizon)",
        font={"size": 13},
        margin={"t": 60, "b": 65, "l": 80, "r": 40},
    )
    return fig

pareto_records = []
for path in paths:
    name = path.stem.removesuffix("_cv_predictions")
    mape_values = 100 * fold_scores[name]["mape__average"]
    pinball_values = fold_pinball[name]
    records = {
        "name": name,
        "time": fit_predict_time_seconds(path),
        "mape": float(np.mean(mape_values)),
        "mape_std": float(np.std(mape_values, ddof=1)),
        "pinball": (
            None if pinball_values is None else float(np.mean(pinball_values))
        ),
        "pinball_std": (
            None if pinball_values is None else float(np.std(pinball_values, ddof=1))
        ),
        "mape_values": mape_values,
        "pinball_values": pinball_values,
        "mape_rows": average_row_loss(frames[name], "mape"),
        "pinball_rows": average_row_loss(frames[name], "pinball"),
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
            textfont={"size": 12},
            cliponaxis=False,
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
        margin={"t": 85, "b": 65, "l": 80, "r": 120},
    )
    return fig


def pareto_difference_figure(records, metric, baseline, title, metric_label):
    baseline_record = next((r for r in records if r["name"] == baseline), None)
    if baseline_record is None or baseline_record[f"{metric}_rows"] is None:
        return None
    data = []
    for r in records:
        if r["time"] is None or r[f"{metric}_rows"] is None:
            continue
        data.append(
            {
                "name": r["name"],
                "time": r["time"],
                "baseline": r["name"] == baseline,
                **paired_difference(
                    r[f"{metric}_rows"], baseline_record[f"{metric}_rows"], DAY, FOLD
                ),
            }
        )
    if not data:
        return None
    fig = go.Figure()
    # one dot per fold, jittered horizontally (the time axis is logarithmic)
    fig.add_trace(
        go.Scatter(
            x=[r["time"] * (1 + 0.015 * (f - 3.5)) for r in data for f in range(len(r["folds"]))],
            y=[v for r in data for v in r["folds"]],
            mode="markers",
            marker={"size": 4, "color": "#1f77b4", "opacity": 0.45},
            text=[f"{r['name']}, fold {f}" for r in data for f in range(len(r["folds"]))],
            hovertemplate="%{text}: %{y:+.3g}<extra></extra>",
            name="per-fold differences",
        )
    )
    above, below = zip(*(error_bar(r) for r in data))
    fig.add_trace(
        go.Scatter(
            x=[r["time"] for r in data],
            y=[r["mean"] for r in data],
            mode="markers+text",
            text=[
                f"{r['name']} (baseline)" if r["baseline"] else r["name"]
                for r in data
            ],
            textposition="top center",
            textfont={"size": 12},
            cliponaxis=False,
            marker={
                "size": [11 if r["baseline"] else 10 for r in data],
                "color": ["#777777" if r["baseline"] else "#1f77b4" for r in data],
                "symbol": ["diamond" if r["baseline"] else "circle" for r in data],
                "line": {"color": "#333333", "width": 1},
            },
            customdata=[
                [r["name"], r["low"], r["high"], r["n_lower"], len(r["folds"])]
                for r in data
            ],
            error_y={
                "type": "data",
                "symmetric": False,
                "array": above,
                "arrayminus": below,
                "color": "#333333",
                "thickness": 1.5,
                "width": 4,
            },
            hovertemplate=(
                "%{customdata[0]}<br>fit + predict: %{x:,.0f} s<br>"
                + metric_label
                + " difference: %{y:+.3g}<br>"
                "95% block bootstrap: [%{customdata[1]:+.3g}, %{customdata[2]:+.3g}]<br>"
                "lower than baseline in %{customdata[3]}/%{customdata[4]} folds"
                "<extra></extra>"
            ),
            name="models (pooled mean)",
        )
    )
    frontier = pareto_frontier(data, "mean")
    if len(frontier) > 1:
        fig.add_trace(
            go.Scatter(
                x=[r["time"] for r in frontier],
                y=[r["mean"] for r in frontier],
                mode="lines+markers",
                line={"color": "#d62728", "dash": "dash"},
                marker={"size": 7, "color": "#d62728"},
                name="Pareto frontier",
                hoverinfo="skip",
            )
        )
    fig.update_xaxes(type="log", title_text="Fit + predict time (s, log scale)")
    fig.update_yaxes(
        title_text=f"Δ {metric_label} vs {baseline} (lower is better)"
    )
    fig.update_layout(
        title=title,
        height=480,
        margin={"t": 85, "b": 65, "l": 80, "r": 120},
    )
    return fig


def write_pareto_page(path, records):
    mape_baseline = baseline_name("mae__{h}h")
    pinball_baseline = baseline_name("d2_pinball_score__{h}h__q_0.5")
    difference_figures = [
        pareto_difference_figure(
            records,
            "mape",
            mape_baseline,
            "MAPE difference vs fit + predict time",
            "MAPE (%)",
        ),
        pareto_difference_figure(
            records,
            "pinball",
            pinball_baseline,
            "Pinball loss difference vs fit + predict time",
            "pinball loss (MW)",
        ),
    ]
    difference_figures = [fig for fig in difference_figures if fig is not None]
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
    difference_parts = "".join(
        f'<div class="chart">{fig.to_html(full_html=False, include_plotlyjs=i == 0, config={"responsive": True, "displaylogo": False})}</div>'
        for i, fig in enumerate(difference_figures)
    )
    absolute_parts = "".join(
        f'<div class="chart">{fig.to_html(full_html=False, include_plotlyjs=not difference_figures and i == 0, config={"responsive": True, "displaylogo": False})}</div>'
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
log-scaled. MAPE uses the median prediction. Pinball loss is averaged over the
5%, 50% and 95% quantiles and all horizons; models without quantiles are omitted
from the pinball figure. Lower is better on both axes. Table values are means
across the 8 folds.</p>
<table>
<thead><tr><th>model</th><th>fit + predict (s)</th><th>MAPE (%)</th><th>pinball loss (MW)</th></tr></thead>
<tbody>{table_rows}</tbody>
</table>
<h2>Paired differences vs baseline</h2>
<p class="note">MAPE is relative to <code>{escape(str(mape_baseline))}</code>;
pinball loss is relative to <code>{escape(str(pinball_baseline))}</code>.
The baseline is shown at zero difference. {INTERVAL_NOTE} Dashed colored lines
mark the Pareto frontiers.</p>
{difference_parts}
<h2>Absolute accuracy vs fit + predict time</h2>
<p class="note">Means across the 8 CV folds, shown without error bars. Dashed
colored lines mark the Pareto frontiers.</p>
{absolute_parts}
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

differences_html = differences_figure().to_html(
    full_html=False,
    include_plotlyjs="cdn",
    config={"responsive": True, "displaylogo": False},
)
bars_html = bars_figure(fold_scores).to_html(
    full_html=False,
    include_plotlyjs=False,
    config={"responsive": True, "displaylogo": False},
)
coverage_width_html = coverage_width_figure(fold_scores).to_html(
    full_html=False,
    include_plotlyjs=False,
    config={"responsive": True, "displaylogo": False},
)
point_baseline = baseline_name("mae__{h}h")
distribution_baseline = baseline_name("d2_pinball_score__{h}h__q_0.5")
baseline_note = (
    f"MAE/MAPE are relative to <code>{escape(str(point_baseline))}</code>; "
    "pinball/width are relative to "
    f"<code>{escape(str(distribution_baseline))}</code>."
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
<h1>Plots</h1>
<p>True vs predicted load per horizon (separate pages, linked below to keep this
page small). For quantile models the plot shows the median as a line and the
5%-95% interval as a shaded band. Each page stacks all models with tied axes:
zooming one chart zooms the others, and models can be hidden.</p>
{combined_section}
{calibration_link}
{pareto_link}
<h1>Paired differences vs baseline</h1>
<p class="note">Each row is a model, with one lane per horizon. {baseline_note}
{INTERVAL_NOTE} The dashed zero line means no difference; lower is better for
every panel (losses and width). A model whose interval excludes zero and whose
dots are all on one side is consistently different from the baseline; dots on
both sides of zero mean the sign depends on the period.</p>
{differences_html}
<h1>Coverage vs interval width</h1>
<p class="note">Each point is one model/horizon, using the mean coverage and
mean interval width across the 8 folds. The vertical dashed line is the nominal
90% coverage. Points close to the line are well calibrated; lower is sharper.</p>
{coverage_width_html}
<h1>Absolute metrics by horizon</h1>
<p class="note">Horizontal bars show the mean across the 8 CV folds, grouped by
horizon (1h, 12h, 24h), without error bars. The table above pools all rows
across folds, so its values can differ slightly, especially for the nonlinear
pinball-d2 metrics.</p>
{bars_html}
{table_js}
</body>
</html>
"""

out = args.dir / "report.html"
out.write_text(page, "utf-8")
print("wrote", out)
print("wrote", pareto_path)
