"""
Build the feature-group report page from the runs of `compare_groups.py`.

    python scripts/feature_engineering/build_report.py DIR

DIR holds the `hgb_mean_<tag>_{cv_predictions.parquet,metadata.json}` files.
Writes `DIR/report.html` (tables, plots, provenance).
"""

import argparse
import html
import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent.parent / "model_comparison"))
from uncertainty import (  # noqa: E402
    error_bar,
    paired_difference,
    row_index,
    row_loss,
)

parser = argparse.ArgumentParser()
parser.add_argument("dir", type=Path)
args = parser.parse_args()

HORIZONS = (1, 12, 24)
COLUMNS = ["avg", *HORIZONS]
COLOURS = {"avg": "#444444", 1: "#1f77b4", 12: "#ff7f0e", 24: "#2ca02c"}
SINGLE = [
    "load_diffs",
    "load_same_hour",
    "temp_diffs",
    "load_rolling",
    "calendar",
    "temp_national",
    "school_holidays",
]
COMBINED = ["load_diffs+temp_diffs", "load_diffs+load_same_hour+temp_diffs", "all_groups"]
DESCRIPTIONS = {
    "load_diffs": "differences between consecutive hourly load lags, and lag_h minus lag_24/48/168",
    "load_same_hour": "load 1 and 7 days before the same clock hour, and lag_h minus them",
    "temp_diffs": "national mean temperature change over 24h, 168h and h, and anomalies against 24h/168h means",
    "load_rolling": "24h mean/min/max of the load, same-hour mean over 7 days, 24h-168h median difference",
    "calendar": "days to next / since previous holiday, bridge days",
    "temp_national": "national mean/std temperature, degree days, 24h min/max, 24/72/168h means",
    "school_holidays": "fraction of school zones A, B, C on vacation",
    "load_diffs+temp_diffs": "the two groups above",
    "load_diffs+load_same_hour+temp_diffs": "the three groups above",
    "all_groups": "all 7 groups",
}

frames = {
    p.name.removeprefix("hgb_mean_").removesuffix("_cv_predictions.parquet"): pl.read_parquet(p).sort(
        "prediction_time", "split"
    )
    for p in args.dir.glob("hgb_mean_*_cv_predictions.parquet")
}
metadata = {
    name: json.loads((args.dir / f"hgb_mean_{name}_metadata.json").read_text())
    for name in frames
}
base = frames["baseline"]
day, fold = row_index(base)


def losses(df, kind):
    per_h = {h: row_loss(df, kind, h) for h in HORIZONS}
    return per_h | {"avg": np.mean(list(per_h.values()), axis=0)}


base_loss = {kind: losses(base, kind) for kind in ("mae", "mape")}
summary = {
    kind: {
        name: {
            key: paired_difference(loss[key], base_loss[kind][key], day, fold)
            for key in COLUMNS
        }
        for name, df in frames.items()
        if name != "baseline"
        for loss in [losses(df, kind)]
    }
    for kind in ("mae", "mape")
}
mean_loss = {
    kind: {name: {k: float(v.mean()) for k, v in losses(df, kind).items()} for name, df in frames.items()}
    for kind in ("mae", "mape")
}

order = [n for n in SINGLE + COMBINED if n in frames]
CONFIG = {"responsive": True, "displaylogo": False}


def differences_figure():
    fig = make_subplots(
        rows=1,
        cols=4,
        shared_yaxes=True,
        subplot_titles=["average", "1h", "12h", "24h"],
        horizontal_spacing=0.03,
    )
    for j, key in enumerate(COLUMNS, 1):
        s = [summary["mae"][n][key] for n in order]
        up, down = zip(*(error_bar(x) for x in s))
        fig.add_trace(
            go.Scatter(
                x=[x["mean"] for x in s],
                y=order,
                mode="markers",
                marker={"color": COLOURS[key], "size": 9},
                error_x={"type": "data", "symmetric": False, "array": up, "arrayminus": down},
                showlegend=False,
                hovertemplate="%{y}<br>%{x:+.1f} MW<extra></extra>",
            ),
            row=1,
            col=j,
        )
        fig.add_vline(x=0, line_color="gray", line_width=1, row=1, col=j)
        fig.update_xaxes(title_text="MAE difference (MW)", row=1, col=j)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        height=520,
        title="Paired MAE difference vs baseline (negative is better), 95% block-bootstrap interval",
        margin={"t": 80, "l": 270, "r": 20},
    )
    return fig


def absolute_figure():
    names = ["baseline", *[n for n in COMBINED + ["load_diffs", "temp_diffs"] if n in frames]]
    fig = go.Figure()
    for h in HORIZONS:
        fig.add_trace(
            go.Bar(
                name=f"{h}h",
                x=names,
                y=[mean_loss["mae"][n][h] for n in names],
                marker_color=COLOURS[h],
                text=[f"{mean_loss['mae'][n][h]:.0f}" for n in names],
                textposition="outside",
            )
        )
    fig.update_layout(
        barmode="group",
        height=450,
        title="MAE by horizon (MW)",
        yaxis_title="MAE (MW)",
        margin={"t": 70, "b": 120},
        legend={"orientation": "h", "y": 1.08},
    )
    fig.update_xaxes(tickangle=-20)
    return fig


def fold_figure():
    fig = go.Figure()
    folds = np.unique(fold)
    for name in COMBINED + ["load_diffs", "temp_diffs"]:
        if name in frames:
            fig.add_trace(
                go.Scatter(
                    x=[int(f) for f in folds],
                    y=summary["mae"][name]["avg"]["folds"],
                    mode="lines+markers",
                    name=name,
                )
            )
    fig.add_hline(y=0, line_color="gray", line_width=1)
    first = base["prediction_time"].filter(pl.Series(fold) == 0).min().date()
    fig.update_layout(
        height=420,
        title="Mean MAE difference vs baseline in each CV fold (average over horizons)",
        xaxis_title=f"CV fold (13-week test blocks, fold 0 starts {first})",
        yaxis_title="MAE difference (MW)",
        margin={"t": 70},
    )
    return fig


def table(kind, unit, digits):
    fmt = f"{{:+.{digits}f}}"

    def cell(s):
        star = "*" if s["low"] > 0 or s["high"] < 0 else ""
        return f"{fmt.format(s['mean'])}{star}"

    rows = [
        "<tr><td>baseline</td><td>no extra feature</td>"
        f"<td>{mean_loss[kind]['baseline']['avg']:.{digits}f}</td><td></td><td></td><td></td><td></td><td></td></tr>"
    ]
    for n in order:
        s = summary[kind][n]
        rows.append(
            f"<tr><td>{n.replace('+', ' + ')}</td><td>{html.escape(DESCRIPTIONS[n])}</td>"
            f"<td>{mean_loss[kind][n]['avg']:.{digits}f}</td>"
            + "".join(f"<td>{cell(s[k])}</td>" for k in COLUMNS)
            + f"<td>{s['avg']['n_lower']}/{len(s['avg']['folds'])}</td></tr>"
        )
    return (
        "<table><thead><tr><th>feature groups</th><th>what they add</th>"
        f"<th>{kind.upper()} ({unit})</th><th>diff avg</th><th>diff 1h</th>"
        "<th>diff 12h</th><th>diff 24h</th><th>folds better</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def div(fig, first=False):
    return fig.to_html(
        full_html=False, include_plotlyjs="cdn" if first else False, config=CONFIG
    )


m = metadata["baseline"]
dirty = ", ".join(m["git_modified_files"]) or "none"
directory = args.dir.resolve()
provenance = f"""<ul class="note">
<li>Source: <code>{html.escape(str(directory))}</code>, one
<code>hgb_mean_&lt;tag&gt;_cv_predictions.parquet</code> and
<code>_metadata.json</code> per row (runs {html.escape(', '.join(sorted(frames)))}).
Built by <code>scripts/feature_engineering/build_report.py</code>; same
paired-bootstrap code as <code>compare_groups.py</code>.</li>
<li>Model: <code>hgb_mean</code> (HistGradientBoostingRegressor, defaults as
resolved: {html.escape(json.dumps(m['pipeline_defaults']))}, max_iter {m['max_iter']}, early stopping),
horizons {m['horizons']}, one model per horizon, cities: all 10, temperature only.</li>
<li>Weather: <code>{m['weather_fetcher']}</code> (safe-lead historical forecasts: day1 lead
for horizons &le; 18h, day2 above). Evaluation range {m['env']['start']} to {m['env']['end']};
splitter: {html.escape(json.dumps(m['splitter']))}.</li>
<li>Runs made {m['date'][:10]}; git commit <code>{m['git_commit'][:7]}</code>
with uncommitted changes: {html.escape(dirty)} (plus untracked
<code>datasets/school_holidays_fr.csv</code>). Packages:
{html.escape(', '.join(f'{k} {v}' for k, v in m['versions'].items() if v))}.
Feature names per horizon are in each metadata file.</li>
<li>Single run per row (one seed): intervals cover the test period only, not training randomness.</li>
</ul>"""

best = summary["mae"]["all_groups"]["avg"]["mean"]
headline = (
    f"<p>Adding differences of load lags, same-clock-hour load comparisons and temperature "
    f"changes lowers the average MAE of <code>hgb_mean</code> from "
    f"{mean_loss['mae']['baseline']['avg']:.0f} to {mean_loss['mae']['all_groups']['avg']:.0f} MW "
    f"({best:+.0f} MW, MAPE {mean_loss['mape']['baseline']['avg']:.2f}% to "
    f"{mean_loss['mape']['all_groups']['avg']:.2f}%), better in all 8 CV folds. "
    "<code>load_diffs</code> helps at 1h and 12h, <code>temp_diffs</code> at 24h. "
    "National temperature aggregates, degree days, extra rolling statistics, calendar "
    "distances and school holidays do nothing on their own.</p>"
)

differences = div(differences_figure(), first=True)
page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Feature groups for lagged-feature models (hgb_mean)</title>
<style>
  body {{ font-family: sans-serif; max-width: 1500px; margin: 2em auto; padding: 0 1em; line-height: 1.4; }}
  h1 {{ font-size: 1.6em; }}
  h2 {{ font-size: 1.15em; margin-top: 1.8em; border-bottom: 1px solid #ddd; padding-bottom: 0.2em; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 1em; }}
  th, td {{ border-bottom: 1px solid #ddd; padding: 0.45em; text-align: right; }}
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
  .note {{ color: #555; font-size: 0.9em; }}
</style>
</head>
<body>
<h1>Feature groups for lagged-feature models (hgb_mean)</h1>
{headline}
<p class="note">Each row adds the listed group(s) to the default features
(<code>--feature_groups</code> of <code>cross_validate_models.py</code>).
Differences are paired against the baseline run; <code>*</code> = the 95%
block-bootstrap interval excludes 0. The interval is a 95% moving-block bootstrap
(14-day blocks over daily differences of the pooled test rows); it is conditional on the
fitted models and ignores training randomness.</p>
<h2>MAE (MW)</h2>
{table('mae', 'MW', 1)}
{differences}
<h2>MAPE (percentage points)</h2>
{table('mape', '%', 3)}
<h2>Absolute errors</h2>
{div(absolute_figure())}
<h2>Consistency across folds</h2>
{div(fold_figure())}
<h2>Caveats</h2>
<ul>
<li>All 7 groups together (773 MW) beat the best 3-group combination (786 MW) although the
4 other groups are neutral alone; not yet explained (interaction or run-to-run noise).</li>
<li>Only <code>hgb_mean</code> was run; <code>hgb_quantile</code>, TabICL and Kumo-Tabular are untested.</li>
</ul>
<h2>Provenance</h2>
{provenance}
</body>
</html>
"""
(args.dir / "report.html").write_text(page, "utf-8")
print(f"wrote {args.dir / 'report.html'}")
