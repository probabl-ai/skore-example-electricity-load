"""Coverage-aware quantile comparison, separate from ``pareto.html``.

Reads the forecast-weather model-comparison directory and writes
``coverage.html`` with:

1. a table of average pinball loss and 90% interval coverage per horizon, with
   cells below the coverage target highlighted and a PASS/FAIL status;
2. the average pinball-vs-time Pareto plot, where models that miss the coverage
   target at any horizon are ringed in red and their per-horizon coverage is in
   the hover;
3. the 12h/24h coverage dot plot against the nominal target.

The 90% target is the central ``q0.05``-``q0.95`` interval. Coverage within a
tolerance below the target still passes (default 0.5 percentage points).

No model is re-run.

    python scripts/model_comparison/build_coverage_report.py
"""

import argparse
import datetime
import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
from build_combined_pareto import SOURCES, load_records
from compare_predictions import quantile_levels_of
from plotly.subplots import make_subplots
from sklearn.metrics import mean_absolute_percentage_error, mean_pinball_loss

from electricity_load_forecasting.metrics import PINBALL_LEVELS

HORIZONS = (1, 12, 24)
FAIL_COLOR = "#c00000"
EXTRA_LABELS = {
    "hgb_quantile_conformal_q7": "HGB q7 conf",
    "tabicl_conformal_n8": "TabICL n8 conf",
    "tabicl_conformal_n8_full": "TabICL n8 conf",
}
LABEL_ORDER = [
    "Chronos-2",
    "t0-gamma",
    "HGB q7",
    "HGB q7 recal",
    "HGB q7 conf",
    "Kumo-Tabular",
    "TabICL n8",
    "TabICL n8 conf",
    "TabICLForecaster",
]


def pick_level(levels, target):
    if not levels:
        return None
    best = min(levels, key=lambda q: abs(q - target))
    return best if abs(best - target) < 1e-9 else None


def fit_score_seconds(path):
    """Fit and score time summed over folds, from the matching scores CSV."""
    target = path.resolve()
    stem = target.stem.removesuffix("_cv_predictions")
    for score_path in dict.fromkeys(
        [target.parent / f"{stem}_scores.csv", *target.parent.glob("*_scores.csv")]
    ):
        if not score_path.exists():
            continue
        scores = pl.read_csv(score_path)
        if {"fit_time", "score_time"} <= set(scores.columns):
            return (
                float(scores["fit_time"].sum()),
                float(scores["score_time"].sum()),
            )
    return None, None


def enrich(records, min_coverage, tolerance):
    """Add per-horizon pinball, coverage and width to each loaded record."""
    for record in records:
        record["label"] = EXTRA_LABELS.get(record["name"], record["label"])
        df = record["frame"]
        horizons = [h for h in HORIZONS if f"{h}h" in df.columns]
        levels = quantile_levels_of(df, horizons[0]) if horizons else []
        q_low = pick_level(levels, 0.05)
        q_high = pick_level(levels, 0.95)
        chosen = [
            q for q in levels if any(abs(q - t) < 1e-9 for t in PINBALL_LEVELS)
        ] or levels
        record["levels"] = levels
        record["horizons"] = horizons
        record["pinball_h"] = {}
        record["coverage_h"] = {}
        record["width_h"] = {}
        record["mape_h"] = {}
        for h in horizons:
            y = df[f"{h}h"].to_numpy()
            point_col = (
                f"pred_{h}h__q_0.5"
                if f"pred_{h}h__q_0.5" in df.columns
                else f"pred_{h}h"
            )
            if point_col in df.columns:
                record["mape_h"][h] = 100 * mean_absolute_percentage_error(
                    y, df[point_col].to_numpy()
                )
            record["pinball_h"][h] = (
                float(
                    np.mean(
                        [
                            mean_pinball_loss(
                                y, df[f"pred_{h}h__q_{q}"].to_numpy(), alpha=q
                            )
                            for q in chosen
                        ]
                    )
                )
                if chosen
                else float("nan")
            )
            if q_low is not None and q_high is not None:
                low = df[f"pred_{h}h__q_{q_low}"].to_numpy()
                high = df[f"pred_{h}h__q_{q_high}"].to_numpy()
                inside = (y >= low) & (y <= high)
                record["coverage_h"][h] = float(inside.mean())
                record["width_h"][h] = float((high - low).mean())
        floor = min_coverage - tolerance
        record["failing"] = [
            h
            for h in horizons
            if record["coverage_h"].get(h) is not None
            and record["coverage_h"][h] < floor - 1e-12
        ]
        record["borderline"] = [
            h
            for h in horizons
            if record["coverage_h"].get(h) is not None
            and floor - 1e-12 <= record["coverage_h"][h] < min_coverage - 1e-12
        ]
        tolerance_pp = 100 * tolerance
        if record["failing"]:
            record["status_detail"] = "misses the target at " + ", ".join(
                f"{h}h" for h in record["failing"]
            )
        elif record["borderline"]:
            record["status_detail"] = (
                f"within {tolerance_pp:g} pp of the target at "
                + ", ".join(f"{h}h" for h in record["borderline"])
            )
        else:
            record["status_detail"] = "meets the target at all horizons"
    return records


def quantile_records(records):
    return [
        r
        for r in records
        if r["time"] is not None and r["pinball"] is not None and r["coverage_h"]
    ]


def status_text(record):
    return record["status_detail"]


def coverage_percent(value):
    return "-" if value is None else f"{100 * value:.2f}%"


def frontier_on(records, x_field):
    out, best = [], float("inf")
    for record in sorted(records, key=lambda r: (r[x_field], r["pinball"])):
        if record["pinball"] < best:
            out.append(record)
            best = record["pinball"]
    return out


COST_AXES = (
    ("time", "Fit + predict time (s, log scale)<br>(sum over all folds)"),
    ("score_time", "Predict time only (s, log scale)<br>(sum over all folds)"),
)


def pinball_figure(records, min_coverage, target_label):
    fig = make_subplots(
        rows=1,
        cols=len(COST_AXES),
        subplot_titles=[title.split(" (")[0] for _, title in COST_AXES],
        shared_yaxes=True,
        horizontal_spacing=0.06,
    )
    # Legend-only entries, so the source marker outline is not taken from a
    # failing point (which is ringed in red).
    for source, (source_label, color, symbol) in SOURCES.items():
        if not any(r["source"] == source for r in records):
            continue
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                marker={
                    "size": 10,
                    "color": color,
                    "symbol": symbol,
                    "line": {"color": "#333333", "width": 1},
                },
                name=source_label,
                hoverinfo="skip",
            ),
            row=1,
            col=1,
        )
    for col, (x_field, x_title) in enumerate(COST_AXES, start=1):
        for source, (source_label, color, symbol) in SOURCES.items():
            data = [
                r
                for r in records
                if r["source"] == source and r.get(x_field) is not None
            ]
            if not data:
                continue
            textposition = "bottom center" if source == "oracle" else "top center"
            for failing in (False, True):
                subset = [r for r in data if bool(r["failing"]) == failing]
                if not subset:
                    continue
                fig.add_trace(
                    go.Scatter(
                        x=[r[x_field] for r in subset],
                        y=[r["pinball"] for r in subset],
                        mode="markers+text",
                        text=[r["label"] for r in subset],
                        textposition=textposition,
                        textfont={"size": 10},
                        cliponaxis=False,
                        marker={
                            "size": 10,
                            "color": color,
                            "symbol": symbol,
                            "line": {
                                "color": FAIL_COLOR if failing else "#333333",
                                "width": 3 if failing else 1,
                            },
                        },
                        showlegend=False,
                        customdata=[
                            [
                                r["label"],
                                r["source"],
                                coverage_percent(r["coverage_h"].get(1)),
                                coverage_percent(r["coverage_h"].get(12)),
                                coverage_percent(r["coverage_h"].get(24)),
                                status_text(r),
                            ]
                            for r in subset
                        ],
                        hovertemplate=(
                            "%{customdata[0]} (%{customdata[1]})"
                            "<br>cost: %{x:,.0f} s"
                            "<br>average pinball loss: %{y:.2f} MW"
                            "<br>90% coverage 1h / 12h / 24h: "
                            "%{customdata[2]} / %{customdata[3]} / %{customdata[4]}"
                            "<br><b>%{customdata[5]}</b><extra></extra>"
                        ),
                    ),
                    row=1,
                    col=col,
                )
            front = frontier_on(data, x_field)
            if len(front) > 1:
                fig.add_trace(
                    go.Scatter(
                        x=[r[x_field] for r in front],
                        y=[r["pinball"] for r in front],
                        mode="lines",
                        line={"color": color, "dash": "dash", "width": 1.5},
                        name=f"{source_label} frontier",
                        showlegend=col == 1,
                        hoverinfo="skip",
                    ),
                    row=1,
                    col=col,
                )
        fig.update_xaxes(type="log", title_text=x_title, row=1, col=col)
    # Legend-only entry explaining the red ring.
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            marker={
                "size": 10,
                "color": "rgba(0,0,0,0)",
                "symbol": "circle",
                "line": {"color": FAIL_COLOR, "width": 3},
            },
            name=f"misses {target_label} at ≥1 horizon",
            hoverinfo="skip",
        ),
        row=1,
        col=1,
    )
    fig.update_yaxes(
        title_text="Average pinball loss (MW, lower is better)", row=1, col=1
    )
    fig.update_layout(
        title=(
            "Average pinball loss vs total and predict cost "
            f"(red ring: {target_label} coverage target missed)"
        ),
        height=560,
        margin={"t": 85, "b": 65, "l": 80, "r": 30},
    )
    return fig


def model_order(records):
    present = {r["label"] for r in records}
    ordered = [label for label in LABEL_ORDER if label in present]
    return ordered + sorted(present - set(ordered))


def coverage_figure(records, min_coverage, tolerance, target_label):
    shown = (12, 24)
    horizons = [h for h in shown if any(h in r["coverage_h"] for r in records)]
    floor = 100 * (min_coverage - tolerance)
    order = model_order(records)
    index = {label: i for i, label in enumerate(order)}
    values = [
        100 * r["coverage_h"][h]
        for r in records
        for h in horizons
        if r["coverage_h"].get(h) is not None
    ]
    low = max(0.0, np.floor(min(values) - 3))
    fig = make_subplots(
        rows=1,
        cols=len(horizons),
        subplot_titles=[f"{h}h horizon" for h in horizons],
        shared_yaxes=True,
        horizontal_spacing=0.05,
    )
    offsets = {"oracle": -0.16, "forecast": 0.16}
    for col, h in enumerate(horizons, start=1):
        fig.add_shape(
            type="rect",
            x0=-0.6,
            x1=len(order) - 0.4,
            y0=low,
            y1=floor,
            fillcolor="#fde0dc",
            opacity=0.55,
            line_width=0,
            layer="below",
            row=1,
            col=col,
        )
        fig.add_hline(
            y=100 * min_coverage,
            line={"color": "#555555", "dash": "dash", "width": 1.5},
            row=1,
            col=col,
        )
        fig.add_hline(
            y=floor,
            line={"color": "#c00000", "dash": "dot", "width": 1},
            row=1,
            col=col,
        )
        for source, (source_label, color, symbol) in SOURCES.items():
            data = [
                r
                for r in records
                if r["source"] == source and r["coverage_h"].get(h) is not None
            ]
            if not data:
                continue
            failing = [h in r["failing"] for r in data]
            fig.add_trace(
                go.Scatter(
                    x=[index[r["label"]] + offsets[source] for r in data],
                    y=[100 * r["coverage_h"][h] for r in data],
                    mode="markers+text",
                    text=[
                        f"{100 * r['coverage_h'][h]:.0f}" if h in r["failing"] else ""
                        for r in data
                    ],
                    textposition="top center",
                    textfont={"size": 9, "color": FAIL_COLOR},
                    marker={
                        "size": 9,
                        "color": color,
                        "symbol": symbol,
                        "line": {
                            "color": [
                                FAIL_COLOR if fail else "#333333" for fail in failing
                            ],
                            "width": [3 if fail else 1 for fail in failing],
                        },
                    },
                    name=source_label,
                    showlegend=col == 1,
                    customdata=[
                        [r["label"], r["source"], status_text(r)] for r in data
                    ],
                    hovertemplate=(
                        "%{customdata[0]} (%{customdata[1]})"
                        f"<br>{h}h 90% coverage: %{{y:.1f}}%"
                        "<br>%{customdata[2]}<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )
        fig.update_xaxes(
            tickvals=list(range(len(order))),
            ticktext=order,
            tickangle=35,
            row=1,
            col=col,
        )
    fig.update_yaxes(
        title_text=f"Observed {target_label} coverage (%)",
        range=[low, 100.5],
        row=1,
        col=1,
    )
    fig.update_layout(
        height=520,
        title=(
            f"Coverage of the {target_label} interval at 12h and 24h "
            f"(dashed: target; dotted: {100 * tolerance:g} pp tolerance floor)"
        ),
        margin={"t": 95, "b": 135, "l": 70, "r": 30},
    )
    return fig


def table_rows(records):
    rows = []
    for record in sorted(records, key=lambda r: r["label"]):
        cells = []
        for h in (12, 24):
            value = record["coverage_h"].get(h)
            if value is None:
                cells.append("<td>-</td>")
                continue
            if h in record["failing"]:
                style = ' class="fail"'
            elif h in record["borderline"]:
                style = ' class="borderline"'
            else:
                style = ""
            cells.append(f"<td{style}>{100 * value:.2f}%</td>")
        status = "PASS" if not record["failing"] else "FAIL"
        status_class = "pass" if not record["failing"] else "fail"
        mape_24 = record["mape_h"].get(24)
        mape_cell = "-" if mape_24 is None else f"{mape_24:.2f}%"
        fit = record["fit_time"]
        predict = record["score_time"]
        fit_cell = "-" if fit is None else f"{fit:,.0f}"
        predict_cell = "-" if predict is None else f"{predict:,.0f}"
        rows.append(
            "<tr>"
            f"<td>{record['label']}</td>"
            f"<td>{fit_cell}</td>"
            f"<td>{predict_cell}</td>"
            f"<td>{mape_cell}</td>"
            f"<td>{record['pinball']:.2f}</td>"
            + "".join(cells)
            + f'<td class="{status_class}">{status}</td>'
            f"<td>{status_text(record)}</td>"
            "</tr>"
        )
    return "".join(rows)


def provenance(args, output_dir):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", ""
    dirty_text = "clean" if not dirty else "dirty (see git status)"
    command = " ".join([Path(sys.argv[0]).name, *sys.argv[1:]])
    return f"""# Coverage-aware quantile report

Generated by `scripts/model_comparison/build_coverage_report.py`.

- command: `python {command}` (run from the repository root)
- date: {datetime.datetime.now().isoformat(timespec="seconds")}
- git commit: `{commit}` ({dirty_text})
- forecast inputs: `{args.forecast_dir.resolve()}`
- target: central {args.min_coverage:.0%} interval (`q0.05`-`q0.95`), per horizon
- tolerance: {100 * args.coverage_tolerance:g} percentage points below target
- times: `fit_time` and `score_time` summed over the 8 folds, from the per-model
  `*_scores.csv`; the Pareto has a fit+predict and a predict-only panel
- table pinball: mean over `q0.05/q0.5/q0.95`, then over horizons, in MW
- coverage: pooled marginal coverage over all CV test folds, not conditional
  on load level; see the source directories' `calibration.html` for conditional
  coverage and reliability diagrams
- no model was re-run; this page is derived from `*_cv_predictions.parquet`
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--forecast_dir",
        type=Path,
        default=Path("results/model_comparison/cv_comparison_forecast"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/model_comparison/summary"),
    )
    parser.add_argument(
        "--min_coverage",
        type=float,
        default=0.90,
        help="minimum acceptable central interval coverage (default 0.90)",
    )
    parser.add_argument(
        "--coverage_tolerance",
        type=float,
        default=0.005,
        help="allowed shortfall in coverage, as a fraction (default 0.005 = 0.5 pp)",
    )
    args = parser.parse_args()

    target_label = f"{args.min_coverage:.0%}"
    records = load_records(args.forecast_dir, "forecast")
    for record in records:
        path = args.forecast_dir / f"{record['name']}_cv_predictions.parquet"
        record["fit_time"], record["score_time"] = fit_score_seconds(path)
    records = quantile_records(
        enrich(records, args.min_coverage, args.coverage_tolerance)
    )
    if not records:
        raise SystemExit("no quantile prediction files found")

    pinball = pinball_figure(records, args.min_coverage, target_label)
    coverage = coverage_figure(
        records, args.min_coverage, args.coverage_tolerance, target_label
    )
    rows = table_rows(records)

    calibration = args.forecast_dir / "calibration.html"
    if calibration.exists():
        href = os.path.relpath(calibration.resolve(), args.output_dir.resolve())
        links_text = (
            f"Conditional coverage and reliability diagrams: "
            f'<a href="{href}">forecast weather</a>.'
        )
    else:
        links_text = (
            "Conditional coverage and reliability diagrams: see "
            "<code>calibration.html</code> in the forecast result directory."
        )

    page = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Coverage-aware quantile comparison</title>
<style>
body {{ font-family: sans-serif; max-width: 1200px; margin: 2em auto; padding: 0 1em; }}
table {{ border-collapse: collapse; margin-bottom: 1.5em; font-size: 0.9em; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 0.35em 0.7em; text-align: right; }}
th:first-child, td:first-child {{ text-align: left; }}
th:nth-last-child(2), td:nth-last-child(2),
th:last-child, td:last-child {{ text-align: left; }}
.fail {{ background: #f8d7da; color: #842029; font-weight: 600; }}
.borderline {{ background: #fff3cd; color: #664d03; }}
.pass {{ color: #146c43; font-weight: 600; }}
.chart {{ width: 100%; margin-bottom: 1.2em; }}
.note {{ color: #555; font-size: 0.9em; }}
</style>
</head>
<body>
<h1>Coverage-aware quantile comparison (forecast weather)</h1>
<p class="note">Pinball loss does not guarantee coverage. The constraint is
{target_label} coverage of the central interval, applied per horizon, with a
{100 * args.coverage_tolerance:g} percentage-point tolerance. Red cells and red
rings mean coverage is below the tolerance floor; amber cells are below
{target_label} but within tolerance. Coverage is pooled over the 8 CV test folds;
it is marginal, not conditional on load level. Fit and predict times are summed
over the 8 folds. Time comparisons mix CPU and GPU runs and are indicative, not
a controlled benchmark.</p>
<table>
<thead><tr><th>model</th><th>fit (s)</th><th>predict (s)</th><th>MAPE 24h</th>
<th>pinball avg (MW)</th><th>cov 12h</th><th>cov 24h</th>
<th>status</th><th>detail</th></tr></thead>
<tbody>{rows}</tbody>
</table>
<h2>Pinball loss vs total and predict cost</h2>
<p class="note">Left: cost of the full CV run (fit + predict). Right: predict
cost only, i.e. the deployment/inference view. Red rings mark coverage failures.
</p>
{links_text}
<div class="chart">{pinball.to_html(full_html=False, include_plotlyjs=True, config={"responsive": True, "displaylogo": False})}</div>
<h2>Coverage at 12h and 24h</h2>
<p class="note">Dashed line: {target_label} target. Dotted line: tolerance floor
({100 * (args.min_coverage - args.coverage_tolerance):.1f}%). The shaded band is
a coverage failure. The same model can look acceptable when averaged across
horizons while failing at 12h or 24h; this panel shows where.</p>
<div class="chart">{coverage.to_html(full_html=False, include_plotlyjs=False, config={"responsive": True, "displaylogo": False})}</div>
</body>
</html>
"""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "coverage.html").write_text(page, "utf-8")
    (args.output_dir / "README.md").write_text(
        provenance(args, args.output_dir), "utf-8"
    )
    print(f"wrote {args.output_dir / 'coverage.html'}")


if __name__ == "__main__":
    main()
