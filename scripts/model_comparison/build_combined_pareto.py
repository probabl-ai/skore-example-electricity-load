"""Combined Pareto plots for oracle-weather and forecast-weather runs.

Reads the two model-comparison directories and writes ``pareto.html`` into the
weather-impact report directory. No model is re-run.
"""

import argparse
import re
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
from compare_predictions import compute_scores, horizons_of, quantile_levels_of
from scipy.stats import t as student_t
from sklearn.metrics import mean_pinball_loss

LABELS = {
    "hgb_mean": "HGB mean",
    "hgb_mean_fast": "HGB mean fast",
    "hgb_quantile_q7": "HGB q7",
    "hgb_quantile_recal_q7": "HGB q7 recal",
    "tabicl_n8_full": "TabICL n8",
    "chronos2_ctx4000": "Chronos-2",
    "kumo_tabular": "Kumo-Tabular",
    "kumo_forecast": "KumoForecast",
}
SOURCES = {
    "oracle": ("Oracle weather", "#999999", "circle"),
    "forecast": ("Forecast weather", "#1f77b4", "square"),
}


def fit_predict_time_seconds(path):
    target = path.resolve()
    stem = target.stem.removesuffix("_cv_predictions")
    for score_path in dict.fromkeys(
        [target.parent / f"{stem}_scores.csv", *target.parent.glob("*_scores.csv")]
    ):
        if not score_path.exists():
            continue
        scores = pl.read_csv(score_path)
        if {"fit_time", "score_time"} <= set(scores.columns):
            return float(scores["fit_time"].sum() + scores["score_time"].sum())
    log_path = target.parent / "run.log"
    if log_path.exists():
        matches = re.findall(r":\s*([\d.]+)s total", log_path.read_text(errors="replace"))
        if matches:
            return float(matches[-1])
    return None


def average_pinball_loss(df):
    hs = horizons_of(df)
    levels = quantile_levels_of(df, hs[0])
    chosen = [
        q
        for q in levels
        if any(abs(q - target) < 1e-9 for target in (0.05, 0.5, 0.95))
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
                mean_pinball_loss(
                    y, df[f"pred_{h}h__q_{q}"].to_numpy(), alpha=q
                )
            )
    return float(np.mean(losses))


def load_records(directory, source):
    records = []
    for path in sorted(directory.glob("*_cv_predictions.parquet")):
        name = path.stem.removesuffix("_cv_predictions")
        df = pl.read_parquet(path)
        splits = sorted(df["split"].unique().to_list())
        fold_scores = [
            compute_scores(df.filter(pl.col("split") == split)) for split in splits
        ]
        fold_pinball = [
            average_pinball_loss(df.filter(pl.col("split") == split))
            for split in splits
        ]
        mape_values = np.array(
            [100 * fold["mape__average"] for fold in fold_scores], dtype=float
        )
        pinball_values = (
            None
            if any(value is None for value in fold_pinball)
            else np.array(fold_pinball, dtype=float)
        )
        records.append(
            {
                "name": name,
                "label": LABELS.get(name, name),
                "source": source,
                "time": fit_predict_time_seconds(path),
                "mape": float(np.mean(mape_values)),
                "mape_std": float(np.std(mape_values, ddof=1)),
                "pinball": (
                    None
                    if pinball_values is None
                    else float(np.mean(pinball_values))
                ),
                "pinball_std": (
                    None
                    if pinball_values is None
                    else float(np.std(pinball_values, ddof=1))
                ),
                "mape_values": mape_values,
                "pinball_values": pinball_values,
            }
        )
    return records


def pinball_cell(value, std=None):
    if value is None:
        return "-"
    return f"{value:.2f} ± {std:.2f}"


def frontier(records, metric):
    out, best = [], float("inf")
    for record in sorted(records, key=lambda r: (r["time"], r[metric])):
        if record[metric] < best:
            out.append(record)
            best = record[metric]
    return out


def figure(records, metric, title, y_title, hover):
    fig = go.Figure()
    for source, (source_label, color, symbol) in SOURCES.items():
        data = [
            r
            for r in records
            if r["source"] == source and r["time"] is not None and r[metric] is not None
        ]
        fig.add_trace(
            go.Scatter(
                x=[r["time"] for r in data],
                y=[r[metric] for r in data],
                mode="markers+text",
                text=[r["label"] for r in data],
                textposition="top center",
                textfont={"size": 11},
                cliponaxis=False,
                marker={"size": 9, "color": color, "symbol": symbol},
                name=source_label,
                customdata=[[r["label"], r["source"]] for r in data],
                hovertemplate=(
                    "%{customdata[0]} (%{customdata[1]})"
                    "<br>fit + predict: %{x:,.0f} s<br>"
                    + hover
                    + "<extra></extra>"
                ),
            )
        )
        front = frontier(data, metric)
        if len(front) > 1:
            fig.add_trace(
                go.Scatter(
                    x=[r["time"] for r in front],
                    y=[r[metric] for r in front],
                    mode="lines",
                    line={"color": color, "dash": "dash", "width": 1.5},
                    name=f"{source_label} frontier",
                    hoverinfo="skip",
                )
            )
    fig.update_xaxes(type="log", title_text="Fit + predict time (s, log scale)")
    fig.update_yaxes(title_text=y_title)
    fig.update_layout(
        title=title,
        height=520,
        margin={"t": 85, "b": 65, "l": 80, "r": 120},
    )
    return fig


def paired_ci(values):
    if len(values) < 2:
        return 0.0
    standard_error = float(np.std(values, ddof=1)) / np.sqrt(len(values))
    return float(student_t.ppf(0.975, len(values) - 1) * standard_error)


def baseline_for(records, source, metric):
    names = [r["name"] for r in records if r["source"] == source]
    if metric == "mape":
        candidates = [n for n in names if n.startswith("hgb_mean") and "fast" not in n]
        prefix = "hgb_mean"
    else:
        candidates = [
            n
            for n in names
            if n.startswith("hgb_quantile_q7") and "recal" not in n
        ]
        prefix = "hgb_quantile"
    if candidates:
        return sorted(candidates)[0]
    fallback = sorted(n for n in names if n.startswith(prefix))
    return fallback[0] if fallback else None


def difference_figure(records, metric, title, metric_label):
    fig = go.Figure()
    for source, (source_label, color, symbol) in SOURCES.items():
        baseline = baseline_for(records, source, metric)
        baseline_record = next(
            (
                r
                for r in records
                if r["source"] == source and r["name"] == baseline
            ),
            None,
        )
        if baseline_record is None or baseline_record[f"{metric}_values"] is None:
            continue
        baseline_values = baseline_record[f"{metric}_values"]
        data = []
        for r in records:
            if (
                r["source"] != source
                or r["time"] is None
                or r[f"{metric}_values"] is None
            ):
                continue
            diffs = r[f"{metric}_values"] - baseline_values
            data.append(
                {
                    "label": r["label"],
                    "time": r["time"],
                    "diff": float(np.mean(diffs)),
                    "ci": paired_ci(diffs),
                    "baseline": r["name"] == baseline,
                }
            )
        fig.add_trace(
            go.Scatter(
                x=[r["time"] for r in data],
                y=[r["diff"] for r in data],
                mode="markers+text",
                text=[
                    f"{r['label']} (baseline)" if r["baseline"] else r["label"]
                    for r in data
                ],
                textposition="top center",
                textfont={"size": 11},
                cliponaxis=False,
                marker={
                    "size": [10 if r["baseline"] else 9 for r in data],
                    "color": color,
                    "symbol": [
                        "diamond" if r["baseline"] else symbol for r in data
                    ],
                },
                name=f"{source_label} (vs {baseline})",
                customdata=[[r["label"], r["ci"]] for r in data],
                error_y={
                    "type": "data",
                    "array": [r["ci"] for r in data],
                    "color": "#333333",
                    "thickness": 1.5,
                    "width": 4,
                },
                hovertemplate=(
                    "%{customdata[0]}<br>fit + predict: %{x:,.0f} s<br>"
                    + metric_label
                    + " difference: %{y:+.3g} ± %{customdata[1]:.3g}"
                    + "<extra></extra>"
                ),
            )
        )
        front = frontier(data, "diff")
        if len(front) > 1:
            fig.add_trace(
                go.Scatter(
                    x=[r["time"] for r in front],
                    y=[r["diff"] for r in front],
                    mode="lines",
                    line={"color": color, "dash": "dash", "width": 1.5},
                    name=f"{source_label} frontier",
                    hoverinfo="skip",
                )
            )
    fig.update_xaxes(type="log", title_text="Fit + predict time (s, log scale)")
    fig.update_yaxes(
        title_text=f"Δ {metric_label} vs source-specific HGB baseline (lower is better)"
    )
    fig.update_layout(
        title=title,
        height=520,
        margin={"t": 85, "b": 65, "l": 80, "r": 120},
    )
    return fig


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--oracle_dir",
        type=Path,
        default=Path("results/model_comparison/cv_comparison"),
    )
    parser.add_argument(
        "--forecast_dir",
        type=Path,
        default=Path("results/model_comparison/cv_comparison_forecast"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/weather_impact/report"),
    )
    args = parser.parse_args()

    records = load_records(args.oracle_dir, "oracle") + load_records(
        args.forecast_dir, "forecast"
    )
    missing = [r["label"] for r in records if r["time"] is None]
    if missing:
        print("warning: no fit+predict time for", ", ".join(missing))

    difference_figs = [
        difference_figure(
            records,
            "mape",
            "MAPE difference vs fit + predict time",
            "MAPE (%)",
        ),
        difference_figure(
            records,
            "pinball",
            "Pinball loss difference vs fit + predict time",
            "pinball loss (MW)",
        ),
    ]
    difference_parts = "".join(
        f'<div class="chart">{fig.to_html(full_html=False, include_plotlyjs=i == 0, config={"responsive": True, "displaylogo": False})}</div>'
        for i, fig in enumerate(difference_figs)
    )
    figs = [
        figure(
            records,
            "mape",
            "Average MAPE vs fit + predict time",
            "Average MAPE (%, lower is better)",
            "MAPE: %{y:.3f}%",
        ),
        figure(
            records,
            "pinball",
            "Average pinball loss vs fit + predict time",
            "Average pinball loss (MW, lower is better)",
            "pinball loss: %{y:.2f} MW",
        ),
    ]
    parts = "".join(
        f'<div class="chart">{fig.to_html(full_html=False, include_plotlyjs=not difference_figs and i == 0, config={"responsive": True, "displaylogo": False})}</div>'
        for i, fig in enumerate(figs)
    )
    rows = "".join(
        "<tr>"
        f"<td>{r['source']}</td>"
        f"<td>{r['label']}</td>"
        f"<td>{r['time']:,.0f}</td>"
        f"<td>{r['mape']:.3f}</td>"
        f"<td>{r['pinball']:.2f}</td>"
        "</tr>"
        if r["pinball"] is not None
        else (
            "<tr>"
            f"<td>{r['source']}</td>"
            f"<td>{r['label']}</td>"
            f"<td>{r['time']:,.0f}</td>"
            f"<td>{r['mape']:.3f}</td>"
            "<td>-</td>"
            "</tr>"
        )
        for r in sorted(
            (r for r in records if r["time"] is not None),
            key=lambda r: (r["source"], r["time"]),
        )
    )
    page = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Pareto plots: oracle vs forecast weather</title>
<style>
body {{ font-family: sans-serif; max-width: 1200px; margin: 2em auto; padding: 0 1em; }}
table {{ border-collapse: collapse; margin-bottom: 1.5em; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 0.4em 0.8em; text-align: right; }}
th:first-child, th:nth-child(2), td:first-child, td:nth-child(2) {{ text-align: left; }}
.chart {{ width: 100%; margin-bottom: 1.2em; }}
.note {{ color: #555; font-size: 0.9em; }}
</style>
</head>
<body>
<h1>Pareto plots: oracle vs forecast weather</h1>
<p class="note">Times are summed over the 8 CV folds (fit + predict/score). The
new forecast-weather HGB runs use cpu-1 with all 8 cores sequentially; the
oracle HGB timings are from the older run with <code>OMP_NUM_THREADS=2</code> and
concurrent HGB jobs, so HGB oracle vs forecast <em>times</em> are not directly
comparable. Accuracy deltas are unaffected. GPU runs use gpu-2; CPU/GPU
hardware differs, so times are indicative rather than a controlled benchmark.
Table values are means across the 8 folds. Lower is better on both axes.
KumoForecast is absent from the pinball chart because it has no quantiles.</p>
<table>
<thead><tr><th>source</th><th>model</th><th>fit + predict (s)</th>
<th>MAPE (%)</th><th>pinball loss (MW)</th></tr></thead>
<tbody>{rows}</tbody>
</table>
<h2>Paired differences vs source-specific baseline</h2>
<p class="note">Each model is compared with its own weather-source HGB baseline
(HGB mean for MAPE, HGB q7 for pinball loss). Error bars are 95% paired
t-intervals across the 8 folds; pairing removes the common seasonal/fold
difficulty. The baseline is shown at zero difference. With only 8 correlated
folds, treat these intervals as descriptive. Dashed colored lines mark the
per-source Pareto frontiers.</p>
{difference_parts}
<h2>Absolute accuracy vs fit + predict time</h2>
<p class="note">Means across the 8 CV folds, shown without error bars. Dashed
colored lines mark the per-source Pareto frontiers.</p>
{parts}
</body>
</html>
"""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "pareto.html").write_text(page, "utf-8")
    print(f"wrote {args.output_dir / 'pareto.html'}")


if __name__ == "__main__":
    main()
