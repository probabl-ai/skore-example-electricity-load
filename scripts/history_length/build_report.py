"""Build the published training-history-length report.

Reads the two result directories produced by ``cross_validate_history.py`` and
writes a single HTML page with MAPE tables (history expressed in months) and
true-vs-predicted plots for the last CV fold.

    python scripts/history_length/build_report.py

Writes ``results/history_length/report/index.html`` and
``results/history_length/report/plots/*.html``. Publish the whole directory with
``tools/publish.py``.
"""

import argparse
import datetime
import sys
from pathlib import Path

import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "model_comparison"))
from compare_predictions import compute_scores

HOURS_PER_MONTH = 365.25 * 24 / 12
MODELS = {
    "hgb_mean": "HGB mean",
    "tabicl": "TabICL",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hgb_dir",
        type=Path,
        default=Path("results/history_length/hgb_mean_cv"),
    )
    parser.add_argument(
        "--tabicl_dir",
        type=Path,
        default=Path("results/history_length/tabicl_n8_common_cv"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/history_length/report"),
    )
    return parser.parse_args()


def month_label(rows):
    if rows == 0:
        return "all"
    return f"{rows / HOURS_PER_MONTH:.3f} mo"


def load_study(directory):
    summary = pl.read_csv(directory / "summary.csv")
    study = []
    for record in summary.iter_rows(named=True):
        predictions = pl.read_parquet(
            directory / f"{record['name']}_cv_predictions.parquet"
        )
        study.append(
            record
            | {
                "label": month_label(record["max_train_rows"]),
                "metrics": compute_scores(predictions),
                "predictions": predictions,
            }
        )
    return sorted(
        study, key=lambda r: (r["max_train_rows"] == 0, r["max_train_rows"])
    )


def pct(value):
    return f"{100 * value:.2f}%"


def table(study, time_column, time_label):
    rows = []
    for record in study:
        metrics = record["metrics"]
        rows.append(
            "<tr>"
            f"<td>{record['label']}</td>"
            f"<td>{pct(metrics['mape__1h'])}</td>"
            f"<td>{pct(metrics['mape__12h'])}</td>"
            f"<td>{pct(metrics['mape__24h'])}</td>"
            f"<td>{pct(metrics['mape__average'])}</td>"
            f"<td>{record[time_column]:.1f} s</td>"
            "</tr>"
        )
    return (
        "<table><thead><tr>"
        "<th>Training history</th><th>MAPE 1h</th><th>MAPE 12h</th>"
        "<th>MAPE 24h</th><th>MAPE average</th>"
        f"<th>{time_label}</th>"
        "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def combined_mape_figure(studies):
    horizons = (1, 12, 24)
    fig = make_subplots(
        rows=1,
        cols=len(horizons),
        shared_yaxes=True,
        subplot_titles=[f"{h}h horizon" for h in horizons],
        horizontal_spacing=0.05,
    )
    styles = {
        "hgb_mean": ("HGB mean", "#E69F00", "circle"),
        "tabicl": ("TabICL", "#D55E00", "square"),
    }
    for j, horizon in enumerate(horizons, 1):
        for model, (label, color, symbol) in styles.items():
            study = [r for r in studies[model] if r["max_train_rows"] > 0]
            fig.add_trace(
                go.Scatter(
                    x=[r["max_train_rows"] / HOURS_PER_MONTH for r in study],
                    y=[
                        100 * r["metrics"][f"mape__{horizon}h"]
                        for r in study
                    ],
                    mode="lines+markers",
                    name=label,
                    line={"color": color},
                    marker={"symbol": symbol, "size": 8},
                    legendgroup=label,
                    showlegend=j == 1,
                    hovertemplate=(
                        f"{label}<br>%{{x:.3f}} months: "
                        "%{y:.2f}%<extra></extra>"
                    ),
                ),
                row=1,
                col=j,
            )
        fig.update_xaxes(
            title_text="Training history (months)",
            rangemode="tozero",
            row=1,
            col=j,
        )
    fig.update_yaxes(title_text="Test MAPE (%)", row=1, col=1)
    fig.update_layout(
        title="Test MAPE vs training-history length",
        height=420,
        autosize=True,
        hovermode="x unified",
        margin={"t": 70, "b": 40, "l": 55, "r": 20},
    )
    return fig


def prediction_figure(study, model, horizon):
    last_split = study[0]["predictions"]["split"].max()
    fig = go.Figure()
    truth = study[0]["predictions"].filter(pl.col("split") == last_split).sort(
        "prediction_time"
    )
    target_time = truth["prediction_time"] + datetime.timedelta(hours=horizon)
    fig.add_trace(
        go.Scatter(
            x=target_time,
            y=truth[f"{horizon}h"],
            mode="lines",
            line={"color": "gray", "dash": "dash"},
            name="actual load",
            hovertemplate="%{x|%Y-%m-%d %H:%M}: %{y}<extra></extra>",
        )
    )
    column = (
        f"pred_{horizon}h__q_0.5"
        if model == "tabicl"
        else f"pred_{horizon}h"
    )
    for record in study:
        data = record["predictions"].filter(pl.col("split") == last_split).sort(
            "prediction_time"
        )
        target_time = data["prediction_time"] + datetime.timedelta(hours=horizon)
        fig.add_trace(
            go.Scatter(
                x=target_time,
                y=data[column],
                mode="lines",
                name=record["label"],
                hovertemplate="%{x|%Y-%m-%d %H:%M}: %{y}<extra></extra>",
            )
        )
    fig.update_layout(
        title=f"{MODELS[model]} - {horizon}h horizon (last CV fold)",
        xaxis_title="Target time",
        yaxis_title="Load (MW)",
        height=420,
        margin={"t": 45, "b": 40, "l": 60, "r": 20},
        legend={"orientation": "h", "y": -0.25},
    )
    return fig


def main():
    args = parse_args()
    plots_dir = args.output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    studies = {
        "hgb_mean": load_study(args.hgb_dir),
        "tabicl": load_study(args.tabicl_dir),
    }

    plots = []
    for model, study in studies.items():
        for horizon in (1, 12, 24):
            name = f"{model}_{horizon}h"
            prediction_figure(study, model, horizon).write_html(
                plots_dir / f"{name}.html",
                include_plotlyjs="cdn",
                config={"responsive": True, "displaylogo": False},
            )
            plots.append((model, horizon, f"plots/{name}.html"))

    chart_html = combined_mape_figure(studies).to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={"responsive": True, "displaylogo": False},
    )

    plot_links = []
    for model, label in MODELS.items():
        items = "".join(
            f'<li><a href="{path}">{horizon}h horizon</a></li>'
            for m, horizon, path in plots
            if m == model
        )
        plot_links.append(f"<h3>{label}</h3><ul>{items}</ul>")

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Training-history length</title>
<style>
  body {{ font-family: sans-serif; max-width: 1100px; margin: 2em auto; padding: 0 1em; line-height: 1.4; }}
  h1 {{ font-size: 1.6em; }}
  h2 {{ font-size: 1.15em; margin-top: 1.8em; border-bottom: 1px solid #ddd; padding-bottom: 0.2em; }}
  h3 {{ font-size: 1em; margin-top: 1.2em; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 1em; }}
  th, td {{ border-bottom: 1px solid #ddd; padding: 0.45em; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
  .note {{ color: #555; font-size: 0.9em; }}
  .charts {{ display: block; }}
  .chart {{ width: 100%; margin-bottom: 1.2em; overflow: visible; }}
</style>
</head>
<body>
<h1>Effect of training-history length</h1>
<p>Walk-forward CV on the same 8 folds as the model comparison. Each
configuration caps the training windows with
<code>make_data_op(max_train_rows=N)</code>, so only the amount of history used
for fitting changes; test dates are identical. Metrics are pooled over all
folds. TabICL uses eight estimators and its median prediction for MAPE.</p>
<p class="note">"all" means no cap: each fold uses all data since 2021-03,
about 26&ndash;47 months depending on the fold, so it has no single months value
and is shown in the tables but not on the month axis. Training-history values
are exact months (rows / {HOURS_PER_MONTH:.1f} hours per month).</p>

<h2>HGB mean (point forecast)</h2>
{table(studies["hgb_mean"], "fit_time_mean", "Mean fit time / fold")}

<h2>TabICL (median forecast)</h2>
{table(studies["tabicl"], "score_time_total", "Total inference time")}

<h2>MAPE vs history by horizon</h2>
<div class="charts">
  <div class="chart">{chart_html}</div>
</div>

<h2>Predicted vs actual load</h2>
<p class="note">Plots show the last CV fold (2025-03-01 to 2025-05-31) so the
curves stay readable; quality metrics above use all 8 folds.</p>
{"".join(plot_links)}

</body>
</html>
"""
    (args.output_dir / "index.html").write_text(page, "utf-8")
    print(f"wrote {args.output_dir / 'index.html'} and {len(plots)} plots")


if __name__ == "__main__":
    main()
