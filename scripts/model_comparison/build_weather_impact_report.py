"""Compare oracle-weather and forecast-weather model results.

Reads the two model-comparison directories and writes an HTML page showing how
much optimistic the oracle-weather scores are. No model is re-run.

    python scripts/model_comparison/build_weather_impact_report.py

Writes ``results/weather_impact/report/index.html``.
"""

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
from compare_predictions import compute_scores
from plotly.subplots import make_subplots
from scipy.stats import t as student_t
from sklearn.metrics import mean_pinball_loss

MODEL_ORDER = [
    "hgb_mean",
    "hgb_quantile_q7",
    "hgb_quantile_recal_q7",
    "tabicl_n8_full",
    "chronos2_ctx4000",
    "kumo_tabular",
    "kumo_forecast",
]
LABELS = {
    "hgb_mean": "HGB mean",
    "hgb_quantile_q7": "HGB q7",
    "hgb_quantile_recal_q7": "HGB q7 recal",
    "tabicl_n8_full": "TabICL n8",
    "chronos2_ctx4000": "Chronos-2",
    "kumo_tabular": "Kumo-Tabular",
    "kumo_forecast": "KumoForecast",
}
HORIZONS = (1, 12, 24)


def load_scores(directory):
    scores = {}
    for path in sorted(directory.glob("*_cv_predictions.parquet")):
        name = path.stem.removesuffix("_cv_predictions")
        scores[name] = compute_scores(pl.read_parquet(path))
    return scores


def average_pinball_loss(df, selected_horizons=HORIZONS):
    """Mean pinball loss (MW) over 5/50/95% quantiles and selected horizons."""
    horizons = [h for h in selected_horizons if f"{h}h" in df.columns]
    levels = sorted(
        {
            float(column.rsplit("__q_", 1)[1])
            for column in df.columns
            if "__q_" in column
        }
    )
    chosen = [q for q in levels if q in {0.05, 0.5, 0.95}]
    if not chosen:
        return None
    losses = []
    for h in horizons:
        y = df[f"{h}h"].to_numpy()
        for q in chosen:
            losses.append(
                mean_pinball_loss(
                    y, df[f"pred_{h}h__q_{q}"].to_numpy(), alpha=q
                )
            )
    return float(np.mean(losses))


def load_fold_scores(directory):
    result = {}
    for path in sorted(directory.glob("*_cv_predictions.parquet")):
        name = path.stem.removesuffix("_cv_predictions")
        df = pl.read_parquet(path)
        splits = sorted(df["split"].unique().to_list())
        fold_frames = [
            df.filter(pl.col("split") == split) for split in splits
        ]
        folds = [compute_scores(frame) for frame in fold_frames]
        pinball = {}
        for h in HORIZONS:
            values = [
                average_pinball_loss(frame, [h]) for frame in fold_frames
            ]
            pinball[h] = (
                None
                if any(value is None for value in values)
                else np.array(values, dtype=float)
            )
        result[name] = {
            "mape": {
                h: np.array([f[f"mape__{h}h"] for f in folds], dtype=float)
                for h in HORIZONS
            },
            "pinball": pinball,
        }
    return result


def paired_ci(values):
    if len(values) < 2:
        return 0.0
    standard_error = float(np.std(values, ddof=1)) / np.sqrt(len(values))
    return float(student_t.ppf(0.975, len(values) - 1) * standard_error)


def oracle_forecast_figure(oracle_folds, forecast_folds, models):
    metrics = [
        ("mape", "MAPE (pp)", "MAPE difference (percentage points)"),
        ("pinball", "Pinball loss (MW)", "Pinball loss difference (MW)"),
    ]
    fig = make_subplots(
        rows=len(HORIZONS),
        cols=2,
        subplot_titles=[
            f"{h}h — {short}" for h in HORIZONS for _, short, _ in metrics
        ],
        horizontal_spacing=0.14,
        vertical_spacing=0.07,
    )
    for row, h in enumerate(HORIZONS, start=1):
        for col, (metric, _, label) in enumerate(metrics, start=1):
            rows = []
            for model in models:
                oracle_values = oracle_folds[model][metric][h]
                forecast_values = forecast_folds[model][metric][h]
                if oracle_values is None or forecast_values is None:
                    continue
                n_folds = min(len(oracle_values), len(forecast_values))
                diffs = oracle_values[:n_folds] - forecast_values[:n_folds]
                if metric == "mape":
                    diffs = 100 * diffs
                wins = int(np.sum(diffs < 0))
                rows.append(
                    (
                        LABELS[model],
                        float(np.mean(diffs)),
                        paired_ci(diffs),
                        f"{wins}/{n_folds}",
                    )
                )
            rows.sort(key=lambda row: row[1])
            means = [row[1] for row in rows]
            cis = [row[2] for row in rows]
            fig.add_trace(
                go.Bar(
                    x=means,
                    y=[row[0] for row in rows],
                    orientation="h",
                    error_x={
                        "type": "data",
                        "array": cis,
                        "color": "#333333",
                        "thickness": 1.5,
                        "width": 4,
                    },
                    marker_color="#1f77b4",
                    text=[row[3] for row in rows],
                    textposition="outside",
                    textfont={"size": 11},
                    cliponaxis=False,
                    hovertemplate=(
                        "%{y}<br>mean oracle − forecast: %{x:+.3g}<br>"
                        "oracle wins: %{text}<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            fig.add_vline(
                x=0,
                line={"color": "gray", "width": 1},
                row=row,
                col=col,
            )
            low = min(0.0, min([mean - ci for mean, ci in zip(means, cis)]))
            high = max(0.0, max([mean + ci for mean, ci in zip(means, cis)]))
            pad = 0.2 * (high - low if high > low else 1.0)
            fig.update_xaxes(
                title_text="oracle − forecast",
                range=[low - pad, high + pad],
                row=row,
                col=col,
            )
            fig.update_yaxes(
                autorange="reversed",
                row=row,
                col=col,
            )
    fig.update_layout(
        height=460 * len(HORIZONS),
        title="Paired oracle − forecast difference (negative = oracle better)",
        margin={"t": 85, "b": 65, "l": 130, "r": 50},
    )
    return fig

def pct(value):
    return f"{100 * value:.2f}%"


def signed(value, digits=1):
    return f"{value:+.{digits}f}"


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

    oracle = load_scores(args.oracle_dir)
    forecast = load_scores(args.forecast_dir)
    oracle_folds = load_fold_scores(args.oracle_dir)
    forecast_folds = load_fold_scores(args.forecast_dir)
    models = [m for m in MODEL_ORDER if m in oracle and m in forecast]
    models_with_impact = [m for m in models if m != "kumo_forecast"]


    metric_rows = []
    quantile_rows = []
    for model in models:
        o, f = oracle[model], forecast[model]
        delta_mae = f["mae__average"] - o["mae__average"]
        delta_mape = f["mape__average"] - o["mape__average"]
        metric_rows.append(
            "<tr>"
            f"<td>{LABELS[model]}</td>"
            f"<td>{o['mae__average']:.1f}</td>"
            f"<td>{f['mae__average']:.1f}</td>"
            f"<td>{signed(delta_mae)} ({100 * delta_mae / o['mae__average']:+.1f}%)</td>"
            f"<td>{pct(o['mape__average'])}</td>"
            f"<td>{pct(f['mape__average'])}</td>"
            f"<td>{100 * delta_mape:+.3f}</td>"
            "</tr>"
        )
        if "d2_pinball_score__average__q_0.5" in o:
            quantile_rows.append(
                "<tr>"
                f"<td>{LABELS[model]}</td>"
                f"<td>{o['d2_pinball_score__average__q_0.05']:.4f} → "
                f"{f['d2_pinball_score__average__q_0.05']:.4f}</td>"
                f"<td>{o['d2_pinball_score__average__q_0.5']:.4f} → "
                f"{f['d2_pinball_score__average__q_0.5']:.4f}</td>"
                f"<td>{o['d2_pinball_score__average__q_0.95']:.4f} → "
                f"{f['d2_pinball_score__average__q_0.95']:.4f}</td>"
                f"<td>{o['coverage_90__average']:.4f} → "
                f"{f['coverage_90__average']:.4f}</td>"
                f"<td>{o['width_90__average']:.0f} → "
                f"{f['width_90__average']:.0f}</td>"
                "</tr>"
            )

    mape_fig = make_subplots(
        rows=1,
        cols=3,
        shared_yaxes=True,
        subplot_titles=[f"{h}h" for h in HORIZONS],
        horizontal_spacing=0.05,
    )
    for j, h in enumerate(HORIZONS, 1):
        mape_fig.add_trace(
            go.Bar(
                x=[LABELS[m] for m in models_with_impact],
                y=[100 * oracle[m][f"mape__{h}h"] for m in models_with_impact],
                name="oracle weather",
                marker_color="#999999",
                showlegend=j == 1,
                hovertemplate="%{x}<br>oracle: %{y:.2f}%<extra></extra>",
            ),
            row=1,
            col=j,
        )
        mape_fig.add_trace(
            go.Bar(
                x=[LABELS[m] for m in models_with_impact],
                y=[100 * forecast[m][f"mape__{h}h"] for m in models_with_impact],
                name="forecast weather",
                marker_color="#1f77b4",
                showlegend=j == 1,
                hovertemplate="%{x}<br>forecast: %{y:.2f}%<extra></extra>",
            ),
            row=1,
            col=j,
        )
        mape_fig.update_xaxes(tickangle=-35, row=1, col=j)
    mape_fig.update_yaxes(title_text="MAPE (%)", row=1, col=1)
    mape_fig.update_layout(
        height=480,
        barmode="group",
        title="MAPE: oracle vs historical forecast weather",
        margin={"t": 70, "b": 90, "l": 55, "r": 20},
    )

    delta_fig = make_subplots(
        rows=1,
        cols=3,
        shared_yaxes=True,
        subplot_titles=[f"{h}h" for h in HORIZONS],
        horizontal_spacing=0.05,
    )
    for j, h in enumerate(HORIZONS, 1):
        delta_fig.add_trace(
            go.Bar(
                x=[LABELS[m] for m in models_with_impact],
                y=[
                    100 * (forecast[m][f"mape__{h}h"] - oracle[m][f"mape__{h}h"])
                    for m in models_with_impact
                ],
                marker_color="#D55E00",
                showlegend=False,
                hovertemplate="%{x}<br>ΔMAPE: %{y:+.3f} pp<extra></extra>",
            ),
            row=1,
            col=j,
        )
        delta_fig.add_hline(y=0, line_color="gray", line_width=1, row=1, col=j)
        delta_fig.update_xaxes(tickangle=-35, row=1, col=j)
    delta_fig.update_yaxes(title_text="ΔMAPE (forecast - oracle, pp)", row=1, col=1)
    delta_fig.update_layout(
        height=430,
        title="Cost of forecast weather: positive means the oracle score was optimistic",
        margin={"t": 70, "b": 90, "l": 60, "r": 20},
    )

    win_html = oracle_forecast_figure(
        oracle_folds, forecast_folds, models
    ).to_html(
        full_html=False,
        include_plotlyjs="cdn",
        config={"responsive": True, "displaylogo": False},
    )
    mape_html = mape_fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"responsive": True, "displaylogo": False},
    )
    delta_html = delta_fig.to_html(
        full_html=False,
        include_plotlyjs=False,
        config={"responsive": True, "displaylogo": False},
    )

    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Weather data impact</title>
<style>
  body {{ font-family: sans-serif; max-width: 1500px; margin: 2em auto; padding: 0 1em; line-height: 1.4; }}
  h1 {{ font-size: 1.6em; }}
  h2 {{ font-size: 1.15em; margin-top: 1.8em; border-bottom: 1px solid #ddd; padding-bottom: 0.2em; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 1em; }}
  th, td {{ border-bottom: 1px solid #ddd; padding: 0.45em; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
  .note {{ color: #555; font-size: 0.9em; }}
</style>
</head>
<body>
<h1>Effect of using oracle weather instead of historical forecasts</h1>
<p>Both comparisons use the same 8 walk-forward CV folds and the same model
settings. The <strong>oracle</strong> comparison uses the Open-Meteo Historical
Forecast files as weather features; the <strong>forecast</strong> comparison
uses the safe-lead historical forecasts (day1 for 1h/12h, day2 for 24h).
Positive deltas mean the forecast-weather run is worse, i.e. the oracle score
was optimistic.</p>
<p class="note">KumoForecast only accepts past covariates. Its run uses
oracle/analysis past weather in both comparisons, so it is shown in the table
but excluded from the weather-impact charts. Its near-zero delta is expected.</p>

<h2>Overall MAE and MAPE</h2>
<table>
<thead><tr><th>Model</th><th>MAE oracle</th><th>MAE forecast</th>
<th>ΔMAE (MW, %)</th><th>MAPE oracle</th><th>MAPE forecast</th>
<th>ΔMAPE (pp)</th></tr></thead>
<tbody>{''.join(metric_rows)}</tbody>
</table>

<h2>Fold-level oracle vs forecast by horizon</h2>
<p class="note">Each panel is one horizon. Bars show the mean paired difference
(oracle − forecast) across the common CV folds; negative means the
oracle-weather error is lower. Error bars are 95% paired t-intervals across the
8 folds. The text at the bar end is the number of folds where the oracle beats
the forecast. Ties count as no win. MAPE uses the median prediction; pinball
loss is averaged over the 5%, 50% and 95% quantiles for that horizon.
KumoForecast has no quantiles, so it is absent from the pinball panels.</p>
{win_html}

<h2>MAPE by horizon</h2>
{mape_html}
{delta_html}

<h2>Quantile and calibration changes</h2>
<table>
<thead><tr><th>Model</th><th>q0.05 pinball</th><th>q0.5 pinball</th>
<th>q0.95 pinball</th><th>90% coverage</th><th>90% width (MW)</th></tr></thead>
<tbody>{''.join(quantile_rows)}</tbody>
</table>

<h2>Pareto plots</h2>
<p><a href="pareto.html">Accuracy vs fit + predict time, oracle vs forecast</a></p>

<h2>Interpretation</h2>
<ul>
<li>Oracle weather makes the scores optimistic, mostly at 12h and 24h; 1h is
nearly unchanged because recent load dominates.</li>
<li>The degradation differs by model. Kumo-Tabular degrades most in absolute
terms, while Chronos-2 and TabICL degrade less.</li>
<li>The top model remains Kumo-Tabular, but its margin over the others shrinks
under forecast weather.</li>
<li>These forecast-weather results are still conservative compared with using
the latest model run (our forecasts have fixed 24h/48h leads).</li>
</ul>
</body>
</html>
"""
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "index.html").write_text(page, "utf-8")
    print(f"wrote {args.output_dir / 'index.html'}")


if __name__ == "__main__":
    main()
