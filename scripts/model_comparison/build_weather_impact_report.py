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
from uncertainty import (
    INTERVAL_NOTE,
    PINBALL_LEVELS,
    error_bar,
    paired_difference,
    row_index,
    row_loss,
)

DATASETS = Path(__file__).resolve().parents[2] / "datasets"
CITIES = [
    "paris",
    "lyon",
    "marseille",
    "toulouse",
    "lille",
    "limoges",
    "nantes",
    "strasbourg",
    "brest",
    "bayonne",
]

MODEL_ORDER = [
    "hgb_mean",
    "hgb_quantile_q7",
    "hgb_quantile_recal_q7",
    "tabicl_n8_full",
    "tabicl_forecast",
    "tabicl_forecast_ctx4096",
    "tabicl_forecast_ctx1024_n4",
    "chronos2_ctx4000",
    "kumo_tabular",
    "kumo_forecast",
]
LABELS = {
    "hgb_mean": "HGB mean",
    "hgb_quantile_q7": "HGB q7",
    "hgb_quantile_recal_q7": "HGB q7 recal",
    "tabicl_n8_full": "TabICL n8",
    "tabicl_forecast": "TabICLForecaster",
    "tabicl_forecast_ctx4096": "TabICLForecaster",
    "tabicl_forecast_ctx1024_n4": "TabICLForecaster",
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


def load_frames(directory):
    return {
        path.stem.removesuffix("_cv_predictions"): pl.read_parquet(path).sort(
            "prediction_time"
        )
        for path in sorted(directory.glob("*_cv_predictions.parquet"))
    }


def load_temperatures(city, days=28):
    """Oracle and 24h/48h-ahead forecast temperature for one city.

    Returns the last ``days`` days of the ``temperature_2m`` series with columns
    ``time``, ``oracle``, ``forecast_24h`` and ``forecast_48h`` (all °C).
    """

    def read(name, column):
        return pl.read_parquet(DATASETS / name).select(
            pl.col("time").dt.cast_time_unit("us"),
            pl.col("temperature_2m").alias(column),
        )

    df = (
        read(f"weather_{city}.parquet", "oracle")
        .join(read(f"weather_forecast_day1_{city}.parquet", "forecast_24h"), on="time")
        .join(read(f"weather_forecast_day2_{city}.parquet", "forecast_48h"), on="time")
        .sort("time")
    )
    return df.filter(pl.col("time") > pl.col("time").max() - pl.duration(days=days))


def temperature_figure(days=28, default_city="paris"):
    """Oracle vs 24h- and 48h-ahead forecast temperature, one city at a time."""
    series = [
        ("oracle", "Oracle", "#222222", "dash"),
        ("forecast_24h", "24h forecast", "#1f77b4", "solid"),
        ("forecast_48h", "48h forecast", "#D55E00", "solid"),
    ]
    fig = go.Figure()
    order = []
    for city in CITIES:
        df = load_temperatures(city, days)
        for column, label, color, dash in series:
            fig.add_trace(
                go.Scatter(
                    x=df["time"],
                    y=df[column],
                    mode="lines",
                    name=label,
                    line={"color": color, "width": 1.4, "dash": dash},
                    visible=city == default_city,
                    showlegend=city == default_city,
                    hovertemplate=(
                        label + " %{x|%Y-%m-%d %H:%M}: %{y:.1f} °C<extra></extra>"
                    ),
                )
            )
            order.append(city)
    buttons = [
        {
            "label": city.title(),
            "method": "update",
            "args": [
                {
                    "visible": [entry == city for entry in order],
                    "showlegend": [entry == city for entry in order],
                },
                {"title": f"Temperature: oracle vs 24h/48h forecasts — {city.title()}"},
            ],
        }
        for city in CITIES
    ]
    fig.update_yaxes(title_text="temperature (°C)")
    fig.update_layout(
        height=480,
        title=f"Temperature: oracle vs 24h/48h forecasts — {default_city.title()}",
        margin={"t": 110, "b": 60, "l": 60, "r": 30},
        updatemenus=[
            {
                "buttons": buttons,
                "direction": "down",
                "x": 1.0,
                "xanchor": "right",
                "y": 1.16,
                "yanchor": "top",
                "showactive": True,
            }
        ],
    )
    return fig


def weather_row_loss(df, metric, h):
    """Per-row MAPE (%) or pinball loss (MW, mean over 5/50/95%) at horizon h."""
    if metric == "mape":
        return row_loss(df, "mape", h)
    losses = [row_loss(df, "pinball", h, q) for q in PINBALL_LEVELS]
    return None if any(loss is None for loss in losses) else np.mean(losses, axis=0)


def oracle_forecast_figure(oracle_frames, forecast_frames, models):
    metrics = [
        ("mape", "MAPE (pp)"),
        ("pinball", "Pinball loss (MW)"),
    ]
    fig = make_subplots(
        rows=len(HORIZONS),
        cols=2,
        subplot_titles=[f"{h}h — {short}" for h in HORIZONS for _, short in metrics],
        horizontal_spacing=0.14,
        vertical_spacing=0.07,
    )
    for row, h in enumerate(HORIZONS, start=1):
        for col, (metric, _) in enumerate(metrics, start=1):
            data = {}
            for model in models:
                o, f = oracle_frames[model], forecast_frames[model]
                if not o.select("prediction_time", "split").equals(
                    f.select("prediction_time", "split")
                ):
                    raise SystemExit(f"{model}: oracle and forecast rows differ")
                oracle_loss = weather_row_loss(o, metric, h)
                forecast_loss = weather_row_loss(f, metric, h)
                if oracle_loss is None or forecast_loss is None:
                    continue
                day, fold = row_index(o)
                data[model] = paired_difference(oracle_loss, forecast_loss, day, fold)
            models_sorted = sorted(data, key=lambda m: data[m]["mean"])
            fold_x = [v for m in models_sorted for v in data[m]["folds"]]
            fold_y = [
                k + (f - 3.5) * 0.04
                for k, m in enumerate(models_sorted)
                for f in range(len(data[m]["folds"]))
            ]
            fig.add_trace(
                go.Scatter(
                    x=fold_x,
                    y=fold_y,
                    mode="markers",
                    marker={"size": 5, "color": "#1f77b4", "opacity": 0.5},
                    text=[
                        f"{LABELS[m]}, fold {f}"
                        for m in models_sorted
                        for f in range(len(data[m]["folds"]))
                    ],
                    hovertemplate="%{text}: %{x:+.3g}<extra></extra>",
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            above, below = zip(*(error_bar(data[m]) for m in models_sorted))
            fig.add_trace(
                go.Scatter(
                    x=[data[m]["mean"] for m in models_sorted],
                    y=list(range(len(models_sorted))),
                    mode="markers",
                    marker={
                        "size": 10,
                        "color": "#1f77b4",
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
                        "width": 4,
                    },
                    customdata=[
                        [
                            LABELS[m],
                            data[m]["low"],
                            data[m]["high"],
                            data[m]["n_lower"],
                            len(data[m]["folds"]),
                        ]
                        for m in models_sorted
                    ],
                    hovertemplate=(
                        "%{customdata[0]}<br>mean oracle − forecast: %{x:+.3g}<br>"
                        "95% block bootstrap: [%{customdata[1]:+.3g}, %{customdata[2]:+.3g}]<br>"
                        "oracle lower in %{customdata[3]}/%{customdata[4]} folds"
                        "<extra></extra>"
                    ),
                    showlegend=False,
                ),
                row=row,
                col=col,
            )
            fig.add_vline(
                x=0,
                line={"color": "gray", "width": 1, "dash": "dash"},
                row=row,
                col=col,
            )
            fig.update_xaxes(title_text="oracle − forecast", row=row, col=col)
            fig.update_yaxes(
                tickvals=list(range(len(models_sorted))),
                ticktext=[LABELS[m] for m in models_sorted],
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
    oracle_frames = load_frames(args.oracle_dir)
    forecast_frames = load_frames(args.forecast_dir)
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
        oracle_frames, forecast_frames, models_with_impact
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
    ts_html = temperature_figure().to_html(
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
<tbody>{"".join(metric_rows)}</tbody>
</table>

<h2>Fold-level oracle vs forecast by horizon</h2>
<p class="note">Each panel is one horizon. Markers show the mean paired difference
(oracle − forecast); negative means the oracle-weather error is lower.
{INTERVAL_NOTE} MAPE uses the median prediction; pinball
loss is averaged over the 5%, 50% and 95% quantiles for that horizon.
KumoForecast is excluded from these panels (see the note above).</p>
{win_html}

<h2>MAPE by horizon</h2>
{mape_html}
{delta_html}

<h2>Temperature: oracle vs 24h/48h forecasts</h2>
<p class="note">Hourly temperature over the last four weeks of the data. The
oracle is the Open-Meteo Historical Forecast value; the 24h and 48h forecasts
are the GFS <code>previous_day1</code>/<code>previous_day2</code> values (valid
time minus 24h/48h). Use the dropdown to switch city.</p>
{ts_html}

<h2>Quantile and calibration changes</h2>
<table>
<thead><tr><th>Model</th><th>q0.05 pinball</th><th>q0.5 pinball</th>
<th>q0.95 pinball</th><th>90% coverage</th><th>90% width (MW)</th></tr></thead>
<tbody>{"".join(quantile_rows)}</tbody>
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
