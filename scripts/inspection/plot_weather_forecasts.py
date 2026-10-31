"""Compare the oracle weather data with 24h-ahead historical forecasts.

Reads ``datasets/weather_<city>.parquet`` (oracle) and
``datasets/weather_forecast_day1_<city>.parquet`` (GFS ``previous_day1``, with JMA
GSM fallback where the GFS archive has gaps). Writes
``results/weather_forecast/oracle_vs_forecast_day1.html``.

    python scripts/inspection/plot_weather_forecasts.py
"""

import datetime
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

ROOT = Path(__file__).resolve().parents[2]
DATASETS = ROOT / "datasets"
OUTPUT = ROOT / "results" / "weather_forecast" / "oracle_vs_forecast_day1.html"

CITIES = [
    "paris", "lyon", "marseille", "toulouse", "lille",
    "limoges", "nantes", "strasbourg", "brest", "bayonne",
]
FALLBACK_START = datetime.datetime(2021, 1, 1, tzinfo=datetime.UTC)
FALLBACK_END = datetime.datetime(2021, 3, 24, tzinfo=datetime.UTC)
GAP_START = datetime.datetime(2023, 12, 30, tzinfo=datetime.UTC)
GAP_END = datetime.datetime(2024, 1, 20, tzinfo=datetime.UTC)


def load(city):
    oracle = (
        pl.read_parquet(DATASETS / f"weather_{city}.parquet")
        .select("time", pl.col("temperature_2m").alias("oracle"))
        .with_columns(pl.col("time").dt.cast_time_unit("us"))
    )
    prev = (
        pl.read_parquet(DATASETS / f"weather_forecast_day1_{city}.parquet")
        .select("time", pl.col("temperature_2m").alias("prev"))
        .with_columns(pl.col("time").dt.cast_time_unit("us"))
    )
    return oracle.join(prev, on="time", how="inner").sort("time")


def gfs_rows(df):
    return df.filter(
        (pl.col("time") >= FALLBACK_END)
        & ~((pl.col("time") >= GAP_START) & (pl.col("time") < GAP_END))
    )


def main():
    data = {city: load(city) for city in CITIES}
    summary = []
    for city, df in data.items():
        d = gfs_rows(df)
        diff = d["prev"] - d["oracle"]
        summary.append(
            {
                "city": city,
                "rows": d.height,
                "bias": float(diff.mean()),
                "mae": float(diff.abs().mean()),
                "rmse": float((diff**2).mean() ** 0.5),
                "corr": float(np.corrcoef(d["oracle"], d["prev"])[0, 1]),
            }
        )

    rows = "".join(
        f"<tr><td>{r['city']}</td><td>{r['rows']}</td>"
        f"<td>{r['bias']:+.2f}</td><td>{r['mae']:.2f}</td>"
        f"<td>{r['rmse']:.2f}</td><td>{r['corr']:.4f}</td></tr>"
        for r in summary
    )

    fig = make_subplots(
        rows=5,
        cols=2,
        subplot_titles=[c.title() for c in CITIES],
        shared_xaxes=True,
        vertical_spacing=0.035,
        horizontal_spacing=0.07,
    )
    for i, city in enumerate(CITIES):
        daily = (
            data[city]
            .group_by_dynamic("time", every="1d")
            .agg(pl.col("oracle").mean(), pl.col("prev").mean())
            .sort("time")
        )
        row, col = divmod(i, 2)
        row, col = row + 1, col + 1
        fig.add_trace(
            go.Scatter(
                x=daily["time"], y=daily["oracle"], mode="lines",
                line={"color": "gray", "dash": "dash"},
                name="oracle", showlegend=i == 0,
                hovertemplate="%{x|%Y-%m-%d}: %{y:.1f} °C<extra>oracle</extra>",
            ),
            row=row, col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=daily["time"], y=daily["prev"], mode="lines",
                line={"color": "#1f77b4"},
                name="24h forecast", showlegend=i == 0,
                hovertemplate="%{x|%Y-%m-%d}: %{y:.1f} °C<extra>24h forecast</extra>",
            ),
            row=row, col=col,
        )
        for start, end in ((FALLBACK_START, FALLBACK_END), (GAP_START, GAP_END)):
            fig.add_vrect(
                x0=start, x1=end, fillcolor="rgba(0,0,0,0.08)",
                line_width=0, row=row, col=col,
            )
    fig.update_yaxes(title_text="temperature (°C)", row=1, col=1)
    fig.update_layout(
        height=1500,
        title="Oracle vs 24h-ahead weather forecast (daily mean temperature)",
        margin={"t": 60, "b": 30, "l": 55, "r": 20},
    )

    zoom = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=["Winter sample (Paris)", "Summer sample (Paris)"],
        horizontal_spacing=0.06,
    )
    paris = data["paris"]
    for j, (start, end, title) in enumerate(
        [
            (
                datetime.datetime(2025, 1, 1, tzinfo=datetime.UTC),
                datetime.datetime(2025, 1, 15, tzinfo=datetime.UTC),
                "Winter",
            ),
            (
                datetime.datetime(2024, 7, 1, tzinfo=datetime.UTC),
                datetime.datetime(2024, 7, 15, tzinfo=datetime.UTC),
                "Summer",
            ),
        ],
        1,
    ):
        d = paris.filter((pl.col("time") >= start) & (pl.col("time") < end))
        zoom.add_trace(
            go.Scatter(
                x=d["time"], y=d["oracle"], mode="lines",
                line={"color": "gray", "dash": "dash"},
                name="oracle", showlegend=j == 1,
                hovertemplate="%{x|%Y-%m-%d %H:%M}: %{y:.1f} °C<extra>oracle</extra>",
            ),
            row=1, col=j,
        )
        zoom.add_trace(
            go.Scatter(
                x=d["time"], y=d["prev"], mode="lines",
                line={"color": "#1f77b4"},
                name="24h forecast", showlegend=j == 1,
                hovertemplate=(
                    "%{x|%Y-%m-%d %H:%M}: %{y:.1f} °C"
                    "<extra>24h forecast</extra>"
                ),
            ),
            row=1, col=j,
        )
    zoom.update_yaxes(title_text="temperature (°C)", row=1, col=1)
    zoom.update_layout(
        height=430,
        title="Hourly forecast vs oracle, Paris",
        margin={"t": 60, "b": 30, "l": 55, "r": 20},
    )

    main_html = fig.to_html(
        full_html=False, include_plotlyjs="cdn",
        config={"responsive": True, "displaylogo": False},
    )
    zoom_html = zoom.to_html(
        full_html=False, include_plotlyjs=False,
        config={"responsive": True, "displaylogo": False},
    )
    page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Oracle vs 24h forecast weather</title>
<style>
  body {{ font-family: sans-serif; max-width: 1500px; margin: 2em auto; padding: 0 1em; line-height: 1.4; }}
  table {{ border-collapse: collapse; margin: 1em 0; }}
  th, td {{ padding: 4px 10px; text-align: right; border-bottom: 1px solid #ddd; }}
  th:first-child, td:first-child {{ text-align: left; }}
  .note {{ color: #555; font-size: 0.9em; }}
</style>
</head>
<body>
<h1>Oracle vs 24h-ahead weather forecasts</h1>
<p>Oracle data is the existing Open-Meteo Historical Forecast weather. The
24h forecast is the Open-Meteo Previous Runs API GFS
<code>temperature_2m_previous_day1</code> (value predicted 24 hours before the
valid time), with JMA GSM filling the small archive gaps.</p>
<p class="note">Grey bands mark periods where the GFS archive is missing and
JMA GSM is used: 2021-01-01 to 2021-03-23 and 2023-12-30 to 2024-01-19.
The summary statistics below exclude those fallback rows.</p>
<h2>Forecast error over GFS-covered rows</h2>
<table>
<thead><tr><th>City</th><th>Rows</th><th>Bias (°C)</th>
<th>MAE (°C)</th><th>RMSE (°C)</th><th>Correlation</th></tr></thead>
<tbody>{rows}</tbody>
</table>
<h2>Full period, daily mean</h2>
{main_html}
<h2>Hourly examples, Paris</h2>
{zoom_html}
</body>
</html>
"""
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(page, "utf-8")
    print(f"wrote {OUTPUT}")
    for r in summary:
        print(r)


if __name__ == "__main__":
    main()
