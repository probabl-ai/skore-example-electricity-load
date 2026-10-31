"""Download historical weather forecasts for the 10 French cities.

The source is the Open-Meteo Previous Runs API. We fetch GFS 2 m temperature at
the 24h and 48h lead times (``temperature_2m_previous_day1/2``), available from
March 2021. JMA GSM fills the GFS archive gaps.

Files are written as ``datasets/weather_prevday<lead>_<city>.parquet`` with the
same ``time, temperature_2m`` layout as the oracle weather files. The values
are forecasts issued ``lead`` days before the target time, not oracle
observations.

Open-Meteo data is CC BY 4.0; the free API is for non-commercial use.

    python scripts/fetch_weather_forecasts.py [--leads 1 2] [--force]
"""

import argparse
import datetime
import json
import time
import urllib.parse
import urllib.request
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
DATASETS = ROOT / "datasets"
URL = "https://previous-runs-api.open-meteo.com/v1/forecast"
START = "2021-01-01"
END = "2025-05-31"

CITIES = [
    {"name": "paris", "latitude": 48.8566, "longitude": 2.3522},
    {"name": "lyon", "latitude": 45.7640, "longitude": 4.8357},
    {"name": "marseille", "latitude": 43.2965, "longitude": 5.3698},
    {"name": "toulouse", "latitude": 43.6047, "longitude": 1.4442},
    {"name": "lille", "latitude": 50.6292, "longitude": 3.0573},
    {"name": "limoges", "latitude": 45.8336, "longitude": 1.2616},
    {"name": "nantes", "latitude": 47.2184, "longitude": -1.5536},
    {"name": "strasbourg", "latitude": 48.5734, "longitude": 7.7521},
    {"name": "brest", "latitude": 48.3904, "longitude": -4.4861},
    {"name": "bayonne", "latitude": 43.4833, "longitude": -1.4667},
]


def fetch(city, model, lead):
    variable = f"temperature_2m_previous_day{lead}"
    params = {
        "latitude": city["latitude"],
        "longitude": city["longitude"],
        "hourly": variable,
        "start_date": START,
        "end_date": END,
        "models": model,
        "timezone": "UTC",
    }
    url = URL + "?" + urllib.parse.urlencode(params)
    for attempt in range(5):
        try:
            with urllib.request.urlopen(url, timeout=120) as response:
                data = json.load(response)
            break
        except Exception:
            if attempt == 4:
                raise
            time.sleep(2**attempt)
    hourly = data["hourly"]
    times = [
        datetime.datetime.fromisoformat(t).replace(tzinfo=datetime.UTC)
        for t in hourly["time"]
    ]
    return pl.DataFrame(
        {
            "time": pl.Series(times, dtype=pl.Datetime("us", "UTC")),
            model: pl.Series(hourly[variable], dtype=pl.Float32),
        }
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--leads", type=int, nargs="+", default=[1, 2])
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    DATASETS.mkdir(exist_ok=True)

    for lead in args.leads:
        for city in CITIES:
            path = DATASETS / f"weather_prevday{lead}_{city['name']}.parquet"
            if path.exists() and not args.force:
                print(f"{path.name} already exists")
                continue
            print(f"downloading {city['name']} day{lead}...", flush=True)
            gfs = fetch(city, "gfs_global", lead)
            jma = fetch(city, "jma_gsm", lead)
            merged = (
                gfs.join(jma, on="time", how="full", coalesce=True)
                .with_columns(
                    pl.coalesce("gfs_global", "jma_gsm").alias("temperature_2m")
                )
                .select("time", "temperature_2m")
                .sort("time")
            )
            n_gfs = gfs["gfs_global"].is_not_null().sum()
            n_jma = merged.height - n_gfs
            if merged["temperature_2m"].null_count():
                raise RuntimeError(f"{path.name}: missing values after fallback")
            merged.write_parquet(path)
            print(
                f"  wrote {path.name}: {merged.height} rows "
                f"({n_gfs} GFS, {n_jma} JMA fallback)"
            )


if __name__ == "__main__":
    main()
