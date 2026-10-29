"""Compare the raw and recalibrated forecasts for example prediction dates.

The data op predicts the next 24 hours (horizons 1 to 24h, one model per
horizon). This script fits the raw and the recalibrated pipelines once, on the
training set of the cross-validation fold that contains ``--date``, then predicts
several dates of that fold's test period (out of sample for the fitted models)
and plots, for each, both fans (median and 5-95% interval) with the surrounding
history and the observed load.

    python scripts/calibration/single_date_calibration.py --date 2023-12-04T09:00 --n_dates 10

Writes ``single_date_calibration.html``.
"""

import argparse
import datetime
import json
import time
from pathlib import Path

import joblib
import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

import electricity_load_forecasting as elf

QUANTILES = (0.05, 0.5, 0.95)
Q_LOW, Q_HIGH = QUANTILES[0], QUANTILES[-1]
COLORS = {"raw": "#d62728", "recalibrated": "#1f77b4"}
OUTPUT = Path("single_date_calibration.html")
DEFAULT_MODELS_DIR = Path("results/single_date_models")
# different hours of the day for the sampled dates
HOUR_OFFSETS = (0, 5, 11, 17, 23, 4, 10, 16, 22, 3)


def make_op(recalibrate, args):
    return elf.make_data_op(
        horizons=tuple(range(1, 25)),
        quantile_strategy="multiple_regressors",
        quantiles=QUANTILES,
        max_iter=args.max_iter,
        recalibrate=recalibrate,
    )


def split_containing(op, date):
    """The CV fold whose test period contains ``date`` (train ends 7 days before)."""
    for split in op.skb.iter_cv_splits(environment=elf.get_env()):
        times = split["X_test"]["prediction_time"]
        if times.min() <= date <= times.max():
            return split
    raise ValueError(f"{date} is not in any CV test fold of the default range")


def sample_dates(split, n_dates):
    """Evenly spaced dates in the test period of the fold, at different hours."""
    times = split["X_test"]["prediction_time"].to_list()
    indices = np.linspace(0, len(times) - 1, n_dates).round().astype(int)
    dates = []
    for k, i in enumerate(indices):
        date = times[int(i)]
        date += datetime.timedelta(hours=HOUR_OFFSETS[k % len(HOUR_OFFSETS)])
        dates.append(date)
    return dates


def fold_config(split, args):
    """Everything that determines the fitted models, to check a cache."""
    return {
        "horizons": list(range(1, 25)),
        "quantiles": list(QUANTILES),
        "max_iter": args.max_iter,
        "train_end": str(split["X_train"]["prediction_time"].max()),
        "test_start": str(split["X_test"]["prediction_time"].min()),
        "test_end": str(split["X_test"]["prediction_time"].max()),
    }


def load_or_fit(split, args):
    """Load the fitted learners from the cache, or fit and save them."""
    models_dir = Path(args.models_dir)
    paths = {
        "raw": models_dir / "raw.joblib",
        "recalibrated": models_dir / "recalibrated.joblib",
    }
    config = fold_config(split, args)
    config_path = models_dir / "config.json"
    cached = (
        not args.refit
        and config_path.exists()
        and all(path.exists() for path in paths.values())
        and json.loads(config_path.read_text()) == config
    )
    if cached:
        print(f"loading models from {models_dir}")
        return (
            joblib.load(paths["raw"]),
            joblib.load(paths["recalibrated"]),
            config,
        )

    models_dir.mkdir(parents=True, exist_ok=True)
    learners = {}
    for recalibrate, name in ((False, "raw"), (True, "recalibrated")):
        print(f"fitting the {name} models...")
        start = time.time()
        learner = make_op(recalibrate, args).skb.make_learner().fit(split["train"])
        print(f"  fitted in {time.time() - start:.0f} s")
        joblib.dump(learner, paths[name], compress=3)
        print(f"  saved {paths[name]} ({paths[name].stat().st_size / 1e6:.0f} MB)")
        learners[name] = learner
    config_path.write_text(json.dumps(config, indent=2))
    return learners["raw"], learners["recalibrated"], config


def add_forecast(fig, prediction, row, name):
    color = COLORS[name]
    r, g, b = (int(color[i : i + 2], 16) for i in (1, 3, 5))
    fig.add_trace(
        go.Scatter(
            x=prediction["time"],
            y=prediction[f"q_{Q_HIGH}"].round(0),
            line=dict(width=0),
            hoverinfo="skip",
            showlegend=False,
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=prediction["time"],
            y=prediction[f"q_{Q_LOW}"].round(0),
            fill="tonexty",
            line=dict(width=0),
            fillcolor=f"rgba({r},{g},{b},0.2)",
            name=f"{name}: {Q_LOW}-{Q_HIGH}",
            hoverinfo="skip",
        ),
        row=row,
        col=1,
    )
    fig.add_trace(
        go.Scatter(
            x=prediction["time"],
            y=prediction["q_0.5"].round(0),
            mode="lines+markers",
            line=dict(color=color),
            marker=dict(size=5),
            name=f"{name}: median",
        ),
        row=row,
        col=1,
    )


def plot(date, raw, recalibrated, history):
    lo = date - datetime.timedelta(days=2)
    hi = date + datetime.timedelta(hours=24)
    window = history.filter((pl.col("time") >= lo) & (pl.col("time") <= hi))
    past = window.filter(pl.col("time") <= date)
    truth = window.filter(pl.col("time") >= date)
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.08,
        subplot_titles=("Raw model", "Recalibrated model"),
    )
    for row, (name, prediction) in enumerate(
        [("raw", raw), ("recalibrated", recalibrated)], 1
    ):
        fig.add_trace(
            go.Scatter(
                x=past["time"],
                y=past["load_mw"].round(0),
                mode="lines+markers",
                line=dict(color="gray", dash="dash"),
                marker=dict(size=3),
                name="history",
                showlegend=row == 1,
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=truth["time"],
                y=truth["load_mw"].round(0),
                mode="lines+markers",
                line=dict(color="black"),
                marker=dict(size=3),
                name="ground truth",
                showlegend=row == 1,
            ),
            row=row,
            col=1,
        )
        add_forecast(fig, prediction, row, name)
        fig.add_vline(
            x=date,
            line=dict(color="gray", dash="dot"),
            row=row,
            col=1,
            exclude_empty_subplots=False,
        )
    values = np.concatenate(
        [
            window["load_mw"].to_numpy(),
            raw[f"q_{Q_LOW}"].to_numpy(),
            raw[f"q_{Q_HIGH}"].to_numpy(),
            recalibrated[f"q_{Q_LOW}"].to_numpy(),
            recalibrated[f"q_{Q_HIGH}"].to_numpy(),
        ]
    )
    margin = 0.05 * (values.max() - values.min())
    y_range = [values.min() - margin, values.max() + margin]
    fig.update_yaxes(title_text="load (MW)", range=y_range)
    fig.update_yaxes(matches="y", row=2, col=1)
    fig.update_layout(
        height=850,
        margin=dict(t=80),
        hovermode="x unified",
        dragmode="zoom",
        title=f"Forecast made at {date:%Y-%m-%d %H:%M} UTC for the next 24h",
    )
    return fig


def coverage_counts(raw, recalibrated, history):
    """How many of the 24 hourly targets each interval contains."""
    truth = history.select("time", pl.col("load_mw").alias("actual"))
    counts = {}
    for name, prediction in (("raw", raw), ("recalibrated", recalibrated)):
        joined = prediction.select("time", f"q_{Q_LOW}", f"q_{Q_HIGH}").join(
            truth, on="time", how="inner"
        )
        inside = (joined["actual"] >= joined[f"q_{Q_LOW}"]) & (
            joined["actual"] <= joined[f"q_{Q_HIGH}"]
        )
        counts[name] = int(inside.sum())
    joined = (
        raw.select(
            "time",
            pl.col(f"q_{Q_LOW}").alias("raw_low"),
            pl.col(f"q_{Q_HIGH}").alias("raw_high"),
        )
        .join(
            recalibrated.select(
                "time",
                pl.col(f"q_{Q_LOW}").alias("recal_low"),
                pl.col(f"q_{Q_HIGH}").alias("recal_high"),
            ),
            on="time",
        )
        .join(truth, on="time", how="inner")
    )
    outside_raw = (joined["actual"] < joined["raw_low"]) | (
        joined["actual"] > joined["raw_high"]
    )
    inside_recal = (joined["actual"] >= joined["recal_low"]) & (
        joined["actual"] <= joined["recal_high"]
    )
    counts["gain"] = int((outside_raw & inside_recal).sum())
    counts["n"] = joined.height
    return counts


PAGE_HEAD = """<!DOCTYPE html>
<html>
<head><meta charset="utf-8"><title>Single-date calibration comparison</title>
<style>
  body {{ font-family: sans-serif; max-width: 1800px; margin: 0.5em auto; line-height: 1.5; padding: 0; }}
  table {{ border-collapse: collapse; margin: 1em 0; }}
  th, td {{ border: 1px solid #ccc; padding: 0.3em 0.7em; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
  code {{ background: #f0f0f0; padding: 0 0.2em; }}
</style></head>
<body>
<h1>Raw vs recalibrated forecasts, example prediction dates</h1>
<p>One model per horizon (1 to 24h), trained once on the data available before
<b>{train_end:%Y-%m-%d}</b> (the training set of the cross-validation fold that contains the
first date; it ends 7 days before that fold's test period). The same fitted models are then
used to predict {n_dates} dates of that fold's test period, so all the forecasts below are out of
sample. The recalibrated model is <code>calibration.Recalibrator</code> (recent
inner folds), applied per horizon.</p>
<p>For each date: the grey line is the observed load before the prediction time, the black
line the observed load after it, and the coloured band the {q_low}&ndash;{q_high} interval.
Drag a rectangle to zoom (both axes), double-click to reset.</p>
<table>
<tr><th>prediction date (UTC)</th><th>raw interval contains<br>(of {n} hours)</th>
<th>recalibrated interval contains</th><th>outside raw,<br>inside recalibrated</th></tr>
{rows}
</table>
"""

PAGE_TAIL = """</body></html>
"""


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--date", default="2023-12-04T09:00", help="first prediction date"
    )
    parser.add_argument("--n_dates", default=10, type=int)
    parser.add_argument(
        "--dates", default=None, help="comma-separated ISO dates instead"
    )
    parser.add_argument("--max_iter", default=2000, type=int)
    parser.add_argument("--models_dir", default=str(DEFAULT_MODELS_DIR))
    parser.add_argument(
        "--refit", action="store_true", help="refit even if cached models exist"
    )
    args = parser.parse_args()
    base_date = datetime.datetime.fromisoformat(args.date)
    if base_date.tzinfo is None:
        base_date = base_date.replace(tzinfo=datetime.UTC)
    base_date = base_date.astimezone(datetime.UTC)

    raw_op = make_op(False, args)
    split = split_containing(raw_op, base_date)
    train_end = split["X_train"]["prediction_time"].max()
    if args.dates:
        dates = []
        for text in args.dates.split(","):
            date = datetime.datetime.fromisoformat(text.strip())
            if date.tzinfo is None:
                date = date.replace(tzinfo=datetime.UTC)
            dates.append(date.astimezone(datetime.UTC))
    else:
        dates = sample_dates(split, args.n_dates)
    print("training ends", train_end)
    print("prediction dates:", [str(d)[:16] for d in dates])

    raw_learner, recal_learner, _ = load_or_fit(split, args)

    history = elf.resample(elf.fetch_load_mw_history())
    rows, figures = [], []
    for k, date in enumerate(dates):
        print("predicting", date)
        raw = raw_learner.predict({"start": date, "end": None})
        recalibrated = recal_learner.predict({"start": date, "end": None})
        counts = coverage_counts(raw, recalibrated, history)
        rows.append(
            f"<tr><td>{date:%Y-%m-%d %H:%M}</td><td>{counts['raw']}</td>"
            f"<td>{counts['recalibrated']}</td><td>{counts['gain']}</td></tr>"
        )
        fig = plot(date, raw, recalibrated, history)
        figures.append(
            f"<h2>Prediction date {date:%Y-%m-%d %H:%M} UTC</h2>"
            + fig.to_html(
                full_html=False,
                include_plotlyjs=(k == 0),
                default_width="100%",
            )
        )
        print(
            f"  interval contains {counts['raw']}/{counts['n']} (raw), "
            f"{counts['recalibrated']}/{counts['n']} (recalibrated); "
            f"outside raw & inside recalibrated: {counts['gain']}"
        )

    head = PAGE_HEAD.format(
        train_end=train_end,
        n_dates=len(dates),
        q_low=Q_LOW,
        q_high=Q_HIGH,
        n=counts["n"],
        rows="\n".join(rows),
    )
    OUTPUT.write_text(head + "\n".join(figures) + PAGE_TAIL, "utf-8")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
