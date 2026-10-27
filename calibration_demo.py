"""Toy demonstration of calibration.Recalibrator.

Run from the repository root:

    python calibration_demo.py

Writes ``calibration_demo.html``: a self-contained page (plotly embedded)
explaining the recalibration and showing the plots.
"""

import datetime
import html
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots
from sklearn.metrics import mean_pinball_loss
from sklearn.model_selection import TimeSeriesSplit

import electricity_load_forecasting as elf
from calibration import Recalibrator
from quantile_regressor import HGBQuantileRegressor

QUANTILES = (0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95)
Q_LOW, Q_HIGH = 0.05, 0.95
N_BINS = 8
OUTPUT = Path("calibration_demo.html")


# ---------------------------------------------------------------------------
# Toy data: hourly "load" with daily + yearly seasonality, an observable
# autocorrelated "weather" signal, and noise proportional to the load level
# (heteroscedastic: peaks are harder to predict than troughs).
# ---------------------------------------------------------------------------
def make_toy_data(n_days=1460, seed=0):
    rng = np.random.default_rng(seed)
    n = 24 * n_days
    hours = np.arange(n)
    hour_of_day = hours % 24
    day_of_year = (hours // 24) % 365
    weather = np.zeros(n)
    for i in range(1, n):
        weather[i] = 0.97 * weather[i - 1] + 0.25 * rng.normal()
    daily = np.sin(2 * np.pi * (hour_of_day - 6) / 24)
    yearly = np.cos(2 * np.pi * day_of_year / 365)
    level = 65 + 15 * daily + 10 * yearly + 4 * weather
    y = level + rng.normal(size=n) * 0.08 * level
    X = np.column_stack(
        [
            np.sin(2 * np.pi * hour_of_day / 24),
            np.cos(2 * np.pi * hour_of_day / 24),
            np.sin(2 * np.pi * day_of_year / 365),
            np.cos(2 * np.pi * day_of_year / 365),
            weather,
        ]
    )
    start = datetime.datetime(2022, 1, 1)
    time = [start + datetime.timedelta(hours=int(h)) for h in hours]
    return time, X, y


class OverconfidentQuantileRegressor(HGBQuantileRegressor):
    """HGB quantile regressor whose intervals are deliberately too narrow.

    The distance between each quantile and the median is shrunk, increasingly
    so for high predicted loads (factor 0.95 at 40 MW down to 0.4 at 90 MW).
    This mimics the kind of miscalibration a real model can show (e.g.
    under-estimated uncertainty at peaks) so that the effect of the
    recalibration is easy to see.
    """

    def predict(self, X):
        pred = super().predict(X)
        median = pred["q_0.5"].to_numpy()
        factor = np.interp(median, [40, 90], [0.95, 0.4])
        return pl.DataFrame(
            {c: median + factor * (pred[c].to_numpy() - median) for c in pred.columns}
        )


# ---------------------------------------------------------------------------
# Metrics and plots
# ---------------------------------------------------------------------------
def levels(pred):
    return {c: float(c.removeprefix("q_")) for c in pred.columns}


def empirical_coverage(y, pred):
    """Fraction of y at or below each predicted quantile (reliability curve)."""
    return {q: float((y <= pred[c].to_numpy()).mean()) for c, q in levels(pred).items()}


def interval_metrics(y, pred):
    """Coverage and mean width of the [Q_LOW, Q_HIGH] interval, via elf."""
    y_df = pl.DataFrame({"1h": y})
    pred_df = pl.DataFrame(
        {
            f"1h__q_{q}": pred[f"q_{q}"].to_numpy()
            for q in (Q_LOW, Q_HIGH)
        }
    )
    kwargs = dict(q_low=Q_LOW, q_high=Q_HIGH)
    # each returns a one-entry dict (a single "1h" column here)
    (coverage,) = elf.coverage(y_df, pred_df, **kwargs).values()
    (width,) = elf.mean_width(y_df, pred_df, **kwargs).values()
    return float(coverage), float(width)


def mean_pinball(y, pred):
    return float(
        np.mean(
            [
                mean_pinball_loss(y, pred[c].to_numpy(), alpha=q)
                for c, q in levels(pred).items()
            ]
        )
    )


COLORS = {"raw": "#d62728", "corrected": "#1f77b4", "true": "black", "ideal": "gray"}


def plot_time_series(time, y, raw, corrected, start, stop):
    sl = slice(start, stop)
    t = time[sl]
    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        shared_yaxes=True,
        vertical_spacing=0.06,
        subplot_titles=("Raw model", "Recalibrated model"),
    )
    for row, (name, pred) in enumerate([("raw", raw), ("corrected", corrected)], 1):
        color = COLORS[name]
        lo = pred[f"q_{Q_LOW}"].to_numpy()[sl]
        hi = pred[f"q_{Q_HIGH}"].to_numpy()[sl]
        fig.add_trace(
            go.Scatter(x=t, y=hi, line=dict(width=0), showlegend=False, hoverinfo="skip"),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=lo,
                fill="tonexty",
                line=dict(width=0),
                fillcolor="rgba(214,39,40,0.25)"
                if name == "raw"
                else "rgba(31,119,180,0.25)",
                name=f"{name}: {Q_LOW}-{Q_HIGH} quantiles",
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=pred["q_0.5"].to_numpy()[sl],
                line=dict(color=color, width=1.5),
                name=f"{name}: median",
            ),
            row=row,
            col=1,
        )
        fig.add_trace(
            go.Scatter(
                x=t,
                y=y[sl],
                mode="lines+markers",
                line=dict(color=COLORS["true"], width=1),
                marker=dict(size=3),
                name="true",
                showlegend=row == 1,
            ),
            row=row,
            col=1,
        )
    fig.update_yaxes(title_text="load (MW)")
    fig.update_layout(height=650, hovermode="x unified", title="Test period (two weeks)")
    return fig


def plot_reliability(y, raw, corrected):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            line=dict(color=COLORS["ideal"], dash="dash"),
            name="perfect calibration",
        )
    )
    for name, pred in [("raw", raw), ("corrected", corrected)]:
        cov = empirical_coverage(y, pred)
        fig.add_trace(
            go.Scatter(
                x=list(cov),
                y=list(cov.values()),
                mode="lines+markers",
                line=dict(color=COLORS[name]),
                name=name,
            )
        )
    fig.update_layout(
        title="Reliability diagram (test set)",
        xaxis_title="nominal quantile level",
        yaxis_title="observed fraction of y_true below the prediction",
        xaxis=dict(range=[0, 1]),
        yaxis=dict(range=[0, 1], scaleanchor="x", scaleratio=1),
        width=620,
        height=620,
    )
    return fig


def plot_reliability_by_level(y, raw, corrected, column="q_0.95", n_bins=N_BINS):
    """Coverage of one quantile, as a function of the predicted value itself."""
    tau = float(column.removeprefix("q_"))
    p = raw[column].to_numpy()
    edges = np.unique(np.quantile(p, np.linspace(0, 1, n_bins + 1)))
    bin_ids = np.searchsorted(edges[1:-1], p, side="right")
    fig = go.Figure()
    fig.add_hline(y=tau, line=dict(color=COLORS["ideal"], dash="dash"))
    for name, pred in [("raw", raw), ("corrected", corrected)]:
        q = pred[column].to_numpy()
        xs, ys = [], []
        for b in np.unique(bin_ids):
            in_bin = bin_ids == b
            xs.append(float(p[in_bin].mean()))
            ys.append(float((y[in_bin] <= q[in_bin]).mean()))
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="lines+markers", line=dict(color=COLORS[name]), name=name
            )
        )
    fig.update_layout(
        title=f"Reliability of {column} by bin of raw predicted value (test set)",
        xaxis_title=f"raw {column} prediction (MW)",
        yaxis_title=f"observed fraction of y_true below {column}",
        yaxis=dict(range=[0, 1]),
        height=450,
    )
    return fig


def plot_shifts(recalibrator):
    fig = go.Figure()
    colors = {f"q_{Q_LOW}": "#2ca02c", f"q_{Q_HIGH}": "#ff7f0e"}
    for column, color in colors.items():
        summary = recalibrator.bin_summary_[column]
        x = summary["mean_prediction"].to_numpy()
        grid = np.linspace(x.min(), x.max(), 100)
        fig.add_trace(
            go.Scatter(
                x=x,
                y=summary["shift"].to_numpy(),
                mode="markers",
                marker=dict(color=color, size=10),
                customdata=np.column_stack(
                    [summary["count"].to_numpy(), summary["empirical_coverage"].to_numpy()]
                ),
                hovertemplate="mean prediction %{x:.1f}<br>shift %{y:.2f}"
                "<br>%{customdata[0]} points"
                "<br>coverage before: %{customdata[1]:.3f}<extra></extra>",
                name=f"{column}: measured shift per bin",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=grid,
                y=recalibrator.shift_models_[column].predict(grid[:, None]),
                mode="lines",
                line=dict(color=color),
                name=f"{column}: fitted spline + ridge",
            )
        )
    fig.update_layout(
        title="Learned corrections (from out-of-fold predictions on the training set)",
        xaxis_title="predicted value (MW)",
        yaxis_title="shift to add (MW)",
        height=450,
    )
    return fig


# ---------------------------------------------------------------------------
# HTML
# ---------------------------------------------------------------------------
def metrics_table(rows):
    header = (
        "<tr><th></th><th>coverage of the 90% interval<br>(target 0.90)</th>"
        "<th>mean 90% interval width (MW)</th><th>mean pinball loss (all quantiles)</th></tr>"
    )
    body = "".join(
        f"<tr><td>{html.escape(name)}</td><td>{cov:.3f}</td><td>{width:.1f}</td>"
        f"<td>{pinball:.3f}</td></tr>"
        for name, cov, width, pinball in rows
    )
    return f"<table>{header}{body}</table>"


PAGE = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Recalibrating a quantile regressor</title>
<style>
  body {{ font-family: sans-serif; max-width: 1000px; margin: 2em auto; line-height: 1.5; padding: 0 1em; }}
  table {{ border-collapse: collapse; margin: 1em 0; }}
  th, td {{ border: 1px solid #ccc; padding: 0.4em 0.8em; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
  code {{ background: #f0f0f0; padding: 0 0.2em; }}
  .note {{ background: #fff8e1; border-left: 4px solid #f0b400; padding: 0.5em 1em; }}
</style>
</head>
<body>
<h1>Recalibrating a quantile regressor</h1>

<h2>The problem</h2>
<p>A quantile regressor claims that, given <code>X</code>, its output <code>q_0.95</code> is the
value (in MW) below which the load will fall 95% of the time. If that claim is true the
model is <em>calibrated</em>: over many predictions, about 95% of the observed values end up
below <code>q_0.95</code>. In practice models are often miscalibrated, typically too
confident (intervals too narrow), and the error may depend on the predicted level (here,
the peaks are the worst).</p>

<h2>The correction (<code>calibration.Recalibrator</code>)</h2>
<p>The recalibrator wraps a quantile regressor. For each predicted quantile level
&tau; (each column <code>q_&tau;</code>) independently:</p>
<ol>
  <li><b>Out-of-fold predictions.</b> Using a walk-forward time-series CV on the
  training data (with a gap between train and test), the wrapped model is fit on the
  past and predicts the future of each fold. We concatenate the predictions
  <code>p</code> with the true values <code>y</code>. These predictions are honest, i.e.
  not made on data the model was trained on.</li>
  <li><b>Bin by predicted value.</b> The out-of-fold predictions are split into
  {n_bins} bins with the same number of points. Within a bin the model claims that
  <code>p</code> is (about) the &tau;-quantile of <code>y</code>.</li>
  <li><b>Measure the actual quantile in each bin.</b> We compute the actual &tau;-quantile
  of the residuals <code>y &minus; p</code> over the points in the bin. This is a number
  in MW: the <em>shift</em> that would make the bin calibrated. If the model is calibrated
  the shift is about 0; if the intervals are too narrow, the shift for
  <code>q_0.95</code> is positive (the prediction should be higher) and the shift for
  <code>q_0.05</code> is negative.</li>
  <li><b>Smooth.</b> The per-bin shifts are noisy, especially for tail quantiles. We fit
  a small model (<code>SplineTransformer</code> with 5 basis functions followed by
  <code>RidgeCV</code>, weighted by the number of points in each bin) that maps a
  predicted value to a shift. The figure below shows the bin measurements and the fitted
  function.</li>
  <li><b>Apply.</b> The wrapped model is refit on all the training data. At prediction time,
  each predicted quantile is corrected by adding the shift predicted for its value:
  <code>q_&tau;_corrected = q_&tau; + shift_&tau;(q_&tau;)</code>. The quantiles are finally
  sorted so that they cannot cross.</li>
</ol>
<p class="note">The correction can only fix a systematic, level-dependent miscalibration. It
does not make the model sharper: if intervals have to be wider to be honest, they will be
wider. It also assumes the calibration measured in the past still holds in the future.</p>

<h2>The toy example</h2>
<p>Hourly synthetic load over {n_days} days: a daily cycle, a yearly cycle, an observable
autocorrelated &quot;weather&quot; signal and Gaussian noise whose standard deviation is 8% of
the load level. The first {train_pct}% is used for training (and for the recalibrator's internal
CV), the rest as the test set. The base model is a gradient boosting quantile regressor (one HGB
per quantile in {quantiles}) whose intervals are then <b>deliberately shrunk towards the
median</b>, more so for high predicted loads (factor 0.95 at 40&nbsp;MW down to 0.4 at 90&nbsp;MW),
to simulate a model that under-estimates uncertainty, mostly at peaks.</p>
<p>The recalibrator uses a walk-forward CV with 4 test folds of 90 days and a 7-day gap
between train and test. Its out-of-fold models must be trained on enough data to be
representative of the final model (here, at least ~2 years, so that the yearly cycle is seen).
If the first folds are trained on very little data, their errors are larger than those of the
final model, and the correction over-shoots (we observed 94% coverage for a 90% interval with a
too-short initial training window).</p>

<h2>Learned corrections</h2>
<p>For each bin of predicted values: the measured shift (dots, hover for details) and the
smooth function used for the correction (line). The upper quantile needs to be raised and
the lower one lowered, increasingly so for high loads.</p>
{shifts}

<h2>Predictions over time</h2>
<p>Raw model on top, recalibrated model below (same y-axis range). The band is the
{q_low}&ndash;{q_high} interval and the line is the median. The raw band is too narrow and
true values often fall outside, especially at the daily peaks; after correction the band
is wider where needed.</p>
{timeseries}

<h2>Reliability diagrams</h2>
<p>A reliability diagram compares each nominal quantile level with the fraction of
observed values actually below the predicted quantile. The dashed diagonal is perfect
calibration. Curves below the diagonal in the upper half (and above it in the lower half)
indicate over-confident intervals.</p>
{reliability}
<p>The same check for one quantile only, but split by the predicted value: this is the
quantity the recalibrator corrects. The raw model is close to the target 0.95 at low loads
but falls well short at high loads; the corrected model is close to the target everywhere.</p>
{reliability_by_level}

<h2>Metrics on the test set</h2>
{metrics}
<p>Coverage and width are computed with <code>electricity_load_forecasting.coverage</code> and
<code>mean_width</code>. The pinball loss (lower is better) is averaged over all the
quantile levels. Recalibration restores the nominal coverage at the cost of
wider intervals (the honest width for this model); the pinball loss, which rewards both
calibration and sharpness, also improves.</p>

</body>
</html>
"""


def main():
    time, X, y = make_toy_data()
    n_train = int(0.7 * len(y))
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]
    time_test = time[n_train:]

    base = OverconfidentQuantileRegressor(
        quantiles=QUANTILES, hgb_params=dict(max_iter=100, random_state=0)
    )
    # walk-forward CV: 4 test folds of 90 days, 7-day gap before each
    cv = TimeSeriesSplit(n_splits=4, test_size=24 * 90, gap=24 * 7)
    recalibrator = Recalibrator(base, cv=cv, n_bins=N_BINS).fit(X_train, y_train)
    raw = recalibrator.estimator_.predict(X_test)
    corrected = recalibrator.predict(X_test)

    rows = []
    for name, pred in [("raw", raw), ("recalibrated", corrected)]:
        cov, width = interval_metrics(y_test, pred)
        print(name, {q: round(v, 3) for q, v in empirical_coverage(y_test, pred).items()})
        rows.append((name, cov, width, mean_pinball(y_test, pred)))
        print(f"{name:13s} coverage={cov:.3f} width={width:.1f} pinball={rows[-1][3]:.3f}")

    # two weeks, starting about a month into the test period
    start = 24 * 30
    stop = start + 24 * 14
    figures = {
        "shifts": plot_shifts(recalibrator),
        "timeseries": plot_time_series(time_test, y_test, raw, corrected, start, stop),
        "reliability": plot_reliability(y_test, raw, corrected),
        "reliability_by_level": plot_reliability_by_level(y_test, raw, corrected),
    }
    parts = {}
    for i, (key, fig) in enumerate(figures.items()):
        parts[key] = fig.to_html(
            full_html=False,
            include_plotlyjs=True if i == 0 else False,
            default_width="100%" if key != "reliability" else None,
        )
    OUTPUT.write_text(
        PAGE.format(
            n_bins=N_BINS,
            n_days=len(y) // 24,
            train_pct=round(100 * n_train / len(y)),
            quantiles=", ".join(map(str, QUANTILES)),
            q_low=Q_LOW,
            q_high=Q_HIGH,
            metrics=metrics_table(rows),
            **parts,
        )
    )
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
