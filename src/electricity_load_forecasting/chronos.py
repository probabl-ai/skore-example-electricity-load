"""
Chronos-2 variant of the load forecasting pipeline.

Everything that is not specific to Chronos (history loading, resampling, X/y
construction, walk-forward splitter, post-processing, scorers, plotting) is
reused from `electricity_load_forecasting`. Only the model-specific parts live
here:

- `build_covariates`: an hourly covariate table (weather + calendar) aligned
  with the resampled load history.
- `ChronosHistoryRegressor`: an sklearn-like estimator which, for each
  prediction time t, builds the context window ending at t (inclusive) and
  asks a forecaster for the next `max(horizons)` hours in a single call.
- Forecasters: `Chronos2Forecaster` (real model) and `SeasonalNaiveForecaster`
  (cheap stub with the same interface, for smoke tests).
- `make_chronos_data_op`: DataOp with the same inputs, outputs and scoring as
  `electricity_load_forecasting.make_data_op`, so reports are comparable.

Causality: the load at `prediction_time` is known when predicting (same
convention as the baseline, whose smallest lag for horizon h is h). Weather and
calendar covariates are used over the forecast horizon, as the baseline does
with its weather "forecasts"; no load value after `prediction_time` is ever
given to a forecaster.
"""

import datetime
import functools

import numpy as np
import polars as pl
import skrub
from polars import selectors as cs
from sklearn.base import BaseEstimator, RegressorMixin

from . import pipeline as elf

_CALENDAR_COLUMNS = (
    "cal_hour_of_day",
    "cal_day_of_week",
    "cal_day_of_year",
    "cal_is_holiday",
)


def build_covariates(
    load_mw_history,
    city_weather_fetcher=elf.fetch_city_weather,
    cities="all",
    temperature_only=True,
    calendar=True,
):
    """
    Build the covariates table used by `ChronosHistoryRegressor`.

    Returns a dataframe with exactly the rows of `load_mw_history` (same
    hourly `time` column, same order) and one float32 column per covariate.
    Weather columns are named like in the baseline pipeline
    (`weather_<variable>_<city>`) and are NULL where the weather data does not
    cover the time grid. Calendar columns come from
    `elf.add_calendar_and_holidays`; `cal_year` is dropped as a trend
    covariate is meaningless for a zero-shot model.

    `cities` is "all" or a sequence of city names (possibly empty).
    """
    grid = load_mw_history.select("time")
    if isinstance(cities, str):
        assert cities == "all"
        cities = elf._ALL_CITIES
    covariates = grid
    for city in cities:
        weather = city_weather_fetcher(city).with_columns(
            pl.col("time").dt.cast_time_unit("us")
        )
        if temperature_only:
            weather = weather.select(pl.col("time"), cs.matches(".*temperature.*"))
        weather = weather.select(
            pl.col("time"),
            (~cs.by_name("time")).as_expr().name.map(f"weather_{{}}_{city}".format),
        )
        covariates = covariates.join(
            weather.collect(), on="time", how="left", maintain_order="left"
        )
    if calendar:
        cal = elf.add_calendar_and_holidays(
            grid.select(pl.col("time").alias("target_time"))
        )
        covariates = covariates.with_columns(
            cal.select(pl.col(c).cast(pl.Float32) for c in _CALENDAR_COLUMNS)
        )
    assert covariates.height == grid.height
    return covariates.with_columns(cs.exclude("time").as_expr().cast(pl.Float32))


def _hourly_positions(prediction_time, grid_time):
    """Index in the hourly `grid_time` of each prediction time."""
    seconds = prediction_time.dt.epoch("s").to_numpy()
    start = grid_time.dt.epoch("s").item(0)
    offsets = seconds - start
    if (offsets % 3600).any():
        raise ValueError("prediction times must be on the full hour")
    positions = offsets // 3600
    if len(positions) and (positions.min() < 0 or positions.max() >= len(grid_time)):
        raise ValueError(
            "prediction times fall outside of the range of the load history "
            f"({grid_time.min()} to {grid_time.max()})"
        )
    return positions


class ChronosHistoryRegressor(RegressorMixin, BaseEstimator):
    """
    Zero-shot multi-horizon forecaster that builds causal windows from history.

    For each row of X (column `prediction_time`) the forecaster receives the
    last `context_hours` of load up to and including the prediction time,
    covariates over that same window and over the next `max(horizons)` hours,
    and returns forecasts for those hours; the steps matching `horizons` are
    kept. `fit` does nothing (the model is pretrained).

    The output has the same layout as the baseline estimators' outputs after
    `concat_horizons`: columns `"{h}h"` (median forecast) if `quantiles` is
    None, else `"{h}h__q_{q}"` for each horizon and (sorted) quantile.

    Parameters
    ----------
    forecaster : object
        Must expose `predict(inputs, prediction_length, quantile_levels)`
        returning an array of shape (len(inputs), prediction_length,
        len(quantile_levels)). Each input is a dict with the keys `target`
        (1-d float array, NaN for missing hours) and, when covariates are
        used, `past_covariates` and `future_covariates` (dicts of 1-d float
        arrays, NaN for missing values). It should be an estimator holding
        only parameters (see `Chronos2Forecaster`) as it gets cloned.
    horizons : tuple of int
        Horizons in hours, same as given to `elf.get_X_y`.
    quantiles : tuple of float or None
        Quantile levels to output, or None for a single median forecast.
    context_hours : int
        Maximum number of past hours given to the forecaster.
    chunk_size : int
        Number of rows for which windows are materialized and forecast per
        forecaster call.
    verbose : bool
        Show a progress bar over chunks.
    """

    def __init__(
        self,
        forecaster=None,
        horizons=(1, 12, 24),
        quantiles=None,
        context_hours=1024,
        chunk_size=512,
        verbose=False,
    ):
        self.forecaster = forecaster
        self.horizons = horizons
        self.quantiles = quantiles
        self.context_hours = context_hours
        self.chunk_size = chunk_size
        self.verbose = verbose

    def fit(self, X, y=None):
        if self.forecaster is None:
            raise ValueError("forecaster must be provided")
        if y is not None:
            expected = [f"{h}h" for h in self.horizons]
            if list(y.columns) != expected:
                raise ValueError(
                    f"y columns {list(y.columns)} do not match horizons {expected}"
                )
        return self

    def _inputs(self, positions, load, covariates, prediction_length):
        for i in positions:
            lo = max(0, i - self.context_hours + 1)
            item = {"target": load[lo : i + 1]}
            if covariates is not None:
                values, names = covariates
                item["past_covariates"] = {
                    n: values[lo : i + 1, j] for j, n in enumerate(names)
                }
                item["future_covariates"] = {
                    n: values[i + 1 : i + 1 + prediction_length, j]
                    for j, n in enumerate(names)
                }
            yield item

    def predict(self, X, history, covariates=None):
        """
        Parameters
        ----------
        X : dataframe with column `prediction_time`.
        history : dataframe [time, load_mw] on a regular hourly grid
            (output of `elf.resample`). May contain data after the prediction
            times; it is never used.
        covariates : dataframe or None
            Output of `build_covariates` for the same `history`.
        """
        if self.forecaster is None:
            raise ValueError("forecaster must be provided")
        horizons = tuple(self.horizons)
        n_steps = max(horizons)
        quantiles = None if self.quantiles is None else sorted(self.quantiles)
        levels = [0.5] if quantiles is None else quantiles

        grid_time = history["time"]
        steps = grid_time.diff().drop_nulls()
        if not (steps == datetime.timedelta(hours=1)).all():
            raise ValueError("history must be on a regular hourly grid")
        load = history["load_mw"].cast(pl.Float32).to_numpy()
        cov = None
        if covariates is not None and covariates.width > 1:
            if not covariates["time"].equals(grid_time):
                raise ValueError("covariates must be aligned with history")
            names = [c for c in covariates.columns if c != "time"]
            values = covariates.select(names).to_numpy().astype(np.float32)
            # NaN padding so that windows over the horizon never run past the end
            padding = np.full((n_steps, len(names)), np.nan, dtype=np.float32)
            cov = (np.vstack([values, padding]), names)

        positions = _hourly_positions(X["prediction_time"], grid_time)
        chunks = [
            positions[i : i + self.chunk_size]
            for i in range(0, len(positions), self.chunk_size)
        ]
        if self.verbose:
            from tqdm import tqdm

            chunks = tqdm(chunks, desc="forecasting", unit="chunk")
        results = []
        for chunk in chunks:
            inputs = list(self._inputs(chunk, load, cov, n_steps))
            out = np.asarray(
                self.forecaster.predict(
                    inputs, prediction_length=n_steps, quantile_levels=levels
                ),
                dtype=np.float64,
            )
            if out.shape != (len(chunk), n_steps, len(levels)):
                raise ValueError(f"unexpected forecaster output shape {out.shape}")
            results.append(out)
        forecasts = (
            np.concatenate(results) if results else np.empty((0, n_steps, len(levels)))
        )

        columns = {}
        for h in horizons:
            for k, q in enumerate(levels):
                name = f"{h}h" if quantiles is None else f"{h}h__q_{q}"
                columns[name] = forecasts[:, h - 1, k]
        return pl.DataFrame(columns)


class SeasonalNaiveForecaster(BaseEstimator):
    """
    Stub forecaster for smoke tests: repeat the value observed one period
    (default one week) before the target time. Quantiles are the point forecast
    scaled by `1 + spread * (q - 0.5)`, which is monotonic but meaningless.
    Ignores covariates.
    """

    def __init__(self, period=24 * 7, spread=0.2):
        self.period = period
        self.spread = spread

    def predict(self, inputs, prediction_length, quantile_levels):
        steps = np.arange(1, prediction_length + 1)
        n_periods_back = -(-steps // self.period)  # ceil
        levels = np.asarray(quantile_levels)
        out = np.empty((len(inputs), prediction_length, len(levels)))
        for i, item in enumerate(inputs):
            context = np.asarray(item["target"], dtype=np.float64)
            n = len(context)
            finite = np.flatnonzero(np.isfinite(context))
            fallback = context[finite[-1]] if len(finite) else np.nan
            idx = n - 1 + steps - n_periods_back * self.period
            point = np.where(
                (idx >= 0) & (idx < n), context[np.clip(idx, 0, n - 1)], np.nan
            )
            point = np.where(np.isfinite(point), point, fallback)
            out[i] = point[:, None] * (1 + self.spread * (levels - 0.5))[None, :]
        return out


@functools.lru_cache(maxsize=2)
def _load_chronos_pipeline(model, device_map, dtype):
    import torch
    from chronos import Chronos2Pipeline

    if device_map is None:
        device_map = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {} if dtype is None else {"dtype": getattr(torch, dtype)}
    return Chronos2Pipeline.from_pretrained(model, device_map=device_map, **kwargs)


class Chronos2Forecaster(BaseEstimator):
    """
    Zero-shot Chronos-2 (`chronos-forecasting` package).

    Holds parameters only: the pipeline is loaded lazily and cached at module
    level, so cloning (done by skrub / skore for each fit) does not copy or
    reload the model.

    Parameters
    ----------
    model : str
        Hugging Face model id or local path, e.g. "amazon/chronos-2" or the
        smaller "autogluon/chronos-2-small".
    device_map : str or None
        None selects cuda when available, else cpu.
    dtype : str or None
        Name of a torch dtype, e.g. "bfloat16". None keeps the default.
    batch_size : int
        Chronos batch size. Note it counts the target and each covariate as
        separate series.
    cross_learning : bool
        Share information across all series of a call. Results then depend on
        batch composition; leave False for independent predictions.
    """

    def __init__(
        self,
        model="amazon/chronos-2",
        device_map=None,
        dtype=None,
        batch_size=256,
        cross_learning=False,
    ):
        self.model = model
        self.device_map = device_map
        self.dtype = dtype
        self.batch_size = batch_size
        self.cross_learning = cross_learning

    def predict(self, inputs, prediction_length, quantile_levels):
        pipeline = _load_chronos_pipeline(self.model, self.device_map, self.dtype)
        quantiles, _ = pipeline.predict_quantiles(
            inputs,
            prediction_length=prediction_length,
            quantile_levels=list(quantile_levels),
            batch_size=self.batch_size,
            cross_learning=self.cross_learning,
        )
        # each element: (n_variates=1, prediction_length, n_quantiles)
        return np.stack([q[0].float().cpu().numpy() for q in quantiles])


def make_chronos_data_op(
    horizons=(1, 12, 24),
    forecaster=None,
    quantiles=None,
    context_hours=1024,
    cities="all",
    temperature_only=True,
    calendar=True,
    chunk_size=512,
    verbose=False,
):
    """
    Same contract as `elf.make_data_op`, with a Chronos-style forecaster.

    The environment keys are the same (`start`, `end`,
    `load_mw_history_fetcher`, `city_weather_fetcher`) and so are the output
    and the scoring (`neg_mape`, plus pinball when `quantiles` is given), so
    results can be compared with the baseline using the same splits.
    `quantiles=None` outputs the median, like the baseline's mean models.

    `cities=()` with `calendar=False` gives a target-only (univariate) model.
    """
    if forecaster is None:
        forecaster = Chronos2Forecaster()
    quantile_regression = quantiles is not None

    range_start = skrub.var("start").skb.set_description(
        "The first time at which a prediction is made."
    )
    range_end = skrub.var("end", None, becomes_default=True).skb.set_description(
        "The last time at which a prediction is made."
    )
    load_mw_history_fetcher = skrub.var(
        "load_mw_history_fetcher", elf.fetch_load_mw_history, becomes_default=True
    )
    city_weather_fetcher = skrub.var(
        "city_weather_fetcher", elf.fetch_city_weather, becomes_default=True
    )
    prediction_time = skrub.deferred(elf.time_range)(range_start, range_end)
    load_mw_history = (
        load_mw_history_fetcher()
        .skb.apply_func(elf.resample)
        .skb.set_description("Historical load data on a regular 1h time grid.")
    )
    X_y = prediction_time.skb.apply_func(elf.get_X_y, load_mw_history, horizons)
    X = X_y["X"].skb.mark_as_X(cv=elf.TimeSeriesSplitter())
    y = X_y["y"].skb.mark_as_y()

    covariates = None
    if len(cities) > 0 or calendar:
        covariates = (
            skrub.deferred(build_covariates)(
                load_mw_history,
                city_weather_fetcher,
                cities=cities,
                temperature_only=temperature_only,
                calendar=calendar,
            )
            .skb.set_name("covariates")
            .skb.set_description(
                "Hourly weather and calendar covariates aligned with the history."
            )
        )

    estimator = ChronosHistoryRegressor(
        forecaster=forecaster,
        horizons=horizons,
        quantiles=quantiles,
        context_hours=context_hours,
        chunk_size=chunk_size,
        verbose=verbose,
    )
    all_pred = X.skb.apply(
        estimator,
        y=y,
        predict_kwargs={"history": load_mw_history, "covariates": covariates},
    ).skb.set_name("pred_multi_horizon")

    multi_horizon_pred = all_pred.skb.apply_func(
        elf.post_process,
        prediction_time,
        range_end,
        quantile_regression=quantile_regression,
    ).skb.with_scoring(
        functools.partial(elf.neg_mape_scorer, quantile_regression=quantile_regression)
    )
    if not quantile_regression:
        return multi_horizon_pred
    return multi_horizon_pred.skb.with_scoring(elf.pinball_scorer)
