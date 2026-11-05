"""
TabICLForecaster variant of the load forecasting pipeline.

Everything that is not specific to TabICLForecaster is reused from
`electricity_load_forecasting`: history loading, resampling, X/y construction,
the walk-forward splitter, covariate building, post-processing and scoring.
This module only holds:

- `TabICLForecastRegressor`: an sklearn-like estimator which, for each
  prediction time t, builds the causal context ending at t and the future
  horizon with known covariates, then delegates to `tabicl.TabICLForecaster`.
- `make_tabicl_forecast_data_op`: same contract as `elf.make_data_op` for a
  quantile forecast, so the same splits and metrics can be used.

The load at `prediction_time` is known when predicting (same convention as the
baseline); no load value after `prediction_time` is ever given to the model.
Weather and calendar covariates are used over the forecast horizon, as in the
Chronos and baseline pipelines.
"""

import datetime
import functools

import numpy as np
import pandas as pd
import polars as pl
import skrub
from sklearn.base import BaseEstimator, RegressorMixin

from . import pipeline as elf
from .chronos import _hourly_positions, build_covariates


def _naive_timestamps(grid_time):
    return pd.to_datetime(grid_time.to_list(), utc=True).tz_localize(None)


def _batch_frames(
    positions,
    times,
    load,
    cov_values,
    cov_names,
    context_hours,
    n_steps,
):
    lengths = [i - max(0, i - context_hours + 1) + 1 for i in positions]
    total_context = sum(lengths)

    context_item = np.repeat(np.arange(len(positions)), lengths)
    context_time = np.concatenate(
        [times[max(0, i - context_hours + 1) : i + 1].to_numpy() for i in positions]
    )
    context_target = np.concatenate(
        [load[max(0, i - context_hours + 1) : i + 1] for i in positions]
    )

    future_item = np.repeat(np.arange(len(positions)), n_steps)
    future_time = np.concatenate(
        [
            (times[i] + pd.to_timedelta(np.arange(1, n_steps + 1), unit="h")).to_numpy()
            for i in positions
        ]
    )
    future_target = np.full(len(positions) * n_steps, np.nan, dtype=np.float32)

    context = {
        "item_id": context_item,
        "timestamp": context_time,
        "target": context_target,
    }
    future = {
        "item_id": future_item,
        "timestamp": future_time,
        "target": future_target,
    }

    if cov_values is not None:
        context_covariates = {
            name: np.concatenate(
                [
                    cov_values[max(0, i - context_hours + 1) : i + 1, k]
                    for i in positions
                ]
            )
            for k, name in enumerate(cov_names)
        }
        future_covariates = {
            name: np.full(len(positions) * n_steps, np.nan, dtype=np.float32)
            for name in cov_names
        }
        for j, i in enumerate(positions):
            indices = np.arange(i + 1, i + 1 + n_steps)
            valid = indices < len(times)
            if not valid.any():
                continue
            start = j * n_steps
            for k, name in enumerate(cov_names):
                future_covariates[name][start : start + n_steps][valid] = cov_values[
                    indices[valid], k
                ]
        context.update(context_covariates)
        future.update(future_covariates)

    assert len(context_item) == total_context
    return pd.DataFrame(context), pd.DataFrame(future)


class TabICLForecastRegressor(RegressorMixin, BaseEstimator):
    """Zero-shot multi-horizon forecaster based on `tabicl.TabICLForecaster`.

    For each row of X (column `prediction_time`) the model receives the last
    `max_context_length` hours of load up to and including the prediction time,
    plus known covariates over that window and over the next `max(horizons)`
    hours. It returns quantile forecasts for the requested horizons.

    Parameters
    ----------
    horizons : tuple of int
        Horizons in hours, same as given to `elf.get_X_y`.
    quantiles : tuple of float
        Quantile levels to output.
    max_context_length : int
        Maximum number of past hours given to TabICLForecaster.
    chunk_size : int
        Number of prediction times processed in one TabICLForecaster call.
    tabicl_config : dict or None
        Passed to TabICLForecaster's `tabicl_config` (e.g. `{"device": "cpu"}`).
    temporal_features : list or None
        Passed to TabICLForecaster. None uses its default temporal features.
    point_estimate : {"mean", "median"}
        Passed to TabICLForecaster.
    """

    def __init__(
        self,
        horizons=(1, 12, 24),
        quantiles=(0.05, 0.5, 0.95),
        max_context_length=4096,
        chunk_size=8,
        tabicl_config=None,
        temporal_features=None,
        point_estimate="mean",
    ):
        self.horizons = horizons
        self.quantiles = quantiles
        self.max_context_length = max_context_length
        self.chunk_size = chunk_size
        self.tabicl_config = tabicl_config
        self.temporal_features = temporal_features
        self.point_estimate = point_estimate

    def fit(self, X, y=None):
        if y is not None:
            expected = [f"{h}h" for h in self.horizons]
            if list(y.columns) != expected:
                raise ValueError(
                    f"y columns {list(y.columns)} do not match horizons {expected}"
                )
        from tabicl import TabICLForecaster

        self.forecaster_ = TabICLForecaster(
            max_context_length=self.max_context_length,
            temporal_features=self.temporal_features,
            point_estimate=self.point_estimate,
            tabicl_config=self.tabicl_config,
        )
        return self

    def predict(self, X, history, covariates=None):
        if not hasattr(self, "forecaster_"):
            raise ValueError("fit must be called before predict")

        horizons = tuple(self.horizons)
        n_steps = max(horizons)
        quantiles = tuple(sorted(self.quantiles))

        grid_time = history["time"]
        steps = grid_time.diff().drop_nulls()
        if not (steps == datetime.timedelta(hours=1)).all():
            raise ValueError("history must be on a regular hourly grid")
        load = history["load_mw"].cast(pl.Float64).to_numpy()
        times = _naive_timestamps(grid_time)

        cov_values = None
        cov_names = []
        if covariates is not None and covariates.width > 1:
            if not covariates["time"].equals(grid_time):
                raise ValueError("covariates must be aligned with history")
            cov_names = [c for c in covariates.columns if c != "time"]
            cov_values = covariates.select(cov_names).to_numpy().astype(np.float32)

        positions = _hourly_positions(X["prediction_time"], grid_time)
        chunks = [
            positions[i : i + self.chunk_size]
            for i in range(0, len(positions), self.chunk_size)
        ]
        results = []
        for chunk in chunks:
            context, future = _batch_frames(
                chunk,
                times,
                load,
                cov_values,
                cov_names,
                self.max_context_length,
                n_steps,
            )
            prediction = self.forecaster_.predict_df(
                context,
                future_df=future,
                quantiles=list(quantiles),
            )
            prediction = prediction.reset_index().sort_values(["item_id", "timestamp"])
            if len(prediction) != len(chunk) * n_steps:
                raise ValueError(
                    f"unexpected forecast row count {len(prediction)} for {len(chunk)} "
                    f"items and {n_steps} steps"
                )
            values = prediction[list(quantiles)].to_numpy()
            results.append(values.reshape(len(chunk), n_steps, len(quantiles)))
        forecasts = (
            np.concatenate(results)
            if results
            else np.empty((0, n_steps, len(quantiles)))
        )

        columns = {}
        for h in horizons:
            for k, q in enumerate(quantiles):
                columns[f"{h}h__q_{q}"] = forecasts[:, h - 1, k]
        return pl.DataFrame(columns)


def make_tabicl_forecast_data_op(
    horizons=(1, 12, 24),
    quantiles=(0.05, 0.5, 0.95),
    max_context_length=4096,
    chunk_size=8,
    cities="all",
    temperature_only=True,
    calendar=True,
    tabicl_config=None,
    temporal_features=None,
    point_estimate="mean",
):
    """Same contract as `elf.make_data_op`, with TabICLForecaster."""
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
        "city_weather_fetcher", elf.fetch_city_weather_forecast, becomes_default=True
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

    estimator = TabICLForecastRegressor(
        horizons=horizons,
        quantiles=quantiles,
        max_context_length=max_context_length,
        chunk_size=chunk_size,
        tabicl_config=tabicl_config,
        temporal_features=temporal_features,
        point_estimate=point_estimate,
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
    kwargs = {"q_low": min(quantiles), "q_high": max(quantiles)}
    return (
        multi_horizon_pred.skb.with_scoring(elf.pinball_scorer)
        .skb.with_scoring(functools.partial(elf.coverage_scorer, **kwargs))
        .skb.with_scoring(functools.partial(elf.mean_width_scorer, **kwargs))
    )
