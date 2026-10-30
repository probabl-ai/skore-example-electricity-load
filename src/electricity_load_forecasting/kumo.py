"""
Kumo-Forecast variant of the load forecasting pipeline.

Everything that is not specific to Kumo is reused from
`electricity_load_forecasting`: history loading, resampling, X/y construction,
the walk-forward splitter, post-processing and scoring. This module only holds:

- `KumoHistoryRegressor`: an sklearn-like estimator which, for each prediction
  time t, builds the causal context window ending at t (inclusive) and asks the
  Kumo forecaster for the next `max(horizons)` hours.
- `KumoForecastForecaster`: wrapper around `sdm.models.KumoForecast`.
- `SeasonalNaivePointForecaster`: cheap point-only stub for smoke tests.
- `make_kumo_data_op`: same contract as `elf.make_data_op` for a point
  forecast, so the same splits and metrics can be used.

Kumo-Forecast only produces point forecasts (no quantiles), and it does not
support known-future covariates: all covariates are passed as past-only.
"""

import datetime
import functools

import numpy as np
import polars as pl
import skrub
from sklearn.base import BaseEstimator, RegressorMixin

from . import pipeline as elf
from .chronos import _hourly_positions, build_covariates


class KumoHistoryRegressor(RegressorMixin, BaseEstimator):
    """Point forecaster that builds causal windows from the load history."""

    def __init__(
        self,
        forecaster=None,
        horizons=(1, 12, 24),
        context_hours=512,
        chunk_size=8,
        verbose=False,
    ):
        self.forecaster = forecaster
        self.horizons = horizons
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

    def _inputs(self, positions, load, covariates):
        length = self.context_hours
        for i in positions:
            lo = max(0, i - length + 1)
            target = load[lo : i + 1]
            if len(target) < length:
                target = np.concatenate(
                    [np.full(length - len(target), np.nan, dtype=np.float32), target]
                )
            item = {"target": target}
            if covariates is not None:
                values, names = covariates
                window = values[lo : i + 1]
                if len(window) < length:
                    window = np.vstack(
                        [
                            np.full(
                                (length - len(window), len(names)),
                                np.nan,
                                dtype=np.float32,
                            ),
                            window,
                        ]
                    )
                item["past_covariates"] = {
                    n: window[:, j] for j, n in enumerate(names)
                }
            yield item

    def predict(self, X, history, covariates=None):
        if self.forecaster is None:
            raise ValueError("forecaster must be provided")
        horizons = tuple(self.horizons)
        n_steps = max(horizons)

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
            cov = (values, names)

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
            inputs = list(self._inputs(chunk, load, cov))
            out = np.asarray(
                self.forecaster.predict(
                    inputs, prediction_length=n_steps, quantile_levels=None
                ),
                dtype=np.float64,
            )
            if out.shape != (len(chunk), n_steps):
                raise ValueError(f"unexpected forecaster output shape {out.shape}")
            results.append(out)
        forecasts = (
            np.concatenate(results) if results else np.empty((0, n_steps))
        )

        return pl.DataFrame(
            {f"{h}h": forecasts[:, h - 1] for h in horizons}
        )


class SeasonalNaivePointForecaster(BaseEstimator):
    """Point-only stub: repeat the value observed one period earlier."""

    def __init__(self, period=24 * 7):
        self.period = period

    def predict(self, inputs, prediction_length, quantile_levels):
        if quantile_levels is not None:
            raise ValueError("Kumo-Forecast only produces point forecasts")
        steps = np.arange(1, prediction_length + 1)
        n_periods_back = -(-steps // self.period)
        out = np.empty((len(inputs), prediction_length))
        for i, item in enumerate(inputs):
            context = np.asarray(item["target"], dtype=np.float64)
            n = len(context)
            finite = np.flatnonzero(np.isfinite(context))
            fallback = context[finite[-1]] if len(finite) else np.nan
            idx = n - 1 + steps - n_periods_back * self.period
            point = np.where(
                (idx >= 0) & (idx < n), context[np.clip(idx, 0, n - 1)], np.nan
            )
            out[i] = np.where(np.isfinite(point), point, fallback)
        return out


@functools.lru_cache(maxsize=2)
def _load_kumo_model(device, cross_channel):
    import sdm
    import torch

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    return sdm.models.KumoForecast(
        cross_channel=cross_channel, device=device
    )


class KumoForecastForecaster(BaseEstimator):
    """Wrapper around `sdm.models.KumoForecast` (point forecasts only)."""

    def __init__(self, device=None, cross_channel=True, num_estimators=1):
        self.device = device
        self.cross_channel = cross_channel
        self.num_estimators = num_estimators

    def predict(self, inputs, prediction_length, quantile_levels):
        if quantile_levels is not None:
            raise ValueError("Kumo-Forecast only produces point forecasts")
        import sdm
        import torch

        model = _load_kumo_model(self.device, self.cross_channel)
        target_device = next(model.parameters()).device

        targets = np.stack(
            [np.asarray(item["target"], dtype=np.float32) for item in inputs]
        )
        length = targets.shape[1]
        covariates = [item.get("past_covariates") or {} for item in inputs]
        names = list(covariates[0])
        if names:
            values = np.stack(
                [
                    np.stack([c[n] for n in names], axis=-1)
                    for c in covariates
                ]
            ).astype(np.float32)
            x_context = sdm.TableTensor.from_tensor(
                torch.from_numpy(values).to(target_device), columns=names
            )
        else:
            x_context = sdm.TableTensor(
                size=(len(inputs), length), device=target_device
            )
        y_context = sdm.TableTensor.from_tensor(
            torch.from_numpy(targets[..., None]).to(target_device),
            columns=["load_mw"],
        )
        x_query = sdm.TableTensor(
            size=(len(inputs), prediction_length), device=target_device
        )
        with torch.amp.autocast(
            target_device.type,
            dtype=torch.float16,
            enabled=target_device.type == "cuda",
        ):
            out = model(
                x_context=x_context,
                y_context=y_context,
                x_query=x_query,
                num_estimators=self.num_estimators,
            )
        return out["load_mw"].numerical.squeeze(-1).float().cpu().numpy()


def make_kumo_data_op(
    horizons=(1, 12, 24),
    forecaster=None,
    context_hours=512,
    cities="all",
    temperature_only=True,
    calendar=True,
    chunk_size=8,
    verbose=False,
):
    """Same contract as `elf.make_data_op`, with a Kumo point forecaster."""
    if forecaster is None:
        forecaster = KumoForecastForecaster()

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

    estimator = KumoHistoryRegressor(
        forecaster=forecaster,
        horizons=horizons,
        context_hours=context_hours,
        chunk_size=chunk_size,
        verbose=verbose,
    )
    all_pred = X.skb.apply(
        estimator,
        y=y,
        predict_kwargs={"history": load_mw_history, "covariates": covariates},
    ).skb.set_name("pred_multi_horizon")

    return all_pred.skb.apply_func(
        elf.post_process, prediction_time, range_end, quantile_regression=False
    ).skb.with_scoring(elf.neg_mape_scorer)
