"""
t0 (The Forecasting Company) variant of the load forecasting pipeline.

Everything that is not specific to t0 is reused from
`electricity_load_forecasting`: history loading, resampling, X/y construction,
the walk-forward splitter, post-processing and scoring, and the generic
`ChronosHistoryRegressor` / `build_covariates` from `chronos.py`. This module
only holds:

- `T0Forecaster`: wrapper around `t0.T0Forecaster` (the `tfc-t0` package),
  exposing the same `predict(inputs, prediction_length, quantile_levels)`
  interface as `Chronos2Forecaster` so it can be dropped into the Chronos
  DataOp unchanged.
- `make_t0_data_op`: same contract as `elf.make_data_op`, delegating to
  `chronos.make_chronos_data_op` with a t0 forecaster.

t0 produces quantile forecasts and conditions on known-future covariates
(weather + calendar) passed as `future_covariates` spanning context + horizon.
"""

import functools

import numpy as np
from sklearn.base import BaseEstimator

from . import chronos as cf


@functools.lru_cache(maxsize=2)
def _load_t0_model(model_id, device, dtype):
    import torch
    from t0 import T0Forecaster as T0Model

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    kwargs = {}
    if dtype is not None:
        kwargs["dtype"] = getattr(torch, dtype)
    model = T0Model.from_pretrained(model_id, **kwargs).eval()
    return model.to(device)


class T0Forecaster(BaseEstimator):
    """
    Zero-shot t0 forecaster (`tfc-t0` package).

    Holds parameters only: the model is loaded lazily and cached at module
    level, so cloning (done by skrub / skore for each fit) does not copy or
    reload it.

    Parameters
    ----------
    model : str
        Hugging Face model id, e.g. "theforecastingcompany/t0-gamma".
    device : str or None
        None selects cuda when available, else cpu.
    dtype : str or None
        Name of a torch dtype, e.g. "bfloat16". None keeps the default.
    trim_leading_missing, symmetric_averaging, clip_nonnegative, snap_integers : bool
        The inference-time adjustments recommended for t0-gamma; each uses only
        the series' own history. `symmetric_averaging` runs the model twice.
    """

    def __init__(
        self,
        model="theforecastingcompany/t0-gamma",
        device=None,
        dtype=None,
        trim_leading_missing=True,
        symmetric_averaging=True,
        clip_nonnegative=True,
        snap_integers=True,
    ):
        self.model = model
        self.device = device
        self.dtype = dtype
        self.trim_leading_missing = trim_leading_missing
        self.symmetric_averaging = symmetric_averaging
        self.clip_nonnegative = clip_nonnegative
        self.snap_integers = snap_integers

    def predict(self, inputs, prediction_length, quantile_levels):
        model = _load_t0_model(self.model, self.device, self.dtype)
        n = len(inputs)
        width = max(len(item["target"]) for item in inputs)
        names = []
        for item in inputs:
            if item.get("past_covariates"):
                names = list(item["past_covariates"])
                break

        context = np.full((n, width), np.nan, dtype=np.float32)
        for i, item in enumerate(inputs):
            target = np.asarray(item["target"], dtype=np.float32)
            context[i, width - len(target) :] = target

        future_covariates = None
        if names:
            future_covariates = np.full(
                (n, len(names), width + prediction_length),
                np.nan,
                dtype=np.float32,
            )
            for i, item in enumerate(inputs):
                past = item["past_covariates"]
                future = item["future_covariates"]
                for j, name in enumerate(names):
                    p = np.asarray(past[name], dtype=np.float32)
                    f = np.asarray(future[name], dtype=np.float32)
                    future_covariates[i, j, width - len(p) : width] = p
                    future_covariates[i, j, width : width + len(f)] = f

        out = model.predict(
            context,
            horizon=prediction_length,
            quantile_levels=list(quantile_levels),
            future_covariates=future_covariates,
            trim_leading_missing=self.trim_leading_missing,
            symmetric_averaging=self.symmetric_averaging,
            clip_nonnegative=self.clip_nonnegative,
            snap_integers=self.snap_integers,
        )
        return out.quantiles.float().cpu().numpy()


def make_t0_data_op(
    horizons=(1, 12, 24),
    forecaster=None,
    quantiles=None,
    context_hours=1024,
    cities="all",
    temperature_only=True,
    calendar=True,
    chunk_size=64,
    verbose=False,
):
    """Same contract as `elf.make_data_op`, with a t0 forecaster.

    Delegates to `chronos.make_chronos_data_op`: the DataOp graph, the generic
    `ChronosHistoryRegressor` and the covariate construction are identical; only
    the forecaster differs.
    """
    if forecaster is None:
        forecaster = T0Forecaster()
    return cf.make_chronos_data_op(
        horizons=horizons,
        forecaster=forecaster,
        quantiles=quantiles,
        context_hours=context_hours,
        cities=cities,
        temperature_only=temperature_only,
        calendar=calendar,
        chunk_size=chunk_size,
        verbose=verbose,
    )
