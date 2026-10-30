"""Kumo-Tabular variant of the tabular quantile pipeline.

Kumo-Tabular is an in-context tabular regression model from
``structured-data-models`` (``sdm.models.KumoTabular``). This module wraps it in
an sklearn-like estimator so it can be used like TabICL in
``pipeline.make_data_op(quantile_strategy="kumo_tabular")``.

The model predicts 999 quantiles (``q001``-``q999``); this wrapper extracts the
requested quantile levels and returns a polars DataFrame with ``q_<level>``
columns, matching ``HGBQuantileRegressor``.
"""

import functools

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, RegressorMixin


@functools.lru_cache(maxsize=2)
def _load_model(size, device):
    import sdm

    return sdm.models.KumoTabular(task="regression", size=size, device=device)


def _table(values, columns, device):
    import sdm
    import torch

    tensor = torch.from_numpy(np.array(values, dtype=np.float32, copy=True))
    return sdm.TableTensor.from_tensor(tensor.to(device), columns=columns)


class KumoTabularRegressor(RegressorMixin, BaseEstimator):
    """In-context quantile regressor backed by ``sdm.models.KumoTabular``."""

    def __init__(
        self,
        size="large",
        device=None,
        n_estimators=1,
        quantiles=(0.05, 0.5, 0.95),
    ):
        self.size = size
        self.device = device
        self.n_estimators = n_estimators
        self.quantiles = quantiles

    def fit(self, X, y):
        self.X_ = np.asarray(X, dtype=np.float32)
        self.y_ = np.asarray(y, dtype=np.float32)
        self.quantiles_ = tuple(self.quantiles)
        return self

    def predict(self, X):
        import torch

        device = self.device or ("cuda" if torch.cuda.is_available() else "cpu")
        model = _load_model(self.size, device)
        X_query = np.asarray(X, dtype=np.float32)
        columns = [f"x{i}" for i in range(self.X_.shape[1])]
        x_context = _table(self.X_, columns, device)
        y_context = _table(self.y_[:, None], ["target"], device)
        x_query = _table(X_query, columns, device)

        with torch.inference_mode(), torch.amp.autocast(
            torch.device(device).type,
            dtype=torch.float16,
            enabled=torch.device(device).type == "cuda",
        ):
            out = model(
                x_context=x_context,
                y_context=y_context,
                x_query=x_query,
                num_estimators=self.n_estimators,
            )

        available = list(out.numerical_columns)
        indices = [round(q * 999) for q in self.quantiles_]
        names = [f"q{i:03d}" for i in indices]
        selected = out.numerical[:, [available.index(name) for name in names]]
        predictions = np.sort(selected.float().cpu().numpy(), axis=1)
        return pl.DataFrame(
            {f"q_{q}": predictions[:, i] for i, q in enumerate(self.quantiles_)}
        )
