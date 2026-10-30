"""TabICL adapter returning the pipeline's polars quantile format."""

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, RegressorMixin


class TabICLQuantileRegressor(RegressorMixin, BaseEstimator):
    """TabICL quantile regressor returning ``q_<level>`` columns."""

    def __init__(self, n_estimators=8, quantiles=(0.05, 0.5, 0.95)):
        self.n_estimators = n_estimators
        self.quantiles = quantiles

    def fit(self, X, y):
        from tabicl import TabICLRegressor

        self.estimator_ = TabICLRegressor(n_estimators=self.n_estimators).fit(X, y)
        return self

    def predict(self, X):
        prediction = self.estimator_.predict(
            X, output_type="quantiles", alphas=self.quantiles
        )
        return pl.DataFrame(
            np.asarray(prediction),
            schema=[f"q_{q}" for q in self.quantiles],
        )
