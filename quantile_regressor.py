"""
copied from:
https://github.com/probabl-ai/forecasting/blob/main/content/python_files/prediction_intervals.py
"""

from scipy.interpolate import interp1d
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.utils.validation import check_is_fitted
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingRegressor
from sklearn.preprocessing import KBinsDiscretizer
from sklearn.utils.validation import check_consistent_length
from sklearn.utils import check_random_state
import numpy as np
import polars as pl


class HGBQuantileRegressor(RegressorMixin, BaseEstimator):
    def __init__(self, quantiles=(0.05, 0.5, 0.95), hgb_params=None):
        self.quantiles = quantiles
        self.hgb_params = hgb_params

    def fit(self, X, y):
        params = (self.hgb_params or {}) | {"loss": "quantile"}
        self.estimators_ = {
            q: HistGradientBoostingRegressor(quantile=q, **params).fit(X, y)
            for q in self.quantiles
        }
        return self

    def predict(self, X):
        return pl.DataFrame({f"q_{q}": e.predict(X) for q, e in self.estimators_.items()})


class BinnedQuantileRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        estimator=None,
        n_bins=100,
        default_quantiles=(0.05, 0.5, 0.95),
        random_state=None,
    ):
        self.n_bins = n_bins
        self.estimator = estimator
        self.default_quantiles = default_quantiles
        self.random_state = random_state

    def fit(self, X, y):
        # Lightweight input validation: most of the input validation will be
        # handled by the sub estimators.
        random_state = check_random_state(self.random_state)
        check_consistent_length(X, y)
        self.target_binner_ = KBinsDiscretizer(
            n_bins=self.n_bins,
            strategy="quantile",
            subsample=200_000,
            encode="ordinal",
            quantile_method="averaged_inverted_cdf",
            random_state=random_state,
        )

        y_binned = (
            self.target_binner_.fit_transform(np.asarray(y).reshape(-1, 1))
            .ravel()
            .astype(np.int32)
        )

        # Fit the multiclass classifier to predict the binned targets from the
        # training set.
        if self.estimator is None:
            estimator = RandomForestClassifier(random_state=random_state)
        else:
            estimator = clone(self.estimator)
        self.estimator_ = estimator.fit(X, y_binned)
        return self

    def predict(self, X, quantiles=None):
        if quantiles is None:
            quantiles = self.default_quantiles
        check_is_fitted(self, "estimator_")
        edges = self.target_binner_.bin_edges_[0]
        n_bins = edges.shape[0] - 1
        expected_shape = (X.shape[0], n_bins)
        y_proba_raw = self.estimator_.predict_proba(X)

        # Some might stay empty on the training set. Typically, classifiers do
        # not learn to predict an explicit 0 probability for unobserved classes
        # so we have to post process their output:
        if y_proba_raw.shape != expected_shape:
            y_proba = np.zeros(shape=expected_shape)
            y_proba[:, self.estimator_.classes_] = y_proba_raw
        else:
            y_proba = y_proba_raw

        # Build the mapper for inverse CDF mapping, from cumulated
        # probabilities to continuous prediction.
        y_cdf = np.zeros(shape=(X.shape[0], edges.shape[0]))
        y_cdf[:, 1:] = np.cumsum(y_proba, axis=1)
        result = np.asarray([interp1d(y_cdf_i, edges)(quantiles) for y_cdf_i in y_cdf])
        return pl.DataFrame(result, schema=[f"q_{q}" for q in quantiles])
