import re

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

# Hourly rows: 7 days between train and test, as in
# electricity_load_forecasting.TimeSeriesSplitter.
_DEFAULT_GAP_ROWS = 24 * 7


class Recalibrator(RegressorMixin, BaseEstimator):
    """Post-hoc correction of a quantile estimator based on its reliability.

    Wraps a single-target quantile regressor whose ``predict`` returns a polars
    DataFrame with columns ``q_<level>`` (e.g. ``q_0.05``, ``q_0.95``), such as
    ``HGBQuantileRegressor`` or ``BinnedQuantileRegressor``.

    Fitting:

    1. Collect out-of-fold predictions of ``estimator`` with ``cv``, and
       concatenate them with the matching ``y``. Unlike sklearn's
       ``cross_val_predict``, the test folds need not partition the data (a
       walk-forward ``cv`` never tests on the first rows); rows that are never
       in a test fold are simply not used for calibration.
    2. Separately for each predicted quantile level tau: bin the out-of-fold
       predictions ``p`` into ``n_bins`` bins with the same number of points.
       In each bin, the estimator claims that ``p`` is the tau-quantile of
       ``y``; measure the actual tau-quantile of the residuals ``y - p``, in MW.
       This is the shift needed to make that bin calibrated.
    3. Fit a smooth model (spline transformer with 5 basis functions + RidgeCV,
       weighted by bin size) mapping the predicted value to the shift.
    4. Refit ``estimator`` on all the data. At prediction time, add the learned
       shift (a function of each predicted value) to its predictions, then sort
       the quantiles so that they do not cross.

    Parameters
    ----------
    estimator : quantile regressor
        See above for the expected ``predict`` output.
    cv : cross-validation splitter, default=None
        Used to obtain out-of-fold predictions. It must respect the temporal
        order. Default: ``TimeSeriesSplit(n_splits=5, gap=24 * 7)``, assuming
        hourly rows.
    n_bins : int, default=8
        Number of bins of predicted values. Keep it small: estimating a tail
        quantile such as 0.95 needs several hundred points per bin to be
        reasonably stable. At least 5 (the number of spline basis functions).

    Attributes
    ----------
    estimator_ : the estimator refit on all the data.
    quantile_levels_ : dict {column name: quantile level}
    shift_models_ : dict {column name: fitted model, predicted value -> shift}
    bin_summary_ : dict {column name: polars DataFrame}
        For inspection: for each bin, the mean prediction, the number of
        points, the measured shift and the empirical coverage before
        correction.
    """

    def __init__(self, estimator, cv=None, n_bins=8):
        self.estimator = estimator
        self.cv = cv
        self.n_bins = n_bins

    def fit(self, X, y):
        if self.n_bins < 5:
            raise ValueError(f"n_bins must be at least 5, got {self.n_bins}")
        X = np.asarray(X)
        y = np.asarray(y, dtype=float).ravel()
        cv = self.cv if self.cv is not None else _default_cv()

        y_oof, pred_oof = [], []
        for train, test in cv.split(X, y):
            fold_estimator = clone(self.estimator).fit(X[train], y[train])
            pred_oof.append(fold_estimator.predict(X[test]))
            y_oof.append(y[test])
        if not y_oof:
            raise ValueError("cv produced no splits")
        pred_oof = pl.concat(pred_oof, how="vertical")
        y_oof = np.concatenate(y_oof)

        self.quantile_levels_ = _quantile_levels(pred_oof)
        self.shift_models_ = {}
        self.bin_summary_ = {}
        for column, tau in self.quantile_levels_.items():
            p = pred_oof[column].to_numpy()
            model, summary = self._fit_shift_model(p, y_oof - p, tau)
            self.shift_models_[column] = model
            self.bin_summary_[column] = summary

        self.estimator_ = clone(self.estimator).fit(X, y)
        return self

    def _fit_shift_model(self, p, residuals, tau):
        # bins with an equal number of points; np.unique in case of ties
        edges = np.unique(np.quantile(p, np.linspace(0, 1, self.n_bins + 1)))
        bin_ids = np.searchsorted(edges[1:-1], p, side="right")
        rows = []
        for b in np.unique(bin_ids):
            in_bin = bin_ids == b
            rows.append(
                {
                    "mean_prediction": float(p[in_bin].mean()),
                    "count": int(in_bin.sum()),
                    "shift": float(np.quantile(residuals[in_bin], tau)),
                    "empirical_coverage": float((residuals[in_bin] <= 0).mean()),
                }
            )
        summary = pl.DataFrame(rows)
        if len(summary) < 5:
            raise ValueError(
                f"Only {len(summary)} distinct bins for quantile {tau}, need at "
                "least 5. Reduce the number of ties or increase n_bins."
            )
        # 5 basis functions: n_knots + degree - 1 = 3 + 3 - 1
        model = make_pipeline(
            SplineTransformer(n_knots=3, degree=3, extrapolation="constant"),
            RidgeCV(alphas=np.logspace(-3, 3, 13)),
        )
        model.fit(
            summary["mean_prediction"].to_numpy()[:, None],
            summary["shift"].to_numpy(),
            ridgecv__sample_weight=summary["count"].to_numpy(),
        )
        return model, summary

    def predict(self, X, **predict_params):
        pred = self.estimator_.predict(np.asarray(X), **predict_params)
        corrected = pred.to_numpy().astype(float, copy=True)
        for i, column in enumerate(pred.columns):
            model = self.shift_models_[column]
            corrected[:, i] += model.predict(corrected[:, [i]])
        # shifts are learned independently per quantile: avoid crossing
        order = np.argsort([self.quantile_levels_[c] for c in pred.columns])
        corrected[:, order] = np.sort(corrected[:, order], axis=1)
        return pl.DataFrame(corrected, schema=pred.columns)


def _default_cv():
    return TimeSeriesSplit(n_splits=5, gap=_DEFAULT_GAP_ROWS)


def _quantile_levels(pred):
    levels = {}
    for column in pred.columns:
        match = re.fullmatch(r"q_(.+)", column)
        if match is None:
            raise ValueError(
                f"Expected quantile columns named 'q_<level>', got {column!r}"
            )
        levels[column] = float(match.group(1))
    return levels
