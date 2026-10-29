import re
import warnings

import numpy as np
import polars as pl
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import TimeSeriesSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

# Default inner CV when none is given (hourly rows: 7 days between train and
# test, as in electricity_load_forecasting.TimeSeriesSplitter, which is what
# make_data_op uses instead).
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
       weighted by bin size, normalized to a mean of 1) mapping the predicted
       value to the shift. If ties leave fewer than 5 distinct bins, a constant
       shift is used instead (with a warning).
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
        hourly rows. ``fit`` can pass extra arguments to its ``split`` method
        (``split_kwargs``).
    n_last_folds : int or None, default=None
        If not None, only calibrate on the out-of-fold predictions of the last
        ``n_last_folds`` folds of ``cv``, so that the correction follows recent
        behaviour rather than the whole history (useful if calibration drifts
        over time). The other folds are not fit, which also saves time. Note
        that the last fold of a walk-forward ``cv`` may be shorter than the others.
    n_bins : int, default=10
        Number of bins of predicted values. Keep it small: estimating a tail
        quantile such as 0.95 needs several hundred points per bin to be
        reasonably stable. At least 5 (the number of spline basis functions).

    Attributes
    ----------
    estimator_ : the estimator refit on all the data.
    n_folds_used_ : number of cv folds used for the calibration.
    oof_ : polars DataFrame
        The out-of-fold data the calibration is learned from: the predicted
        quantiles (columns ``q_<level>``), ``y``, ``row`` (position in the
        ``X`` given to ``fit``) and ``inner_fold`` (index of the fold in
        ``cv.split``).
    quantile_levels_ : dict {column name: quantile level}
    shift_models_ : dict {column name: fitted model, predicted value -> shift}
    bin_summary_ : dict {column name: polars DataFrame}
        For inspection: for each bin, the mean prediction, the number of
        points, the measured shift and the empirical coverage before
        correction.
    """

    def __init__(self, estimator, cv=None, n_bins=10, n_last_folds=4):
        self.estimator = estimator
        self.cv = cv
        self.n_bins = n_bins
        self.n_last_folds = n_last_folds

    def fit(self, X, y, split_kwargs=None):
        """Fit the estimator and its correction.

        ``split_kwargs`` is a dict of extra keyword arguments for
        ``cv.split(X, y, **split_kwargs)``, for example the prediction times
        when ``X`` is an array that does not contain them.
        """
        if self.n_bins < 5:
            raise ValueError(f"n_bins must be at least 5, got {self.n_bins}")
        if self.n_last_folds is not None and self.n_last_folds < 1:
            raise ValueError(f"n_last_folds must be at least 1, got {self.n_last_folds}")
        X = np.asarray(X)
        y = np.asarray(y, dtype=float).ravel()
        cv = self.cv if self.cv is not None else _default_cv()

        folds = list(cv.split(X, y, **(split_kwargs or {})))
        n_skipped = 0
        if self.n_last_folds is not None:
            n_skipped = max(len(folds) - self.n_last_folds, 0)
            folds = folds[n_skipped:]
        self.n_folds_used_ = len(folds)

        y_oof, pred_oof, row_oof, fold_oof = [], [], [], []
        for fold, (train, test) in enumerate(folds, start=n_skipped):
            fold_estimator = clone(self.estimator).fit(X[train], y[train])
            pred_oof.append(fold_estimator.predict(X[test]))
            y_oof.append(y[test])
            row_oof.append(np.asarray(test))
            fold_oof.append(np.full(len(test), fold))
        if not y_oof:
            raise ValueError("cv produced no splits")
        pred_oof = pl.concat(pred_oof, how="vertical")
        y_oof = np.concatenate(y_oof)
        # keep the out-of-fold data, for inspection
        self.oof_ = pred_oof.with_columns(
            row=pl.Series(np.concatenate(row_oof)),
            inner_fold=pl.Series(np.concatenate(fold_oof)),
            y=pl.Series(y_oof),
        )

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
            # many identical predictions (e.g. a saturated model) can leave too
            # few distinct bins for the spline: fall back to a constant shift
            # rather than failing in the middle of a long CV
            warnings.warn(
                f"Only {len(summary)} distinct bins for quantile {tau} (many tied "
                "predictions): using a constant shift.",
                stacklevel=2,
            )
            model = DummyRegressor(
                strategy="constant", constant=float(np.quantile(residuals, tau))
            ).fit(summary[["mean_prediction"]].to_numpy(), summary["shift"].to_numpy())
            return model, summary
        # 5 basis functions: n_knots + degree - 1 = 3 + 3 - 1
        model = make_pipeline(
            SplineTransformer(n_knots=3, degree=3, extrapolation="constant"),
            RidgeCV(alphas=np.logspace(-3, 3, 13)),
        )
        # Weights proportional to the bin sizes, but normalized to a mean of 1:
        # the ridge penalty is not scaled by the weights, so with raw counts
        # (hundreds or thousands) the alpha grid would be ineffective, and the
        # regularization would depend on the size of the training set.
        weights = summary["count"].to_numpy() / summary["count"].mean()
        model.fit(
            summary["mean_prediction"].to_numpy()[:, None],
            summary["shift"].to_numpy(),
            ridgecv__sample_weight=weights,
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
        # explicit orientation: polars guesses it for a square array
        return pl.DataFrame(corrected, schema=pred.columns, orient="row")


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
