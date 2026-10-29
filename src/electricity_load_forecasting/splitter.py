import datetime

import polars as pl

_TRAIN_TEST_GAP_DAYS = 7


def _split_indices(X, test_start_date, test_length_days):
    train = (
        X.with_row_index()
        .filter(
            pl.col("prediction_time")
            < test_start_date - datetime.timedelta(_TRAIN_TEST_GAP_DAYS)
        )["index"]
        .to_numpy()
    )
    test = (
        X.with_row_index()
        .filter(
            (pl.col("prediction_time") >= test_start_date)
            & (
                pl.col("prediction_time")
                < test_start_date + datetime.timedelta(days=test_length_days)
            )
        )["index"]
        .to_numpy()
    )
    return train, test


class TimeSeriesSplitter:
    """
    Walk-forward CV with a gap of ``_TRAIN_TEST_GAP_DAYS`` between train and test.

    The test folds are aligned towards the end of the time range: they are
    consecutive, all ``test_length_days`` long (the default, 13 weeks, is about
    3 months), and the last one ends at the last prediction time. The training set
    of each fold is all the data that precedes its test fold by more than the
    gap. The number of folds is the largest such that the first training set
    spans at least ``min_train_days``; it can be longer, by less than
    ``test_length_days``.

    ``split`` reads the prediction times from the ``prediction_time`` column of
    ``X``, or uses its ``prediction_time`` argument if provided (for example when
    ``X`` is a numpy array of features from which the time column was dropped).
    """

    def __init__(self, min_train_days=365 * 2, test_length_days=7 * 13):
        self.min_train_days = min_train_days
        self.test_length_days = test_length_days

    def split(self, X, y=None, groups=None, prediction_time=None):
        if prediction_time is not None:
            X = pl.DataFrame({"prediction_time": prediction_time})
        times = X["prediction_time"]
        # the test folds are half-open intervals, the last one must include the
        # last prediction time: it ends one time step after it
        time_steps = times.unique().sort().diff().drop_nulls()
        if not len(time_steps):
            return
        end = times.max() + time_steps.min()
        test_length = datetime.timedelta(days=self.test_length_days)
        available = (
            end
            - times.min()
            - datetime.timedelta(days=_TRAIN_TEST_GAP_DAYS + self.min_train_days)
        )
        n_splits = max(available // test_length, 0)
        for i in range(n_splits, 0, -1):
            train, test = _split_indices(
                X, end - i * test_length, test_length_days=self.test_length_days
            )
            if len(train) and len(test):
                yield train, test

    def get_n_splits(self, X, y=None, groups=None, prediction_time=None):
        return len(list(self.split(X, y, prediction_time=prediction_time)))


def train_test_split(X, y, test_start_date="2025-01-01"):
    if isinstance(test_start_date, str):
        test_start_date = datetime.datetime.fromisoformat(test_start_date).astimezone(
            datetime.UTC
        )
    train, test = _split_indices(
        X, test_start_date=test_start_date, test_length_days=24 * 7
    )
    return X[train], X[test], y[train], y[test]
