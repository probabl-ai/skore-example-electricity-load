import re
import datetime
import json
from pathlib import Path
import subprocess

import holidays
import polars as pl
from polars import selectors as cs
import skrub
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.ensemble import HistGradientBoostingRegressor
import plotly.graph_objects as go

_ALL_CITIES = (
    "paris",
    "lyon",
    "marseille",
    "toulouse",
    "lille",
    "limoges",
    "nantes",
    "strasbourg",
    "brest",
    "bayonne",
)
_TRAIN_TEST_GAP_DAYS = 7


def project_dir():
    return Path(__file__).parent


def data_dir():
    return project_dir() / "datasets"


def get_output_dir(prefix=""):
    output_dir = (
        project_dir() / "results" / f"{prefix}{datetime.datetime.now().isoformat()}"
    )
    output_dir.mkdir(parents=True)
    (output_dir / "metadata.json").write_text(
        json.dumps(
            {
                "commit": last_commit_hash(),
                "date": datetime.datetime.now().isoformat(),
            }
        ),
        "utf-8",
    )
    return output_dir


def last_commit_hash():
    # TODO: either create a commit in a dedicated branch or assert that working
    # tree is clean
    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=str(project_dir()),
        check=True,
        encoding="utf-8",
        capture_output=True,
    ).stdout.strip()


def fetch_load_mw_history():
    """
    Fetch the historical electricity grid load in MW.

    Returns a dataframe with columns [time, load_mw].
    """
    return (
        pl.scan_csv(
            data_dir() / "Total Load - Day Ahead*.csv", null_values=["N/A", "-"]
        )
        .drop_nulls()
        .select(
            pl.col("Time (UTC)")
            .str.split(by=" - ")
            .list.first()
            .str.to_datetime("%d.%m.%Y %H:%M", time_zone="UTC")
            .alias("time"),
            pl.col("Actual Total Load [MW] - BZN|FR").alias("load_mw"),
        )
        .collect()
    )


def time_range(start, end):
    """
    Build a 1-hour-spaced datetime range from start to end.
    """
    if isinstance(start, str):
        start = datetime.datetime.fromisoformat(start)
    if isinstance(end, str):
        end = datetime.datetime.fromisoformat(end)
    return pl.DataFrame().with_columns(
        pl.datetime_range(
            start=start,
            end=end,
            time_zone="UTC",
            interval="1h",
        ).alias("time"),
    )


def resample(load_mw_history):
    """
    Resample the load history on a regular time grid to have exactly 1 row every hour.

    Parts where sampling was finer (eg every 15 minutes) are averaged over 1h
    intervals, and if some hours are missing a corresponding row is inserted
    containing explicit NULL values (rather than a missing row).
    """
    averaged = load_mw_history.group_by(pl.col("time").dt.truncate("1h")).agg(
        pl.col("load_mw").mean()
    )
    all_times = averaged["time"]
    return time_range(all_times.min(), all_times.max()).join(
        averaged, on="time", how="left", maintain_order="left"
    )


class GetXy(BaseEstimator):
    """
    Get query (prediction time grid) and actual loads (at different horizons) to predict.

    During training and for cross-validation, rows for which some data is
    missing (we have no actual load for one of the horizons) are dropped.
    During inference no rows are dropped (there is no ground truth anyway).

    data must be a dict with keys 'load_mw_history', 'prediction_time', 'horizons'.
    Returns a dictionary with keys 'X' and 'y'.

    y is constructed by shifting the historical load back, ie aligning future
    load with the current date (during inference y will be None).
    """

    def fit_transform(self, data, y=None):
        load = (
            data["load_mw_history"]
            .select(
                pl.col("time"),
                *[
                    pl.col("load_mw").shift(-h).alias(f"{h}h")
                    for h in data["horizons"]
                ],
            )
            .drop_nulls()
        )
        X_y = data["prediction_time"].join(
            load, on="time", how="inner", maintain_order="left"
        )
        return {
            "X": X_y.select(pl.col("time").alias("prediction_time")),
            "y": X_y.drop("time"),
        }

    def transform(self, data):
        return {"X": data["prediction_time"], "y": None}

    def fit(self, data, y=None):
        return self


def add_lagged_features(target_time, load_mw_history, horizon):
    """
    Build lagged features for the given horizon.

    horizon must be <= 24 (hours). Only features that would be available at
    prediction time, ie that require data at least horizon hours in the past,
    are created.
    """
    assert horizon <= 24
    lags = (
        pl.col("load_mw").shift(lag).alias(f"lag_{lag}")
        for lag in list(range(horizon, 24)) + [24, 24 * 2, 24 * 7]
    )

    rolling_lags = sorted(set((horizon, 24)))
    rolling_widths = (24, 24 * 7)

    def rolling(e, name):
        return [
            e.rolling(
                index_column="time", period=f"{width}h", offset=f"{-width -lag}h"
            ).alias(f"lag_{lag}_width_{width}_{name}")
            for lag in rolling_lags
            for width in rolling_widths
        ]

    medians = rolling(pl.col("load_mw").median(), "median")
    iqr = rolling(
        (pl.col("load_mw").quantile(0.75) - pl.col("load_mw").quantile(0.25)), "iqr"
    )
    features = load_mw_history.select(pl.col("time"), *lags, *medians, *iqr)
    return target_time.join(
        features,
        left_on="target_time",
        right_on="time",
        how="left",
        maintain_order="left",
    )


def fetch_city_weather(city):
    return pl.scan_parquet(data_dir() / f"weather_{city}.parquet")


def add_weather(
    target_time,
    city_names="all",
    temperature_only=False,
    city_weather_fetcher=fetch_city_weather,
):
    """Add weather information for the required cities."""
    # NOTE: here ideally we should retrieve the exact weather forecast
    # corresponding to the horizon. But we do not have it available in the
    # historical data. We just take the only forecast we have.
    if isinstance(city_names, str):
        assert city_names == "all"
        city_names = _ALL_CITIES
    with_weather = target_time.lazy()
    for city in city_names:
        with_weather = with_weather.join(
            city_weather_fetcher(city)
            .with_columns(pl.col("time").dt.cast_time_unit("us"))
            .select(
                (pl.col("time"), cs.matches(".*temperature.*"))
                if temperature_only
                else pl.all()
            )
            .select(
                pl.col("time"),
                (~cs.by_name("time"))
                .as_expr()
                .name.map(f"weather_{{}}_{city}".format),
            ),
            left_on="target_time",
            right_on="time",
            how="left",
            maintain_order="left",
        )
    return with_weather.collect()


def add_calendar_and_holidays(target_time):
    """Add calendar features and holiday information."""
    fr_time = pl.col("target_time").dt.convert_time_zone("Europe/Paris")
    fr_year_min = target_time.select(fr_time.dt.year().min()).item()
    fr_year_max = target_time.select(fr_time.dt.year().max()).item()
    holidays_fr = holidays.country_holidays(
        "FR", years=range(fr_year_min, fr_year_max + 1)
    )
    return target_time.with_columns(
        fr_time.dt.hour().alias("cal_hour_of_day"),
        fr_time.dt.weekday().alias("cal_day_of_week"),
        fr_time.dt.ordinal_day().alias("cal_day_of_year"),
        fr_time.dt.year().alias("cal_year"),
        fr_time.dt.date().is_in(holidays_fr.keys()).alias("cal_is_holiday"),
    )


def add_target_time(df, horizon):
    return df.with_columns(
        (pl.col("prediction_time") + pl.duration(hours=horizon)).alias("target_time")
    )


def add_features(
    df, horizon, temperature_only, city_names, load_mw_history, city_weather_fetcher
):
    df = add_target_time(df, horizon=horizon)
    df = add_weather(
        df,
        temperature_only=temperature_only,
        city_names=city_names,
        city_weather_fetcher=city_weather_fetcher,
    )
    df = add_calendar_and_holidays(df)
    df = add_lagged_features(df, load_mw_history=load_mw_history, horizon=horizon)
    return df


def concat_horizons(all_pred):
    """
    Consolidate predictions of models for different horizons in one dataframe.
    """
    return pl.DataFrame({f"{h}h": v for h, v in all_pred.items()})


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
    def split(self, X, y=None, groups=None):
        min_train_days = 365 * 2
        test_length_days = 24 * 7  # 24 weeks
        test_start_dates = pl.date_range(
            X["prediction_time"].min()
            + datetime.timedelta(days=min_train_days + _TRAIN_TEST_GAP_DAYS),
            X["prediction_time"].max(),
            interval=datetime.timedelta(days=test_length_days),
            closed="left",
            eager=True,
        )
        for test_start in test_start_dates:
            train, test = _split_indices(
                X, test_start, test_length_days=test_length_days
            )
            if len(train) and len(test):
                yield train, test

    def get_n_splits(self, X, y=None, groups=None):
        return len(list(self.split(X, y)))


def train_test_split(X, y, test_start_date="2025-01-01"):
    if isinstance(test_start_date, str):
        test_start_date = datetime.datetime.fromisoformat(test_start_date).astimezone(
            datetime.UTC
        )
    train, test = _split_indices(
        X, test_start_date=test_start_date, test_length_days=24 * 7
    )
    return X[train], X[test], y[train], y[test]


def make_data_op(horizons=(1, 2, 12, 24)):
    """
    Prepare a skrub dataop for multiple horizon prediction.

    The dataop contains the following inputs (expected keys in the environment
    passed to the learner):

    - start : datetime or ISO datetime string
        The start of the date range to predict

    - end : datetime or ISO datetime string
        The end of the date range to predict

    start and end correspond to the time at which the prediction is made. the
    prediction is made about that time + the horizon.

    - load_mw_history_fetcher : function, optional
       Use this to override how historical data is loaded. It is called with no
       arguments and must return a dataframe with columns time, load_mw.

    - city_weather_fetcher : function, optional
       Use this to override how weather forecasts are loaded. It is called
       with a city name and must return a lazyframe with columns similar to
       those of datasets/weather_paris.parquet.

    """
    range_start = skrub.var("start").skb.set_description(
        "The first time at which a prediction is made."
    )
    range_end = skrub.var("end").skb.set_description(
        "The last time at which a prediction is made."
    )
    load_mw_history_fetcher = (
        skrub.as_data_op(fetch_load_mw_history)
        .skb.set_name("load_mw_history_fetcher")
        .skb.set_description(
            "Function that loads the historical load data. "
            "See signature of electricity_load_forecasting.fetch_load_mw_history."
        )
    )
    city_weather_fetcher = (
        skrub.as_data_op(fetch_city_weather)
        .skb.set_name("city_weather_fetcher")
        .skb.set_description(
            "Function that loads the weather forecast for a city. "
            "See signature of electricity_load_forecasting.fetch_city_weather."
        )
    )
    prediction_time = skrub.deferred(time_range)(range_start, range_end)
    load_mw_history = (
        load_mw_history_fetcher()
        .skb.apply_func(resample)
        .skb.set_description("Historical load data on a regular 1h time grid.")
    )
    X_y = skrub.as_data_op(
        {
            "prediction_time": prediction_time,
            "load_mw_history": load_mw_history,
            "horizons": horizons,
        }
    ).skb.apply(GetXy())
    X = X_y["X"].skb.mark_as_X(cv=TimeSeriesSplitter())
    y = (
        X_y["y"]
        .skb.mark_as_y()
        .skb.set_description(
            "Actual loads at different horizons. The column Xh corresponds to "
            "what happened X hours after the date in the "
            "corresponding 'prediction_time' column of X."
        )
    )

    temperature_only = skrub.choose_bool(name="temperature_only", default=False)
    cities = skrub.choose_from(["all", ["paris", "lyon", "marseille"]], name="cities")

    regressor = HistGradientBoostingRegressor(
        random_state=0,
        loss=skrub.choose_from(["squared_error", "poisson", "gamma"], name="loss"),
        learning_rate=skrub.choose_float(
            0.01, 0.7, default=0.1, log=True, name="learning_rate"
        ),
        max_leaf_nodes=skrub.choose_int(
            3, 300, default=30, log=True, name="max_leaf_nodes"
        ),
    )

    all_pred = {}
    for h in horizons:
        pred = (
            X.skb.apply_func(
                add_features,
                horizon=h,
                temperature_only=temperature_only,
                city_names=cities,
                load_mw_history=load_mw_history,
                city_weather_fetcher=city_weather_fetcher,
            )
            .skb.set_name(f"feat_{h}h")
            .skb.set_description(
                f"Features to use for predicting the {h}h horizon. "
                f"They only use data available at least {h} hours before the target time."
            )
            .skb.drop(["prediction_time", "target_time"])
            .skb.apply(regressor, y=y[f"{h}h"])
            .skb.set_name(f"pred_{h}h")
            .skb.set_description(
                f"Predicted load {h} hours after the 'prediction_time' in X."
            )
        )
        all_pred[h] = pred

    multi_horizon_pred = (
        skrub.as_data_op(all_pred)
        .skb.apply_func(concat_horizons)
        .skb.set_name("pred_multi_horizon")
        .skb.set_description(
            "Output of the pipeline: predicted loads at multiple horizons. "
            "Column Xh contains the predicted load X hours after "
            "the 'prediction_time' in X."
        )
        .skb.with_scoring("neg_mean_absolute_percentage_error")
    )
    return multi_horizon_pred


def get_env():
    """
    Default environment for experimenting with / validating the dataop defined
    in this module.
    """
    return {
        "start": "2021-03-23",
        "end": "2025-05-31",
    }


def concat_X_y_predictions(X_test, y_test, prediction):
    return pl.concat(
        [
            X_test,
            y_test,
            prediction.rename("pred_{}".format),
        ],
        how="horizontal",
    )


def cross_val_predict(data_op, environment=None):
    """
    Get cross-validated predictions for different horizons.
    """
    all_predictions, all_scores = [], {"mape": []}
    for i, split in enumerate(data_op.skb.iter_cv_splits(environment=environment)):
        learner = data_op.skb.make_learner()
        learner.fit(split["train"])
        prediction = learner.predict(split["test"])
        all_predictions.append(
            concat_X_y_predictions(
                split["X_test"], split["y_test"], prediction
            ).with_columns(split=pl.lit(i)),
        )
        mape = mean_absolute_percentage_error(
            split["y_test"], prediction, multioutput="raw_values"
        )
        print(
            split["X_test"]["prediction_time"].min().strftime("%Y-%m-%d")
            + ": "
            + " ".join([f"{h}: {m:.1%}" for h, m in zip(prediction.columns, mape)])
        )
        all_scores["mape"].append(mape.tolist())

    all_predictions = pl.concat(all_predictions, how="vertical")
    return all_predictions, all_scores


def plot_predictions(results, horizons=None):
    if horizons is None:
        horizons = [
            int(m.group(1))
            for c in results.columns
            if (m := re.match(r"^pred_(\d+)h$", c)) is not None
        ]
    fig = go.Figure()
    for i, h in enumerate(horizons):
        target_time = results["prediction_time"] + datetime.timedelta(hours=h)
        if not i:
            fig.add_trace(
                go.Scatter(
                    x=target_time,
                    y=results[f"{h}h"],
                    mode="lines+markers",
                    name="true_load_mw",
                    hovertemplate="%{x|%Y-%m-%d} (%{x|%A}): %{y}<extra></extra>",
                )
            )
        fig.add_trace(
            go.Scatter(
                x=target_time,
                y=results[f"pred_{h}h"],
                mode="lines+markers",
                name=f"predicted_load_mw_{h}h",
                hovertemplate="%{x|%Y-%m-%d} (%{x|%A}): %{y}<extra></extra>",
            )
        )
    fig.update_layout(height=600, title=f"CV predicted load mw")
    return fig


def get_report_predictions(report):
    """
    Get predictions out of a skore report.
    """
    all_predictions = []
    for i, r in enumerate(report.estimator_reports_):
        all_predictions.append(
            concat_X_y_predictions(
                r.X_test, r.y_test, r.get_predictions(data_source="test")
            ).with_columns(split=pl.lit(i))
        )
    return pl.concat(all_predictions, how="vertical")
