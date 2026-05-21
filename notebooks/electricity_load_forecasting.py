from pathlib import Path
import polars as pl


def data_dir():
    return Path(".").resolve().parent / "datasets"


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


# %%
import skrub

raw_load_mw_history = skrub.deferred(fetch_load_mw_history)()
raw_load_mw_history

# %%
import datetime


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


range_start = skrub.var("start", "2021-03-23")
range_end = skrub.var("end", "2025-05-31")

prediction_time = skrub.deferred(time_range)(range_start, range_end)
prediction_time


# %%
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


load_mw_history = raw_load_mw_history.skb.apply_func(resample)
load_mw_history


# %%
HORIZONS = (1, 12, 24)


def get_X_y(prediction_time, load_mw_history, horizons, mode=skrub.eval_mode()):
    if mode in ("fit", "fit_transform", "preview"):
        load = load_mw_history.select(
            pl.col("time"),
            *[pl.col("load_mw").shift(-h).alias(f"{h}h") for h in horizons],
        ).drop_nulls()
        X_y = prediction_time.join(load, on="time", how="inner", maintain_order="left")
        return {
            "X": X_y.select(pl.col("time").alias("prediction_time")),
            "y": X_y.drop("time"),
        }
    else:
        return {"X": prediction_time}


X_y = prediction_time.skb.apply_func(get_X_y, load_mw_history, HORIZONS)
X_y["X"]

# %%
X_y["y"]

# %%
X_val = X_y["X"].skb.preview()
X_val

# %%
TRAIN_TEST_GAP_DAYS = 7


def _split_indices(X, test_start_date, test_length_days):
    train = (
        X.with_row_index()
        .filter(
            pl.col("prediction_time")
            < test_start_date - datetime.timedelta(TRAIN_TEST_GAP_DAYS)
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
            + datetime.timedelta(days=min_train_days + TRAIN_TEST_GAP_DAYS),
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


next(iter(TimeSeriesSplitter().split(X_val)))

# %%
X = X_y["X"].skb.mark_as_X(cv=TimeSeriesSplitter())


# %%
def add_target_time(df, horizon):
    return df.with_columns(
        (pl.col("prediction_time") + pl.duration(hours=horizon)).alias("target_time")
    )


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


feat_12h = X.skb.apply_func(add_target_time, 12).skb.apply_func(
    add_lagged_features, load_mw_history, 12
)
feat_12h

# %%
from polars import selectors as cs

ALL_CITIES = (
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


def fetch_city_weather(city):
    return pl.scan_parquet(data_dir() / f"weather_{city}.parquet")


def add_weather(
    target_time,
    city_names="all",
):
    """Add weather information for the required cities."""
    # NOTE: here ideally we should retrieve the exact weather forecast
    # corresponding to the horizon. But we do not have it available in the
    # historical data. We just take the only forecast we have.
    if isinstance(city_names, str):
        assert city_names == "all"
        city_names = ALL_CITIES
    with_weather = target_time.lazy()
    for city in city_names:
        with_weather = with_weather.join(
            fetch_city_weather(city)
            .with_columns(pl.col("time").dt.cast_time_unit("us"))
            .select((pl.col("time"), cs.matches(".*temperature.*")))
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


feat_12h_with_weather = feat_12h.skb.apply_func(add_weather)
feat_12h_with_weather

# %%
import holidays


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


feat_12h_with_calendar = feat_12h_with_weather.skb.apply_func(
    add_calendar_and_holidays
)
feat_12h_with_calendar


# %%
def add_features(df, horizon, load_mw_history):
    df = add_target_time(df, horizon=horizon)
    df = add_lagged_features(df, load_mw_history=load_mw_history, horizon=horizon)
    df = add_weather(df)
    df = add_calendar_and_holidays(df)
    return df


# %%
from sklearn.ensemble import HistGradientBoostingRegressor

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


def apply_predictor(X, y, horizon):
    return (
        X.skb.apply_func(
            add_features,
            horizon=horizon,
            load_mw_history=load_mw_history,
        )
        .skb.set_name(f"feat_{horizon}h")
        .skb.drop(["prediction_time", "target_time"])
        .skb.apply(regressor, y=y)
        .skb.set_name(f"pred_{horizon}h")
    )


pred_12h = apply_predictor(X, X_y["y"]["12h"].skb.mark_as_y(), 12)
pred_12h

# %%
pred_12h.skb.with_scoring("neg_mean_absolute_percentage_error").skb.cross_validate()

# %%
y = X_y["y"].skb.mark_as_y()

all_pred = {}
for h in HORIZONS:
    all_pred[h] = apply_predictor(X, y[f"{h}h"], h)


def concat_horizons(all_pred):
    """
    Consolidate predictions of models for different horizons in one dataframe.
    """
    return pl.DataFrame({f"{h}h": v for h, v in all_pred.items()})


multi_horizon_pred = (
    skrub.as_data_op(all_pred)
    .skb.apply_func(concat_horizons)
    .skb.set_name("pred_multi_horizon")
    .skb.with_scoring("neg_mean_absolute_percentage_error")
)
multi_horizon_pred

# %%
multi_horizon_pred.skb.cross_validate()

# %%
split = multi_horizon_pred.skb.train_test_split()
learner = multi_horizon_pred.skb.make_learner()
learner.fit(split["train"])
first_split_prediction = learner.predict(split["test"])
first_split_prediction

# %%
from sklearn.metrics import mean_absolute_percentage_error

mean_absolute_percentage_error(
    split["y_test"], first_split_prediction, multioutput="raw_values"
)


# %%
def concat_X_y_predictions(X_test, y_test, prediction):
    return pl.concat(
        [
            X_test,
            y_test,
            prediction.rename("pred_{}".format),
        ],
        how="horizontal",
    )


concat_X_y_predictions(split["X_test"], split["y_test"], first_split_prediction)

# %%


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


cv_predictions, scores = cross_val_predict(multi_horizon_pred)

# %%
import re

import plotly.graph_objects as go


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


plot_predictions(cv_predictions)
