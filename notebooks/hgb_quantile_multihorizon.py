"""
A simplified version that does not have any options to control the quantile
strategy, single or multi-horizon etc. always does multi horizon + quantiles
with histogram gradient boosting.
"""

import pickle
import re
import datetime
import json
from pathlib import Path

import holidays
import polars as pl
from polars import selectors as cs
import skrub
from sklearn.metrics import mean_absolute_percentage_error, d2_pinball_score
from sklearn.base import BaseEstimator, RegressorMixin, clone
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


def data_dir():
    return Path(__file__).resolve().parents[1] / "datasets"


def fetch_consumption_history():
    return (
        pl.read_csv(data_dir() / "Total Load - Day Ahead*.csv", null_values=["N/A", "-"])
        .drop_nulls()
        .select(
            pl.col("Time (UTC)")
            .str.split(by=" - ")
            .list.first()
            .str.to_datetime("%d.%m.%Y %H:%M", time_zone="UTC")
            .alias("time"),
            pl.col("Actual Total Load [MW] - BZN|FR").cast(pl.Float32).alias("load_mw"),
        )
    )


def time_range(start, end=None):
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
        )
        .dt.truncate("1h")
        .alias("time"),
    )


def resample(consumption_history):
    averaged = consumption_history.group_by(pl.col("time").dt.truncate("1h")).agg(
        pl.col("load_mw").mean()
    )
    all_times = averaged["time"]
    return time_range(
        all_times.min(), all_times.max() + datetime.timedelta(hours=48)
    ).join(averaged, on="time", how="left", maintain_order="left")


def get_X_y(df, consumption_history, horizons, mode=skrub.eval_mode()):
    df = df.rename({"time": "prediction_time"})
    if mode in ("fit", "fit_transform", "preview"):
        consumption = consumption_history.select(
            pl.col("time"),
            *[pl.col("load_mw").shift(-h).alias(f"{h}h") for h in horizons],
        ).drop_nulls()
        X_y = df.join(
            consumption,
            left_on="prediction_time",
            right_on="time",
            how="inner",
            maintain_order="left",
        )
        return {
            "X": X_y.select(pl.col("prediction_time")),
            "y": X_y.drop("prediction_time"),
        }
    else:
        return {"X": prediction_time}


def add_target_time(df, horizon):
    return df.with_columns(
        (pl.col("prediction_time") + pl.duration(hours=horizon)).alias("target_time")
    )


def add_lagged_features(df, consumption_history, horizon):
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
    features = consumption_history.select(pl.col("time"), *lags, *medians, *iqr)
    return df.join(
        features,
        left_on="target_time",
        right_on="time",
        how="left",
        maintain_order="left",
    )


def fetch_city_weather(city):
    return pl.read_parquet(data_dir() / f"weather_{city}.parquet")


def add_weather(
    df,
    horizon,
    city_names="all",
    temperature_only=True,
    city_weather_fetcher=fetch_city_weather,
):
    del horizon
    if isinstance(city_names, str):
        assert city_names == "all"
        city_names = _ALL_CITIES
    with_weather = df
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
                (~cs.by_name("time")).as_expr().name.map(f"weather_{{}}_{city}".format),
            ),
            left_on="target_time",
            right_on="time",
            how="left",
            maintain_order="left",
        )
    return with_weather


def add_calendar_and_holidays(df):
    fr_time = pl.col("target_time").dt.convert_time_zone("Europe/Paris")
    fr_year_min = df.select(fr_time.dt.year().min()).item()
    fr_year_max = df.select(fr_time.dt.year().max()).item()
    holidays_fr = holidays.country_holidays(
        "FR", years=range(fr_year_min, fr_year_max + 1)
    )
    return df.with_columns(
        fr_time.dt.hour().alias("cal_hour_of_day"),
        fr_time.dt.weekday().alias("cal_day_of_week"),
        fr_time.dt.ordinal_day().alias("cal_day_of_year"),
        fr_time.dt.year().alias("cal_year"),
        fr_time.dt.date().is_in(holidays_fr.keys()).alias("cal_is_holiday"),
    )


def add_features(
    df, horizon, temperature_only, city_names, consumption_history, city_weather_fetcher
):
    df = add_target_time(df, horizon=horizon)
    df = add_weather(
        df,
        horizon=horizon,
        temperature_only=temperature_only,
        city_names=city_names,
        city_weather_fetcher=city_weather_fetcher,
    )
    df = add_calendar_and_holidays(df)
    df = add_lagged_features(df, consumption_history=consumption_history, horizon=horizon)
    return df.drop(["prediction_time", "target_time"])


def concat_horizons(all_pred, mode=skrub.eval_mode()):
    if mode == "fit":
        return all_pred
    return pl.concat(
        [v.rename(f"{h}h__{{}}".format) for h, v in all_pred.items()], how="horizontal"
    )


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
            train, test = _split_indices(X, test_start, test_length_days=test_length_days)
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


def split_by_quantile(pred):
    quantile_cols = {}
    for c in pred.columns:
        quantile_cols.setdefault(c.split("__")[1], []).append(c)
    return {
        q: pred.select(cols).rename(lambda c: c.split("__")[0])
        for q, cols in quantile_cols.items()
    }


def neg_mape(y_true, y_pred):
    quantile_predictions = split_by_quantile(y_pred)
    scores = {}
    for q, q_pred in quantile_predictions.items():
        scores[f"neg_mape__average__{q}"] = -mean_absolute_percentage_error(
            y_true, q_pred
        )
        detail = mean_absolute_percentage_error(y_true, q_pred, multioutput="raw_values")
        scores.update(
            {f"neg_mape__{c}__{q}": -float(s) for c, s in zip(y_true.columns, detail)}
        )
    return scores


def neg_mape_scorer(estimator, X, y):
    return neg_mape(y, estimator.predict(X))


def pinball(y_true, y_pred):
    quantile_predictions = split_by_quantile(y_pred)
    scores = {}
    for q, q_pred in quantile_predictions.items():
        scores[f"d2_pinball_score__average__{q}"] = d2_pinball_score(
            y_true, q_pred, alpha=float(q.removeprefix("q_"))
        )
        detail = d2_pinball_score(y_true, q_pred, multioutput="raw_values")
        scores.update(
            {
                f"d2_pinball_score__{c}__{q}": float(s)
                for c, s in zip(y_true.columns, detail)
            }
        )
    return scores


def pinball_scorer(estimator, X, y):
    return pinball(y, estimator.predict(X))


class QuantileRegressor(RegressorMixin, BaseEstimator):
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


def make_data_op(horizons=(1, 12, 24), quantiles=(0.05, 0.5, 0.95)):
    range_start = skrub.var("start")
    range_end = skrub.var("end")
    history_fetcher = skrub.var(
        "history_fetcher", fetch_consumption_history, becomes_default=True
    )
    weather_fetcher = skrub.var(
        "weather_fetcher", fetch_city_weather, becomes_default=True
    )
    prediction_time = skrub.deferred(time_range)(range_start, range_end)
    history = history_fetcher().skb.apply_func(resample)
    X_y = prediction_time.skb.apply_func(get_X_y, history, horizons)
    X = X_y["X"].skb.mark_as_X(cv=TimeSeriesSplitter())
    y = X_y["y"].skb.mark_as_y()
    temperature_only = skrub.choose_bool(name="temperature_only", default=True)
    cities = skrub.choose_from(["all", ["paris", "lyon", "marseille"]], name="cities")
    learning_rate = skrub.choose_float(
        0.01, 0.7, default=0.1, log=True, name="learning_rate"
    )
    max_leaf_nodes = skrub.choose_int(3, 300, default=30, log=True, name="max_leaf_nodes")
    hgb_params = dict(
        random_state=0,
        max_iter=500,
        early_stopping=True,
        n_iter_no_change=100,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
    )
    predictor = QuantileRegressor(hgb_params=hgb_params, quantiles=quantiles)
    all_pred = {}
    for h in horizons:
        all_pred[h] = (
            X.skb.apply_func(
                add_features,
                horizon=h,
                temperature_only=temperature_only,
                city_names=cities,
                consumption_history=history,
                city_weather_fetcher=weather_fetcher,
            )
            .skb.apply(predictor, y=y[f"{h}h"])
            .skb.set_name(f"pred_{h}h")
        )
    return (
        skrub.deferred(concat_horizons)(all_pred)
        .skb.with_scoring(neg_mape_scorer)
        .skb.with_scoring(pinball_scorer)
    )


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
    all_predictions, all_scores = [], []
    for i, split in enumerate(data_op.skb.iter_cv_splits(environment=environment)):
        learner = data_op.skb.make_learner().fit(split["train"])
        score, predictions = learner.score(split["test"], return_predictions=True)
        all_predictions.append(
            concat_X_y_predictions(
                split["X_test"], split["y_test"], predictions["predict"]
            ).with_columns(split=pl.lit(i)),
        )
        print(i, split["X_test"]["prediction_time"].min().isoformat())
        print(score)
        all_scores.append(score | {"split": i})
    all_predictions = pl.concat(all_predictions, how="vertical")
    all_scores = pl.DataFrame(all_scores)
    return all_predictions, all_scores


def plot_predictions(results, horizons=None, start="2025-03-01"):
    if start is not None:
        results = results.filter(
            pl.col("prediction_time")
            > datetime.datetime.fromisoformat(start).astimezone(datetime.UTC)
        )
    if horizons is None:
        horizons = sorted(
            {
                int(m.group(1))
                for c in results.columns
                if (m := re.match(r"^pred_(\d+)h.*$", c)) is not None
            }
        )
    fig = go.Figure()
    for i, h in enumerate(horizons):
        target_time = results["prediction_time"] + datetime.timedelta(hours=h)
        if not i:
            fig.add_trace(
                go.Scatter(
                    x=target_time,
                    y=results[f"{h}h"],
                    mode="lines+markers",
                    line={"dash": "dash"},
                    name="true_load_mw",
                    hovertemplate="%{x|%Y-%m-%d} (%{x|%A}): %{y}<extra></extra>",
                )
            )
        for col in filter(lambda c: f"pred_{h}h" in c, results.columns):
            fig.add_trace(
                go.Scatter(
                    x=target_time,
                    y=results[col],
                    mode="lines",
                    name=col,
                    hovertemplate="%{x|%Y-%m-%d} (%{x|%A}): %{y}<extra></extra>",
                )
            )
    fig.update_layout(height=600, title=f"CV predicted load mw")
    return fig


def get_report_predictions(report):
    all_predictions = []
    for i, r in enumerate(report.reports_):
        all_predictions.append(
            concat_X_y_predictions(
                r.X_test, r.y_test, r.get_predictions(data_source="test")
            ).with_columns(split=pl.lit(i))
        )
    return pl.concat(all_predictions, how="vertical")


if __name__ == "__main__":

    env = {
        "start": "2021-03-23",
        "end": "2025-05-31",
    }

    pred = make_data_op()
    pred.skb.full_report(env)
    split = pred.skb.train_test_split(environment=env, split_func=train_test_split)

    storage = f"sqlite:///optuna.db"
    study_name = "randomized_search"
    search = pred.skb.make_randomized_search(
        backend="optuna",
        n_iter=2,
        n_jobs=2,
        refit="d2_pinball_score__average__q_0.5",
        storage=storage,
        study_name=study_name,
    )
    search.fit(split["train"])
    with open("search.pickle", "wb") as f:
        pickle.dump(search, f)
    scores, predictions = search.best_learner_.score(
        split["test"], return_predictions=True
    )
    results = concat_X_y_predictions(
        split["X_test"], split["y_test"], predictions["predict"]
    )
    print(scores)
    fig = plot_predictions(results, horizons=(1,))
    fig.show(renderer="browser")
    fig = plot_predictions(results, horizons=(24,))
    fig.show(renderer="browser")
