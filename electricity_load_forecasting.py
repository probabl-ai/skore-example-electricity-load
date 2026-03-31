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


def fetch_load_mw_history(how="local"):
    if callable(how):
        return how()
    assert isinstance(how, str) and how == "local"
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


def get_1h_average(load_mw_history):
    return load_mw_history.group_by(pl.col("time").dt.truncate("1h")).agg(
        pl.col("load_mw").mean()
    )


def time_range(start, end):
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


def resample(df):
    all_times = df["time"]
    return time_range(all_times.min(), all_times.max()).join(
        df, on="time", how="left", maintain_order="left"
    )


class GetXy(BaseEstimator):
    def fit_transform(self, data, y=None):
        new_data = data["target_time"].join(
            data["load_mw_history"].drop_nulls(),
            on="time",
            how="inner",
            maintain_order="left",
        )
        return {
            "X": new_data[["time"]],
            "y": new_data[f"load_mw"],
        }

    def transform(self, data):
        return {"X": data["target_time"], "y": None}

    def fit(self, data, y=None):
        return self


def add_lagged_features(target_time, load_mw_history, horizon):
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
    features = resample(load_mw_history).select(pl.col("time"), *lags, *medians, *iqr)
    return target_time.join(features, on="time", how="left", maintain_order="left")


def add_weather(target_time, horizon, city_names="all", temp_only=False, how="local"):
    assert horizon <= 24
    # NOTE: here ideally we should retrieve the exact weather forecast
    # corresponding to the horizon. But we do not have it available in the
    # historical data. We just take the only forecast we have.
    if callable(how):
        return how(target_time, city_names)
    assert isinstance(how, str) and how == "local"
    if isinstance(city_names, str):
        assert city_names == "all"
        city_names = _ALL_CITIES
    with_weather = target_time.lazy()
    for city in city_names:
        with_weather = with_weather.join(
            pl.scan_parquet(data_dir() / f"weather_{city}.parquet")
            .with_columns(pl.col("time").dt.cast_time_unit("us"))
            .select(
                (pl.col("time"), cs.matches(".*temperature.*"))
                if temp_only
                else pl.all()
            )
            .select(
                pl.col("time"),
                (~cs.by_name("time"))
                .as_expr()
                .name.map(f"weather_{{}}_{city}".format),
            ),
            on="time",
            how="left",
            maintain_order="left",
        )
    return with_weather.collect()


def add_holidays(target_time):
    fr_time = pl.col("time").dt.convert_time_zone("Europe/Paris")
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


def _split_indices(X, test_start_date, test_length_days):
    train = (
        X.with_row_index()
        .filter(
            pl.col("time") < test_start_date - datetime.timedelta(_TRAIN_TEST_GAP_DAYS)
        )["index"]
        .to_numpy()
    )
    test = (
        X.with_row_index()
        .filter(
            (pl.col("time") >= test_start_date)
            & (
                pl.col("time")
                < test_start_date + datetime.timedelta(days=test_length_days)
            )
        )["index"]
        .to_numpy()
    )
    return train, test


class Splitter:
    def split(self, X, y=None, groups=None):
        min_train_days = 365 * 2
        test_length_days = 24 * 7  # 24 weeks
        test_start_dates = pl.date_range(
            X["time"].min()
            + datetime.timedelta(days=min_train_days + _TRAIN_TEST_GAP_DAYS),
            X["time"].max(),
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


def make_data_op(horizon=24):
    range_start = skrub.var("start")
    range_end = skrub.var("end")
    load_mw_source = skrub.var("load_mw_source")
    weather_source = skrub.var("weather_source")

    horizon = skrub.as_data_op(horizon).skb.set_name("horizon")

    load_mw_history = load_mw_source.skb.apply_func(
        fetch_load_mw_history
    ).skb.apply_func(get_1h_average)

    target_time = skrub.deferred(time_range)(range_start, range_end)
    filtered = skrub.as_data_op(
        {"target_time": target_time, "load_mw_history": load_mw_history}
    ).skb.apply(GetXy())
    X = filtered["X"].skb.mark_as_X()
    y = filtered["y"].skb.mark_as_y()

    temp_only = skrub.choose_bool(name="temp_only", default=False)
    cities = skrub.choose_from(["all", ["paris", "lyon", "marseille"]], name="cities")
    features = (
        X.skb.apply_func(add_lagged_features, load_mw_history, horizon=horizon)
        .skb.apply_func(
            add_weather,
            how=weather_source,
            horizon=horizon,
            temp_only=temp_only,
            city_names=cities,
        )
        .skb.apply_func(add_holidays)
    )

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
    pred = features.skb.apply(regressor, y=y)

    return pred


def get_env():
    return {
        "start": "2021-03-23",
        "end": "2025-05-31",
        "load_mw_source": "local",
        "weather_source": "local",
    }


def cross_val_predict(data_op, environment=None):
    all_predictions, all_scores = [], {"mape": []}
    for i, split in enumerate(
        data_op.skb.iter_cv_splits(cv=Splitter(), environment=environment)
    ):
        learner = data_op.skb.make_learner()
        learner.fit(split["train"])
        prediction = learner.predict(split["test"])
        all_predictions.append(
            split["X_test"].with_columns(
                true_load_mw=split["y_test"],
                predicted_load_mw=prediction,
                split=pl.lit(i),
            )
        )
        mape = mean_absolute_percentage_error(split["y_test"], prediction)
        print(f"{mape:.1%}")
        all_scores["mape"].append(mape)

    all_predictions = pl.concat(all_predictions, how="vertical")
    return all_predictions, all_scores


def plot_predictions(results):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=results["time"],
            y=results["true_load_mw"],
            mode="lines+markers",
            name="true_load_mw",
            hovertemplate="%{x|%Y-%m-%d} (%{x|%A}): %{y}<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=results["time"],
            y=results["predicted_load_mw"],
            mode="lines+markers",
            name="predicted_load_mw",
            hovertemplate="%{x|%Y-%m-%d} (%{x|%A}): %{y}<extra></extra>",
        )
    )
    fig.update_layout(height=600, title="CV predicted load mw")
    return fig


def get_report_predictions(report):
    all_predictions = []
    for i, r in enumerate(report.estimator_reports_):
        all_predictions.append(
            r.X_test.with_columns(
                true_load_mw=r.y_test,
                predicted_load_mw=r.get_predictions(data_source="test"),
                split=pl.lit(i),
            )
        )
    return pl.concat(all_predictions, how="vertical")
