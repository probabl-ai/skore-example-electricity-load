import datetime
import functools
import json
import re
import subprocess
import sys
import time
from pathlib import Path

import holidays
import plotly.graph_objects as go
import polars as pl
import skrub
from polars import selectors as cs
from sklearn.ensemble import HistGradientBoostingRegressor

from .calibration import Recalibrator
from .metrics import coverage_scorer, mean_width_scorer, neg_mape_scorer, pinball_scorer
from .quantile_regressor import BinnedQuantileRegressor, HGBQuantileRegressor
from .splitter import TimeSeriesSplitter

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


def project_dir():
    # repository root: src/electricity_load_forecasting/pipeline.py
    return Path(__file__).resolve().parents[2]


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
                "argv": sys.argv,
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
            pl.col("Actual Total Load [MW] - BZN|FR").cast(pl.Float32).alias("load_mw"),
        )
        .collect()
    )


def time_range(start, end=None):
    """
    Build a 1-hour-spaced datetime range from start to end.

    Times are truncated to the nearest full hour.

    If end is None, we get a time range containing only the start time.
    """
    if end is None:
        end = start
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
    return time_range(
        all_times.min(), all_times.max() + datetime.timedelta(hours=48)
    ).join(averaged, on="time", how="left", maintain_order="left")


def get_X_y(prediction_time, load_mw_history, horizons, mode=skrub.eval_mode()):
    """
    Compute input and target variables.

    For fitting (and validation), this builds the targets y by applying
    appropriate shifts to the historical data. The targets y and prediction
    times X are aligned, and rows with missing ground truth are dropped.
    Returns a dictionary with keys X and y, ready to be split for
    cross-validation or used to fit a model.

    For prediction, simply returns `prediction_time` in a dictionary with a
    single key X.
    """
    if isinstance(horizons, int):
        single_horizon = True
        horizons = (horizons,)
    else:
        single_horizon = False
    prediction_time = prediction_time.rename({"time": "prediction_time"})
    if mode in ("fit", "fit_transform", "preview"):
        # For those modes we need the ground truth; restrict to rows for which
        # there is y
        load = load_mw_history.select(
            pl.col("time"),
            *[pl.col("load_mw").shift(-h).alias(f"{h}h") for h in horizons],
        ).drop_nulls()
        X_y = prediction_time.join(
            load,
            left_on="prediction_time",
            right_on="time",
            how="inner",
            maintain_order="left",
        )
        return {
            "X": X_y.select(pl.col("prediction_time")),
            "y": (
                X_y[f"{horizons[0]}h"]
                if single_horizon
                else X_y.drop("prediction_time")
            ),
        }
    else:
        # In predict mode there is no y and we return unmodified query
        return {"X": prediction_time}


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
                index_column="time", period=f"{width}h", offset=f"{-width - lag}h"
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
    horizon,
    city_names="all",
    temperature_only=False,
    city_weather_fetcher=fetch_city_weather,
):
    """Add weather information for the required cities."""
    # NOTE: here ideally we should retrieve the exact weather forecast
    # corresponding to the horizon. But we do not have it available in the
    # historical data. We just take the only forecast we have.
    del horizon
    if isinstance(city_names, str):
        assert city_names == "all"
        city_names = _ALL_CITIES
    with_weather = target_time.lazy()

    rolls = [(lag * 3, 3) for lag in range(8)]
    rolling_means = [
        cs.matches(".*temperature.*")
        .as_expr()
        .mean()
        .rolling(index_column="time", period=f"{width}h", offset=f"{-width - lag}h")
        .name.map(f"{{}}_lag_{lag}_width_{width}".format)
        for lag, width in rolls
    ]

    for city in city_names:
        with_weather = with_weather.join(
            city_weather_fetcher(city)
            .with_columns(pl.col("time").dt.cast_time_unit("us"))
            .select(
                (pl.col("time"), cs.matches(".*temperature.*"))
                if temperature_only
                else pl.all()
            )
            .with_columns(*rolling_means)
            .select(
                pl.col("time"),
                (~cs.by_name("time")).as_expr().name.map(f"weather_{{}}_{city}".format),
            ),
            left_on="target_time",
            right_on="time",
            how="left",
            maintain_order="left",
        )
    return with_weather.collect()


def fetch_holidays(years):
    return holidays.country_holidays("FR", years=years)


def add_calendar_and_holidays(df, holidays_fetcher=fetch_holidays):
    """Add calendar features and holiday information."""
    fr_time = pl.col("target_time").dt.convert_time_zone("Europe/Paris")
    fr_year_min = df.select(fr_time.dt.year().min()).item()
    fr_year_max = df.select(fr_time.dt.year().max()).item()
    holidays_fr = fetch_holidays(years=range(fr_year_min, fr_year_max + 1))
    return df.with_columns(
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
    df,
    horizon,
    temperature_only,
    city_names,
    load_mw_history,
    city_weather_fetcher,
    holidays_fetcher,
):
    df = add_target_time(df, horizon=horizon)
    df = add_weather(
        df,
        horizon=horizon,
        temperature_only=temperature_only,
        city_names=city_names,
        city_weather_fetcher=city_weather_fetcher,
    )
    df = add_calendar_and_holidays(df, holidays_fetcher=holidays_fetcher)
    df = add_lagged_features(df, load_mw_history=load_mw_history, horizon=horizon)
    return df


def concat_horizons(all_pred, quantile_regression=False, mode=skrub.eval_mode()):
    """
    Consolidate predictions of models for different horizons in one dataframe.

    Output columns look like '12h__q_0.05'
    """
    if mode == "fit":
        return all_pred
    if not quantile_regression:
        return pl.DataFrame({f"{h}h": v for h, v in all_pred.items()})
    return pl.concat(
        [v.rename(f"{h}h__{{}}".format) for h, v in all_pred.items()], how="horizontal"
    )


def transpose_pred(prediction_date, prediction):
    date = [
        prediction_date + datetime.timedelta(hours=int(c.removesuffix("h")))
        for c in prediction.columns
    ]
    load = prediction.row()
    return pl.DataFrame({"time": date, "load_mw": load})


def post_process(pred, prediction_time, range_end, quantile_regression):
    if range_end is not None:
        return pred
    pred_time = prediction_time["time"].to_list()[0]
    horizons, q = zip(
        *(re.match(r"(\d+)h(?:__(q_.*))?", c).groups() for c in pred.columns)
    )
    date = [pred_time + datetime.timedelta(hours=int(h)) for h in horizons]
    load = pred.row()
    if not quantile_regression:
        return pl.DataFrame({"time": date, "load_mw": load})
    return pl.DataFrame({"time": date, "load_mw": load, "quantile": q}).pivot(
        on="quantile", values="load_mw", maintain_order=True, sort_columns=False
    )


def _recalibrator_cv():
    """
    Inner walk-forward CV of the Recalibrator, on the training set of an outer
    fold (the prediction times are passed with ``split_kwargs``). Same splitter,
    test fold length and gap as the outer CV, but starting after 1 year of
    training data instead of 2, so that the smallest outer training set still
    gets several inner splits. Caveat: the model of the first inner splits (1
    year of data) is weaker than the final one, which makes the measured errors
    a bit larger.
    """
    return TimeSeriesSplitter(min_train_days=365)


def tabicl_quantiles_to_df(prediction, quantiles, mode=skrub.eval_mode()):
    if mode == "fit":
        return prediction
    return pl.DataFrame(prediction, schema=[f"q_{q}" for q in quantiles])


def limit_train_size(df, size=9000, mode=skrub.eval_mode()):
    if mode in ("fit", "fit_transform", "preview"):
        return df.tail(size)
    else:
        return df


def make_data_op(
    horizons=(1, 12, 24),
    quantile_strategy=None,
    quantiles=(0.05, 0.5, 0.95),
    tabicl_n_estimators=1,
    max_train_rows=None,
    max_iter=3000,
    recalibrate=False,
):
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

    tabicl_n_estimators : int, default=1
        Number of estimators for ``TabICLRegressor`` (ignored unless
        ``quantile_strategy == "tabicl"``). TabICL ensembles over estimators
        fit on different row/feature permutations; more estimators costs
        proportionally more inference compute.

    max_train_rows : int or None, default=None
        Max number of rows used for training.

    recalibrate : bool, default=False
        If True, wrap the quantile regressor of each horizon in a
        ``calibration.Recalibrator``, which corrects its predicted quantiles
        based on their out-of-fold reliability (see that class). Only
        supported for ``quantile_strategy == "multiple_regressors"``. The
        recalibrator's inner CV is ``_recalibrator_cv()``.
    """
    if recalibrate and quantile_strategy != "multiple_regressors":
        raise ValueError(
            "recalibrate=True is only supported with "
            f"quantile_strategy='multiple_regressors', got {quantile_strategy!r}"
        )
    range_start = skrub.var("start").skb.set_description(
        "The first time at which a prediction is made."
    )
    range_end = skrub.var("end", None, becomes_default=True).skb.set_description(
        "The last time at which a prediction is made."
    )
    load_mw_history_fetcher = skrub.var(
        "load_mw_history_fetcher", fetch_load_mw_history, becomes_default=True
    ).skb.set_description(
        "Function that loads the historical load data. "
        "See signature of electricity_load_forecasting.fetch_load_mw_history."
    )
    city_weather_fetcher = skrub.var(
        "city_weather_fetcher", fetch_city_weather, becomes_default=True
    ).skb.set_description(
        "Function that loads the weather forecast for a city. "
        "See signature of electricity_load_forecasting.fetch_city_weather."
    )
    holidays_fetcher = skrub.var(
        "holidays_fetcher", fetch_holidays, becomes_default=True
    )
    prediction_time = skrub.deferred(time_range)(range_start, range_end)
    load_mw_history = (
        load_mw_history_fetcher()
        .skb.apply_func(resample)
        .skb.set_description("Historical load data on a regular 1h time grid.")
    )
    X_y = prediction_time.skb.apply_func(get_X_y, load_mw_history, horizons)
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

    if max_train_rows is not None:
        X = X.skb.apply_func(limit_train_size, size=max_train_rows)
        y = y.skb.apply_func(limit_train_size, size=max_train_rows)
    temperature_only = skrub.choose_bool(name="temperature_only", default=True)
    cities = skrub.choose_from(["all", ["paris", "lyon", "marseille"]], name="cities")

    quantiles_to_predict = skrub.var("quantiles", quantiles, becomes_default=True)
    quantile_regression = quantile_strategy is not None
    learning_rate = skrub.choose_float(
        0.01, 0.7, default=0.1, log=True, name="learning_rate"
    )
    max_leaf_nodes = skrub.choose_int(
        2, 128, default=31, log=True, name="max_leaf_nodes"
    )
    min_samples_leaf = skrub.choose_int(
        2, 128, default=20, log=True, name="min_samples_leaf"
    )
    hgb_params = dict(
        random_state=0,
        max_iter=max_iter,
        early_stopping=True,
        n_iter_no_change=100,
        max_depth=None,
        learning_rate=learning_rate,
        max_leaf_nodes=max_leaf_nodes,
        min_samples_leaf=min_samples_leaf,
    )
    hgb_regressor = HistGradientBoostingRegressor(
        loss=skrub.choose_from(["squared_error", "poisson", "gamma"], name="loss"),
        **hgb_params,
    )
    hgb_q_regressor = HGBQuantileRegressor(quantiles=quantiles, hgb_params=hgb_params)
    binned_q_regressor = BinnedQuantileRegressor()
    predict_kwargs = None
    fit_kwargs = None
    if quantile_strategy == "binning":
        predictor = binned_q_regressor
        predict_kwargs = {"quantiles": quantiles_to_predict}
    elif quantile_strategy == "multiple_regressors":
        predictor = hgb_q_regressor
        if recalibrate:
            predictor = Recalibrator(hgb_q_regressor, cv=_recalibrator_cv())
            # the inner CV needs the prediction times of the training rows, which
            # are not in the numpy array of features that the estimator receives
            fit_kwargs = {"split_kwargs": {"prediction_time": X["prediction_time"]}}
    elif quantile_strategy == "tabicl":
        from tabicl import TabICLRegressor

        predictor = TabICLRegressor(n_estimators=tabicl_n_estimators)
        predict_kwargs = {"output_type": "quantiles", "alphas": quantiles_to_predict}
    elif quantile_strategy is None:
        predictor = hgb_regressor
    else:
        raise ValueError(f"Bad quantile strategy: {quantile_strategy!r}")
    all_pred = {}
    for h in horizons:
        h_pred = (
            X.skb.apply_func(
                add_features,
                horizon=h,
                temperature_only=temperature_only,
                city_names=cities,
                load_mw_history=load_mw_history,
                city_weather_fetcher=city_weather_fetcher,
                holidays_fetcher=holidays_fetcher,
            )
            .skb.set_name(f"feat_{h}h")
            .skb.set_description(
                f"Features to use for predicting the {h}h horizon. "
                f"They only use data available at least {h} hours before the target time."
            )
            .skb.drop(["prediction_time", "target_time"])
            .skb.apply(skrub.ToFloat())
            .to_numpy()
            .skb.apply(
                predictor,
                y=y[f"{h}h"],
                fit_kwargs=fit_kwargs,
                predict_kwargs=predict_kwargs,
            )
        )
        if quantile_strategy == "tabicl":
            h_pred = h_pred.skb.apply_func(tabicl_quantiles_to_df, quantiles_to_predict)
        all_pred[h] = h_pred.skb.set_name(f"pred_{h}h").skb.set_description(
            f"Predicted load {h} hours after the 'prediction_time' in X."
        )

    multi_horizon_pred = (
        skrub.deferred(concat_horizons)(
            all_pred, quantile_regression=quantile_regression
        )
        .skb.set_name("pred_multi_horizon")
        .skb.set_description(
            "Output of the pipeline: predicted loads at multiple horizons. "
            "Column Xh contains the predicted load X hours after "
            "the 'prediction_time' in X."
        )
        .skb.apply_func(
            post_process,
            prediction_time,
            range_end,
            quantile_regression=quantile_regression,
        )
        .skb.with_scoring(
            functools.partial(neg_mape_scorer, quantile_regression=quantile_regression)
        )
    )
    if not quantile_regression:
        return multi_horizon_pred
    kwargs = {"q_low": min(quantiles), "q_high": max(quantiles)}
    return (
        multi_horizon_pred.skb.with_scoring(pinball_scorer)
        .skb.with_scoring(functools.partial(coverage_scorer, **kwargs))
        .skb.with_scoring(functools.partial(mean_width_scorer, **kwargs))
    )


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
    all_predictions, all_scores = [], []
    for i, split in enumerate(data_op.skb.iter_cv_splits(environment=environment)):
        t0 = time.perf_counter()
        learner = data_op.skb.make_learner().fit(split["train"])
        fit_time = time.perf_counter() - t0
        t0 = time.perf_counter()
        score, prediction = learner.score(split["test"], return_predictions=True)
        score_time = time.perf_counter() - t0
        all_predictions.append(
            concat_X_y_predictions(
                split["X_test"], split["y_test"], prediction["predict"]
            ).with_columns(split=pl.lit(i)),
        )
        print(split["X_test"]["prediction_time"].min().isoformat())
        print(score)
        all_scores.append(
            score | {"split": i, "fit_time": fit_time, "score_time": score_time}
        )
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
    fig.update_layout(height=600, title="CV predicted load mw")
    return fig


def get_skore_report_predictions(report):
    """
    Get predictions out of a skore report.
    """
    all_predictions = []
    for i, r in enumerate(report.reports_):
        all_predictions.append(
            concat_X_y_predictions(
                r.X_test, r.y_test, r.get_predictions(data_source="test")
            ).with_columns(split=pl.lit(i))
        )
    return pl.concat(all_predictions, how="vertical")


def get_new_date(add_hours=0):
    """Get the first hour out of the range of historical data"""
    return (
        (fetch_load_mw_history()["time"] - datetime.timedelta(seconds=1)).dt.truncate(
            "1h"
        )
        + datetime.timedelta(hours=add_hours)
    ).max()
