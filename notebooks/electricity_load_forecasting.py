# %% [markdown]
# # Electricity load forecasting
#
# We build a pipeline that for a given prediction time, predicts the future
# electricity load. We start by doing it for 1 horizon, then we extend to
# predicting multiple horizons in the same pipeline.
#
# Our pipeline has as inputs:
#
# - The start and end of a time range. Every hour between start and end will
#   be a prediction time, and we will output the prediction that would have
#   been made at that time.
# - The function used to access historical load data. It is optional and by
#   default we fetch from the data directory in this repository.
# - The function used to access weather forecasts.  It is optional and by
#   default we fetch from the data directory in this repository.
#
# For each prediction time, it outputs predicted loads for 1 or several
# horizons.


# %%
from pathlib import Path
import polars as pl


def data_dir():
    return Path(".").resolve().parent / "datasets"


def results_dir():
    out = Path(".").resolve() / "results"
    out.mkdir(exist_ok=True)
    return out


def fetch_load_mw_history():
    """
    Fetch the historical electricity grid load in MW from the default data dir.

    Returns a dataframe with columns [time, load_mw].
    """
    return (
        pl.read_csv(data_dir() / "Total Load - Day Ahead*.csv", null_values=["N/A", "-"])
        .drop_nulls()
        .select(
            pl.col("Time (UTC)")
            .str.split(by=" - ")
            .list.first()
            .str.to_datetime("%d.%m.%Y %H:%M", time_zone="UTC")
            .alias("time"),
            pl.col("Actual Total Load [MW] - BZN|FR").alias("load_mw"),
        )
    )


# %% [markdown]
# We will use this function to fetch the historical load. However we may want
# to fetch data from a difference source when we use our trained model later,
# so we wrap it in a DataOp and set a name on it. This allows passing a
# different function to use instead of it whenever we evaluate our DataOp / use
# our SkrubLearner.

# %%
import skrub

# Note: after https://github.com/skrub-data/skrub/pull/2082 we can write this as:
# raw_load_mw_history = skrub.var(
#     "load_mw_history_fetcher", fetch_load_mw_history, store_default=True
# )()
raw_load_mw_history = skrub.as_data_op(fetch_load_mw_history).skb.set_name(
    "load_mw_history_fetcher"
)()
raw_load_mw_history


# %% [markdown]
# We now build up the range of prediction times. We declare 2 variables (with
# example values which will be used to compute preview results during
# interactive development) for the start and end time.

# %%
import datetime


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


range_start = skrub.var("start", "2021-03-23")
range_end = skrub.var("end", "2025-05-31")

prediction_time = skrub.deferred(time_range)(range_start, range_end)
prediction_time

# %% [markdown]
# The historical data is sampled irregularly, sometimes every hour, sometimes
# every 15 min, and with missing rows. We define a function to resample it on a
# regular 1h-spaced grid.
#
# As this will serve as the basis for our lagged features, we add a buffer of
# empty rows beyond the range of our data. We do not have the actual load for
# those rows, but lagged loads can be defined for them and joined onto the
# feature set we are building.


# %%
def resample(load_mw_history):
    """
    Resample the load history on a regular time grid to have exactly 1 row every hour.

    Parts where sampling was finer (eg every 15 minutes) are averaged over 1h
    intervals, and if some hours are missing a corresponding row is inserted
    containing explicit NULL values (rather than a missing row).

    We add an extra empty 48h at the end to receive lags that can be used to
    predict beyond the range of the available data.
    """
    averaged = load_mw_history.group_by(pl.col("time").dt.truncate("1h")).agg(
        pl.col("load_mw").mean()
    )
    all_times = averaged["time"]
    return time_range(
        all_times.min(), all_times.max() + datetime.timedelta(hours=48)
    ).join(averaged, on="time", how="left", maintain_order="left")


load_mw_history = raw_load_mw_history.skb.apply_func(resample)
load_mw_history

# %% [markdown]
# The prediction time range we built above is the input query to our system.
# For each row, it outputs a prediction.
#
# We use it to build the ground truth y, by shifting the historical load by the
# horizon. Moreover, there is missing data in our ground truth. Therefore, when
# we are fitting a model or doing cross-validation, we restrict the data to
# timestamps for which we have a ground truth. At inference, when making a
# prediction we keep all the query timestamps.
#
# This function is almost the same for handling single or multiple horizons so
# we anticipate a little bit the need for multiple horizons and make it general
# enough to accomodate both.


# %%
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
                X_y[f"{horizons[0]}h"] if single_horizon else X_y.drop("prediction_time")
            ),
        }
    else:
        # In predict mode there is no y and we return unmodified query
        return {"X": prediction_time}


# Example output for 3 horizons: 1, 12 and 24 hours
X_y = prediction_time.skb.apply_func(get_X_y, load_mw_history, (1, 12, 24))
X_y["X"]

# %%
X_y["y"]

# %% [markdown]
# ## Single horizon
#
# We now build the pipeline for a single horizon, 12 hours as an example.
# We create our X and y for the single 12 hour horizon.

# %%
EXAMPLE_HORIZON = 12
X_y = prediction_time.skb.apply_func(get_X_y, load_mw_history, EXAMPLE_HORIZON)
X_y["y"]

# %% [markdown]
# ### Cross-validation splitter
#
# The first thing we need to do in our pipeline now that we have X and y is to
# define how they are split into training and testing sets. So now we define a
# time-based cross-validation splitter.
#
# We do not use the scikit-learn TimeSeries split because it is based on
# positional indices but here we do not have a regular grid because of dropping
# rows with missing ground truth. Also it is arguably easier to check the code
# for splitting based on actual dates and a datetime column than based on
# positional indices. Finally, the splitter here is a simple example of
# something that needs to be done frequently, because forecasting problems
# often have specific setups for when splits happen (to mimick the actual setup
# of the deployed system, for example, on the first Tuesday of every month make
# a prediction for every day of the following month, ...) that require a custom
# datetime-based splitter.
#
# When we want an actual value to inspect, experiment with or debug, we can
# always call .skb.preview(). It gives us the output of the pipeline for the
# preview example data we set on the variables. Getting it is cheap because it
# is precomputed eagerly when we define the dataop so it is readily available.
# Here for example we grab the value of X (a dataframe) and we can use it to
# test our splitter and debug it.

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
        test_length_days = 24 * 7
        test_start_dates = pl.date_range(
            X["prediction_time"].min()
            + datetime.timedelta(days=min_train_days + TRAIN_TEST_GAP_DAYS),
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


# Example output for our X value
next(iter(TimeSeriesSplitter().split(X_val)))

# %% [markdown]
# We now use mark_as_X and mark_as_y to indicate that these DataOps are the
# nodes to split when doing cross-validation. When we need a train/test split
# or cross-validate our pipeline, skrub will first materialize those values,
# then use the splitter to separate them into train and test, and finally train
# a model on the train set and score it on the test set.
#
# We pass a TimeSeriesSplitter as the cv parameter of `mark_as_X` to set it as
# the default splitting strategy to use for our pipeline.

# %%
X = X_y["X"].skb.mark_as_X(cv=TimeSeriesSplitter())
y = X_y["y"].skb.mark_as_y()


# %% [markdown]
# ### Feature engineering
#
# Now that we have our query and the ground-truth answers for it, we can start
# building the rest of our predictive pipeline: creating the features and
# adding a supervised predictor.

# %% [markdown]
#
# Feature engineering takes _target time_ into account. In X we have the
# prediction time, the time at which we make the prediction. We also want to
# take into account the target time, ie the time about which we make a
# prediction. For example if we are predicting what the load will be on Tuesday
# at 3pm, we want to know what the weather will be, whether Tuesday is a
# holiday, and what the load was on Monday at 3pm and the previous Tuesday at
# 3pm. Those features are driven by the target time. So our first step is to
# add it to the dataframe of features we are building up.
#
# Next we have a function for adding lagged features (such as load on the same
# day of the previous week). It needs the input dataframe (which so far only
# contains prediction and target time), the historical data that will be used
# to build the lagged features and join them to the input. The horizon
# (difference between target and prediction time) is also needed to ensure that
# we do not include lags that would not be available after deployment: for
# example if we are creating a pipeline for a 12 h horizon we cannot include
# the 3-hour lagged load (because it would only become available 9 hours after
# the deadline for our prediction).


# %%
def add_target_time(df, horizon):
    return df.with_columns(
        (pl.col("prediction_time") + pl.duration(hours=horizon)).alias("target_time")
    )


def add_lagged_features(df, load_mw_history, horizon):
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
    return df.join(
        features,
        left_on="target_time",
        right_on="time",
        how="left",
        maintain_order="left",
    )


with_lags = X.skb.apply_func(add_target_time, EXAMPLE_HORIZON).skb.apply_func(
    add_lagged_features, load_mw_history, EXAMPLE_HORIZON
)
with_lags

# %% [markdown]
#
# Now we add weather data.

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
    return pl.read_parquet(data_dir() / f"weather_{city}.parquet")


fetch_city_weather("paris")

# %% [markdown]
#
# As for fetching the historical load, will we make the
# function that fetches the weather forecast a named DataOp so we can pass a
# different one if needed.
#
# We are not sure if it is best to use all cities or only a few big ones. Also,
# we don't know which features to use, temperature is probably the most
# important one so we may want to try using all features or the temperature
# only. Therefore the function we define has parameters for controlling that.
#
# Skrub lets us create "choice" objects, nodes in our pipeline that can take
# different values for hyperparameter search. We use this for the choice of
# city names and of temperature only vs all features.
#
# Our function also accepts the horizon, like `add_lagged_features` did. In
# theory, it should use it to retrieve the correct forecast that was available
# at this horizon, which is why we make it a parameter. However we do not have
# that level of detail in our historical weather data so the parameter is
# ignored :/


# %%
def add_weather(
    df,
    horizon,
    cities="all",
    temperature_only=True,
    city_weather_fetcher=fetch_city_weather,
):
    """Add weather information for the required cities."""
    # NOTE: here ideally we should retrieve the exact weather forecast
    # corresponding to the horizon. But we do not have it available in the
    # historical data. Therefore we just take the only forecast we have and
    # ignore the horizon.
    del horizon
    if isinstance(cities, str):
        assert cities == "all"
        cities = ALL_CITIES
    with_weather = df
    for city in cities:
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


city_weather_fetcher = skrub.as_data_op(fetch_city_weather).skb.set_name(
    "city_weather_fetcher"
)
temperature_only = skrub.choose_bool(name="temperature_only", default=True)
cities = skrub.choose_from(["all", ["paris", "lyon", "marseille"]], name="cities")

with_weather = with_lags.skb.apply_func(
    add_weather,
    EXAMPLE_HORIZON,
    cities=cities,
    temperature_only=temperature_only,
    city_weather_fetcher=city_weather_fetcher,
)
with_weather

# %% [markdown]
# In the preview above, we see the output of the pipeline for the default
# values of the choices, ie using temperature only and all cities.

# %%
print(with_weather.skb.describe_param_grid())

# %% [markdown]
# We now add calendar features and holidays.

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


with_calendar = with_weather.skb.apply_func(add_calendar_and_holidays)
with_calendar

# %% [markdown]
# Now we are done with all the feature engineering steps. For later reuse we
# group the steps we just created into one function:


# %%
def add_features(df, horizon, load_mw_history, cities, temperature_only):
    df = add_target_time(df, horizon=horizon)
    df = add_lagged_features(df, load_mw_history=load_mw_history, horizon=horizon)
    df = add_weather(
        df,
        horizon,
        cities=cities,
        temperature_only=temperature_only,
        city_weather_fetcher=city_weather_fetcher,
    )
    df = add_calendar_and_holidays(df)
    return df


# %% [markdown]
# ### Supervised predictor

# %%
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor

loss = skrub.choose_from(["squared_error", "poisson", "gamma"], name="loss")

regressor = HistGradientBoostingRegressor(
    random_state=0,
    loss=loss,
    learning_rate=skrub.choose_float(
        0.01, 0.7, default=0.1, log=True, name="learning_rate"
    ),
    max_leaf_nodes=skrub.choose_int(3, 300, default=30, log=True, name="max_leaf_nodes"),
)

# If the log is squared_error, we want to try with and without log-transforming the targets.
# Otherwise no log-transform.

use_log_transform = loss.match(
    {"squared_error": skrub.choose_bool(name="use_log_transform", default=True)},
    default=False,
)


def log_transform_maybe(y_true, use_log_transform):
    return y_true.log() if use_log_transform else y_true


def exp_transform_maybe(estimator_output, use_log_transform):
    if not use_log_transform:
        return estimator_output
    if isinstance(estimator_output, np.ndarray):
        return np.exp(estimator_output)
    if isinstance(estimator_output, pl.Series):
        return estimator_output.exp()
    # in 'fit' mode, the output of the final estimator will be the fitted
    # estimator itself.
    return estimator_output


def apply_predictor(X, y, horizon):
    return (
        X.skb.apply_func(
            add_features,
            horizon=horizon,
            load_mw_history=load_mw_history,
            cities=cities,
            temperature_only=temperature_only,
        )
        .skb.set_name(f"feat_{horizon}h")
        .skb.drop(["prediction_time", "target_time"])
        .skb.apply(regressor, y=y.skb.apply_func(log_transform_maybe, use_log_transform))
        .skb.apply_func(exp_transform_maybe, use_log_transform)
        .skb.set_name(f"pred_{horizon}h")
    )


pred = apply_predictor(X, y, EXAMPLE_HORIZON).skb.with_scoring(
    "neg_mean_absolute_percentage_error"
)
pred

# %% [markdown]
# We can run the cross-validation. Note the scores we see are for the default
# values of the choices, as we are not doing any hyperparameter search yet.

# %%
pred.skb.cross_validate()

# %% [markdown]
# For further inspection of predictions, we will collect the cross-validated
# prediction into a dataframe. To easily inspect the output of the pipeline and
# debug our cross-validation loop, we perform one train/test split to have an
# example to work with.

# %%
split = pred.skb.train_test_split()

# %%
split["X_test"]

# %%
split["y_test"]

# %%
pred.skb.make_learner().fit(split["train"]).predict(split["test"])

# %% [markdown]
# Now we can collect predictions for all splits and plot them.

# %%
cv_predictions = pl.concat(
    [
        split["X_test"].with_columns(
            split["y_test"],
            **{
                f"pred_{EXAMPLE_HORIZON}h": pred.skb.make_learner()
                .fit(split["train"])
                .predict(split["test"]),
                "split": i,
            },
        )
        for i, split in enumerate(pred.skb.iter_cv_splits())
    ]
)
cv_predictions

# %%
import plotly.graph_objects as go

target_time = cv_predictions["prediction_time"] + datetime.timedelta(
    hours=EXAMPLE_HORIZON
)


def plot_line(x, y):
    return go.Scatter(
        x=x,
        y=y,
        mode="lines+markers",
        name=y.name,
        hovertemplate="%{x|%Y-%m-%dT%H} (%{x|%A}): %{y}<extra></extra>",
    )


fig = go.Figure()
fig.add_trace(
    plot_line(target_time, cv_predictions[f"{EXAMPLE_HORIZON}h"].rename("y_true"))
)
fig.add_trace(plot_line(target_time, cv_predictions[f"pred_{EXAMPLE_HORIZON}h"]))
fig.update_layout(height=700)


# %% [markdown]
# ## Multiple horizons
#
# We now have a pipeline that makes predictions for 1 horizon. To predict
# multiple horizons, we just need to make one prediction for each horizon and
# group them in a single dataframe.


# %%
def concat_horizons(predictions):
    """
    Consolidate predictions of models for different horizons in one dataframe.
    """
    return pl.DataFrame({f"{h}h": v for h, v in predictions.items()})


def make_multi_horizon_pred(horizons):
    """
    Create a full DataOp for predicting the specified horizons.
    """
    X_y = prediction_time.skb.apply_func(get_X_y, load_mw_history, horizons)
    X = X_y["X"].skb.mark_as_X(cv=TimeSeriesSplitter())
    y = X_y["y"].skb.mark_as_y()
    predictions = {h: apply_predictor(X, y[f"{h}h"], h) for h in horizons}
    return skrub.deferred(concat_horizons)(predictions).skb.set_name("pred_multi_horizon")


# We inspect the pipeline on an example with only 3 horizons so that it is fast
# and reasonably easy to visualize. Later we will cross-validate a pipeline for
# all horizons between 1 and 25 hours.
pred = make_multi_horizon_pred((1, 12, 24))
pred

# %% [markdown]
# We want to define a scorer that will produce the Mean Absolute Percentage
# Error (MAPE) for each of the horizons, and also averaged across horizons. To
# easily try our metric function on actual values and debug it, we collect
# ground truth and predictions on an example train/test split.

# %%
split = pred.skb.train_test_split()
learner = pred.skb.make_learner().fit(split["train"])
predicted_y_test = learner.predict(split["test"])
predicted_y_test

# %% [markdown]
# For multioutput regression, we can `mean_absolute_percentage_error` to return
# the error for each target, without averaging:

# %%
from sklearn.metrics import mean_absolute_percentage_error

mean_absolute_percentage_error(
    split["y_test"], predicted_y_test, multioutput="raw_values"
)

# %% [markdown]
# We will therefore use `multioutput='raw_values'` and return all the errors in
# a dictionary, after adding the averaged error.
#
# Once we have defined this function of true and predicted electricity loads,
# (what scikit-learn calls a 'metric'), we wrap it in a 'scorer', a function
# that takes an estimator, X and y. Scorers can return a single score, or a
# dictionary mapping metric names (in our case 'neg_mape_1h', 'neg_mape_2h', ...) to
# scores.


# %%
def neg_mape(y_true, y_pred):
    average = mean_absolute_percentage_error(y_true, y_pred)
    detail = mean_absolute_percentage_error(y_true, y_pred, multioutput="raw_values")
    return {"neg_mape_average": -average} | {
        f"neg_mape_{c}": -float(s) for c, s in zip(y_true.columns, detail)
    }


def neg_mape_scorer(estimator, X, y):
    return neg_mape(y, estimator.predict(X))


# We set this as the default scorer on our pipeline.
pred = make_multi_horizon_pred((1, 12, 24)).skb.with_scoring(neg_mape_scorer)
pred

# %%
print(pred.skb.describe_param_grid())

# %%
# run this if in an environment where it is possible to open a browser tab,
# useful for inspecting the whole pipeline:
#
# pred.skb.full_report()

# %% [markdown]
# Now that we have configured the scorer, we can check the score of our
# pipeline on the example split:

# %%
pred.skb.make_learner().fit(split["train"]).score(split["test"])

# %% [markdown]
# **Out-of-sample check:**
# it is always good that our pipeline can make a prediction on some truly
# left-out data, as a sanity check which could find bugs in the way we set it
# up or did the cross-validation.

# %%
history_dates = raw_load_mw_history["time"].skb.preview()
history_dates.max()

# %%
new_date = (
    (history_dates - datetime.timedelta(seconds=1)).dt.truncate("1h")
    + datetime.timedelta(hours=1)
).max()
new_date

# %%
# fit on all available data
learner = pred.skb.make_learner(fitted=True)
future_pred = learner.predict({"start": new_date, "end": None})
future_pred

# %% [markdown]
# Note that the new_date is 15 minutes past our last data point, not exactly on
# that point, so some lags will already be missing. Also we do not have weather
# data for that date.
#
# This can be easily inspected by creating a report for the prediction:
#
# ```
# learner.report(environment={'start': new_date, 'end': None}, mode='predict')
# ```
#
# Luckily the HistGradientBoostingRegressor is able to deal with missing values
# and produces a reasonable prediction.


# %%
def transpose_pred(prediction_date, prediction):
    date = [
        prediction_date + datetime.timedelta(hours=int(c.removesuffix("h")))
        for c in prediction.columns
    ]
    load = prediction.row()
    return pl.DataFrame({"time": date, "load_mw": load})


future_pred_tall = transpose_pred(new_date, future_pred)

# %%
history_tail = load_mw_history.skb.preview().filter(
    pl.col("time") > new_date - datetime.timedelta(days=8)
)

fig = go.Figure()
fig.add_trace(plot_line(history_tail["time"], history_tail["load_mw"]))
fig.add_trace(plot_line(future_pred_tall["time"], future_pred_tall["load_mw"]))
fig.update_layout(height=700)

# %% [markdown]
# It seems getting the predictions in long rather than wide format indexed by
# date for a single prediction date might be a frequent need. We can add a
# little post-processor to the pipeline to optionally do that.


# %%
def post_process(pred):
    if range_end is not None:
        return pred
    return transpose_pred(prediction_time["time"].to_list()[0], pred)


pred = (
    make_multi_horizon_pred((1, 12, 24))
    .skb.apply_func(post_process)
    .skb.with_scoring(neg_mape_scorer)
)

# %%
learner = pred.skb.make_learner().fit(split["train"])

# %%
# by default we get the same format as before
learner.predict(split["test"])

# %%
# but if we pass end=None (predict only 1 date, start), we get the more convenient format
learner.predict({"start": new_date, "end": None})

# %% [markdown]
# Finally we compute the full cross-validation.

# %%
pred.skb.cross_validate()


# %% [markdown]
# We now want to plot the true and predicted loads for the different horizons.
# This is similar to what we did before; we just need to put all the
# predictions and true loads for the different horizons in a single dataframe.


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


concat_X_y_predictions(split["X_test"], split["y_test"], predicted_y_test)


# %%
def cross_val_predict(data_op, environment=None):
    """
    Get cross-validated predictions for different horizons.
    """
    all_predictions, all_scores = [], []
    for i, split in enumerate(data_op.skb.iter_cv_splits(environment=environment)):
        prediction = data_op.skb.make_learner().fit(split["train"]).predict(split["test"])
        all_predictions.append(
            concat_X_y_predictions(
                split["X_test"], split["y_test"], prediction
            ).with_columns(split=pl.lit(i)),
        )
        split_neg_mape = neg_mape(split["y_test"], prediction)
        split_start = split["X_test"]["prediction_time"].min()
        fmt_mape = " ".join(
            f"{k.removeprefix('neg_mape_')}: {-v:.1%}" for k, v in split_neg_mape.items()
        )
        print(f"{split_start:%Y-%m-%d}: {fmt_mape}")
        all_scores.append(split_neg_mape | {"split": i})
    all_predictions = pl.concat(all_predictions, how="vertical")
    all_scores = pl.DataFrame(all_scores)
    return all_predictions, all_scores


cv_predictions, cv_scores = cross_val_predict(pred)
cv_scores

# %% [markdown]
# Now we have the predictions, we can plot them. We notice that the 1h horizon
# qualitatively seems to stick better to the ground truth, which is expected
# and also corresponds to what we see in the MAPE.

# %%
import re

import plotly.graph_objects as go


def plot_predictions(cv_predictions, horizons=None):
    if horizons is None:
        horizons = [
            int(m.group(1))
            for c in cv_predictions.columns
            if (m := re.match(r"^pred_(\d+)h$", c)) is not None
        ]
    fig = go.Figure()
    for i, h in enumerate(horizons):
        target_time = cv_predictions["prediction_time"] + datetime.timedelta(hours=h)
        if i == 0:
            fig.add_trace(
                plot_line(target_time, cv_predictions[f"{h}h"].rename("true_load"))
            )
        fig.add_trace(plot_line(target_time, cv_predictions[f"pred_{h}h"]))
    fig.update_layout(height=700)
    return fig


plot_predictions(cv_predictions, (24,))

# %%
plot_predictions(cv_predictions)

# %% [markdown]
# We can save this pipeline for future reuse:

# %%
import pickle

with open(results_dir() / "learner_3_horizons.pickle", "wb") as f:
    pickle.dump(pred.skb.make_learner(), f)

# %% [markdown]
# Finally, we run the cross-validation for all 24 horizons

# %%
HORIZONS = tuple(range(1, 25))

pred_24_horizons = (
    make_multi_horizon_pred(HORIZONS)
    .skb.apply_func(post_process)
    .skb.with_scoring(neg_mape_scorer)
)
cv_scores = pred_24_horizons.skb.cross_validate(verbose=2)
cv_scores

# %%
learner = pred_24_horizons.skb.make_learner(fitted=True)
future_pred = learner.predict({"start": new_date, "end": None})
fig = go.Figure()
fig.add_trace(plot_line(history_tail["time"], history_tail["load_mw"]))
fig.add_trace(plot_line(future_pred["time"], future_pred["load_mw"]))
fig.update_layout(height=700)


# %% [markdown]
# We can plot horizon vs MAPE to see if shorter horizons are easier to predict:

# %%
from matplotlib import pyplot as plt

(cv_scores.filter(regex="test_neg_mape_.*h") * -1).rename(
    columns=lambda c: c.removeprefix("test_neg_mape_")
).boxplot()
plt.xticks(rotation=45)
plt.xlabel("Horizon")
plt.ylabel("MAPE")

# %%
with open(results_dir() / "learner_24_horizons.pickle", "wb") as f:
    pickle.dump(pred_24_horizons.skb.make_learner(), f)
