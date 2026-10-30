"""
Smoke tests for the Kumo-Forecast pipeline (electricity_load_forecasting/kumo.py).

By default only the stub forecaster (seasonal naive, no model, no GPU) is used.
It checks window construction, horizon alignment, causality, the DataOp contract
and scoring. With `--forecaster kumo` the real Kumo-Forecast model additionally
forecasts `--n_rows` prediction times.

    python scripts/checks/smoke_kumo.py
    python scripts/checks/smoke_kumo.py --forecaster kumo --device cuda
"""

import argparse
import datetime
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import electricity_load_forecasting as elf
from electricity_load_forecasting import kumo as kf

HORIZONS = (1, 12, 24)
CONTEXT_HOURS = 200

parser = argparse.ArgumentParser()
parser.add_argument("--forecaster", choices=["stub", "kumo"], default="stub")
parser.add_argument("--device", default=None)
parser.add_argument("--n_rows", type=int, default=2)
args = parser.parse_args()


def utc(s):
    return datetime.datetime.fromisoformat(s).replace(tzinfo=datetime.UTC)


class RecordingForecaster(kf.SeasonalNaivePointForecaster):
    def __init__(self, period=24 * 7):
        super().__init__(period=period)
        self.calls = []

    def predict(self, inputs, prediction_length, quantile_levels):
        self.calls.append((inputs, prediction_length, quantile_levels))
        return super().predict(inputs, prediction_length, quantile_levels)


history = elf.resample(elf.fetch_load_mw_history())
covariates = kf.build_covariates(history)
cov_names = [c for c in covariates.columns if c != "time"]


def check_windows():
    last = history["time"].max()
    times = [
        history["time"].min() + datetime.timedelta(hours=5),
        utc("2023-06-15T13:00"),
        utc("2024-12-31T23:00"),
        last - datetime.timedelta(hours=3),
    ]
    rec = RecordingForecaster()
    est = kf.KumoHistoryRegressor(
        forecaster=rec, horizons=HORIZONS, context_hours=CONTEXT_HOURS, chunk_size=3
    )
    X = pl.DataFrame({"prediction_time": times}).with_columns(
        pl.col("prediction_time").dt.cast_time_unit("us")
    )
    est.predict(X, history=history, covariates=covariates)
    assert len(rec.calls) == 2, "chunk_size should split 4 rows in 2 calls"
    inputs = [item for call in rec.calls for item in call[0]]
    assert all(call[1] == max(HORIZONS) for call in rec.calls)
    for t, item in zip(times, inputs):
        past = history.filter(pl.col("time") <= t).tail(CONTEXT_HOURS)
        past_cov = covariates.filter(pl.col("time") <= t).tail(CONTEXT_HOURS)
        pad = CONTEXT_HOURS - past.height
        expected_target = np.concatenate(
            [np.full(pad, np.nan, dtype=np.float32), past["load_mw"].to_numpy()]
        )
        assert len(item["target"]) == CONTEXT_HOURS
        np.testing.assert_array_equal(item["target"], expected_target)
        assert set(item["past_covariates"]) == set(cov_names)
        for name in cov_names:
            expected_cov = np.concatenate(
                [
                    np.full(pad, np.nan, dtype=np.float32),
                    past_cov[name].to_numpy(),
                ]
            )
            np.testing.assert_array_equal(
                item["past_covariates"][name], expected_cov
            )
        assert "future_covariates" not in item
        assert past["time"].max() == t, "context must end at the prediction time"
    print("OK windows")


def check_outputs_and_alignment():
    t = utc("2024-03-05T10:00")
    est = kf.KumoHistoryRegressor(
        forecaster=kf.SeasonalNaivePointForecaster(),
        horizons=HORIZONS,
        context_hours=CONTEXT_HOURS,
    )
    X = pl.DataFrame({"prediction_time": [t]})
    out = est.predict(X, history=history, covariates=None)
    assert out.columns == ["1h", "12h", "24h"]
    for h in HORIZONS:
        ref = history.filter(
            pl.col("time") == t + datetime.timedelta(hours=h, days=-7)
        )["load_mw"].item()
        assert np.isclose(out[f"{h}h"].item(), ref), (h, out[f"{h}h"].item(), ref)
    print("OK output layout and horizon alignment")


def make_op(**kwargs):
    kwargs.setdefault("forecaster", kf.SeasonalNaivePointForecaster())
    kwargs.setdefault("context_hours", CONTEXT_HOURS)
    return kf.make_kumo_data_op(horizons=HORIZONS, **kwargs)


def check_data_op():
    pred = make_op()
    learner = pred.skb.make_learner()
    env = {"start": "2023-01-01", "end": "2023-03-01"}
    learner.fit(env)

    day = learner.predict({"start": "2024-03-05T00:00", "end": "2024-03-05T23:00"})
    assert day.columns == ["1h", "12h", "24h"] and day.height == 24, day.shape
    t = utc("2024-03-05T10:00")
    single = learner.predict({"start": t, "end": None})
    assert single.columns == ["time", "load_mw"] and single.height == len(HORIZONS)
    assert single["time"].to_list() == [
        t + datetime.timedelta(hours=h) for h in HORIZONS
    ]
    np.testing.assert_allclose(single["load_mw"].to_numpy(), day.row(10))

    def corrupted_fetcher():
        return elf.fetch_load_mw_history().with_columns(
            pl.when(pl.col("time") > t)
            .then(pl.lit(1e9, dtype=pl.Float32))
            .otherwise(pl.col("load_mw"))
            .alias("load_mw")
        )

    corrupted = learner.predict(
        {"start": t, "end": None, "load_mw_history_fetcher": corrupted_fetcher}
    )
    assert corrupted.equals(single), "prediction depends on data after t"
    print("OK data op predict / single date / causality")


def check_scoring():
    env = elf.get_env()
    pred = make_op()
    split = pred.skb.train_test_split(
        environment=env, split_func=elf.train_test_split
    )
    learner = pred.skb.make_learner().fit(split["train"])
    score = learner.score(split["test"])
    assert set(score) == {
        "neg_mape__average",
        "neg_mape__1h",
        "neg_mape__12h",
        "neg_mape__24h",
    }, score
    mape = -score["neg_mape__average"]
    print(f"  n_test={split['X_test'].height}, MAPE={mape:.4f}")
    assert 0 < mape < 0.2
    first = next(iter(make_op().skb.iter_cv_splits(environment=env)))
    assert {"X_train", "X_test", "y_train", "y_test"} <= set(first)
    print("OK scoring and CV splits")


def check_univariate():
    pred = make_op(cities=(), calendar=False)
    learner = pred.skb.make_learner().fit({"start": "2023-01-01", "end": "2023-03-01"})
    out = learner.predict({"start": "2024-03-05T00:00", "end": "2024-03-05T23:00"})
    assert out.height == 24
    print("OK target-only (no covariates)")


def check_real_model():
    forecaster = kf.KumoForecastForecaster(device=args.device)

    def split_func(X, y):
        X_train, X_test, y_train, y_test = elf.train_test_split(X, y)
        n = args.n_rows
        return X_train.head(10), X_test.head(n), y_train.head(10), y_test.head(n)

    env = elf.get_env()
    results = {}
    for name, kwargs in {
        "kumo + past covariates": dict(
            forecaster=forecaster, context_hours=512
        ),
        "kumo target only": dict(
            forecaster=forecaster,
            context_hours=512,
            cities=(),
            calendar=False,
        ),
    }.items():
        pred = make_op(verbose=True, **kwargs)
        split = pred.skb.train_test_split(environment=env, split_func=split_func)
        learner = pred.skb.make_learner().fit(split["train"])
        score = learner.score(split["test"])
        assert all(np.isfinite(v) for v in score.values()), score
        results[name] = -score["neg_mape__average"]
    print(f"MAPE on {args.n_rows} prediction times from 2025-01-01:")
    for name, mape in results.items():
        print(f"  {name}: {mape:.4f}")
    assert all(m < 0.3 for m in results.values())
    print("OK real model")


check_windows()
check_outputs_and_alignment()
check_data_op()
check_scoring()
check_univariate()
if args.forecaster == "kumo":
    check_real_model()
print("all smoke tests passed")
