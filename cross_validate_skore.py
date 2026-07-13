from pathlib import Path
import argparse
import pickle
import datetime

import skore

import electricity_load_forecasting as elf

# %%
parser = argparse.ArgumentParser()
parser.add_argument("--n_trials", type=int, default=None)
parser.add_argument(
    "--quantile_strategy",
    default=None,
    choices=["multiple_regressors", "tabicl", "binning"],
)
parser.add_argument("--hub", action="store_true")
args = parser.parse_args()
quantile_strategy = args.quantile_strategy
n_trials = args.n_trials
do_search = n_trials is not None

horizons = (1, 12, 24)

# %%
pred = elf.make_data_op(horizons=horizons, quantile_strategy=quantile_strategy)
env = elf.get_env()

# %%
split = pred.skb.train_test_split(environment=env, split_func=elf.train_test_split)

if do_search:
    # cannot store in the report dir as it is not yet created
    db_name = f"optuna_{datetime.datetime.now().isoformat()}.db"
    storage = f"sqlite:///{db_name}"
    study_name = "randomized_search"
    learner = pred.skb.make_randomized_search(
        backend="optuna",
        n_iter=n_trials,
        n_jobs=2,
        refit="neg_mape__average",
        storage=storage,
        study_name=study_name,
    )
else:
    learner = pred.skb.make_learner()

# %%
report = skore.EstimatorReport(
    learner, train_data=split["train"], test_data=split["test"]
)
for metric in set(report.metrics.available()) - {"score", "fit_time", "predict_time"}:
    report.metrics.remove(metric)


# %%
print(report.metrics.summarize())

# %%
project = skore.Project(name="electricity_forecasting", mode="local")
report_name = "__".join(
    [
        "hgb_mean" if quantile_strategy is None else quantile_strategy,
        f"search_{n_trials}" if do_search else "default",
        "with_temperature_lags"
    ]
)
report_path = project.put(report_name, report)
print(report_path)

# %%
if do_search:
    Path(db_name).rename(report_path / "user" / "optuna.db")
    with open(report_path / "user" / "best_learner.pickle", "wb") as f:
        pickle.dump(report.learner_.best_learner_, f)
    fig = report.learner_.plot_results(show_scores=["neg_mape__average"], show_times=[])
    fig.write_html(report_path / "user" / "parallel_coord.html")

# %%
results = elf.concat_X_y_predictions(
    report.X_test, report.y_test, report.get_predictions(data_source="test")
)

fig = elf.plot_predictions(results, horizons=(1,))
fig.show(renderer="browser")
fig.write_html(report_path / "user" / "1h.html")
fig = elf.plot_predictions(results, horizons=(24,))
fig.show(renderer="browser")
fig.write_html(report_path / "user" / "24h.html")

# %%
# store in hub
if args.hub and quantile_strategy is None:
    import dotenv
    import os

    dotenv.load_dotenv()

    skore.login()
    project = skore.Project(
        name="electricity_forecasting",
        mode="hub",
        workspace=os.environ["SKORE_HUB_WORKSPACE"],
    )
    project.put(report_name, report)


# %%
pred.skb.full_report(environment=env, output_dir=report_path / "user" / "full_report")

# %%
learner = report.learner_
try:
    learner = learner.best_learner_
except AttributeError:
    pass

learner.report(
    environment=split["test"],
    mode="predict",
    title="predict",
    output_dir=report_path / "user" / "predict_report",
)
learner.report(
    environment={"start": elf.get_new_date()},
    mode="predict",
    title="predict_single_date",
    output_dir=report_path / "user" / "predict_single_date_report",
)
