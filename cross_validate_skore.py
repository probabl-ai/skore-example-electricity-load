import argparse
import pickle

import skore

import electricity_load_forecasting as elf


# %%
parser = argparse.ArgumentParser()
parser.add_argument(
    "--quantile_strategy", default=None, choices=["tabicl", "multiple_regressors", "binning"]
)
parser.add_argument("--skip_reports", action="store_true")
args = parser.parse_args()

quantile_strategy = args.quantile_strategy
skip_reports = args.skip_reports
horizons = (1, 24) if quantile_strategy == 'binning' else (1, 12, 24)

# %%
output_dir = elf.get_output_dir("cross_validate_skore_")

env = elf.get_env()
# quantile prediction prevents skore hub upload because it calls some metrics and displays
# that fail due to the different y pred shape
pred = elf.make_data_op(horizons=horizons, quantile_strategy=quantile_strategy)
# pred = elf.make_data_op(horizons=(1, 12, 24), quantile_strategy=None)

# %%
# optional: make skrub reports
if not skip_reports:
    # pred.skb.full_report(environment=env, output_dir=output_dir / "full_report")
    split = pred.skb.train_test_split(environment=env, split_func=elf.train_test_split)
    learner = pred.skb.make_learner()
    learner.report(
        environment=split["train"],
        mode="fit",
        title="fit",
        output_dir=output_dir / "fit_report",
    )
    learner.report(
        environment=split["test"],
        mode="predict",
        title="predict",
        output_dir=output_dir / "predict_report",
    )
    learner.report(
        environment={"start": elf.get_new_date()},
        mode="predict",
        title="predict_single_date",
        output_dir=output_dir / "predict_single_date_report",
    )

# %%
report = skore.CrossValidationReport(pred, data=env, splitter=elf.TimeSeriesSplitter())

# %%
with open(output_dir / "skore_report.pickle", "wb") as f:
    pickle.dump(report, f)

# %%
for metric in set(report.metrics.available()) - {"score", "fit_time", "predict_time"}:
    report.metrics.remove(metric)

print(report.metrics.summarize().frame())
# %%
# store in local
# only possible after merging new local storage in skore

# project = skore.Project(name="electricity_forecasting", mode="local")
# project.put(f"quantile-strategy_{quantile_strategy}", report)

# %%
# store in hub
import dotenv
import os
dotenv.load_dotenv()

skore.login()
project = skore.Project(name="electricity_forecasting", mode="hub", workspace=os.environ["SKORE_HUB_WORKSPACE"])
project.put(f"quantile-strategy_{quantile_strategy}", report)


# %%
cv_predictions = elf.get_report_predictions(report)
cv_predictions.write_parquet(output_dir / "cv_predictions.parquet")

fig = elf.plot_predictions(cv_predictions, horizons=(1,))
fig.write_html(output_dir / "cv_predictions_plot_1h.html")
fig.show(renderer="browser")

fig = elf.plot_predictions(cv_predictions, horizons=(24,))
fig.write_html(output_dir / "cv_predictions_plot_24h.html")
fig.show(renderer="browser")

# %%
# checks
# print(report.checks.summarize(ignore=["SKD008"]))
