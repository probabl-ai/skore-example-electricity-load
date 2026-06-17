import argparse
import pickle

import skore

import electricity_load_forecasting as elf

output_dir = elf.get_output_dir("cross_validate_skore_")

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
print(report.metrics.summarize(metric="score").frame())

# %%
with open(output_dir / "skore_report.pickle", "wb") as f:
    pickle.dump(report, f)

# %%
# store in local
for metric in set(report.metrics.available()) - {"score", "fit_time", "predict_time"}:
    report.metrics.remove(metric)
project = skore.Project("jerome-workspace-1/electricity_forecasting", mode="local")
project.put(f"{quantile_strategy}_default", report)

# %%
cv_predictions = elf.get_report_predictions(report)
cv_predictions.write_parquet(output_dir / "cv_predictions.parquet")

fig = elf.plot_predictions(cv_predictions, horizons=(24,))
fig.write_html(output_dir / "cv_predictions_plot.html")
fig.show(renderer="browser")

# %%
# checks
# print(report.checks.summarize(ignore=["SKD008"]))
