"""
Walk-forward CV for the Chronos pipeline, structured exactly like
`scripts/cross_validate.py` (no skore): the train/test split is only used to produce
the skrub fit/predict/single-date reports; the actual scores come from a real
cross-validation (`elf.cross_val_predict`, iterating over every
`electricity_load_forecasting.splitter.TimeSeriesSplitter` fold), same as the baseline.

    python scripts/chronos_comparison_scripts/cross_validate_chronos.py
    python scripts/chronos_comparison_scripts/cross_validate_chronos.py --forecaster stub
    python scripts/chronos_comparison_scripts/cross_validate_chronos.py --model amazon/chronos-2 --device cuda
    python scripts/chronos_comparison_scripts/cross_validate_chronos.py --quantiles 0.05 0.5 0.95 --no_weather
"""

import argparse

import electricity_load_forecasting as elf
from electricity_load_forecasting import chronos as cf

parser = argparse.ArgumentParser()
parser.add_argument("--forecaster", choices=["chronos", "stub"], default="chronos")
parser.add_argument("--model", default="amazon/chronos-2")
parser.add_argument("--device", default=None, help="default: cuda if available")
parser.add_argument("--dtype", default=None, help="eg bfloat16")
parser.add_argument("--batch_size", type=int, default=256, help="Chronos batch size")
parser.add_argument("--context_hours", type=int, default=1024)
parser.add_argument("--quantiles", type=float, nargs="+", default=None)
parser.add_argument("--no_weather", action="store_true")
parser.add_argument("--no_calendar", action="store_true")
args = parser.parse_args()

if args.forecaster == "stub":
    forecaster = cf.SeasonalNaiveForecaster()
else:
    forecaster = cf.Chronos2Forecaster(
        model=args.model,
        device_map=args.device,
        dtype=args.dtype,
        batch_size=args.batch_size,
    )

quantiles = None if args.quantiles is None else tuple(args.quantiles)

output_dir = elf.get_output_dir("cross_validate_chronos_")

env = elf.get_env()
pred = cf.make_chronos_data_op(
    horizons=(1, 12, 24),
    forecaster=forecaster,
    quantiles=quantiles,
    context_hours=args.context_hours,
    cities=() if args.no_weather else "all",
    calendar=not args.no_calendar,
)

# %%
# optional: make skrub reports
# pred.skb.report(environment=env, output_dir=output_dir / "full_report")
split = pred.skb.train_test_split(environment=env)
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
cv_predictions, scores = elf.cross_val_predict(pred, env)
cv_predictions.write_parquet(output_dir / "cv_predictions.parquet")
scores.write_csv(output_dir / "scores.csv")

fig = elf.plot_predictions(cv_predictions)
fig.write_html(output_dir / "cv_predictions_plot.html")
fig.show(renderer="browser")


fig = elf.plot_predictions(cv_predictions, horizons=(1,))
fig.write_html(output_dir / "cv_predictions_1h_plot.html")
fig.show(renderer="browser")

fig = elf.plot_predictions(cv_predictions, horizons=(24,))
fig.write_html(output_dir / "cv_predictions_24h_plot.html")
fig.show(renderer="browser")

# %%
index = """
<!DOCTYPE html>
<html>
    <head>
        <meta charset="utf-8">
        <title>chronos cross-validation result</title>
    </head>
    <body>
        <ul>
            <li><a href="report/index.html">data op report</a></li>
            <li><a href="fit_report/index.html">data op fit report</a></li>
            <li><a href="predict_report/index.html">data op predict report</a></li>
            <li><a href="predict_single_date_report/index.html">data op predict a single date report</a></li>
            <li><a href="cv_predictions_plot.html">multiple horizon predictions</a></li>
            <li><a href="cv_predictions_1h_plot.html">1-h horizon predictions</a></li>
            <li><a href="cv_predictions_24h_plot.html">24-h horizon predictions</a></li>
            <li><a href="scores.csv">scores</a></li>
        </ul>
    </body>
</html>
"""
(output_dir / "index.html").write_text(index, "utf-8")
