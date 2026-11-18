# Model comparison

Scripts for running and comparing walk-forward CV predictions from several
model families on the same folds: `hgb_mean`, `hgb_quantile`,
`hgb_quantile_recal`, `hgb_quantile_conformal`, `tabicl`, `tabicl_recal`,
`tabicl_conformal`, `kumo_tabular`, `kumo_tabular_recal`, `chronos2`, `kumo_forecast`,
`tabicl_forecast` and `t0_gamma`.

Run them from the repository root. A typical study directory is
`results/model_comparison/<study-name>/`.

## Running models

Use `cross_validate_models.py` once per model/configuration, pointing all runs
at the same `--output_dir`:

```bash
python scripts/model_comparison/cross_validate_models.py \
    --model hgb_quantile --output_dir results/model_comparison/my_study
python scripts/model_comparison/cross_validate_models.py \
    --model hgb_quantile_recal --max_iter 2000 \
    --output_dir results/model_comparison/my_study
python scripts/model_comparison/cross_validate_models.py \
    --model tabicl --output_dir results/model_comparison/my_study
python scripts/model_comparison/cross_validate_models.py \
    --model kumo_tabular --max_train_rows 0 --output_dir results/model_comparison/my_study
python scripts/model_comparison/cross_validate_models.py \
    --model tabicl_recal --max_train_rows 0 --output_dir results/model_comparison/my_study
python scripts/model_comparison/cross_validate_models.py \
    --model kumo_tabular_recal --max_train_rows 0 --output_dir results/model_comparison/my_study
python scripts/model_comparison/cross_validate_models.py \
    --model chronos2 --device cuda --output_dir results/model_comparison/my_study
python scripts/model_comparison/cross_validate_models.py \
    --model tabicl_forecast --device cuda --output_dir results/model_comparison/my_study
python scripts/model_comparison/cross_validate_models.py \
    --model t0_gamma --device cuda --output_dir results/model_comparison/my_study
```

`tabicl_forecast` uses `tabicl.TabICLForecaster` (the time-series interface,
not the tabular `TabICLRegressor`); install it with `tabicl[forecast]`. It uses
known weather/calendar covariates over the forecast horizon, like the Chronos
pipeline, and returns 5/50/95% quantiles by default. Its `--tabicl_context_hours`
(default 4096) and `--tabicl_n_estimators` (default 8) control the context
window and ensemble size. It fits one TabICL regressor per forecast origin, so
it is much slower than the tabular `tabicl` model and should be run on a GPU.

`t0_gamma` is the `theforecastingcompany/t0-gamma` foundation model from The
Forecasting Company (`tfc-t0` package, install with `tfc-t0[torch]>=0.6.0`). It
is a zero-shot quantile forecaster that conditions on known-future
weather/calendar covariates, like the Chronos pipeline, and returns 5/50/95%
quantiles by default. It uses the recommended inference-time adjustments
(`trim_leading_missing`, `symmetric_averaging`, `clip_nonnegative`,
`snap_integers`); `symmetric_averaging` runs the model twice. `--device`,
`--dtype` and `--context_hours` apply to it.

Weather defaults to the safe-lead historical forecasts (`--weather forecast`):
day1 for horizons <=18h and day2 above. `--weather oracle` is meant to select the
oracle files, but **currently it is a silent no-op** (the default fetcher is now
the forecast one and the flag only overrides it for `forecast`), so it still uses
the forecast files; see `_scratch/REVIEW_REPORT_2.md`, finding 1.1. Note also that
the Chronos, Kumo and TabICLForecaster pipelines always use the day2 files for
every horizon.

Each run writes `<model>_cv_predictions.parquet` and `<model>_scores.csv`.
Use `--tag` to keep multiple runs of the same model side by side. Add
`--reports` to also write the skrub fit/predict/single-date reports for that
model.

To add the recalibrated HGB to an existing comparison, run the command above
with `--model hgb_quantile_recal` and the same `--output_dir`; it will then be
picked up by the report scripts. This is different from
`scripts/calibration/run_recalibration.py`, which is the in-depth study of the
recalibrator itself.

`hgb_quantile_conformal` is the same HGB quantile model wrapped in
`calibration.ConformalRecalibrator`: the same out-of-fold residuals as
`hgb_quantile_recal`, but corrected by one global constant per quantile level
(split conformal) instead of a spline + ridge function of the predicted load.
`tabicl_conformal` does the same for TabICL.

`cross_validate.py` is the small convenience wrapper for a single CV run of the
default pipeline.

## Building reports

```bash
python scripts/model_comparison/build_cv_report.py results/model_comparison/my_study
python scripts/model_comparison/build_calibration_report.py results/model_comparison/my_study
python scripts/model_comparison/plot_cv_comparison.py results/model_comparison/my_study
```

- `build_cv_report.py` writes a metrics table, one true-vs-predicted plot per
  model/horizon, and a `pareto.html` page with accuracy vs total and predict
  time. `--models a,b,...` restricts the page to those models,
  `--summary_labels` uses the short publication labels, and
  `--pareto_only PATH` writes only the Pareto page (with `--figure_height` /
  `--page_width`). The curated summary page is built with:
  ```bash
  python scripts/model_comparison/build_cv_report.py \
      results/model_comparison/cv_comparison_forecast \
      --models hgb_mean,hgb_mean_fast,tabicl_n8_full,tabicl_forecast_ctx1024_n4,kumo_tabular,kumo_forecast,chronos2_ctx4000,hgb_quantile_q7,hgb_quantile_recal_q7 \
      --summary_labels --pareto_only results/model_comparison/summary/pareto.html \
      --figure_height 700 --page_width 1600
  ```
- `build_calibration_report.py` adds coverage, conditional coverage and
  reliability diagrams for all quantile models.
- `plot_cv_comparison.py` overlays all models on one plot per horizon.
- `compare_predictions.py` recomputes MAE, MAPE and pinball-d2 directly from the
  saved parquet files.
- `build_combined_pareto.py` and `build_weather_impact_report.py` build the
  oracle vs forecast weather pages (`results/weather_impact/report/`).
- `build_summary_figures.py` writes four static SVG summary figures to
  `results/model_comparison/summary/figures/`: paired MAPE/pinball differences vs
  baseline against fit + predict and predict-only time, the binned 90% interval
  coverage from the calibration summary, and true-vs-predicted 24h load for five
  models over the first two weeks of April 2025, ordered by whole-dataset MAPE.
- `build_summary_calibration.py` writes a separate
  `results/model_comparison/summary/calibration.html` page with binned interval
  coverage and binned reliability at 5/50/95 for the forecast-weather summary
  models. Point-only models are skipped.
- `build_coverage_report.py` writes a separate
  `results/model_comparison/summary/coverage.html` page from the forecast-weather
  run only. It flags models missing the nominal 90% interval coverage target per
  horizon (0.5 pp tolerance by default), alongside the pinball Pareto and a
  12h/24h coverage panel. It does not modify `pareto.html`.
- `uncertainty.py` is the shared helper for paired differences between models,
  used by the three report scripts above. For two models on identical rows it
  gives the per-fold mean differences (one dot per fold in the plots) and a 95%
  moving-block bootstrap interval (14-day blocks) on the daily mean difference
  of the pooled test rows. The interval is conditional on the fitted models
  (training randomness is not included) and the folds are not treated as
  independent samples, so there is no t-interval across folds. Use it for any
  new paired comparison instead of re-implementing intervals.

## Provenance

Every result directory and every generated page must record what produced it
(command, commit and dirty state, package versions and hardware, the weather
source and features actually used, resolved hyperparameters, and the source
directories of derived results). See "Provenance (mandatory)" in `AGENTS.md`.
When adding a `cross_validate_models.py` run to a study directory, write its
`README.md` entry at the same time.
