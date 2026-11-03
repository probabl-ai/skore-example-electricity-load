# Model comparison

Scripts for running and comparing walk-forward CV predictions from several
model families on the same folds: `hgb_mean`, `hgb_quantile`,
`hgb_quantile_recal`, `tabicl`, `tabicl_recal`, `kumo_tabular`,
`kumo_tabular_recal`, `chronos2` and `kumo_forecast`.

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
```

Weather defaults to the safe-lead historical forecasts (`--weather forecast`):
day1 for horizons <=18h and day2 above. `--weather oracle` is meant to select the
oracle files, but **currently it is a silent no-op** (the default fetcher is now
the forecast one and the flag only overrides it for `forecast`), so it still uses
the forecast files; see `_scratch/REVIEW_REPORT_2.md`, finding 1.1. Note also that
the Chronos and Kumo pipelines always use the day2 files for every horizon.

Each run writes `<model>_cv_predictions.parquet` and `<model>_scores.csv`.
Use `--tag` to keep multiple runs of the same model side by side. Add
`--reports` to also write the skrub fit/predict/single-date reports for that
model.

To add the recalibrated HGB to an existing comparison, run the command above
with `--model hgb_quantile_recal` and the same `--output_dir`; it will then be
picked up by the report scripts. This is different from
`scripts/calibration/run_recalibration.py`, which is the in-depth study of the
recalibrator itself.

`cross_validate.py` is the small convenience wrapper for a single CV run of the
default pipeline.

## Building reports

```bash
python scripts/model_comparison/build_cv_report.py results/model_comparison/my_study
python scripts/model_comparison/build_calibration_report.py results/model_comparison/my_study
python scripts/model_comparison/plot_cv_comparison.py results/model_comparison/my_study
```

- `build_cv_report.py` writes a metrics table and one true-vs-predicted plot per
  model/horizon.
- `build_calibration_report.py` adds coverage, conditional coverage and
  reliability diagrams for all quantile models.
- `plot_cv_comparison.py` overlays all models on one plot per horizon.
- `compare_predictions.py` recomputes MAE, MAPE and pinball-d2 directly from the
  saved parquet files.
- `build_combined_pareto.py` and `build_weather_impact_report.py` build the
  oracle vs forecast weather pages (`results/weather_impact/report/`).
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
