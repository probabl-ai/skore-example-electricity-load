# Model comparison

Scripts for running and comparing walk-forward CV predictions from several
model families on the same folds: `hgb_mean`, `hgb_quantile`,
`hgb_quantile_recal`, `tabicl`, `kumo_tabular`, `chronos2` and `kumo`.

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
    --model chronos2 --device cuda --output_dir results/model_comparison/my_study
```

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
