# Chronos vs. baseline comparison scripts

Ad hoc tooling built to compare `chronos_forecasting.py` against the baseline
pipelines (`hgb_mean`, `hgb_quantile`, `tabicl`) with a full walk-forward CV
(`elf.cross_val_predict`, no skore -- see `cross_validate.py` at the repo
root for the pattern these follow). Kept separate from the repo root because
there are several of them and they're specific to this investigation rather
than part of the core pipeline; run them from the repository root, e.g.:

```
python chronos_comparison_scripts/cross_validate_compare.py --model chronos2 \
    --output_dir results/cv_compare_xxx --device cuda
```

- `cross_validate_compare.py` -- runs one pipeline (`hgb_mean`, `hgb_quantile`,
  `hgb_quantile_recal`, `tabicl` or `chronos2`) through all CV folds and saves
  predictions (`*_cv_predictions.parquet`) and per-fold scores (`*_scores.csv`)
  to `--output_dir`. `hgb_quantile_recal` is HGB with the
  `calibration.Recalibrator` post-hoc correction (default settings);
  `--max_iter` applies to the HGB models, and `--quantiles` to all models. Run
  once per model/configuration you want to compare, pointing all of them at the
  same `--output_dir`; use `--tag` to keep multiple runs of the same model
  (e.g. different `tabicl_n_estimators`) side by side instead of overwriting.
- `compare_predictions.py` -- computes MAE, MAPE, pinball-d2 and
  coverage/width (for quantile models) straight from the saved
  `*_cv_predictions.parquet` files, no re-running needed. Also a library:
  `compute_scores(df)` / `compare(paths)` are imported by the two scripts
  below.
- `build_cv_report.py` -- single HTML report: the metrics table from
  `compare_predictions`, plus links to one true-vs-predicted plot per
  model/horizon (median line + shaded quantile interval for quantile models),
  written as separate files under `<dir>/plots/` to keep the main page small.
- `build_calibration_report.py` -- reliability / calibration diagrams in the
  style of `compare_calibration.py`'s report: marginal coverage curve,
  conditional coverage of the outermost interval by predicted load, and
  per-quantile conditional reliability (equal-count bins and isotonic),
  overlaying every model found in the directory. Writes `<dir>/calibration.html`
  (linked from `report.html`).
- `plot_cv_comparison.py` -- alternative view: overlays every model's median
  prediction on one plot per horizon (no shading), instead of one plot per
  model.

For a report-focused single run (skrub fit/predict/single-date reports, same
structure as `cross_validate.py`), use `cross_validate_chronos.py` at the repo
root instead -- it also scores via the full walk-forward CV, just for one
model configuration with reports rather than many configurations compared
side by side.
