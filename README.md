# Electricity load forecasting

This example is adapted from https://github.com/probabl-ai/forecasting.

The goal is to have a dataset and prediction pipeline that can be used to
experiment with features of skore and other packages.

## Layout

- `src/electricity_load_forecasting/` — core package: data loading, feature
  engineering, the skrub DataOp pipeline, CV splitter, metrics and plotting.
- `scripts/` — analysis code grouped by study/theme:
  `calibration/`, `model_comparison/`, `tuning/`, `inspection/` and `checks/`.
- `datasets/` — committed historical load and weather data.
- `results/` — generated experiment outputs (git-ignored), organized by theme.
  Legacy pre-reorganization results are kept under `results/old/`.
- `docs/` — published HTML reports and the experiment index.

Each experiment result directory should contain a short `README.md` describing
what was run, when, and with which settings.

## Quick start

With pixi:

```bash
pixi install
pixi run python scripts/model_comparison/cross_validate.py
pixi run python scripts/model_comparison/cross_validate_models.py \
    --model hgb_quantile --output_dir results/model_comparison/my_study
```

See the published reports and experiment overview in `docs/index.html`.
