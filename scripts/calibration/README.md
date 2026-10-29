# Calibration

In-depth study of quantile calibration and the effect of
`electricity_load_forecasting.calibration.Recalibrator`.

## Scripts

- `run_recalibration.py` runs the raw vs recalibrated walk-forward CV and writes
  `results/calibration/recalibration_<timestamp>/` with predictions, scores,
  learned corrections and inner out-of-fold data. The report is built at the end
  of the run; use `--report_only DIR` to rebuild it from an existing directory.
- `recalibration_report.py` contains the report-generation code. It is normally
  imported by `run_recalibration.py`.
- `single_date_calibration.py` compares raw and recalibrated forecasts for
  selected dates of one CV fold; writes
  `results/calibration/single_date_calibration.html` and caches fitted models in
  `results/calibration/single_date_models/`.
- `calibration_demo.py` is a pedagogical toy example; writes
  `results/calibration/calibration_demo.html`.

General model-comparison reliability diagrams live in
`scripts/model_comparison/build_calibration_report.py`.
