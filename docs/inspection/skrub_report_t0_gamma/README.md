# skrub report `skrub_report_t0_gamma`

Built without any CV: only the DataOp graph and its previews on `2025-05-01`
to `2025-05-31` (shortened with --start).

- Command: `python scripts/inspection/make_skrub_report.py --model t0_gamma --start 2025-05-01` (from the repository root)
- Date: 2026-11-24T04:58:47+00:00
- Git commit: a5c9c9aa4f76a5f1ea13794d630a3a06e0125b53; modified files (untracked excluded):
  M scripts/checks/smoke_chronos.py,  M scripts/inspection/make_skrub_report.py,  M scripts/model_comparison/build_weather_impact_report.py,  M scripts/model_comparison/cross_validate_models.py,  M src/electricity_load_forecasting/__init__.py,  M src/electricity_load_forecasting/chronos.py,  M src/electricity_load_forecasting/pipeline.py
- Host: ip-10-3-68-243, x86_64
- Packages: python 3.12.3, polars 1.44.2,
  scikit-learn 1.9.1, skrub 0.11.0
  (installed from PyPI release@), torch 2.15.0, tfc-t0 0.6.1
- Weather: default `fetch_city_weather_forecast`; the window given to the model gets day1 (24h lead) for the past and future steps <= 18, day2 (48h lead) for later steps (one table each, `covariates` and `covariates_long`); load from
  `fetch_load_mw_history`; public holidays from `holidays.country_holidays("FR")`
- Evaluation range (`get_env()`, `start` overridden if shortened): {'start': '2025-05-01', 'end': '2025-05-31'}
- `make_t0_data_op` arguments as resolved: {'horizons': (1, 12, 24), 'forecaster': "T0Forecaster(device='cuda', dtype='bfloat16')", 'quantiles': (0.05, 0.5, 0.95), 'context_hours': 1024, 'cities': 'all', 'temperature_only': True, 'calendar': True, 'chunk_size': 64, 'verbose': False, 'forecaster_params': {'clip_nonnegative': True, 'device': 'cuda', 'dtype': 'bfloat16', 'model': 'theforecastingcompany/t0-gamma', 'snap_integers': True, 'symmetric_averaging': True, 'trim_leading_missing': True}}
- Model defaults as resolved by `describe_defaults()`:
  {}
