# skrub report `skrub_report_hgb_quantile_all_groups`

Built without any CV: only the DataOp graph and its previews on `2021-03-23`
to `2025-05-31`.

- Command: `python scripts/inspection/make_skrub_report.py --model hgb_quantile --feature_groups load_diffs load_same_hour temp_diffs` (from the repository root)
- Date: 2026-11-21T07:50:45+00:00
- Git commit: 41a6cbd69d99717ec82803a01e5a441d8607be87; modified files (untracked excluded):
  none (clean)
- Host: 6fd98e65a156, x86_64
- Packages: python 3.14.7, polars 1.44.2,
  scikit-learn 1.9.1, skrub 0.12.dev0
  (installed from https://github.com/skrub-data/skrub.git@a14648149ea1ad8bd689c8ed6796d5047a4beea3)
- Weather: default `fetch_city_weather_forecast` (historical
  forecasts, day1 lead for horizons <= 18h, day2 above); load from
  `fetch_load_mw_history`; public holidays from `holidays.country_holidays("FR")`
- Evaluation range (`get_env()`): {'start': '2021-03-23', 'end': '2025-05-31'}
- Feature groups: ['load_diffs', 'load_same_hour', 'temp_diffs']
- `make_data_op` arguments as resolved: {'horizons': (1, 12, 24), 'model': 'hgb_quantile', 'quantiles': (0.05, 0.5, 0.95), 'max_train_rows': None, 'max_iter': 3000, 'recalibrate': False, 'feature_groups': ['load_diffs', 'load_same_hour', 'temp_diffs']}
- Model defaults as resolved by `describe_defaults()`:
  {"temperature_only": "True", "cities": "all", "learning_rate": "0.1", "max_leaf_nodes": "31", "min_samples_leaf": "20"}
