# Scripts

The analysis code is organized by study/theme. Run scripts from the repository
root; the `electricity_load_forecasting` package is installed from `src/`.

- `calibration/` — raw vs recalibrated quantile study, in-depth calibration
  reports, single-date case studies and the pedagogical demo.
- `model_comparison/` — run comparable walk-forward CV predictions for several
  model families and build metrics/calibration comparison reports.
- `tuning/` — hyperparameter search with Optuna.
- `inspection/` — data/pipeline diagnostics and stable stakeholder reports.
- `checks/` — contract and smoke checks for optional model backends.

Results go under `results/<theme>/...` and each experiment directory should have
a short `README.md` with its provenance. Legacy results are kept under
`results/old/`. Published HTML reports live in `docs/`.
