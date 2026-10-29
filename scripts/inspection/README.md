# Inspection and diagnostics

Ad hoc tools for understanding the data and the pipeline. They are not
experiments: outputs are written to stable paths under `results/inspection/`.

- `eda.py` opens skrub reports for the raw datasets and load history.
- `draw_graph.py` writes the DataOp graph to `results/inspection/graph.{svg,png}`.
- `make_skrub_report.py` regenerates the skrub report for the current pipeline
  under `results/inspection/skrub_report_<pipeline>/`.
- `show_train_test_splits.py` writes
  `results/inspection/train_test_splits.html`.
- `single_date_plot.py` writes example single-date forecast plots under
  `results/inspection/`.
