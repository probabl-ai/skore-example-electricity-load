from pathlib import Path
import datetime

import plotly.graph_objects as go
import polars as pl

import electricity_load_forecasting as elf

pred = elf.make_data_op(horizons=(1,), quantile_strategy=None)
env = elf.get_env()


history = elf.resample(elf.fetch_load_mw_history())

figures = []

for split in pred.skb.iter_cv_splits(environment=env):
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=history["time"],
            y=history["load_mw"],
            mode="lines",
            line={"dash": "dash", "color": "gray"},
        )
    )
    fig.add_trace(
        go.Scatter(
            x=split["X_train"]["prediction_time"] + datetime.timedelta(hours=1),
            y=split["y_train"]["1h"],
            mode="lines+markers",
            name="train",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=split["X_test"]["prediction_time"] + datetime.timedelta(hours=1),
            y=split["y_test"]["1h"],
            mode="lines+markers",
            name="test",
        )
    )
    fig.update_layout(height=600)
    figures.append(fig.to_html(full_html=False))


fig_snippets = "\n".join(f"<p>\n{f}\n</p>\n" for f in figures)
figures_html = f"""
<!DOCTYPE html>
<html>
<head>
<title>train/test splits</title>
<meta charset="UTF-8" />
</head>
<body>
{fig_snippets}
</body>
</html>
"""

Path("train_test_splits.html").write_text(figures_html, "UTF-8")
