# %%
import argparse
import datetime
import sys
from pathlib import Path

import plotly.graph_objects as go
import polars as pl

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import electricity_load_forecasting as elf

OUTPUT_DIR = ROOT / "results" / "inspection"

parser = argparse.ArgumentParser()
parser.add_argument("--model", default="tabicl")
args = parser.parse_args()
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# %%

pred = elf.make_data_op(horizons=tuple(range(1, 25)), model="hgb_mean")
env = elf.get_env()

split = pred.skb.train_test_split(environment=env)
new_date = elf.get_new_date()

learner = pred.skb.make_learner().fit(split["train"])
prediction = learner.predict({"start": new_date, "end": None})

# %%
history = elf.resample(elf.fetch_load_mw_history())
history_tail = history.filter(pl.col("time") > new_date - datetime.timedelta(days=8))

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=history_tail["time"],
        y=history_tail["load_mw"],
        mode="lines+markers",
        line={"dash": "dash", "color": "gray"},
    )
)
fig.add_trace(
    go.Scatter(
        x=prediction["time"],
        y=prediction["load_mw"],
        mode="lines+markers",
    )
)
fig.update_layout(height=700, showlegend=False)
fig.show(renderer="browser")
fig.write_html(OUTPUT_DIR / "example-mean.html")

# %%
pred = elf.make_data_op(
    horizons=tuple(range(1, 25)),
    model=args.model,
    quantiles=(0.05, 0.5, 0.95),
)
learner = pred.skb.make_learner().fit(split["train"])
prediction = learner.predict({"start": new_date, "end": None})

# %%

fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=history_tail["time"],
        y=history_tail["load_mw"],
        mode="lines+markers",
        line={"dash": "dash", "color": "gray"},
        name="historical data",
    )
)
fig.add_trace(
    go.Scatter(
        x=prediction["time"],
        y=prediction["q_0.5"],
        mode="lines+markers",
        name="median prediction",
    )
)

fig.add_trace(
    go.Scatter(
        x=prediction["time"],
        y=prediction["q_0.95"],
        mode="lines",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    )
)

fig.add_trace(
    go.Scatter(
        x=prediction["time"],
        y=prediction["q_0.05"],
        mode="lines",
        line=dict(width=0),
        fill="tonexty",
        fillcolor="rgba(0, 100, 250, 0.2)",
        showlegend=True,
        hoverinfo="skip",
        name="5% - 95% confidence interval",
    )
)

fig.update_layout(height=700)
fig.show(renderer="browser")

# %%
fig.write_html(OUTPUT_DIR / f"example-{args.model}-90.html")
