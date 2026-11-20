"""
Build the HTML page comparing models with the selected feature groups against
the same models with default features (round 2, see
`results/feature_engineering/round2_selected_groups/README.md`).

    python scripts/feature_engineering/build_report.py [DIR]

DIR defaults to `results/feature_engineering/round2_selected_groups`; it holds
`<model>/<model>_fg3_{cv_predictions.parquet,metadata.json}`. The earlier runs
with default features are read from `results/model_comparison/`. Writes
`DIR/report.html` (same paired-bootstrap code as `compare_runs.py`).
"""

import html
import json
import sys
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots

sys.path.insert(0, str(Path(__file__).parent.parent / "model_comparison"))
from uncertainty import error_bar, paired_difference, row_index, row_loss  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DIR = (
    Path(sys.argv[1])
    if len(sys.argv) > 1
    else ROOT / "results/feature_engineering/round2_selected_groups"
)
OLD = ROOT / "results/model_comparison"
OLD_RUNS = {
    "hgb_mean": "hgb_mean_forecast_safe/hgb_mean_forecast_safe",
    "hgb_quantile": "hgb_quantile_q7_forecast_safe/hgb_quantile_forecast_safe",
    "hgb_quantile_recal": "hgb_quantile_recal_q7_forecast_safe/hgb_quantile_recal_forecast_safe",
    "hgb_quantile_conformal": "hgb_quantile_conformal_q7_forecast_safe/hgb_quantile_conformal_forecast_safe",
    "tabicl": "tabicl_n8_full_forecast_safe/tabicl_n8_full_forecast_safe",
    "kumo_tabular": "kumotabular_n8_full_forecast_safe/kumo_tabular_n8_full_forecast_safe",
}
HORIZONS = (1, 12, 24)
KEYS = ("avg", *HORIZONS)
COLOURS = {"avg": "#444444", 1: "#1f77b4", 12: "#ff7f0e", 24: "#2ca02c"}
CONFIG = {"responsive": True, "displaylogo": False}


def read(path):
    return pl.read_parquet(path).sort("prediction_time", "split")


def losses(df, kind):
    per_h = {h: row_loss(df, kind, h) for h in HORIZONS}
    return per_h | {"avg": np.mean(list(per_h.values()), axis=0)}


def pinball(df):
    """Mean pinball loss over the 5/50/95% levels, per horizon and averaged."""
    per_h = {
        h: np.mean([row_loss(df, "pinball", h, q) for q in (0.05, 0.5, 0.95)], axis=0)
        for h in HORIZONS
    }
    return per_h | {"avg": np.mean(list(per_h.values()), axis=0)}


models = {}
for name, old in OLD_RUNS.items():
    new_df = read(DIR / name / f"{name}_fg3_cv_predictions.parquet")
    old_df = read(OLD / f"{old}_cv_predictions.parquet")
    assert (new_df["prediction_time"] == old_df["prediction_time"]).all()
    day, fold = row_index(new_df)
    entry = {
        "metadata": json.loads((DIR / name / f"{name}_fg3_metadata.json").read_text())
    }
    for kind in ("mae", "mape", "pinball"):
        if kind == "pinball" and "pred_1h__q_0.05" not in new_df.columns:
            continue
        new = losses(new_df, kind) if kind != "pinball" else pinball(new_df)
        old_l = losses(old_df, kind) if kind != "pinball" else pinball(old_df)
        entry[kind] = {
            k: {
                "old": float(old_l[k].mean()),
                "new": float(new[k].mean()),
                **paired_difference(new[k], old_l[k], day, fold),
            }
            for k in KEYS
        }
    models[name] = entry


def cell(s, digits):
    star = "*" if s["low"] > 0 or s["high"] < 0 else ""
    return f"{s['old']:.{digits}f} &rarr; {s['new']:.{digits}f} ({s['mean']:+.{digits}f}){star}"


def table(kind, unit, digits):
    rows = "".join(
        f"<tr><td>{name}</td>"
        + "".join(f"<td>{cell(m[kind][k], digits)}</td>" for k in KEYS)
        + f"<td>{m[kind]['avg']['n_lower']}/8</td></tr>"
        for name, m in models.items()
        if kind in m
    )
    return (
        f"<table><thead><tr><th>model</th><th>{kind.upper()} ({unit}) average</th>"
        "<th>1h</th><th>12h</th><th>24h</th><th>folds better (avg)</th></tr></thead>"
        f"<tbody>{rows}</tbody></table>"
    )


def differences_figure():
    fig = make_subplots(
        rows=1,
        cols=4,
        shared_yaxes=True,
        horizontal_spacing=0.03,
        subplot_titles=["average", "1h", "12h", "24h"],
    )
    names = list(models)
    for j, key in enumerate(KEYS, 1):
        s = [models[n]["mae"][key] for n in names]
        up, down = zip(*(error_bar(x) for x in s))
        fig.add_trace(
            go.Scatter(
                x=[x["mean"] for x in s],
                y=names,
                mode="markers",
                marker={"color": COLOURS[key], "size": 9},
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": up,
                    "arrayminus": down,
                },
                showlegend=False,
                hovertemplate="%{y}<br>%{x:+.1f} MW<extra></extra>",
            ),
            row=1,
            col=j,
        )
        fig.add_vline(x=0, line_color="gray", line_width=1, row=1, col=j)
        fig.update_xaxes(title_text="MAE difference (MW)", row=1, col=j)
    fig.update_yaxes(autorange="reversed")
    fig.update_layout(
        height=420,
        margin={"t": 80, "l": 190, "r": 20},
        title="MAE with new features minus MAE with default features (negative is better), 95% block-bootstrap interval",
    )
    return fig


def absolute_figure():
    names = list(models)
    fig = go.Figure()
    for label, which, colour in (
        ("default features", "old", "#bbbbbb"),
        ("selected features", "new", "#1f77b4"),
    ):
        fig.add_trace(
            go.Bar(
                name=label,
                x=names,
                y=[models[n]["mae"]["avg"][which] for n in names],
                marker_color=colour,
                text=[f"{models[n]['mae']['avg'][which]:.0f}" for n in names],
                textposition="outside",
            )
        )
    fig.update_layout(
        barmode="group",
        height=430,
        title="Average MAE over horizons (MW)",
        yaxis_title="MAE (MW)",
        margin={"t": 70, "b": 100},
        legend={"orientation": "h", "y": 1.1},
    )
    return fig


def fold_figure():
    fig = go.Figure()
    for name, m in models.items():
        fig.add_trace(
            go.Scatter(
                x=list(range(8)),
                y=m["mae"]["avg"]["folds"],
                mode="lines+markers",
                name=name,
            )
        )
    fig.add_hline(y=0, line_color="gray", line_width=1)
    fig.update_layout(
        height=420,
        title="Mean MAE difference in each CV fold (average over horizons)",
        xaxis_title="CV fold (13-week test blocks, in time order)",
        yaxis_title="MAE difference (MW)",
        margin={"t": 70},
    )
    return fig


def div(fig, first=False):
    return fig.to_html(
        full_html=False, include_plotlyjs="cdn" if first else False, config=CONFIG
    )


runs = "".join(
    f"<tr><td>{name}</td><td><code>{html.escape(m['metadata']['command'])}</code></td>"
    f"<td>{m['metadata']['host']['node']}, python {m['metadata']['versions']['python']}, "
    f"{html.escape(', '.join(f'{k} {v}' for k, v in m['metadata']['versions'].items() if v and k != 'python'))}</td>"
    f"<td>{m['metadata']['git_commit'][:7]}"
    f"{' (modified: ' + html.escape(', '.join(m['metadata']['git_modified_files'])) + ')' if m['metadata']['git_modified_files'] else ' (clean)'}</td>"
    f"<td>{OLD_RUNS[name].split('/')[0]}</td></tr>"
    for name, m in models.items()
)
meta = models["hgb_mean"]["metadata"]
page = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Selected feature groups on all lagged-feature models</title>
<style>
  body {{ font-family: sans-serif; max-width: 1500px; margin: 2em auto; padding: 0 1em; line-height: 1.4; }}
  h1 {{ font-size: 1.6em; }}
  h2 {{ font-size: 1.15em; margin-top: 1.8em; border-bottom: 1px solid #ddd; padding-bottom: 0.2em; }}
  table {{ border-collapse: collapse; width: 100%; margin-bottom: 1em; }}
  th, td {{ border-bottom: 1px solid #ddd; padding: 0.45em; text-align: right; }}
  th:first-child, td:first-child {{ text-align: left; }}
  .runs td {{ text-align: left; font-size: 0.85em; vertical-align: top; }}
  .note {{ color: #555; font-size: 0.9em; }}
</style>
</head>
<body>
<h1>Selected feature groups on all lagged-feature models</h1>
<p>The three feature groups that helped in <a href="round1_hgb_mean.html">round 1</a>
(<code>load_diffs</code>, <code>load_same_hour</code>, <code>temp_diffs</code>) are added to the
default features of every lagged-feature model. Each model is compared with its earlier run with
default features, with the full 8-fold walk-forward CV and safe-lead forecast weather. All
average differences are significant, and the new features are better in 8/8 folds except TabICL
(6/8, no gain at 24h). Kumo-Tabular remains the best model (655 vs 757 MW for the best HGB) but
the gap shrinks.</p>
<ul>
<li><code>load_diffs</code>: differences between consecutive hourly load lags, and
<code>lag_h</code> minus <code>lag_24/48/168</code>.</li>
<li><code>load_same_hour</code>: load 1 and 7 days before the same clock hour (as extra lags),
and <code>lag_h</code> minus them.</li>
<li><code>temp_diffs</code>: national mean temperature change over 24h, 168h and h, and anomalies
against its 24h / 168h means.</li>
</ul>
<p class="note">Cells: default &rarr; new (difference); <code>*</code> = the 95% moving-block
bootstrap interval of the paired difference (14-day blocks over daily differences of the pooled
test rows) excludes 0. It covers test-period variability only, not training randomness.
Pinball is the mean over the 5%, 50% and 95% levels.</p>
<h2>MAE (MW)</h2>
{table("mae", "MW", 1)}
{div(differences_figure(), first=True)}
<h2>MAPE (%)</h2>
{table("mape", "%", 3)}
<h2>Pinball loss (MW)</h2>
{table("pinball", "MW", 1)}
<h2>Absolute errors</h2>
{div(absolute_figure())}
<h2>Consistency across folds</h2>
{div(fold_figure())}
<h2>Caveats</h2>
<ul>
<li>The runs with default features were not repeated: they come from other commits, hosts and
package versions (see their READMEs in <code>results/model_comparison/</code>). The
<code>hgb_mean</code> result with these groups (786.4 MW) is the same on the dev machine and
cpu-1, so HGB results do not depend on the host.</li>
<li>TabICL and Kumo-Tabular ran with torch 2.11.0 where their default-feature runs used 2.15.0,
and are stochastic ensembles: differences of a few MW may be noise. One run per row, no seed
repetition.</li>
<li>The recalibrated and conformal HGBs are still worse than plain <code>hgb_quantile</code> on MAE.</li>
</ul>
<h2>Provenance</h2>
<p class="note">Built by <code>scripts/feature_engineering/build_report.py</code> from
<code>{html.escape(str(DIR))}</code> (run date 2026-11-21) and
<code>{html.escape(str(OLD))}</code>. Evaluation range {meta["env"]["start"]} to {meta["env"]["end"]},
horizons {meta["horizons"]}, weather <code>{meta["weather_fetcher"]}</code> (day1 lead for
horizons &le; 18h, day2 above), splitter {html.escape(json.dumps(meta["splitter"]))},
HGB defaults as resolved {html.escape(json.dumps(meta["pipeline_defaults"]))}, max_iter {meta["max_iter"]}.
Feature names per horizon are in each <code>*_metadata.json</code>.</p>
<table class="runs"><thead><tr><th>model</th><th>command</th><th>host and packages</th><th>commit</th><th>compared with (<code>results/model_comparison/</code>)</th></tr></thead>
<tbody>{runs}</tbody></table>
</body>
</html>
"""
(DIR / "report.html").write_text(page, "utf-8")
print(f"wrote {DIR / 'report.html'}")
