"""Measure how CV metrics drift with time since retraining.

Each CV test fold is a contiguous ~13-week block and the model is retrained at
the start of each fold. This bins every test fold by elapsed time since the
fold start (the retraining point) and reports MAPE, pinball-d2, 90% interval
coverage and width. Within-fold slopes control for the fact that different
folds cover different seasons.

    python scripts/inspection/retraining_frequency.py \
        results/model_comparison/cv_comparison
"""

import argparse
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
import polars as pl
from plotly.subplots import make_subplots
from sklearn.metrics import d2_pinball_score

HORIZONS = (1, 12, 24)
QUANTILES = (0.05, 0.5, 0.95)
MODEL_ORDER = (
    "chronos2_ctx4000",
    "hgb_mean",
    "hgb_quantile_q7",
    "hgb_quantile_recal_q7",
    "tabicl_n8_full",
    "kumo",
)
ZERO_SHOT_MODELS = ("chronos2_ctx4000", "kumo")
MODEL_COLORS = {
    "chronos2_ctx4000": "#0072B2",
    "hgb_mean": "#E69F00",
    "hgb_quantile_q7": "#009E73",
    "hgb_quantile_recal_q7": "#CC79A7",
    "tabicl_n8_full": "#D55E00",
    "kumo": "#56B4E9",
}
FALLBACK_COLORS = ("#0072B2", "#E69F00", "#009E73", "#CC79A7", "#D55E00", "#56B4E9")


def ordered_models(pooled):
    present = pooled["model"].unique().to_list()
    ordered = [m for m in MODEL_ORDER if m in present]
    ordered += sorted(m for m in present if m not in ordered)
    return ordered


def model_color(name, index):
    return MODEL_COLORS.get(name, FALLBACK_COLORS[index % len(FALLBACK_COLORS)])


def metric_figure(pooled, metric, title, y_title, bin_days, scale=1.0, hline=None):
    fig = make_subplots(
        rows=1,
        cols=len(HORIZONS),
        subplot_titles=[f"{h}h horizon" for h in HORIZONS],
        horizontal_spacing=0.06,
    )
    models = ordered_models(pooled)
    for j, h in enumerate(HORIZONS, 1):
        if hline is not None:
            fig.add_hline(
                y=hline,
                line=dict(color="gray", dash="dash"),
                row=1,
                col=j,
                exclude_empty_subplots=False,
            )
        for i, name in enumerate(models):
            sub = pooled.filter(
                (pl.col("model") == name) & (pl.col("horizon") == h)
            ).sort("bin")
            if sub.height == 0 or sub[metric].is_nan().all():
                continue
            x = sub["bin"].to_numpy() * bin_days + bin_days / 2
            y = sub[metric].to_numpy() * scale
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    mode="lines+markers",
                    line=dict(color=model_color(name, i), width=2),
                    marker=dict(size=6),
                    name=name,
                    legendgroup=name,
                    showlegend=j == 1,
                    hovertemplate=(
                        f"{name}<br>%{{x:.0f}} days since retraining"
                        "<br>%{y:.4g}<extra></extra>"
                    ),
                ),
                row=1,
                col=j,
            )
        fig.update_xaxes(title_text="days since retraining", row=1, col=j)
    fig.update_yaxes(title_text=y_title, row=1, col=1)
    fig.update_layout(height=430, title=title, margin=dict(t=70))
    return fig


def write_plots(pooled, slopes, output_dir, bin_days):
    figures = [
        metric_figure(
            pooled,
            "mape",
            "MAPE by time since retraining",
            "MAPE (%)",
            bin_days,
            scale=100,
        ),
        metric_figure(
            pooled,
            "pinball_d2",
            "Pinball-d2 by time since retraining",
            "pinball-d2 (mean over q0.05/0.5/0.95)",
            bin_days,
        ),
        metric_figure(
            pooled,
            "coverage_90",
            "90% interval coverage by time since retraining",
            "coverage",
            bin_days,
            hline=0.9,
        ),
    ]
    parts = [
        fig.to_html(
            full_html=False,
            include_plotlyjs=i == 0,
            config={"responsive": True, "displaylogo": False},
        )
        for i, fig in enumerate(figures)
    ]
    rows = ""
    for r in slopes.sort(["model", "horizon"]).iter_rows(named=True):
        mape = r["mape_slope_30d"]
        pinball = r["pinball_d2_slope_30d"]
        coverage = r["coverage_90_slope_30d"]
        rows += (
            f"<tr><td>{r['model']}</td><td>{r['horizon']}h</td>"
            f"<td>{mape * 100:+.3f}</td>"
            f"<td>{'-' if np.isnan(pinball) else f'{pinball:+.4f}'}</td>"
            f"<td>{'-' if np.isnan(coverage) else f'{coverage:+.4f}'}</td></tr>"
        )
    page = f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>Retraining frequency</title>
<style>
body {{ font-family: sans-serif; max-width: 1500px; margin: 1em auto; padding: 0 1em; line-height: 1.45; }}
table {{ border-collapse: collapse; margin: 1em 0; }}
th, td {{ padding: 4px 10px; text-align: right; border-bottom: 1px solid #ddd; }}
th:first-child, td:first-child {{ text-align: left; }}
</style>
</head>
<body>
<h1>Retraining frequency: metric drift since the last retraining</h1>
<p>Each CV test fold is binned by elapsed days since the fold start (the
retraining point), in two-week bins. The last partial week is excluded. Points
are pooled over the 8 folds; the table below shows within-fold slopes averaged
over folds, which removes most fold/season differences.</p>
<p>Models are retrained at the start of each 13-week fold. If predictions
degrade noticeably with elapsed time, retraining more often may help.</p>
<p><strong>Chronos-2 and Kumo are excluded:</strong> they are zero-shot and
always consume history up to the prediction time, so they are never stale with
respect to the retraining schedule. Any within-fold drift they show is due to
other time-varying factors, not retraining frequency. TabICL (8 estimators,
full-history context) is included because its in-context training set is
refreshed only once per fold.</p>
{"".join(parts)}
<h2>Within-fold slopes (per 30 days)</h2>
<p>MAPE is in percentage points; pinball-d2 and coverage are absolute changes.
For MAPE, positive is worse. For pinball-d2, negative is worse. Coverage is
relative to the nominal 0.90 level.</p>
<table>
<thead><tr><th>model</th><th>horizon</th><th>MAPE (pp/30d)</th>
<th>pinball-d2 /30d</th><th>coverage /30d</th></tr></thead>
<tbody>{rows}</tbody>
</table>
<p>Caveat: only 8 folds; the slopes are linear approximations and residual
seasonal effects remain. The final partial week is excluded because it is only
7 days and is seasonally noisy.</p>
</body>
</html>
"""
    (output_dir / "index.html").write_text(page, "utf-8")


def metrics(df, horizon, has_quantiles):
    y = df[f"{horizon}h"].to_numpy().astype(float)
    median_column = (
        f"pred_{horizon}h__q_0.5" if has_quantiles else f"pred_{horizon}h"
    )
    median = df[median_column].to_numpy().astype(float)
    out = {
        "mape": float(np.mean(np.abs(y - median) / np.abs(y))),
        "n": df.height,
    }
    if not has_quantiles:
        out.update(pinball_d2=np.nan, coverage_90=np.nan, width_90=np.nan)
        return out
    low = df[f"pred_{horizon}h__q_0.05"].to_numpy().astype(float)
    high = df[f"pred_{horizon}h__q_0.95"].to_numpy().astype(float)
    out["coverage_90"] = float(np.mean((y >= low) & (y <= high)))
    out["width_90"] = float(np.mean(high - low))
    out["pinball_d2"] = float(
        np.mean(
            [
                d2_pinball_score(
                    y, df[f"pred_{horizon}h__q_{q}"].to_numpy().astype(float), alpha=q
                )
                for q in QUANTILES
            ]
        )
    )
    return out


def slope(rows, metric, bin_days):
    values = np.array(
        [(bin_index * bin_days, row[metric]) for bin_index, row in rows], dtype=float
    )
    return float(np.polyfit(values[:, 0], values[:, 1], 1)[0])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "directory",
        nargs="?",
        type=Path,
        default=Path("results/model_comparison/cv_comparison"),
    )
    parser.add_argument("--bin_days", type=int, default=14)
    parser.add_argument(
        "--include_partial",
        action="store_true",
        help="include the last incomplete bin (84-90 days); by default only "
        "complete two-week bins are used",
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/inspection/retraining_frequency"),
    )
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    n_bins = 7 if args.include_partial else 6

    pooled_rows, slope_rows = [], []
    for path in sorted(args.directory.glob("*_cv_predictions.parquet")):
        name = path.stem.removesuffix("_cv_predictions")
        if name in ZERO_SHOT_MODELS:
            print(f"skipping {name}: zero-shot, always uses fresh history")
            continue
        df = pl.read_parquet(path)
        starts = df.group_by("split").agg(
            pl.col("prediction_time").min().alias("fold_start")
        )
        df = df.join(starts, on="split", how="left").with_columns(
            (
                (pl.col("prediction_time") - pl.col("fold_start")).dt.total_seconds()
                / 86400
            ).alias("elapsed_days")
        )
        df = df.with_columns(
            (pl.col("elapsed_days") // args.bin_days)
            .clip(0, n_bins - 1)
            .cast(pl.Int32)
            .alias("bin")
        )
        has_quantiles = all(
            f"pred_{h}h__q_0.5" in df.columns for h in HORIZONS
        )

        # Metrics on the pooled data, per bin and horizon.
        for b in range(n_bins):
            sub = df.filter(pl.col("bin") == b)
            if sub.height == 0:
                continue
            for h in HORIZONS:
                row = metrics(sub, h, has_quantiles)
                pooled_rows.append(
                    {
                        "model": name,
                        "horizon": h,
                        "bin": b,
                        "days": f"{b * args.bin_days}-{min((b + 1) * args.bin_days, 91) - 1}",
                        **row,
                    }
                )

        # Within-fold slope per horizon, averaged over folds.
        for h in HORIZONS:
            per_fold = []
            for s in sorted(df["split"].unique().to_list()):
                fold_rows = []
                for b in range(n_bins):
                    sub = df.filter((pl.col("split") == s) & (pl.col("bin") == b))
                    if sub.height:
                        fold_rows.append((b, metrics(sub, h, has_quantiles)))
                if len(fold_rows) >= 3:
                    per_fold.append(fold_rows)
            if not per_fold:
                continue
            row = {"model": name, "horizon": h, "n_folds": len(per_fold)}
            for metric in ("mape", "pinball_d2", "coverage_90"):
                if not has_quantiles and metric != "mape":
                    row[f"{metric}_slope_30d"] = np.nan
                    row[f"{metric}_worsening_folds"] = np.nan
                    continue
                slopes = [slope(rows, metric, args.bin_days) for rows in per_fold]
                row[f"{metric}_slope_30d"] = float(np.mean(slopes) * 30)
                if metric == "mape":
                    row[f"{metric}_worsening_folds"] = int(np.sum(np.array(slopes) > 0))
                else:
                    row[f"{metric}_worsening_folds"] = int(
                        np.sum(np.array(slopes) < 0)
                    )
            slope_rows.append(row)

    pooled = pl.DataFrame(pooled_rows)
    pooled.write_csv(args.output_dir / "metrics_by_bin.csv")
    slopes = pl.DataFrame(slope_rows)
    slopes.write_csv(args.output_dir / "within_fold_slopes.csv")

    # Simple average across horizons.
    avg = (
        pooled.group_by(["model", "bin", "days"])
        .agg(
            pl.col("mape").mean().alias("mape"),
            pl.col("pinball_d2").mean().alias("pinball_d2"),
            pl.col("coverage_90").mean().alias("coverage_90"),
            pl.col("width_90").mean().alias("width_90"),
            pl.col("n").sum().alias("n"),
        )
        .sort(["model", "bin"])
    )
    avg.write_csv(args.output_dir / "metrics_by_bin_avg.csv")
    write_plots(pooled, slopes, args.output_dir, args.bin_days)
    print(f"wrote {args.output_dir}")


if __name__ == "__main__":
    main()
