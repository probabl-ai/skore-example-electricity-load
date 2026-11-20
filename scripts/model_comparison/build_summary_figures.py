"""Static SVG versions of the summary figures.

1. Paired MAPE/pinball differences vs baseline, against fit + predict time.
2. The same against predict-only time.
3. The first row of the calibration summary page: binned coverage of the 90%
   interval by predicted median, per horizon.
4. Mean 90% interval width vs marginal coverage, one subplot per horizon.
5. True vs predicted 24h load for five models over 2025-03-23 to 2025-04-10,
   one subplot per model, ordered by window MAPE.

    python scripts/model_comparison/build_summary_figures.py

Writes five SVG files (plus matching PNGs under
``results/model_comparison/summary/figures/png/``) and ``figures.md`` under
``results/model_comparison/summary/figures/``. No model is re-run.
"""

import argparse
import datetime
import subprocess
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, HPacker, TextArea
from matplotlib.patches import Patch
from model_registry import model_colour
from sklearn.metrics import mean_absolute_percentage_error, mean_pinball_loss
from uncertainty import average_row_loss, paired_difference, row_index

from electricity_load_forecasting import diagnostics as dg

MODEL_ORDER = [
    "hgb_mean",
    "hgb_mean_fast",
    "tabicl_n8_full",
    "tabicl_recal_n8_full",
    "tabicl_forecast_ctx1024_n4",
    "kumo_tabular",
    "kumo_forecast",
    "chronos2_ctx4000",
    "t0_gamma",
    "hgb_quantile_q7",
    "hgb_quantile_recal_q7",
]
LABELS = {
    "hgb_mean": "HGB mean",
    "hgb_mean_fast": "HGB mean fast",
    "tabicl_n8_full": "TabICL",
    "tabicl_recal_n8_full": "TabICL recalibrated",
    "tabicl_forecast_ctx1024_n4": "TabICL forecast",
    "kumo_tabular": "Kumo-Tabular",
    "kumo_forecast": "KumoForecast",
    "chronos2_ctx4000": "Chronos-2",
    "t0_gamma": "t0-gamma",
    "hgb_quantile_q7": "HGB quantile",
    "hgb_quantile_recal_q7": "HGB quantile recalibrated",
}
CALIBRATION_ORDER = [
    "hgb_quantile_q7",
    "hgb_quantile_recal_q7",
    "tabicl_n8_full",
    "tabicl_recal_n8_full",
    "tabicl_forecast_ctx1024_n4",
    "kumo_tabular",
    "chronos2_ctx4000",
    "t0_gamma",
]
BLUE = "#1f77b4"
GREY = "#777777"
RED = "#d62728"
TEXT = "#333333"
TRUE_VS_PRED_MODELS = [
    "chronos2_ctx4000",
    "hgb_mean_fast",
    "hgb_quantile_recal_q7",
    "kumo_tabular",
    "tabicl_n8_full",
    "t0_gamma",
]
TRUE_VS_PRED_START = datetime.datetime(2025, 4, 1, tzinfo=datetime.timezone.utc)
TRUE_VS_PRED_END = datetime.datetime(2025, 4, 14, 23, 59, tzinfo=datetime.timezone.utc)


def save_both(fig, out):
    """Save the figure as SVG at ``out`` and as a matching PNG in ``out/png/``."""
    fig.savefig(out, format="svg", bbox_inches="tight")
    png_dir = out.parent / "png"
    png_dir.mkdir(parents=True, exist_ok=True)
    fig.savefig(png_dir / out.with_suffix(".png").name, bbox_inches="tight")


def fit_score_seconds(path):
    target = path.resolve()
    stem = target.stem.removesuffix("_cv_predictions")
    for score_path in dict.fromkeys(
        [target.parent / f"{stem}_scores.csv", *target.parent.glob("*_scores.csv")]
    ):
        if not score_path.exists():
            continue
        scores = pl.read_csv(score_path)
        if {"fit_time", "score_time"} <= set(scores.columns):
            fit = float(scores["fit_time"].sum())
            score = float(scores["score_time"].sum())
            return fit, score, fit + score
    return None, None, None


def load_records(directory):
    records = {}
    reference = None
    for name in MODEL_ORDER:
        path = directory / f"{name}_cv_predictions.parquet"
        if not path.exists():
            raise SystemExit(f"missing {path}")
        df = pl.read_parquet(path).sort("prediction_time")
        keys = df.select("prediction_time", "split")
        if reference is None:
            reference = keys
        elif not keys.equals(reference):
            raise SystemExit(f"{name} is not on the same rows as the others")
        fit, score, total = fit_score_seconds(path)
        records[name] = {
            "label": LABELS[name],
            "df": df,
            "fit": fit,
            "score": score,
            "total": total,
            "mape_rows": average_row_loss(df, "mape"),
            "pinball_rows": average_row_loss(df, "pinball"),
        }
    day, fold = row_index(records[MODEL_ORDER[0]]["df"])
    return records, day, fold


def difference_records(records, metric, baseline, x_field, day, fold):
    base_rows = records[baseline][f"{metric}_rows"]
    out = []
    for name in MODEL_ORDER:
        record = records[name]
        rows = record[f"{metric}_rows"]
        x = record[x_field]
        if rows is None or x is None:
            continue
        out.append(
            {
                "name": name,
                "label": record["label"],
                "x": x,
                "baseline": name == baseline,
                **paired_difference(rows, base_rows, day, fold),
            }
        )
    return out


def frontier(records):
    out, best = [], float("inf")
    for record in sorted(records, key=lambda r: (r["x"], r["mean"])):
        if record["mean"] < best:
            out.append(record)
            best = record["mean"]
    return out


def style_difference_axis(ax, data, y_label, title):
    for record in data:
        folds = np.asarray(record["folds"], dtype=float)
        xs = record["x"] * (1 + 0.01 * (np.arange(len(folds)) - 3.5))
        color = GREY if record["baseline"] else BLUE
        ax.scatter(xs, folds, s=9, color=color, alpha=0.4, zorder=2)
    for record in data:
        low = record["mean"] - record["low"]
        high = record["high"] - record["mean"]
        ax.errorbar(
            record["x"],
            record["mean"],
            yerr=[[low], [high]],
            fmt="none",
            ecolor=TEXT,
            elinewidth=1.1,
            capsize=3,
            zorder=3,
        )
    points = frontier(data)
    if len(points) > 1:
        ax.plot(
            [r["x"] for r in points],
            [r["mean"] for r in points],
            color=RED,
            ls="--",
            lw=1.2,
            marker="o",
            ms=4,
            zorder=4,
        )
    for record in data:
        marker = "D" if record["baseline"] else "o"
        color = GREY if record["baseline"] else BLUE
        ax.scatter(
            record["x"],
            record["mean"],
            marker=marker,
            s=65 if record["baseline"] else 42,
            color=color,
            edgecolor=TEXT,
            linewidth=0.6,
            zorder=5,
        )
    offsets = [(0, 8), (0, -14), (10, 0), (-10, 0)]
    order = sorted(range(len(data)), key=lambda i: (data[i]["x"], data[i]["mean"]))
    for rank, index in enumerate(order):
        record = data[index]
        label = record["label"] + (" (baseline)" if record["baseline"] else "")
        ax.annotate(
            label,
            (record["x"], record["mean"]),
            textcoords="offset points",
            xytext=offsets[rank % len(offsets)],
            fontsize=8,
            color=TEXT,
            zorder=6,
        )
    ax.set_xscale("log")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.18)


def plot_differences(records, day, fold, x_field, x_label, suptitle, out):
    specs = [
        (
            "mape",
            "hgb_mean",
            "Δ MAPE (%) vs HGB mean (lower is better)",
            "MAPE",
        ),
        (
            "pinball",
            "hgb_quantile_q7",
            "Δ pinball loss (MW) vs HGB quantile (lower is better)",
            "Pinball loss",
        ),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))
    for ax, (metric, baseline, y_label, title) in zip(axes, specs):
        data = difference_records(records, metric, baseline, x_field, day, fold)
        style_difference_axis(ax, data, y_label, title)
        ax.set_xlabel(x_label)
    handles = [
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=BLUE,
            markeredgecolor=TEXT,
            label="model",
        ),
        Line2D(
            [0],
            [0],
            marker="D",
            color="w",
            markerfacecolor=GREY,
            markeredgecolor=TEXT,
            label="baseline",
        ),
        Line2D([0], [0], color=RED, ls="--", label="Pareto frontier"),
        Line2D(
            [0],
            [0],
            marker="o",
            color="w",
            markerfacecolor=BLUE,
            alpha=0.4,
            label="per-fold difference",
        ),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False)
    fig.suptitle(suptitle, fontsize=14)
    fig.tight_layout(rect=[0, 0.05, 1, 0.96])
    save_both(fig, out)
    plt.close(fig)


def plot_calibration_coverage(records, n_bins, out):
    fig, axes = plt.subplots(1, 3, figsize=(18, 5.6), sharey=True)
    for ax, horizon in zip(axes, (1, 12, 24)):
        ax.axhline(0.9, color="gray", ls="--", lw=1)
        for index, name in enumerate(CALIBRATION_ORDER):
            record = records[name]
            df = record["df"]
            columns = dg.pred_columns(df)[horizon]
            y = df[f"{horizon}h"].to_numpy()
            xs, coverage = dg.interval_coverage_by_level(
                y,
                df[columns[0.05]].to_numpy(),
                df[columns[0.95]].to_numpy(),
                df[columns[0.5]].to_numpy(),
                n_bins,
            )
            ax.plot(
                xs,
                coverage,
                marker="o",
                ms=4,
                lw=1.3,
                color=model_colour(name),
                label=record["label"],
            )
        ax.set_title(f"{horizon}h horizon")
        ax.set_xlabel("predicted median (MW)")
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.2)
    axes[0].set_ylabel("observed coverage of the 0.05–0.95 interval")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(
        handles,
        labels,
        loc="lower center",
        ncol=len(labels),
        frameon=False,
        bbox_to_anchor=(0.5, 0.01),
    )
    fig.suptitle("Binned coverage of the 90% interval", fontsize=14)
    fig.tight_layout(rect=[0, 0.10, 1, 0.95])
    save_both(fig, out)
    plt.close(fig)


COVERAGE_WIDTH_LABELS = {
    "hgb_quantile_q7": "HGB q7",
    "hgb_quantile_recal_q7": "HGB q7 recal",
    "tabicl_n8_full": "TabICL",
    "tabicl_recal_n8_full": "TabICL recal",
    "tabicl_forecast_ctx1024_n4": "TabICL forecast",
    "kumo_tabular": "Kumo-Tabular",
    "chronos2_ctx4000": "Chronos-2",
    "t0_gamma": "t0-gamma",
}


def plot_coverage_width(records, out):
    """Mean 90% interval width vs marginal coverage, one subplot per horizon.

    One marker per quantile model, labelled next to the dot, with a vertical
    dashed line at the nominal 90% coverage. Models that meet the target and sit
    low are sharp and honest.
    """
    fig, axes = plt.subplots(3, 1, figsize=(10, 12), sharex=True)
    for index, name in enumerate(CALIBRATION_ORDER):
        record = records[name]
        df = record["df"]
        columns = dg.pred_columns(df)
        label = COVERAGE_WIDTH_LABELS[name]
        for ax, horizon in zip(axes, (1, 12, 24)):
            y = df[f"{horizon}h"].to_numpy()
            low = df[columns[horizon][0.05]].to_numpy()
            high = df[columns[horizon][0.95]].to_numpy()
            coverage = float(((y >= low) & (y <= high)).mean())
            width = float((high - low).mean())
            ax.scatter(
                coverage,
                width,
                color=model_colour(name),
                s=40,
                edgecolor=TEXT,
                linewidth=0.5,
            )
            ax.annotate(
                label,
                (coverage, width),
                textcoords="offset points",
                xytext=(6, 3),
                fontsize=8,
                color=TEXT,
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "none"},
            )
    for ax, horizon in zip(axes, (1, 12, 24)):
        ax.axvline(0.9, color="gray", ls="--", lw=1)
        ax.set_title(f"{horizon}h horizon")
        ax.set_ylabel("mean 90% interval width (MW)")
        ax.grid(True, alpha=0.2)
    axes[-1].set_xlabel("marginal coverage of the 0.05–0.95 interval")
    fig.suptitle("Interval sharpness vs coverage (dashed: 90% target)", fontsize=14)
    fig.tight_layout(rect=[0, 0.02, 1, 0.97])
    save_both(fig, out)
    plt.close(fig)


def whole_dataset_metrics(record):
    df = record["df"]
    y = df["24h"].to_numpy()
    has_quantiles = "pred_24h__q_0.5" in df.columns
    prediction = df["pred_24h__q_0.5" if has_quantiles else "pred_24h"].to_numpy()
    metrics = {
        "mape": 100 * mean_absolute_percentage_error(y, prediction),
        "pinball": None,
        "coverage": None,
        "width": None,
    }
    if has_quantiles:
        metrics["pinball"] = float(
            np.mean(
                [
                    mean_pinball_loss(y, df[f"pred_24h__q_{q}"].to_numpy(), alpha=q)
                    for q in (0.05, 0.5, 0.95)
                ]
            )
        )
        low = df["pred_24h__q_0.05"].to_numpy()
        high = df["pred_24h__q_0.95"].to_numpy()
        metrics["coverage"] = float(((y >= low) & (y <= high)).mean())
        metrics["width"] = float((high - low).mean())
    return metrics


def true_vs_pred_series(record):
    df = record["df"].with_columns(
        (pl.col("prediction_time") + pl.duration(hours=24)).alias("target_time")
    )
    sub = df.filter(
        (pl.col("target_time") >= TRUE_VS_PRED_START)
        & (pl.col("target_time") <= TRUE_VS_PRED_END)
    )
    times = sub["target_time"].to_list()
    y = sub["24h"].to_numpy()
    has_quantiles = "pred_24h__q_0.5" in sub.columns
    prediction = sub["pred_24h__q_0.5" if has_quantiles else "pred_24h"].to_numpy()
    low = sub["pred_24h__q_0.05"].to_numpy() if has_quantiles else None
    high = sub["pred_24h__q_0.95"].to_numpy() if has_quantiles else None
    return {
        "times": times,
        "y": y,
        "prediction": prediction,
        "low": low,
        "high": high,
        **whole_dataset_metrics(record),
    }


def true_vs_pred_title(item):
    if item["pinball"] is None:
        text = f"{item['label']} — MAPE {item['mape']:.2f}% (over the full dataset)"
        return HPacker(
            children=[TextArea(text, textprops=dict(fontsize=10, color=TEXT))],
            align="baseline",
            pad=0,
            sep=0,
        )
    return HPacker(
        children=[
            TextArea(
                f"{item['label']} — MAPE {item['mape']:.2f}% | "
                f"pinball {item['pinball']:.1f} MW | ",
                textprops=dict(fontsize=10, color=TEXT),
            ),
            TextArea(
                f"coverage {100 * item['coverage']:.2f}%",
                textprops=dict(fontsize=10, color=TEXT),
            ),
            TextArea(
                f" | width {item['width']:,.0f} MW",
                textprops=dict(fontsize=10, color=TEXT),
            ),
            TextArea(
                " (over the full dataset)",
                textprops=dict(fontsize=10, color=TEXT),
            ),
        ],
        align="baseline",
        pad=0,
        sep=0,
    )


def plot_true_vs_pred(records, out):
    series = []
    for name in TRUE_VS_PRED_MODELS:
        item = {
            "name": name,
            "label": LABELS[name],
            **true_vs_pred_series(records[name]),
        }
        series.append(item)
    series.sort(key=lambda item: item["mape"])
    n = len(series)
    fig, axes = plt.subplots(
        n, 1, figsize=(16, 2.6 * n + 1.8), sharex=True, sharey=True
    )
    axes = np.atleast_1d(axes)
    values = np.concatenate(
        [item["y"] for item in series]
        + [item["prediction"] for item in series]
        + [
            array
            for item in series
            for array in (item["low"], item["high"])
            if array is not None
        ]
    )
    y_min, y_max = float(values.min()), float(values.max())
    padding = 0.03 * (y_max - y_min)
    for ax, item in zip(axes, series):
        ax.plot(item["times"], item["y"], color="gray", ls="--", lw=1.1)
        ax.plot(item["times"], item["prediction"], color=BLUE, lw=1.2)
        if item["low"] is not None:
            ax.fill_between(
                item["times"],
                item["low"],
                item["high"],
                color=BLUE,
                alpha=0.18,
                linewidth=0,
            )
        ax.add_artist(
            AnnotationBbox(
                true_vs_pred_title(item),
                (0, 1),
                xycoords="axes fraction",
                xybox=(0, 6),
                boxcoords="offset points",
                box_alignment=(0, 0),
                frameon=False,
                annotation_clip=False,
            )
        )
        ax.set_ylabel("load (MW)")
        ax.set_ylim(y_min - padding, y_max + padding)
        ax.grid(True, alpha=0.2)
    locator = mdates.DayLocator(interval=2)
    axes[-1].xaxis.set_major_locator(locator)
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[-1].set_xlim(series[0]["times"][0], series[0]["times"][-1])
    handles = [
        Line2D([0], [0], color="gray", ls="--", lw=1.2, label="true load"),
        Line2D([0], [0], color=BLUE, lw=1.5, label="prediction (median)"),
        Patch(facecolor=BLUE, alpha=0.18, label="5–95% interval"),
    ]
    fig.legend(
        handles=handles,
        loc="upper center",
        ncol=3,
        frameon=False,
        bbox_to_anchor=(0.5, 0.98),
    )
    fig.suptitle(
        "Example predictions (first 2 weeks of April 2025)",
        fontsize=14,
        y=1.00,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.95], h_pad=2.4)
    save_both(fig, out)
    plt.close(fig)
    print(
        "true vs pred order:",
        ", ".join(f"{item['label']} {item['mape']:.2f}%" for item in series),
    )


def write_provenance(args, output_dir):
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
        dirty = subprocess.check_output(
            ["git", "status", "--porcelain"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        commit, dirty = "unknown", ""
    command = " ".join(sys.argv)
    text = f"""# Summary figures (SVG)

Generated by `scripts/model_comparison/build_summary_figures.py`.

- command: `python {command}` (run from the repository root)
- date: {datetime.datetime.now().isoformat(timespec="seconds")}
- git commit: `{commit}` ({"clean" if not dirty else "dirty (see git status)"})
- input directory: `{args.dir.resolve()}`
- weather: safe-lead forecast (this directory), oracle not used
- models: {", ".join(LABELS[name] for name in MODEL_ORDER)}
- calibration models: {", ".join(LABELS[name] for name in CALIBRATION_ORDER)}
- files: one SVG plus one PNG per figure (PNGs under `png/`):
  `diff_fit_predict`, `diff_predict_only`, `calibration_binned_coverage`,
  `coverage_width`, `true_vs_pred_24h_2025-04-01_2025-04-14`
- coverage vs width: mean 90% interval width (MW) vs marginal coverage per
  horizon (1h/12h/24h), one marker per calibration model, dashed line at the
  nominal 90% coverage
- true vs predicted: 24h horizon, 2025-04-01 to 2025-04-14 (first two weeks of
  April 2025), models
  {", ".join(LABELS[name] for name in TRUE_VS_PRED_MODELS)}, ordered by the
  whole-dataset 24h MAPE; quantile models show MAPE, pinball, coverage and
  mean interval width, point-only models show MAPE only. All title metrics are
  computed over the whole CV test set at 24h
- metrics: MAPE and pinball (mean over q0.05/q0.5/q0.95 and horizons), paired
  differences vs HGB mean / HGB quantile with 95% moving-block bootstrap
  intervals; coverage of the 0.05-0.95 interval in {args.n_bins} equal-count
  bins by predicted median
- no model was re-run; derived from `*_cv_predictions.parquet`
"""
    (output_dir / "figures.md").write_text(text, "utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dir",
        type=Path,
        default=Path("results/model_comparison/cv_comparison_forecast"),
    )
    parser.add_argument(
        "--output_dir",
        type=Path,
        default=Path("results/model_comparison/summary/figures"),
    )
    parser.add_argument("--n_bins", type=int, default=10)
    args = parser.parse_args()

    records, day, fold = load_records(args.dir)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    plot_differences(
        records,
        day,
        fold,
        "total",
        "Fit + predict time (s, log scale)\n(sum over all folds)",
        "Paired differences vs baseline — fit + predict time",
        args.output_dir / "diff_fit_predict.svg",
    )
    plot_differences(
        records,
        day,
        fold,
        "score",
        "Predict time only (s, log scale)\n(sum over all folds)",
        "Paired differences vs baseline — predict-only time",
        args.output_dir / "diff_predict_only.svg",
    )
    plot_calibration_coverage(
        records, args.n_bins, args.output_dir / "calibration_binned_coverage.svg"
    )
    plot_coverage_width(records, args.output_dir / "coverage_width.svg")
    plot_true_vs_pred(
        records,
        args.output_dir / "true_vs_pred_24h_2025-04-01_2025-04-14.svg",
    )
    write_provenance(args, args.output_dir)
    print(f"wrote 5 SVG files to {args.output_dir}")


if __name__ == "__main__":
    main()
