"""
Overlay true vs predicted load for hgb_mean, tabicl and chronos2 on the same
plot, from the `*_cv_predictions.parquet` files written by
`cross_validate_compare.py` (no re-running).

Reuses `elf.plot_predictions`, which already draws one actual-value trace plus
one line per column matching `pred_{h}h*` -- so the three models' predictions
just need to be joined into a single dataframe with distinctly-named
prediction columns. Horizon/quantile detection is shared with
`compare_predictions.py` (`horizons_of`, `quantile_levels_of`) rather than
duplicated here.

Run from the repository root:

    python chronos_comparison_scripts/plot_cv_comparison.py results/cv_compare_remote

Writes one HTML file per horizon (1h, 12h, 24h) into `<dir>/plots/`. See also
`build_cv_report.py` for a report with separate (non-overlaid) plots per
model, shaded confidence intervals, and a metrics table.
"""

import argparse
import sys
from pathlib import Path

import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import electricity_load_forecasting as elf

from compare_predictions import horizons_of, quantile_levels_of

parser = argparse.ArgumentParser()
parser.add_argument("dir", type=Path)
args = parser.parse_args()

paths = sorted(args.dir.glob("*_cv_predictions.parquet"))
if not paths:
    raise SystemExit(f"no *_cv_predictions.parquet found in {args.dir}")
frames = {p.stem.removesuffix("_cv_predictions"): pl.read_parquet(p) for p in paths}

horizons = sorted({h for df in frames.values() for h in horizons_of(df)})

# actual values should be identical across models (same splits); take them
# from whichever frame, keyed on prediction_time
first_name, first_df = next(iter(frames.items()))
combined = first_df.select(["prediction_time"] + [f"{h}h" for h in horizons])

for name, df in frames.items():
    pred_cols = {}
    for h in horizons:
        levels = quantile_levels_of(df, h)
        median = min(levels, key=lambda q: abs(q - 0.5)) if levels else None
        col = f"pred_{h}h__q_{median}" if median is not None else f"pred_{h}h"
        pred_cols[f"pred_{h}h_{name}"] = col
    renamed = df.select(
        "prediction_time", *(pl.col(v).alias(k) for k, v in pred_cols.items())
    )
    combined = combined.join(renamed, on="prediction_time", how="inner")

assert combined.height == first_df.height, "predictions could not be aligned by prediction_time"

output_dir = args.dir / "plots"
output_dir.mkdir(exist_ok=True)
for h in horizons:
    fig = elf.plot_predictions(combined, horizons=(h,), start=None)
    fig.update_layout(title=f"{h}h horizon: true vs predicted ({', '.join(frames)})")
    out = output_dir / f"{h}h_comparison.html"
    fig.write_html(out)
    print("wrote", out)
