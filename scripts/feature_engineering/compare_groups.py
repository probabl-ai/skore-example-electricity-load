"""
Paired comparison of runs with different `--feature_groups` against a baseline.

    python scripts/feature_engineering/compare_groups.py DIR [--baseline TAG]

DIR holds the `<model>_<tag>_cv_predictions.parquet` files written by
`cross_validate_models.py --tag <tag>`. Writes `DIR/comparison.md`: MAE and MAPE
differences versus the baseline (per horizon and averaged), with the 95%
moving-block bootstrap interval of `uncertainty.py` and the number of folds
where the run beats the baseline.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent / "model_comparison"))
from uncertainty import paired_difference, row_index, row_loss  # noqa: E402

parser = argparse.ArgumentParser()
parser.add_argument("dir", type=Path)
parser.add_argument("--baseline", default="baseline")
args = parser.parse_args()

frames = {
    p.name.removesuffix("_cv_predictions.parquet").split("_", 2)[-1]
    if p.name.startswith("hgb_mean_")
    else p.name: pl.read_parquet(p).sort("prediction_time", "split")
    for p in args.dir.glob("*_cv_predictions.parquet")
}
base = frames[args.baseline]
day, fold = row_index(base)
horizons = (1, 12, 24)


def losses(df, kind):
    return {h: row_loss(df, kind, h) for h in horizons} | {
        "avg": np.mean([row_loss(df, kind, h) for h in horizons], axis=0)
    }


lines = [f"# Feature groups vs `{args.baseline}` (paired, pooled over folds)\n"]
for kind, unit in (("mae", "MW"), ("mape", "% points")):
    base_loss = losses(base, kind)
    lines += [
        f"\n## {kind.upper()} ({unit}); difference vs baseline, negative is better\n",
        "Each cell: difference [95% block-bootstrap interval]; `*` if the interval excludes 0.\n",
        f"| run | avg {kind} | diff avg | diff 1h | diff 12h | diff 24h | folds better (avg) |",
        "|---|---:|---|---|---|---|---:|",
    ]
    fmt = "{:+.1f}" if kind == "mae" else "{:+.3f}"
    for name in sorted(frames, key=lambda n: (n != args.baseline, n)):
        loss = losses(frames[name], kind)
        cells = []
        for key in ("avg", 1, 12, 24):
            if name == args.baseline:
                cells.append("")
                continue
            s = paired_difference(loss[key], base_loss[key], day, fold)
            star = "*" if s["low"] > 0 or s["high"] < 0 else ""
            cells.append(
                f"{fmt.format(s['mean'])} [{fmt.format(s['low'])}, "
                f"{fmt.format(s['high'])}]{star}"
            )
            if key == "avg":
                n_better = f"{s['n_lower']}/{len(s['folds'])}"
        mean = loss["avg"].mean()
        lines.append(
            f"| {name} | {mean:.1f} | " + " | ".join(cells) + f" | "
            f"{'' if name == args.baseline else n_better} |"
            if kind == "mae"
            else f"| {name} | {mean:.3f} | "
            + " | ".join(cells)
            + f" | {'' if name == args.baseline else n_better} |"
        )
(args.dir / "comparison.md").write_text("\n".join(lines) + "\n")
print("\n".join(lines))
