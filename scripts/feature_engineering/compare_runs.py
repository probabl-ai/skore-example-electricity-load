"""
Paired comparison of runs with new features against earlier runs of the same
models (default features).

    python scripts/feature_engineering/compare_runs.py NEW.parquet=OLD.parquet ...

Prints MAE, MAPE and pinball (mean over the 5% and 95% levels, if available)
differences NEW - OLD
(negative is better) with the 95% moving-block bootstrap interval of
`uncertainty.py`, averaged over horizons and per horizon; `*` if it excludes 0.
"""

import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).parent.parent / "model_comparison"))
from uncertainty import (  # noqa: E402
    average_row_loss,
    paired_difference,
    row_index,
    row_loss,
)

HORIZONS = (1, 12, 24)


def read(path):
    return pl.read_parquet(path).sort("prediction_time", "split")


def cell(new, old, day, fold, digits):
    s = paired_difference(new, old, day, fold)
    star = "*" if s["low"] > 0 or s["high"] < 0 else ""
    return (
        f"{old.mean():.{digits}f} -> {new.mean():.{digits}f} "
        f"({s['mean']:+.{digits}f} [{s['low']:+.{digits}f}, {s['high']:+.{digits}f}]){star}"
    )


print("| model | metric | average | 1h | 12h | 24h | folds better |")
print("|---|---|---|---|---|---|---:|")
for arg in sys.argv[1:]:
    new_path, old_path = arg.split("=")
    new, old = read(new_path), read(old_path)
    assert (new["prediction_time"] == old["prediction_time"]).all()
    day, fold = row_index(new)
    name = Path(new_path).parent.name
    for kind, digits in (("mae", 1), ("mape", 3), ("pinball", 1)):
        if kind == "pinball":
            a, b = average_row_loss(new, "pinball"), average_row_loss(old, "pinball")
            if a is None or b is None:
                continue
            cols = [(a, b)] + [
                tuple(
                    np.mean(
                        [row_loss(d, "pinball", h, q) for q in (0.05, 0.95)],
                        axis=0,
                    )
                    for d in (new, old)
                )
                for h in HORIZONS
            ]
        else:
            cols = [
                (
                    np.mean([row_loss(new, kind, h) for h in HORIZONS], axis=0),
                    np.mean([row_loss(old, kind, h) for h in HORIZONS], axis=0),
                )
            ] + [(row_loss(new, kind, h), row_loss(old, kind, h)) for h in HORIZONS]
        n_better = paired_difference(*cols[0], day, fold)["n_lower"]
        print(
            f"| {name} | {kind} | "
            + " | ".join(cell(n, o, day, fold, digits) for n, o in cols)
            + f" | {n_better}/{len(np.unique(fold))} |"
        )
