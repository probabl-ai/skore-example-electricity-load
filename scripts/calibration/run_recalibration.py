"""Run the raw vs recalibrated walk-forward CV.

This writes the prediction, score, correction and inner out-of-fold files under
``results/calibration/recalibration_<timestamp>/``; the HTML report is built by
``recalibration_report.py`` (or by passing ``--report_only``).
"""

import argparse
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from electricity_load_forecasting import pipeline as elf
from recalibration_report import QUANTILES, build_report, read_config


# ---------------------------------------------------------------------------
# Running the CV
# ---------------------------------------------------------------------------
def run_cv(recalibrate, horizons, quantiles, max_iter, env):
    """Walk-forward CV like ``elf.cross_val_predict``, keeping the corrections."""
    data_op = elf.make_data_op(
        horizons=horizons,
        quantile_strategy="multiple_regressors",
        quantiles=quantiles,
        max_iter=max_iter,
        recalibrate=recalibrate,
    )
    predictions, scores, corrections, oofs, n_iters = [], [], [], [], []
    for i, split in enumerate(data_op.skb.iter_cv_splits(environment=env)):
        learner = data_op.skb.make_learner().fit(split["train"])
        score, prediction = learner.score(split["test"], return_predictions=True)
        test_start = split["X_test"]["prediction_time"].min()
        print(f"[{'recalibrated' if recalibrate else 'raw'}] split {i}: {test_start}")
        predictions.append(
            elf.concat_X_y_predictions(
                split["X_test"], split["y_test"], prediction["predict"]
            ).with_columns(split=pl.lit(i))
        )
        scores.append(score | {"split": i, "test_start": test_start})
        for h in horizons:
            fitted = learner.find_fitted_estimator(f"pred_{h}h")
            hgb_q = fitted.estimator_ if recalibrate else fitted
            n_iters += [
                {"split": i, "horizon": h, "quantile": q, "n_iter": int(e.n_iter_)}
                for q, e in hgb_q.estimators_.items()
            ]
            if recalibrate:
                corrections.append(corrections_of(fitted, i, h))
                train_times = split["X_train"]["prediction_time"]
                oofs.append(
                    fitted.oof_.with_columns(
                        prediction_time=train_times[fitted.oof_["row"].to_numpy()],
                        split=pl.lit(i),
                        horizon=pl.lit(h),
                    )
                )
    extra = {
        "corrections": pl.concat(corrections) if corrections else None,
        "inner_oof": pl.concat(oofs) if oofs else None,
        "n_iter": pl.DataFrame(n_iters),
    }
    return pl.concat(predictions), pl.DataFrame(scores), extra


def corrections_of(recalibrator, split, horizon):
    """Bins (measured shifts) and fitted curve of a fitted Recalibrator."""
    frames = []
    for column, summary in recalibrator.bin_summary_.items():
        x = summary["mean_prediction"].to_numpy()
        grid = np.linspace(x.min(), x.max(), 50)
        curve = recalibrator.shift_models_[column].predict(grid[:, None])
        bins = pl.DataFrame(
            {
                "kind": "bin",
                "x": x,
                "y": summary["shift"].to_numpy(),
                "count": summary["count"].to_numpy(),
                "empirical_coverage": summary["empirical_coverage"].to_numpy(),
            }
        )
        curve = pl.DataFrame(
            {
                "kind": "curve",
                "x": grid,
                "y": curve,
                "count": pl.Series([None] * len(grid), dtype=pl.Int64),
                "empirical_coverage": pl.Series([None] * len(grid), dtype=pl.Float64),
            }
        )
        frames += [
            part.with_columns(
                column=pl.lit(column), split=pl.lit(split), horizon=pl.lit(horizon)
            )
            for part in (bins, curve)
        ]
    return pl.concat(frames)



def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_iter", default=1000, type=int)
    parser.add_argument(
        "--reuse_raw",
        type=Path,
        help="copy the 'raw' (no recalibration) results from this existing "
        "output directory instead of recomputing them (max_iter, horizons and "
        "quantiles must match; checked)",
    )
    parser.add_argument("--horizons", default=(1, 12, 24), nargs="+", type=int)
    parser.add_argument("--quantiles", default=QUANTILES, nargs="+", type=float)
    parser.add_argument(
        "--report_only",
        type=Path,
        help="only rebuild the report from this existing output directory",
    )
    args = parser.parse_args()
    if args.report_only is not None:
        build_report(args.report_only)
        return

    horizons, quantiles = tuple(args.horizons), tuple(args.quantiles)
    config = dict(
        max_iter=args.max_iter,
        # default of calibration.Recalibrator, used by make_data_op
        n_last_folds=4,
        horizons=list(horizons),
        quantiles=list(quantiles),
    )
    if args.reuse_raw is not None:
        # the raw results must come from the same settings
        reused = read_config(args.reuse_raw)
        for key in ("max_iter", "horizons", "quantiles"):
            if reused.get(key) != config[key]:
                raise SystemExit(
                    f"--reuse_raw: {key} differs from the reused run "
                    f"({reused.get(key)!r} there, {config[key]!r} here)"
                )
    output_dir = elf.get_output_dir("calibration/recalibration_")
    print("output directory:", output_dir)
    env = elf.get_env()
    (output_dir / "config.json").write_text(json.dumps(config, indent=2))
    for recalibrate in (False, True):
        name = "recalibrated" if recalibrate else "raw"
        if not recalibrate and args.reuse_raw is not None:
            for suffix in ("cv_predictions.parquet", "scores.csv", "n_iter.csv"):
                if (args.reuse_raw / f"raw_{suffix}").exists():  # n_iter: newer runs
                    shutil.copy(
                        args.reuse_raw / f"raw_{suffix}", output_dir / f"raw_{suffix}"
                    )
            continue
        predictions, scores, extra = run_cv(
            recalibrate, horizons, quantiles, args.max_iter, env
        )
        predictions.write_parquet(output_dir / f"{name}_cv_predictions.parquet")
        scores.write_csv(output_dir / f"{name}_scores.csv")
        extra["n_iter"].write_csv(output_dir / f"{name}_n_iter.csv")
        if extra["corrections"] is not None:
            extra["corrections"].write_parquet(output_dir / "corrections.parquet")
            extra["inner_oof"].write_parquet(output_dir / "inner_oof.parquet")
    build_report(output_dir)


if __name__ == "__main__":
    main()
