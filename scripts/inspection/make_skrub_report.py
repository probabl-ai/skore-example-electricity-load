"""Regenerate the skrub report for the current pipeline.

The output path is stable and depends on the pipeline options, so the report can
be published or shared again after a pipeline change.
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
import electricity_load_forecasting as elf

parser = argparse.ArgumentParser()
parser.add_argument(
    "--quantile_strategy",
    default="multiple_regressors",
    choices=["hgb_mean", "multiple_regressors", "tabicl", "binning"],
)
parser.add_argument("--recalibrate", action="store_true")
args = parser.parse_args()

quantile_strategy = None if args.quantile_strategy == "hgb_mean" else args.quantile_strategy
name = args.quantile_strategy
if args.recalibrate:
    name += "_recalibrated"
output_dir = ROOT / "results" / "inspection" / f"skrub_report_{name}"

pred = elf.make_data_op(
    quantile_strategy=quantile_strategy, recalibrate=args.recalibrate
)
pred.skb.report(environment=elf.get_env(), output_dir=output_dir)
print(f"wrote {output_dir}")
