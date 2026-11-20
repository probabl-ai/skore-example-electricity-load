"""Regenerate the skrub report for the current pipeline.

The output path is stable and depends on the pipeline options, so the report can
be published or shared again after a pipeline change. A README.md with the
provenance (command, code, packages, data, features, model) is written next to
the report.
"""

import argparse
import datetime
import importlib.metadata as md
import inspect
import json
import platform
import subprocess
import sys
from pathlib import Path

import electricity_load_forecasting as elf

ROOT = Path(__file__).resolve().parents[2]

parser = argparse.ArgumentParser()
parser.add_argument(
    "--model",
    default="hgb_quantile",
    choices=["hgb_mean", "hgb_quantile", "tabicl", "binning"],
)
parser.add_argument("--recalibrate", action="store_true")
parser.add_argument(
    "--feature_groups", nargs="*", choices=elf.FEATURE_GROUPS, default=[]
)
args = parser.parse_args()

name = args.model
if args.recalibrate:
    name += "_recalibrated"
if set(args.feature_groups) == set(elf.FEATURE_GROUPS):
    name += "_all_groups"
elif args.feature_groups:
    name += "_fg-" + "-".join(args.feature_groups)
output_dir = ROOT / "results" / "inspection" / f"skrub_report_{name}"

pred = elf.make_data_op(
    model=args.model,
    recalibrate=args.recalibrate,
    feature_groups=tuple(args.feature_groups),
)
env = elf.get_env()
pred.skb.report(environment=env, output_dir=output_dir, overwrite=True)


def git(*cmd):
    return subprocess.run(
        ["git", *cmd], cwd=ROOT, capture_output=True, text=True
    ).stdout.strip()


def skrub_source():
    url = json.loads(md.distribution("skrub").read_text("direct_url.json") or "{}")
    return (
        url.get("url", "PyPI release")
        + "@"
        + url.get("vcs_info", {}).get("commit_id", "")
    )


make_data_op_defaults = {
    k: v.default
    for k, v in inspect.signature(elf.make_data_op).parameters.items()
    if v.default is not inspect.Parameter.empty
}
resolved = make_data_op_defaults | {
    "model": args.model,
    "recalibrate": args.recalibrate,
    "feature_groups": args.feature_groups,
}
(output_dir / "README.md").write_text(f"""# skrub report `{output_dir.name}`

Built without any CV: only the DataOp graph and its previews on `{env["start"]}`
to `{env["end"]}`.

- Command: `{" ".join(["python", *sys.argv])}` (from the repository root)
- Date: {datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
- Git commit: {git("rev-parse", "HEAD")}; modified files (untracked excluded):
  {git("status", "--porcelain", "--untracked-files=no").replace(chr(10), ", ") or "none (clean)"}
- Host: {platform.node()}, {platform.processor() or platform.machine()}
- Packages: python {platform.python_version()}, polars {md.version("polars")},
  scikit-learn {md.version("scikit-learn")}, skrub {md.version("skrub")}
  (installed from {skrub_source()})
- Weather: default `{elf.fetch_city_weather_forecast.__name__}` (historical
  forecasts, day1 lead for horizons <= 18h, day2 above); load from
  `{elf.fetch_load_mw_history.__name__}`; public holidays from `holidays.country_holidays("FR")`
- Evaluation range (`get_env()`): {env}
- Feature groups: {args.feature_groups or "none (default features)"}
- `make_data_op` arguments as resolved: {resolved}
- Model defaults as resolved by `describe_defaults()`:
  {json.dumps({str(k): str(v) for k, v in pred.skb.describe_defaults().items()})}
""")
print(f"wrote {output_dir}")
