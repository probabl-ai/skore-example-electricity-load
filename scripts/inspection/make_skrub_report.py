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
    choices=["hgb_mean", "hgb_quantile", "tabicl", "binning", "t0_gamma"],
)
parser.add_argument("--recalibrate", action="store_true")
parser.add_argument(
    "--feature_groups", nargs="*", choices=elf.FEATURE_GROUPS, default=[]
)
parser.add_argument(
    "--start",
    help="first prediction time (default: start of elf.get_env()); a short range "
    "keeps the previews of a zero-shot model like t0_gamma fast",
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

if args.model == "t0_gamma":
    # same settings as scripts/model_comparison/cross_validate_models.py
    from electricity_load_forecasting import t0 as t0mod

    make_op = t0mod.make_t0_data_op
    op_kwargs = dict(
        horizons=(1, 12, 24),
        forecaster=t0mod.T0Forecaster(device="cuda", dtype="bfloat16"),
        quantiles=(0.05, 0.5, 0.95),
        context_hours=1024,
        chunk_size=64,
    )
else:
    make_op = elf.make_data_op
    op_kwargs = dict(
        model=args.model,
        recalibrate=args.recalibrate,
        feature_groups=tuple(args.feature_groups),
    )
pred = make_op(**op_kwargs)
env = elf.get_env() | ({"start": args.start} if args.start else {})
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


op_defaults = {
    k: v.default
    for k, v in inspect.signature(make_op).parameters.items()
    if v.default is not inspect.Parameter.empty
}
resolved = op_defaults | {
    k: str(v) if k == "forecaster" else v for k, v in op_kwargs.items()
}
if args.model == "t0_gamma":
    weather = (
        f"default `{elf.fetch_city_weather_forecast.__name__}`; the window given to "
        f"the model gets day1 (24h lead) for the past and future steps <= "
        f"{elf.DAY1_MAX_HORIZON}, day2 (48h lead) for later steps (one table each, "
        "`covariates` and `covariates_long`)"
    )
    resolved["forecaster_params"] = op_kwargs["forecaster"].get_params()
else:
    weather = (
        f"default `{elf.fetch_city_weather_forecast.__name__}` (historical "
        "forecasts, day1 lead for horizons <= 18h, day2 above)"
    )

feature_groups_line = (
    ""
    if args.model == "t0_gamma"
    else f"- Feature groups: {args.feature_groups or 'none (default features)'}\n"
)


def version(package):
    try:
        return md.version(package)
    except md.PackageNotFoundError:
        return None


(output_dir / "README.md").write_text(f"""# skrub report `{output_dir.name}`

Built without any CV: only the DataOp graph and its previews on `{env["start"]}`
to `{env["end"]}`{" (shortened with --start)" if args.start else ""}.

- Command: `{" ".join(["python", *sys.argv])}` (from the repository root)
- Date: {datetime.datetime.now().astimezone().isoformat(timespec="seconds")}
- Git commit: {git("rev-parse", "HEAD")}; modified files (untracked excluded):
  {git("status", "--porcelain", "--untracked-files=no").replace(chr(10), ", ") or "none (clean)"}
- Host: {platform.node()}, {platform.processor() or platform.machine()}
- Packages: python {platform.python_version()}, polars {md.version("polars")},
  scikit-learn {md.version("scikit-learn")}, skrub {md.version("skrub")}
  (installed from {skrub_source()}), torch {version("torch")}, tfc-t0 {version("tfc-t0")}
- Weather: {weather}; load from
  `{elf.fetch_load_mw_history.__name__}`; public holidays from `holidays.country_holidays("FR")`
- Evaluation range (`get_env()`, `start` overridden if shortened): {env}
{feature_groups_line}- `{make_op.__name__}` arguments as resolved: {resolved}
- Model defaults as resolved by `describe_defaults()`:
  {json.dumps({str(k): str(v) for k, v in pred.skb.describe_defaults().items()})}
""")
print(f"wrote {output_dir}")
