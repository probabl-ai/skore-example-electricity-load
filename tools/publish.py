import argparse
import html
import re
import shutil
from datetime import datetime
from pathlib import Path
from urllib.parse import quote

DOCS = Path(__file__).resolve().parent.parent / "docs"
SKIP_HTML = re.compile(r"^node_.*\.html$")
SECTION_TITLES = {
    "calibration": "Calibration",
    "model-comparison": "Model comparison",
    "tuning": "Hyperparameter tuning",
    "inspection": "Inspection and diagnostics",
    "checks": "Checks",
    "misc": "Miscellaneous",
    "reports": "Other reports",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Copy HTML reports into docs/ and rebuild docs/index.html "
        "for GitHub Pages (main branch, /docs folder)."
    )
    parser.add_argument(
        "sources",
        nargs="*",
        type=Path,
        help="HTML files or result directories to publish. A file named "
        "index.html is published as docs/<parent-directory>/index.html. With no "
        "source, only the index is regenerated.",
    )
    parser.add_argument(
        "--name",
        help="name in docs/ for the single given source, e.g. calibration-v2",
    )
    parser.add_argument(
        "--theme",
        help="subdirectory in docs/ for this report, e.g. calibration or "
        "model-comparison",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="when a source is a directory, copy every file instead of only HTML",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="replace a destination that already exists",
    )
    parser.add_argument(
        "--no-index",
        action="store_true",
        help="copy the files without regenerating docs/index.html",
    )
    return parser.parse_args()


def check_name(name):
    path = Path(name)
    if path.is_absolute() or ".." in path.parts or not path.parts:
        raise SystemExit(f"invalid --name: {name!r}")
    return path


def destination_for(source, name, theme=None):
    base = DOCS / check_name(theme) if theme else DOCS
    if name:
        return base / check_name(name)
    if source.is_file() and source.name == "index.html":
        parent = source.parent.name
        if not parent:
            raise SystemExit(f"cannot derive a name from {source}; use --name")
        return base / parent / "index.html"
    return base / source.name


def copy_source(source, destination, copy_all):
    if source.is_dir():
        if destination.exists():
            if destination.is_dir():
                shutil.rmtree(destination)
            else:
                destination.unlink()
        if copy_all:
            shutil.copytree(source, destination)
        else:
            htmls = list(source.rglob("*.html"))
            if not htmls:
                raise SystemExit(
                    f"no HTML files in {source}; use --all to copy everything"
                )
            for path in htmls:
                relative = path.relative_to(source)
                target = destination / relative
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return destination


def html_title(path):
    with path.open("r", encoding="utf-8", errors="replace") as f:
        chunk = f.read(65536)
    match = re.search(r"<title[^>]*>(.*?)</title>", chunk, re.IGNORECASE | re.DOTALL)
    if not match:
        return None
    title = re.sub(r"\s+", " ", html.unescape(match.group(1))).strip()
    return title or None


def human_size(size):
    for unit in ("B", "KiB", "MiB", "GiB"):
        if size < 1024 or unit == "GiB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.1f} {unit}"
        size /= 1024


def report_files():
    def walk(directory):
        index = directory / "index.html"
        if index.exists():
            return [index]
        files = [
            path for path in directory.glob("*.html") if not SKIP_HTML.match(path.name)
        ]
        if len(files) == 1:
            return files
        for child in directory.iterdir():
            if child.is_dir() and not child.name.startswith("."):
                files.extend(walk(child))
        return files

    files = []
    for path in DOCS.iterdir():
        if path.name.startswith("."):
            continue
        if path.is_file():
            if (
                path.suffix == ".html"
                and path.name != "index.html"
                and not SKIP_HTML.match(path.name)
            ):
                files.append(path)
        elif path.is_dir():
            files.extend(walk(path))
    return sorted(files, key=lambda p: (-p.stat().st_mtime, str(p)))


def build_index():
    groups = {}
    for path in report_files():
        relative = path.relative_to(DOCS)
        key = relative.parts[0] if len(relative.parts) > 1 else "reports"
        groups.setdefault(key, []).append(path)

    order = [key for key in SECTION_TITLES if key in groups]
    order += sorted(key for key in groups if key not in SECTION_TITLES)
    sections = []
    for key in order:
        rows = []
        for path in groups[key]:
            relative = path.relative_to(DOCS)
            title = html_title(path) or (
                str(relative.parent) if path.name == "index.html" else path.stem
            )
            stat = path.stat()
            rows.append(
                "<tr>"
                f'<td><a href="{quote(relative.as_posix())}" target="_blank" rel="noopener">{html.escape(title)}</a>'
                f'<div class="path">{html.escape(relative.as_posix())}</div></td>'
                f"<td>{datetime.fromtimestamp(stat.st_mtime):%Y-%m-%d %H:%M}</td>"
                f"<td>{human_size(stat.st_size)}</td>"
                "</tr>"
            )
        title = SECTION_TITLES.get(key, key.replace("-", " ").title())
        sections.append(
            f"<h2>{html.escape(title)}</h2>\n"
            "<table>"
            "<thead><tr><th>Report</th><th>Modified</th><th>Size</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody>"
            "</table>"
        )

    body = "\n".join(sections) or "<p>No reports yet.</p>"
    index = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Electricity load forecasting experiments</title>
<style>
  body {{ font-family: sans-serif; max-width: 1100px; margin: 2em auto; padding: 0 1em; line-height: 1.4; }}
  h1 {{ font-size: 1.6em; }}
  h2 {{ font-size: 1.15em; margin-top: 1.8em; border-bottom: 1px solid #ddd; padding-bottom: 0.2em; }}
  table {{ border-collapse: collapse; width: 100%; }}
  th, td {{ border-bottom: 1px solid #ddd; padding: 0.5em; text-align: left; vertical-align: top; }}
  th {{ white-space: nowrap; }}
  .path {{ color: #666; font-family: monospace; font-size: 0.85em; }}
</style>
</head>
<body>
<h1>Electricity load forecasting experiments</h1>
<p>Study code lives in <code>scripts/</code>, the core package in
<code>src/electricity_load_forecasting/</code>, and result provenance in each
experiment's <code>README.md</code>. This page lists the published reports,
grouped by theme.</p>
{body}
</body>
</html>
"""
    (DOCS / "index.html").write_text(index, "utf-8")


def main():
    args = parse_args()
    if args.name and len(args.sources) != 1:
        raise SystemExit("--name requires exactly one source")

    DOCS.mkdir(exist_ok=True)
    (DOCS / ".nojekyll").touch()

    plan = []
    seen = {}
    for source in args.sources:
        if not source.exists():
            raise SystemExit(f"no such file or directory: {source}")
        destination = destination_for(source, args.name, args.theme)
        if destination == DOCS / "index.html":
            raise SystemExit(
                f"{source} would overwrite the landing page at {destination}; "
                "use --name to publish it elsewhere"
            )
        if destination in seen:
            raise SystemExit(
                f"{seen[destination]} and {source} would both publish to "
                f"{destination}; use --name one at a time"
            )
        seen[destination] = source
        if destination.exists() and not args.force:
            raise SystemExit(
                f"destination already exists: {destination}\n"
                "use --name NAME to publish under a different name, or --force "
                "to replace it"
            )
        plan.append((source, destination))

    for source, destination in plan:
        copy_source(source, destination, args.all)
        print(f"copied {source} -> {destination}")

    if not args.no_index:
        build_index()
        print(f"wrote {DOCS / 'index.html'}")

    if args.sources:
        print("next: git add docs && git commit -m 'Publish report' && git push")


if __name__ == "__main__":
    main()
