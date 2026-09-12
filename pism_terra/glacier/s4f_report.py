"""
Run the Snow4Flow calibration and benchmark tools on a project and write an HTML report.

``pism-s4f-report RUN_DIR --output-path REPORT`` runs, in order,

* ``pism-glacier-importance-sampling`` (posterior weights of the UQ members
  from observed elevation change),
* ``pism-glacier-usgs-benchmark-glaciers`` (glacier-wide balances against
  the USGS benchmark glaciers) and
* ``pism-glacier-usgs-benchmark-stakes`` (stake balances and gradients),

each into its own sub-directory of the report, and renders ``index.html``
plus one page per tool in the layout of the pism-terra documentation, with
the Snow4Flow logo. A tool that fails is reported on its page with the
traceback and does not stop the others; ``--no-run`` renders the pages from
outputs already on disk and ``--skip`` leaves a tool out.
"""

from __future__ import annotations

import logging
import re
import shutil
import sys
import time
import traceback
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Sequence
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from pism_terra.glacier import (
    importance_sampling,
    usgs_benchmark_glaciers,
    usgs_benchmark_stakes,
)
from pism_terra.glacier.usgs import DEFAULT_DATA_DIR as DEFAULT_USGS_DATA_DIR
from pism_terra.glacier.usgs import plot_year
from pism_terra.likelihood import REDUCTIONS

logger = logging.getLogger(__name__)

TEMPLATE_DIR = Path(__file__).resolve().parent.parent / "templates" / "report"
LOGO = TEMPLATE_DIR / "snow4flow_logo.svg"
RGI_PATTERN = re.compile(r"RGI2000-v7\.0-[A-Z]-\d{2}-\d+")
MAX_TABLE_ROWS = 300

TOOLS: tuple[dict[str, str], ...] = (
    {
        "key": "importance",
        "subdir": "importance_sampling",
        "href": "importance_sampling.html",
        "label": "Importance sampling",
        "blurb": "Posterior weights, effective sample sizes and parameter histograms of the UQ ensemble "
        "against the observed elevation change, per glacier and jointly.",
    },
    {
        "key": "glaciers",
        "subdir": "usgs_glaciers",
        "href": "usgs_glaciers.html",
        "label": "USGS benchmark glaciers",
        "blurb": "Modelled glacier-wide annual and seasonal balances against the USGS benchmark-glacier record.",
    },
    {
        "key": "stakes",
        "subdir": "usgs_stakes",
        "href": "usgs_stakes.html",
        "label": "USGS benchmark stakes",
        "blurb": "Modelled surface mass balance at the stake locations, balance gradients and skill scores.",
    },
)


def _table(path: Path, label: str, table_id: str, *, index: bool = False) -> dict[str, Any] | None:
    """
    Render a CSV file as an HTML table for the report.

    Parameters
    ----------
    path : Path
        CSV file; a missing file yields ``None``.
    label : str
        Section heading.
    table_id : str
        HTML anchor.
    index : bool, optional
        Keep the first column as an index.

    Returns
    -------
    dict or None
        ``{"id", "label", "html", "note"}`` or ``None`` when the file is missing or empty.
    """
    if not path.is_file():
        return None
    try:
        df = pd.read_csv(path, index_col=0 if index else None)
    except pd.errors.EmptyDataError:
        return None
    note = f"{len(df)} rows; source <code>{path.name}</code>."
    if len(df) > MAX_TABLE_ROWS:
        note = f"first {MAX_TABLE_ROWS} of {len(df)} rows; full table in <code>{path.name}</code>."
        df = df.head(MAX_TABLE_ROWS)
    html = df.to_html(classes="table", index=index, float_format=lambda v: f"{v:.4g}", border=0, na_rep="")
    return {"id": table_id, "label": label, "html": html, "note": note}


def _figures(directory: Path, base: Path, pattern: str = "*.png") -> list[dict[str, str]]:
    """
    List figures of a directory as report entries with paths relative to the report root.

    Parameters
    ----------
    directory : Path
        Directory searched (not recursively).
    base : Path
        Report root the hrefs are made relative to.
    pattern : str, optional
        Glob pattern.

    Returns
    -------
    list of dict
        ``{"href", "caption"}`` per figure, sorted by name.
    """
    if not directory.is_dir():
        return []
    return [
        {"href": p.relative_to(base).as_posix(), "caption": p.stem.replace("_", " ")}
        for p in sorted(directory.glob(pattern))
    ]


def _glacier_sections(output_dir: Path, base: Path, table_specs: Sequence[tuple[str, str]] = ()) -> list[dict]:
    """
    One section per ``<rgi_id>/`` sub-directory with its figures and optional tables.

    Parameters
    ----------
    output_dir : Path
        Tool output directory.
    base : Path
        Report root.
    table_specs : sequence of (glob, label), optional
        CSV files inside each glacier directory to tabulate.

    Returns
    -------
    list of dict
        Sections ``{"id", "label", "figures", "tables"}`` sorted by glacier.
    """
    sections: list[dict[str, Any]] = []
    if not output_dir.is_dir():
        return sections
    for sub in sorted(p for p in output_dir.iterdir() if p.is_dir() and RGI_PATTERN.fullmatch(p.name)):
        tables = []
        for glob, label in table_specs:
            for csv in sorted(sub.glob(glob)):
                t = _table(csv, label, f"{sub.name}-{csv.stem}")
                if t:
                    tables.append(t)
        sections.append({"id": sub.name, "label": sub.name, "figures": _figures(sub, base), "tables": tables})
    return sections


def collect_importance(output_dir: Path, base: Path) -> dict[str, Any]:
    """
    Gather the importance-sampling outputs for its page.

    Parameters
    ----------
    output_dir : Path
        ``<report>/importance_sampling``.
    base : Path
        Report root.

    Returns
    -------
    dict
        ``tables`` (summary), ``sections`` (joint first, then per glacier) and ``n_glaciers``.
    """
    tables = [t for t in (_table(output_dir / "importance_sampling_summary.csv", "Summary", "summary"),) if t]
    sections = []
    joint = output_dir / "joint"
    if joint.is_dir():
        joint_tables = [
            t
            for csv in sorted(joint.glob("importance_joint_*.csv"))
            if (t := _table(csv, csv.stem, f"joint-{csv.stem}", index=True))
        ]
        sections.append(
            {"id": "joint", "label": "Joint posterior", "figures": _figures(joint, base), "tables": joint_tables}
        )
    glaciers = _glacier_sections(output_dir, base, [("importance_*.csv", "Members")])
    return {"tables": tables, "sections": sections + glaciers, "n_glaciers": len(glaciers)}


def collect_usgs_glaciers(output_dir: Path, base: Path) -> dict[str, Any]:
    """
    Gather the USGS glacier benchmark outputs for its page.

    Parameters
    ----------
    output_dir : Path
        ``<report>/usgs_glaciers``.
    base : Path
        Report root.

    Returns
    -------
    dict
        ``tables`` (skill, matches), ``sections`` per glacier and ``n_glaciers``.
    """
    tables = [
        t
        for t in (
            _table(output_dir / "usgs_benchmark_skill.csv", "Skill scores", "skill"),
            _table(output_dir / "usgs_benchmark_rgi_match.csv", "USGS glaciers matched to RGI", "matches"),
        )
        if t
    ]
    glaciers = _glacier_sections(output_dir, base)
    return {"tables": tables, "sections": glaciers, "n_glaciers": len(glaciers)}


def collect_usgs_stakes(output_dir: Path, base: Path) -> dict[str, Any]:
    """
    Gather the USGS stake benchmark outputs for its page.

    Parameters
    ----------
    output_dir : Path
        ``<report>/usgs_stakes``.
    base : Path
        Report root.

    Returns
    -------
    dict
        ``tables`` (skill, gradients, matches), ``sections`` per glacier and ``n_glaciers``.
    """
    tables = [
        t
        for t in (
            _table(output_dir / "usgs_benchmark_stakes_skill.csv", "Skill scores", "skill"),
            _table(output_dir / "usgs_benchmark_stakes_gradients.csv", "Balance gradients", "gradients"),
            _table(output_dir / "usgs_benchmark_stakes_rgi_match.csv", "USGS glaciers matched to RGI", "matches"),
        )
        if t
    ]
    glaciers = _glacier_sections(output_dir, base, [("*_gradient.csv", "Gradient fits")])
    return {"tables": tables, "sections": glaciers, "n_glaciers": len(glaciers)}


COLLECTORS = {"importance": collect_importance, "glaciers": collect_usgs_glaciers, "stakes": collect_usgs_stakes}


def run_tools(
    run_dir: Path,
    output_path: Path,
    *,
    data_path: Path | None,
    usgs_data_path: Path,
    rgi_file: Path | None,
    skip: set[str],
    fudge_factors: Sequence[float],
    reduction: str,
    bootstrap: bool,
    n_jobs: int,
    plot_years: tuple[float | None, float | None],
) -> dict[str, dict[str, Any]]:
    """
    Run each tool into its sub-directory, recording status, elapsed time and errors.

    Parameters
    ----------
    run_dir : Path
        Project directory the tools search for model output.
    output_path : Path
        Report root; tools write to ``<output_path>/<subdir>``.
    data_path : Path or None
        Staging tree with the observations for the importance sampling.
    usgs_data_path : Path
        Cache for the USGS archives and RGI outlines.
    rgi_file : Path or None
        Outline file for the USGS matching.
    skip : set of str
        Tool keys not to run.
    fudge_factors : sequence of float
        Importance-sampling fudge factors.
    reduction : str
        Likelihood reduction of the importance sampling.
    bootstrap : bool
        Run the block-bootstrap RMSE ranking.
    n_jobs : int
        Worker processes for the USGS tools.
    plot_years : tuple
        Plot limits for the USGS tools.

    Returns
    -------
    dict
        ``{key: {"status", "elapsed", "error", "command"}}``.
    """
    results: dict[str, dict[str, Any]] = {}
    for tool in TOOLS:
        key, out = tool["key"], output_path / tool["subdir"]
        if key in skip:
            results[key] = {"status": "skipped", "elapsed": None, "error": None, "command": ""}
            continue
        out.mkdir(parents=True, exist_ok=True)
        command, call = _tool_call(
            key,
            run_dir,
            out,
            data_path=data_path,
            usgs_data_path=usgs_data_path,
            rgi_file=rgi_file,
            fudge_factors=fudge_factors,
            reduction=reduction,
            bootstrap=bootstrap,
            n_jobs=n_jobs,
            plot_years=plot_years,
        )
        logger.info("running %s -> %s", tool["label"], out)
        t0 = time.time()
        try:
            call()
            results[key] = {"status": "ok", "elapsed": _elapsed(time.time() - t0), "error": None, "command": command}
        except Exception:  # pylint: disable=broad-exception-caught
            err = traceback.format_exc()
            logger.error("%s failed:\n%s", tool["label"], err)
            results[key] = {"status": "failed", "elapsed": _elapsed(time.time() - t0), "error": err, "command": command}
    return results


def _tool_call(key: str, run_dir: Path, out: Path, **opts: Any) -> tuple[str, Any]:
    """
    Build the command-line equivalent and the callable of one tool.

    Parameters
    ----------
    key : str
        Tool key: ``importance``, ``glaciers`` or ``stakes``.
    run_dir : Path
        Project directory.
    out : Path
        The tool's output directory.
    **opts : Any
        The options of :func:`run_tools`.

    Returns
    -------
    str
        The equivalent command line, shown on the tool's page.
    callable
        Zero-argument function running the tool.
    """
    if key == "importance":
        command = f"pism-glacier-importance-sampling {run_dir} --output-path {out}"
        if opts["data_path"]:
            command += f" --data-path {opts['data_path']}"

        def call_importance():
            """
            Run the importance sampling into ``out``.

            Returns
            -------
            pandas.DataFrame
                The tool's summary.
            """
            return importance_sampling.run_pipeline(
                run_dir,
                data_path=opts["data_path"],
                output_path=out,
                fudge_factors=opts["fudge_factors"],
                reduction=opts["reduction"],
                bootstrap=opts["bootstrap"],
            )

        return command, call_importance
    module = usgs_benchmark_glaciers if key == "glaciers" else usgs_benchmark_stakes
    name = "glaciers" if key == "glaciers" else "stakes"
    command = f"pism-glacier-usgs-benchmark-{name} {run_dir} --data-path {opts['usgs_data_path']} --output-path {out}"

    def call_usgs():
        """
        Run the USGS benchmark into ``out``.

        Returns
        -------
        pandas.DataFrame
            The tool's summary.
        """
        return module.run_pipeline(
            run_dir,
            data_dir=opts["usgs_data_path"],
            output_dir=out,
            rgi_file=opts["rgi_file"],
            n_jobs=opts["n_jobs"],
            plot_years=opts["plot_years"],
        )

    return command, call_usgs


def _elapsed(seconds: float) -> str:
    """
    Format a duration for the report.

    Parameters
    ----------
    seconds : float
        Duration.

    Returns
    -------
    str
        ``"12 s"``, ``"3.4 min"`` or ``"1.2 h"``.
    """
    if seconds < 90:
        return f"{seconds:.0f} s"
    if seconds < 5400:
        return f"{seconds / 60:.1f} min"
    return f"{seconds / 3600:.1f} h"


def render(
    run_dir: Path,
    output_path: Path,
    results: dict[str, dict[str, Any]],
    *,
    title: str,
    logo: Path,
    setup: Sequence[tuple[str, str]],
) -> list[Path]:
    """
    Render ``index.html`` and one page per tool.

    Parameters
    ----------
    run_dir : Path
        Project directory shown in the header.
    output_path : Path
        Report root.
    results : dict
        Per-tool status from :func:`run_tools` (``skipped`` entries for tools not run).
    title : str
        Report title.
    logo : Path
        Logo file copied to ``_static/``.
    setup : sequence of (str, str)
        Key/value rows for the Setup table of the index page.

    Returns
    -------
    list of Path
        The pages written.
    """
    static = output_path / "_static"
    static.mkdir(parents=True, exist_ok=True)
    shutil.copy2(TEMPLATE_DIR / "report.css", static / "report.css")
    logo_name = f"logo{logo.suffix.lower()}"
    shutil.copy2(logo, static / logo_name)
    env = Environment(loader=FileSystemLoader(TEMPLATE_DIR), autoescape=True, undefined=StrictUndefined)
    try:
        pkg_version = version("pism-terra")
    except PackageNotFoundError:
        pkg_version = "unknown"
    common = {
        "title": title,
        "run_dir": str(run_dir),
        "logo_name": logo_name,
        "generated": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "version": pkg_version,
        "nav": [{"href": "index.html", "label": "Overview"}]
        + [{"href": t["href"], "label": t["label"]} for t in TOOLS],
    }
    pages = []
    cards = []
    for tool in TOOLS:
        key = tool["key"]
        res = results.get(key, {"status": "skipped", "elapsed": None, "error": None, "command": ""})
        collected = COLLECTORS[key](output_path / tool["subdir"], output_path)
        status = res["status"]
        if status == "ok" and not collected["sections"] and not collected["tables"]:
            status = "missing"
        if status == "skipped" and (collected["sections"] or collected["tables"]):
            status = "ok"  # rendered from outputs already on disk
        page = output_path / tool["href"]
        page.write_text(
            env.get_template("tool.html.j2").render(
                **common,
                page_title=tool["label"],
                current=tool["href"],
                blurb=tool["blurb"],
                command=res["command"] or f"(not run) outputs read from {tool['subdir']}/",
                elapsed=res["elapsed"],
                error=res["error"],
                status=status,
                output_dir=tool["subdir"],
                tables=collected["tables"],
                sections=collected["sections"],
            ),
            encoding="utf-8",
        )
        pages.append(page)
        cards.append(
            {
                **tool,
                "status": status,
                "elapsed": res["elapsed"],
                "error": res["error"],
                "n_glaciers": collected["n_glaciers"],
            }
        )
    index = output_path / "index.html"
    index.write_text(
        env.get_template("index.html.j2").render(
            **common, page_title="Overview", current="index.html", sections=[], tools=cards, setup=list(setup)
        ),
        encoding="utf-8",
    )
    return [index] + pages


def main(argv: Sequence[str] | None = None) -> list[Path]:
    """
    Command-line entry point.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Arguments; ``None`` reads ``sys.argv``.

    Returns
    -------
    list of Path
        The pages written.
    """
    parser = ArgumentParser(
        description="Run the Snow4Flow calibration and benchmark tools on a project and write an HTML report.",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("RUN_DIR", help="Project directory searched recursively for model output.")
    parser.add_argument(
        "--output-path", default="report", help="Report directory; each tool writes to a sub-directory of it."
    )
    parser.add_argument(
        "--data-path", default=None, help="Staging tree with <rgi_id>/input/obs_<rgi_id>.nc; defaults to RUN_DIR."
    )
    parser.add_argument(
        "--usgs-data-path", default=DEFAULT_USGS_DATA_DIR, help="Cache for the USGS archives and RGI outlines."
    )
    parser.add_argument("--rgi-glacier-file", default=None, help="Outline file the USGS tools match against.")
    parser.add_argument("--title", default="Snow4Flow report", help="Report title.")
    parser.add_argument("--logo", default=str(LOGO), help="Logo image copied into the report (SVG or PNG).")
    parser.add_argument(
        "--skip", action="append", choices=[t["key"] for t in TOOLS], default=[], help="Tool to leave out; repeatable."
    )
    parser.add_argument(
        "--no-run", action="store_true", default=False, help="Render from outputs already under --output-path."
    )
    parser.add_argument(
        "--fudge-factors",
        type=lambda s: tuple(float(x) for x in s.split(",")),
        default=importance_sampling.DEFAULT_FUDGE_FACTORS,
        help="Importance sampling: comma-separated multipliers on the observed error.",
    )
    parser.add_argument(
        "--reduction",
        choices=REDUCTIONS,
        default=importance_sampling.DEFAULT_REDUCTION,
        help="Importance sampling: likelihood reduction.",
    )
    parser.add_argument(
        "--no-bootstrap", action="store_true", default=False, help="Importance sampling: skip the RMSE ranking."
    )
    parser.add_argument("--n-jobs", type=int, default=1, help="USGS tools: worker processes.")
    parser.add_argument("--plot-start", default=None, help="USGS tools: first year of the plots.")
    parser.add_argument("--plot-end", default=None, help="USGS tools: last year of the plots.")
    args = parser.parse_args(list(argv) if argv is not None else None)

    run_dir = Path(args.RUN_DIR).expanduser().resolve()
    output_path = Path(args.output_path).expanduser().resolve()
    output_path.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", datefmt="%H:%M:%S")
    data_path = Path(args.data_path).expanduser().resolve() if args.data_path else None
    usgs_data_path = Path(args.usgs_data_path).expanduser()
    rgi_file = Path(args.rgi_glacier_file).expanduser() if args.rgi_glacier_file else None

    if args.no_run:
        results = {t["key"]: {"status": "skipped", "elapsed": None, "error": None, "command": ""} for t in TOOLS}
    else:
        results = run_tools(
            run_dir,
            output_path,
            data_path=data_path,
            usgs_data_path=usgs_data_path,
            rgi_file=rgi_file,
            skip=set(args.skip),
            fudge_factors=args.fudge_factors,
            reduction=args.reduction,
            bootstrap=not args.no_bootstrap,
            n_jobs=args.n_jobs,
            plot_years=(plot_year(args.plot_start), plot_year(args.plot_end)),
        )
    setup = [
        ("run directory", str(run_dir)),
        ("observations", str(data_path or run_dir)),
        ("USGS cache", str(usgs_data_path)),
        ("fudge factors", ", ".join(f"{f:g}" for f in args.fudge_factors)),
        ("reduction", args.reduction),
        ("command", " ".join(["pism-s4f-report"] + list(argv if argv is not None else sys.argv[1:]))),
    ]
    pages = render(run_dir, output_path, results, title=args.title, logo=Path(args.logo).expanduser(), setup=setup)
    logger.info("report written to %s", pages[0])
    return pages


def cli(argv: Sequence[str] | None = None) -> int:
    """
    Console-script wrapper around :func:`main`.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Arguments; ``None`` reads ``sys.argv``.

    Returns
    -------
    int
        Exit status.
    """
    main(argv)
    return 0


if __name__ == "__main__":
    sys.exit(cli())
