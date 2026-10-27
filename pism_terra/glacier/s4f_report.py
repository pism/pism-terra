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

import json
import logging
import re
import shutil
import sys
import time
import traceback
from argparse import SUPPRESS, ArgumentDefaultsHelpFormatter, ArgumentParser
from collections.abc import Sequence
from datetime import datetime, timezone
from html import escape
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from statistics import NormalDist
from typing import Any

import pandas as pd
from jinja2 import Environment, FileSystemLoader, StrictUndefined

from pism_terra.glacier import (
    importance_sampling,
    sensitivity_indices,
    usgs_benchmark_glaciers,
    usgs_benchmark_stakes,
)
from pism_terra.glacier.usgs import DEFAULT_DATA_DIR as DEFAULT_USGS_DATA_DIR
from pism_terra.glacier.usgs import plot_year
from pism_terra.likelihood import REDUCTIONS
from pism_terra.ntasks_option import add_ntasks_argument

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
        "against the observed elevation change, per glacier and jointly over all glaciers (the members are "
        "shared parameter draws, so their per-glacier log-likelihoods add up; see the note below).",
    },
    {
        "key": "sensitivity",
        "subdir": "sensitivity_indices",
        "href": "sensitivity_indices.html",
        "label": "Sensitivity indices",
        "blurb": "First-order Sobol and Borgonovo delta indices of the yearly ice mass (or another target) to the "
        "UQ parameters, per complex and per glacier, with bootstrap confidence bands.",
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


REDUCTION_TEXT = {
    "blocks": "the per-cell log-likelihood is <em>summed</em> and divided by the square of the block side, i.e. "
    "each block of decorrelation-length side counts as one independent observation. A glacier therefore enters "
    "the joint posterior with a weight proportional to its number of independent blocks: large glaciers with "
    "short-range correlation count more than small ones.",
    "mean": "the per-cell log-likelihood is <em>averaged</em> over the cells, so every glacier enters the joint "
    "posterior with the same weight whatever its size, and the posterior is strongly tempered.",
    "sum": "the per-cell log-likelihood is <em>summed</em> over all cells, so every cell counts as an independent "
    "observation and a glacier's weight in the joint posterior is proportional to its cell count.",
}


def joint_posterior_note(summary_csv: Path) -> dict[str, str] | None:
    """
    Explain how the joint posterior weighs the glaciers, with this run's numbers.

    Parameters
    ----------
    summary_csv : Path
        ``importance_sampling_summary.csv`` of the run.

    Returns
    -------
    dict or None
        ``{"title", "html"}`` for the page, or ``None`` when the summary is missing.
    """
    if not summary_csv.is_file():
        return None
    try:
        df = pd.read_csv(summary_csv)
    except pd.errors.EmptyDataError:
        return None
    glaciers = df[df["rgi_id"] != "joint"].drop_duplicates("rgi_id")
    reduction = str(glaciers["reduction"].iloc[0]) if "reduction" in glaciers and len(glaciers) else "blocks"
    html = [
        "<p>Each ensemble member is one set of parameter values shared by every glacier, so the glaciers' "
        "evidence about a member adds up: for every member that finished on <em>all</em> glaciers, the "
        "per-glacier log-likelihoods are summed, and the sums are turned into weights, counts and an "
        "effective sample size exactly as for a single glacier. The fudge factor scales the observed "
        "uncertainty before the likelihood, per glacier and jointly alike.</p>",
        f"<p>How much a glacier counts in that sum is fixed by the likelihood reduction, here "
        f"<code>{reduction}</code>: {REDUCTION_TEXT.get(reduction, '')}</p>",
    ]
    cols = [c for c in ("n_valid_cells", "block_size", "decorrelation_length") if c in glaciers]
    if cols and len(glaciers):
        rows = []
        for _, r in glaciers.iterrows():
            n_valid = int(r["n_valid_cells"]) if pd.notna(r.get("n_valid_cells")) else None
            block = int(r["block_size"]) if pd.notna(r.get("block_size")) else None
            blocks = round(n_valid / block**2) if n_valid and block else None
            length = f"{r['decorrelation_length'] / 1e3:.1f} km" if pd.notna(r.get("decorrelation_length")) else ""
            rows.append(
                f"<tr><td>{r['rgi_id']}</td><td>{n_valid if n_valid is not None else ''}</td><td>{length}</td>"
                f"<td>{block if block is not None else ''}</td><td>{blocks if blocks is not None else ''}</td></tr>"
            )
        html.append(
            '<div class="table-wrap"><table class="table"><thead><tr><th>glacier</th><th>cells with observations</th>'
            "<th>decorrelation length</th><th>block side (cells)</th><th>independent blocks</th></tr></thead>"
            f"<tbody>{''.join(rows)}</tbody></table></div>"
        )
        if reduction == "blocks":
            html.append(
                "<p>With the <code>blocks</code> reduction the last column is, up to the misfit itself, each "
                "glacier's weight in the joint posterior.</p>"
            )
    if not (df["rgi_id"] == "joint").any():
        html.append("<p>No joint posterior was formed: it needs at least two glaciers with a common member.</p>")
    return {"title": "How the joint posterior is formed", "html": "".join(html)}


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
        sections.append({"id": "joint", "label": "Joint posterior", "figures": _figures(joint, base), "tables": []})
    # The per-member tables run to hundreds of rows per glacier; they stay in the
    # CSV files next to the figures rather than on the page.
    glaciers = _glacier_sections(output_dir, base)
    note = joint_posterior_note(output_dir / "importance_sampling_summary.csv")
    return {
        "tables": tables,
        "sections": sections + glaciers,
        "n_glaciers": len(glaciers),
        "notes": [note] if note else [],
    }


GLACIER_METHOD_NOTE = {
    "title": "How the numbers are calculated",
    "html": "<h4>1. Observations</h4>"
    "<p>The USGS benchmark-glacier release publishes geodetically calibrated glacier-wide winter "
    "(<code>Bw</code>), summer (<code>Bs</code>) and annual (<code>Ba</code>) balances in m w.e. A balance year "
    "runs from one annual minimum to the next, both dated in the release (<code>Ba_Date</code>), and is split "
    "at the winter maximum (<code>Bw_Date</code>). Each glacier is assigned the RGI v7 outline holding most of "
    "its stakes, or the outline nearest their centroid within 5 km (<em>USGS glaciers matched to RGI</em>).</p>"
    "<h4>2. Model series</h4>"
    "<p>The model values come from the post-processed per-glacier scalar files (<code>scalar_G_*.nc</code>) of "
    "that RGI outline. <code>surface</code> rows integrate <code>tendency_of_ice_mass_due_to_surface_mass_flux"
    "</code>, the mass a stake network can see, over the release's own dates: <code>Bw</code> from the "
    "previous year's <code>Ba_Date</code> to this year's <code>Bw_Date</code>, <code>Bs</code> from there to "
    "this year's <code>Ba_Date</code>, and <code>Ba = Bw + Bs</code>. Monthly values are weighted by the days "
    "they share with the season, so this needs monthly output; a season the model record does not fully cover "
    "is left out. The <code>total</code> row is the annual mean of <code>tendency_of_ice_mass</code>, which "
    "also holds discharge and flow across the outline.</p>"
    "<h4>3. Per unit area</h4>"
    "<p>Model mass changes (Gt) are divided by the model's own ice area of the balance year (the ice-covered "
    "fraction <code>sftgif</code> summed over the outline times the cell area; the outline area where the file "
    "has no <code>sftgif</code>; the observed area where it has neither) and the density of water, 1000 kg "
    "m<sup>-3</sup>, giving m w.e. like the observations.</p>"
    "<h4>4. Ensemble and skill</h4>"
    "<p>With several runs the median over runs is scored. Over the balance years both sides have: <code>n"
    "</code> their number; <code>r</code> the Pearson correlation (with at least 3 years that are not "
    "constant); <code>mae</code> the mean of |model &minus; observed|; <code>bias</code> the mean of model "
    "&minus; observed, so a negative bias is too little accumulation or too much melt. All in m w.e. per "
    "year.</p>",
}


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
    return {"tables": tables, "sections": glaciers, "n_glaciers": len(glaciers), "notes": [GLACIER_METHOD_NOTE]}


START_AT_TEXT = {
    "summer_surface": "<p>A winter or annual balance is measured against the <em>previous summer surface</em>, "
    "the lowest the surface gets at the end of the melt season, not against the surface on the day of the fall "
    "visit. A visit before the melt season ends (July on the Juneau Icefield, August or September elsewhere) "
    "therefore leaves the melt that follows it out of the measurement: on the next visit it is below the "
    "summer surface the stake is read against. The modelled <code>bw</code> and <code>ba</code> start where the "
    "model's own balance since the fall visit is lowest: the visit itself, or the end of a month up to "
    "{days} days later (and before the closing visit). Where the model accumulates from the visit on, the start "
    "stays at the visit. <code>bs</code> and <code>db</code> start at the visit. The table "
    "<em>Start of the modelled winter and annual balances</em> below says how far the starts moved and how much "
    "balance that left out; <code>model_from_visit</code> in the per-glacier CSV files is each balance "
    "integrated from the visit instead.</p>",
    "visit": "<p>The modelled <code>bw</code> and <code>ba</code> start on the day of the previous fall visit, "
    "so melt between the visit and the end of the melt season is counted, although the stake, read against "
    "the previous summer surface, does not see it. Run with <code>--start-at summer-surface</code> to start "
    "at the modelled summer surface instead.</p>",
}


def stake_method_note(output_dir: Path) -> dict[str, str]:
    """
    Explain how every number on the stake page is calculated, with this run's settings.

    Parameters
    ----------
    output_dir : Path
        ``<report>/usgs_stakes``, holding the settings the benchmark wrote.

    Returns
    -------
    dict
        ``{"title", "html"}`` for the page.
    """
    settings: dict[str, Any] = {}
    path = output_dir / usgs_benchmark_stakes.SETTINGS_FILE
    if path.is_file():
        settings = json.loads(path.read_text(encoding="utf-8"))
    sampling = settings.get("sampling", "nearest")
    start_at = settings.get("start_at", "visit")
    days = settings.get("summer_surface_search_days", usgs_benchmark_stakes.SUMMER_SURFACE_SEARCH_DAYS)
    rho = settings.get("rho_water", 1000.0)
    min_years = settings.get("skill_min_years", 3)
    min_points = settings.get("gradient_min_points", usgs_benchmark_stakes.GRADIENT_MIN_POINTS)
    n_files = f" ({settings['n_files']} files)" if "n_files" in settings else ""
    sampled = (
        "the value of the grid cell the stake lies in"
        if sampling == "nearest"
        else "a bilinear interpolation of the four surrounding cells (the stake's own cell where one of them has "
        "no value; the ice mask is always the stake's own cell)"
    )
    html = [
        (
            ""
            if path.is_file()
            else "<p><em>The benchmark wrote no settings file, so the defaults of an older "
            "version are assumed below: nearest-cell sampling and balances from the visit date.</em></p>"
        ),
        "<h4>1. Stakes and glaciers</h4>",
        "<p>The stake positions and measurements are the USGS benchmark-glacier release (glaciological data: "
        "<code>bw</code> at the spring visit, <code>ba</code> at the fall visit, sub-seasonal <code>db</code> "
        "between two visits). Each glacier is assigned the RGI v7 outline holding most of its stakes, or the "
        "outline nearest their centroid within 5 km (<em>USGS glaciers matched to RGI</em>).</p>",
        "<h4>2. Model values at a stake</h4>",
        f"<p>Every monthly <code>spatial_*.nc</code> file under the run directory is read at the stakes, as "
        f"{sampled}{n_files}; files that end before "
        "the first measurement (initialisation legs) are left out. The surface mass balance rate is "
        f"<code>(surface_accumulation_flux &minus; surface_runoff_flux) / &rho;<sub>w</sub></code> with "
        f"&rho;<sub>w</sub> = {rho:g} kg m<sup>-3</sup>, in m w.e. per year; without those two fields, "
        "<code>tendency_of_ice_mass_due_to_surface_mass_flux</code> divided by the cell area and "
        "&rho;<sub>w</sub>. Accumulation is the snowfall and runoff the melt that does not refreeze; rain "
        "does not enter the balance. Months in which the model has no ice at the stake "
        "(<code>thk</code> &le; 0) have no value, and an interval containing such a month has no modelled "
        "balance.</p>",
        "<h4>3. Intervals</h4>",
        "<p>Each measurement is compared with the model over its own interval: <code>bw</code> from the "
        "previous fall visit (or, without one, the previous balance year's glacier-wide <code>Ba_Date</code>) "
        "to the spring visit; <code>bs</code> from the spring to the fall visit, observed as "
        "<code>ba &minus; bw</code>; <code>ba</code> from the previous to this fall visit; <code>db</code> "
        "between its two visits. The monthly rates are integrated with each month weighted by the days it "
        "shares with the interval, divided by 365; an interval the model record does not fully cover has no "
        "modelled value.</p>",
        "<h4>4. Start of the modelled winter and annual balances</h4>",
        START_AT_TEXT.get(start_at, "").format(days=days),
        "<h4>5. Ensemble</h4>",
        "<p>Each spatial file is one run. With several runs the scores, scatter and gradient plots use the "
        "median over the runs of each measurement, and the time series show the 5&ndash;95 % range over runs "
        "as a band.</p>",
        "<h4>6. Skill scores</h4>",
        "<p>Per site, and pooled over all sites of a glacier (<code>site = all</code>), per season, over the "
        "measurements with both an observed and a modelled value: <code>n</code> their number; <code>r</code> "
        f"the Pearson correlation (only with at least {min_years} values that are not constant); "
        "<code>mae</code> the mean of |model &minus; observed|; <code>bias</code> the mean of "
        "model &minus; observed, so a negative bias is too little accumulation or too much melt; "
        "<code>elevation_offset</code> the mean modelled surface over the interval minus the stake's surveyed "
        "elevation; <code>n_ice_free_months</code> the months the model had no ice at the stake. All balances "
        "are in m w.e.</p>",
        "<h4>7. Balance gradients</h4>",
        "<p>For each season the observed and the modelled balances (ensemble median) are fitted against the "
        "stake elevations with two straight lines that meet at zero balance, the ELA: "
        "<code>b(z) = k<sub>below</sub> (z &minus; ELA)</code> below it and <code>k<sub>above</sub> "
        "(z &minus; ELA)</code> above. For each candidate ELA the slopes are a least-squares fit; the ELA with "
        "the smallest squared misfit is kept, searched over the stakes' elevation range widened by that range "
        f"on either side, with at least {min_points} stakes on each side that has any. <code>ela_in_range</code> "
        "is 1 when the ELA lies between the lowest and highest stake; the gradients are in m w.e. per m, "
        "<code>rmse</code> in m w.e.</p>",
    ]
    return {"title": "How the numbers are calculated", "html": "".join(html)}


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
            _table(
                output_dir / usgs_benchmark_stakes.MODEL_START_FILE,
                "Start of the modelled winter and annual balances",
                "model-start",
            ),
            _table(output_dir / "usgs_benchmark_stakes_rgi_match.csv", "USGS glaciers matched to RGI", "matches"),
        )
        if t
    ]
    glaciers = _glacier_sections(output_dir, base, [("*_gradient.csv", "Gradient fits")])
    return {
        "tables": tables,
        "sections": glaciers,
        "n_glaciers": len(glaciers),
        "notes": [stake_method_note(output_dir)],
    }


FREQUENCY_TEXT = {
    "yearly": "averaged to calendar years",
    "monthly": "averaged to calendar months",
    "none": "used at the times the files hold",
}


def sensitivity_method_note(output_dir: Path) -> dict[str, str]:
    """
    Explain how the sensitivity indices are calculated, with this run's settings.

    Parameters
    ----------
    output_dir : Path
        ``<report>/sensitivity_indices``, holding the settings the tool wrote.

    Returns
    -------
    dict
        ``{"title", "html"}`` for the page.
    """
    settings: dict[str, Any] = {}
    path = output_dir / sensitivity_indices.SETTINGS_FILE
    if path.is_file():
        settings = json.loads(path.read_text(encoding="utf-8"))
    target = escape(str(settings.get("target", sensitivity_indices.DEFAULT_TARGET)))
    freq = str(settings.get("frequency", "yearly"))
    resamples = settings.get("num_resamples", 100)
    min_members = settings.get("min_members", 8)
    conf = float(settings.get("conf_level", sensitivity_indices.CONF_LEVEL))
    parts = [
        "" if path.is_file() else "<p><em>The tool wrote no settings file, so the defaults are assumed below.</em></p>",
        "<h4>1. Ensemble</h4>",
        "<p>Each member of the UQ ensemble is one draw of the uncertain parameters, listed in the complex's "
        "<code>uq.csv</code>. Its response is read from the post-processed scalar files, <code>scalar_C_*</code> "
        "for the whole complex and <code>scalar_G_*</code> for each of its glaciers, as the variable "
        f"<code>{target}</code>, {FREQUENCY_TEXT.get(freq, escape(freq))}. A complex with fewer than "
        f"{min_members} finished members is skipped: the estimator needs comfortably more members than "
        "parameters, and with fewer members than parameters plus two no index is computed at all.</p>",
        "<h4>2. Indices</h4>",
        "<p>The indices are computed separately at every time step, from the members' parameter values and "
        "responses alone (the <em>given-data</em> method of Plischke et al. 2013, SALib's "
        "<code>delta.analyze</code>), so no special sampling design is needed. For each parameter the members are "
        "sorted by its value and split into classes of equal size, about N<sup>1/4</sup> of them for N members "
        "up to a few hundred (at most 48). The first-order Sobol index <code>S1</code> is the share of the response variance "
        "explained by that parameter alone: the variance of the class means of the response, weighted by class "
        "size, over the total variance. Borgonovo's <code>delta</code> asks how much the whole distribution of "
        "the response shifts when the parameter is known: half the area between the response's density within a "
        "class and its overall density (both kernel estimates), averaged over the classes. It is 0 for a "
        "parameter without influence and 1 at most, and catches effects on the spread or shape of the response "
        "that leave its variance alone. Both are dimensionless. The <code>S1</code> of all parameters add up to "
        "1 at most; what is missing is due to interactions between parameters.</p>",
        "<h4>3. Confidence bands</h4>",
        f"<p>The members are resampled with replacement {resamples} times; the shaded bands are the index plus "
        f"or minus {conf:.0%} confidence half-widths, {_z(conf):.2f} times the standard deviation over the "
        "resamples. With a few dozen members, <code>delta</code> of a parameter without influence can still come "
        "out clearly above zero; compare parameters with that in mind.</p>",
        "<h4>4. Summary columns</h4>",
        "<p>In the per-complex CSV files, <code>S1_mean</code> and <code>delta_mean</code> average the index over "
        "all time steps, <code>S1_last</code> and <code>delta_last</code> over the last 10.</p>",
    ]
    complexes = settings.get("complexes", [])
    if complexes:
        rows = "".join(
            f"<tr><td>{escape(str(c['rgi_id']))}</td><td>{escape(str(c['kind']))}</td>"
            f"<td>{c['n_files']}</td><td>{c.get('n_members', '')}</td><td>{c.get('n_parameters', '')}</td>"
            f"<td>{escape(str(c.get('skipped', '')))}</td></tr>"
            for c in complexes
        )
        parts.append(
            '<div class="table-wrap"><table class="table"><thead><tr><th>complex</th><th>kind</th>'
            "<th>member files</th><th>members analyzed</th><th>parameters</th><th>skipped</th></tr></thead>"
            f"<tbody>{rows}</tbody></table></div>"
        )
    return {"title": "How the indices are calculated", "html": "".join(parts)}


def _z(conf_level: float) -> float:
    """
    Two-sided standard-normal quantile of a confidence level.

    Parameters
    ----------
    conf_level : float
        Confidence level, e.g. 0.95.

    Returns
    -------
    float
        The quantile, 1.96 for 0.95.
    """
    return float(NormalDist().inv_cdf(0.5 + conf_level / 2))


def collect_sensitivity(output_dir: Path, base: Path) -> dict[str, Any]:
    """
    Gather the sensitivity-index outputs for their page.

    Parameters
    ----------
    output_dir : Path
        ``<report>/sensitivity_indices``.
    base : Path
        Report root.

    Returns
    -------
    dict
        ``tables`` (none), ``sections`` per complex with its figures, and
        ``n_glaciers``.
    """
    # Figures only: the summary and per-glacier index tables run to hundreds of
    # rows and stay in the CSV files next to the figures.
    glaciers = _glacier_sections(output_dir, base)
    return {
        "tables": [],
        "sections": glaciers,
        "n_glaciers": len(glaciers),
        "notes": [sensitivity_method_note(output_dir)],
    }


COLLECTORS = {
    "importance": collect_importance,
    "sensitivity": collect_sensitivity,
    "glaciers": collect_usgs_glaciers,
    "stakes": collect_usgs_stakes,
}


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
    target: str = sensitivity_indices.DEFAULT_TARGET,
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
    target : str, optional
        Variable the sensitivity indices are computed for.

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
            target=target,
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
    if key == "sensitivity":
        command = f"pism-glacier-sensitivity-indices {run_dir} --output-path {out} --target {opts['target']}"

        def call_sensitivity():
            return sensitivity_indices.run_pipeline(
                run_dir, output_path=out, target=opts["target"], n_jobs=opts["n_jobs"] if opts["n_jobs"] > 1 else None
            )

        return command, call_sensitivity
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
                notes=collected.get("notes", []),
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
    # The bundled logo is named relative to the package in the help rather
    # than by wherever the package happens to be installed.
    parser.add_argument(
        "--logo",
        default=SUPPRESS,
        help="Logo image copied into the report (SVG or PNG). "
        f"(default: {LOGO.relative_to(TEMPLATE_DIR.parents[2])})",
    )
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
    add_ntasks_argument(
        parser,
        default=1,
        help_text="USGS tools and sensitivity indices: worker processes.",
        dest="n_jobs",
        deprecated=("--n-jobs",),
    )
    parser.add_argument(
        "--target", default=sensitivity_indices.DEFAULT_TARGET, help="Sensitivity indices: variable to analyze."
    )
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
            target=args.target,
        )
    setup = [
        ("run directory", str(run_dir)),
        ("observations", str(data_path or run_dir)),
        ("USGS cache", str(usgs_data_path)),
        ("fudge factors", ", ".join(f"{f:g}" for f in args.fudge_factors)),
        ("reduction", args.reduction),
        ("command", " ".join(["pism-s4f-report"] + list(argv if argv is not None else sys.argv[1:]))),
    ]
    pages = render(
        run_dir,
        output_path,
        results,
        title=args.title,
        logo=Path(getattr(args, "logo", LOGO)).expanduser(),
        setup=setup,
    )
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
