# Copyright (C) 2026 Andy Aschwanden
#
# This file is part of pism-terra.
#
# PISM-TERRA is free software; you can redistribute it and/or modify it under the
# terms of the GNU General Public License as published by the Free Software
# Foundation; either version 3 of the License, or (at your option) any later
# version.
#
# PISM-TERRA is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
# FOR A PARTICULAR PURPOSE.  See the GNU General Public License for more
# details.
#
# You should have received a copy of the GNU General Public License
# along with PISM; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

# pylint: disable=too-many-locals,too-many-statements,too-many-arguments,too-many-positional-arguments,too-many-branches
"""
Describe an ISMIP7 perturbed parameter ensemble (PPE) as an Excel appendix.

The ISMIP7 protocol asks each group for a spreadsheet that links every
``set_counter`` to the parameter and modelling choices of that run. The
runners write :data:`MEMBERS_CSV` beside each run with exactly that
information and copy the run config and UQ TOML next to it. This module turns
a directory holding those files -- a local output path or an S3 prefix of
PISM-Cloud jobs -- into a workbook in the style of the ISMIP7
``AppendixPPE.xlsx``:

- ``PPE``: one row per ``set_counter`` (counter, experiment, start and end
  year, ESM, member ids), whether the run ended successfully and the last
  year it wrote, a caption describing the ensemble and a legend of the
  physics every member shares, read off the run config. A run "completed"
  when its ISMIP7 output reaches the config's end year; a PISM state saved
  by an aborting leg marks it "failed".
- ``Members``: the sampled value of every perturbed parameter per member and
  how many of the member's runs completed.
- ``Priors``: distribution, bounds, posterior percentiles and the ensemble
  range of every perturbed parameter.

Cells hold values, not formulas, so the file is complete as written and
needs no spreadsheet application to recalculate it.
"""

from __future__ import annotations

import logging
import re
import tomllib
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import Any, Sequence

import fsspec
import pandas as pd
from openpyxl import Workbook
from openpyxl.comments import Comment
from openpyxl.styles import Alignment, Border, Font, Side
from openpyxl.utils import get_column_letter

logger = logging.getLogger(__name__)

#: Per-run record of what each ``set_counter`` was, written beside the run
#: scripts by the ISMIP7 runners. The protocol's counter "links to entries in
#: a spreadsheet specifying parameter and modelling choices for this particular
#: experiment"; this is that spreadsheet.
MEMBERS_CSV = "ismip7_members.csv"

#: Columns that identify the run, before the sampled parameters.
MEMBER_ID_COLUMNS = (
    "set_counter",
    "ism_member_id",
    "forcing_member_id",
    "esm_id",
    "experiment_id",
    "set_id",
    "uq_draw",
)
#: Columns :func:`run_status` adds to the member table.
STATUS_COLUMNS = ("status", "last_output_year", "output_files")

#: ISMIP7 output files: ``output/<domain>/<group>/<model>/<set>/<counter>/<var>_..._<y0>-<y1>.nc``.
_ISMIP7_OUTPUT = re.compile(
    r"(^|/)output/[^/]+/[^/]+/[^/]+/[^/]+/(?P<counter>[^/]+)/[^/]+_(?P<y0>\d{4})-(?P<y1>\d{4})\.nc$"
)
#: PISM state a leg saved when it aborted: ``output/state/state_..._id_<esm>_uq_<draw>_..._failed.nc``.
_FAILED_STATE = re.compile(r"(^|/)output/state/[^/]*_id_(?P<esm>.+?)_uq_(?P<draw>\d+)_[^/]*_failed\.nc$")
#: How deep the listing goes: the PISM-Cloud layout is ``<job>/output/<5 dirs>/<file>``.
_MAXDEPTH = 8

#: Posterior sample the ice-dynamics parameters are drawn from when the
#: ensemble is built with ``--posterior-file``; looked for when none is given.
DEFAULT_POSTERIOR = Path(__file__).resolve().parents[1] / "uq" / "X_gris_posterior_1e6_samples_11_2021.csv.gz"
DEFAULT_POSTERIOR_REFERENCE = (
    "Aschwanden, A. & Brinkerhoff, D. J. (2022): Calibrated mass loss predictions for the Greenland Ice Sheet, "
    "Geophys. Res. Lett. 49, e2022GL099058"
)
DEFAULT_SPINUP_NOTE = "All members share the same thermal spin-up; no separate spin-up per member."

#: Description, units and Excel number format of the parameters a PPE
#: perturbs. Unknown parameters fall back to ``--pism-config-cdl`` or blanks.
PARAMETER_DOCS: dict[str, tuple[str, str, str]] = {
    "stress_balance.sia.enhancement_factor": ("Flow enhancement factor for the SIA", "1", "0.000"),
    "stress_balance.ssa.Glen_exponent": ("Glen exponent in the ice flow law for the SSA", "1", "0.000"),
    "stress_balance.ssa.enhancement_factor": ("Flow enhancement factor for the SSA", "1", "0.000"),
    "basal_resistance.pseudo_plastic.q": ("Exponent of the pseudo-plastic basal sliding law", "1", "0.000"),
    "basal_yield_stress.mohr_coulomb.till_effective_fraction_overburden": (
        "δ: effective till pressure as a fraction of overburden (Bueler & van Pelt, 2015)",
        "1",
        "0.0000",
    ),
    "basal_yield_stress.mohr_coulomb.topg_to_phi.phi_min": (
        "Lower till friction angle (bed below topg_min)",
        "degrees",
        "0.00",
    ),
    "basal_yield_stress.mohr_coulomb.topg_to_phi.phi_max": (
        "Upper till friction angle (bed above topg_max)",
        "degrees",
        "0.00",
    ),
    "basal_yield_stress.mohr_coulomb.topg_to_phi.topg_min": ("Bed elevation at which phi_min applies", "m", "0"),
    "basal_yield_stress.mohr_coulomb.topg_to_phi.topg_max": ("Bed elevation at which phi_max applies", "m", "0"),
    "calving.vonmises_calving.sigma_max": ("Maximum tensile stress of the von Mises calving law", "Pa", "#,##0"),
    "calving.thickness_calving.threshold": ("Thickness below which floating ice calves", "m", "0"),
    "frontal_melt.routing.parameter_a": (
        "Parameter A of the frontal-melt law, eqn. 1 of Xu et al. (2013)",
        "m^-α day^(α-1) °C^-β",
        "0.000E+00",
    ),
    "frontal_melt.routing.parameter_b": (
        "Parameter B of the frontal-melt law, eqn. 1 of Xu et al. (2013)",
        "m day^(α-1) °C^-β",
        "0.000",
    ),
    "ocean.plume.melt_rate_parameter": ("PLUME melt-rate parameter (Pelle et al., 2019)", "m yr^-1 K^-2", "0.000"),
    "ocean.plume.length_scale_factor": (
        "Multiplier on the PLUME plume length scale l (Pelle et al., 2019, eqn. 11)",
        "1",
        "0.000",
    ),
    "ocean.plume.drag_coefficient": ("PLUME drag coefficient", "1", "0.00000"),
    "ocean.plume.power_alpha": (
        "Plume-velocity exponent of the subglacial-discharge melt rate (Pelle et al., 2023, eqns. 13–14)",
        "1",
        "0.000",
    ),
    "ocean.plume.power_beta": (
        "Exponent on the thermal forcing in the PLUME ambient melt function (Pelle et al., 2019, eqn. 10)",
        "1",
        "0.000",
    ),
    "surface.pdd.factor_ice": ("Positive-degree-day factor for ice", "m K^-1 day^-1", "0.0000"),
    "surface.pdd.factor_snow": ("Positive-degree-day factor for snow", "m K^-1 day^-1", "0.0000"),
}

#: Human-readable names of the model choices a run config makes, for the legend.
MODEL_NAMES: dict[str, dict[str, str]] = {
    "stress_balance": {
        "hybrid": "hybrid SIA+SSA dynamics",
        "ssa+sia": "hybrid SIA+SSA dynamics",
        "blatter": "Blatter-Pattyn dynamics",
        "sia": "SIA dynamics",
        "ssa": "SSA dynamics",
    },
    "energy": {
        "enthalpy": "enthalpy-based thermodynamics",
        "cold": "cold-ice thermodynamics",
        "none": "isothermal ice",
    },
    "hydrology": {
        "routing": "routing subglacial hydrology",
        "null": "no subglacial water transport (null hydrology)",
        "distributed": "distributed subglacial hydrology",
    },
    "ocean": {
        "plume_sgd": "PLUME sub-shelf melt with subglacial discharge",
        "plume": "PLUME sub-shelf melt",
        "picop_sgd": "PICOP sub-shelf melt with subglacial discharge",
        "picop": "PICOP sub-shelf melt",
        "pico": "PICO sub-shelf melt",
        "th": "three-equation (TH) sub-shelf melt",
        "given": "prescribed sub-shelf melt",
        "constant": "constant sub-shelf melt",
    },
    "frontal_melt": {
        "routing": "frontal melt after Xu et al. (2013) from routed subglacial discharge",
        "discharge_given": "frontal melt after Xu et al. (2013) from prescribed discharge",
        "off": "no frontal melt",
        "none": "no frontal melt",
    },
    "bed_deformation": {"lc": "Lingle-Clark bed deformation", "iso": "ELRA isostasy", "none": "fixed bedrock"},
    "surface": {
        "ismip7": "ISMIP7 surface forcing (SMB and surface temperature with elevation gradients)",
        "ismip7_forcing": "ISMIP7 surface forcing with force-to-thickness",
        "given": "prescribed SMB and surface temperature",
        "pdd": "positive-degree-day surface model",
        "debm_simple": "dEBM-simple surface model",
    },
}
CALVING_NAMES = {
    "vonmises_calving": "von Mises calving",
    "eigen_calving": "eigen-calving",
    "hayhurst_calving": "Hayhurst calving",
    "thickness_calving": "thickness calving",
    "float_kill": "float-kill calving",
}

# --- styles, copied from the ISMIP7 AppendixPPE workbook ----------------------
_FONT = "Calibri"
_HDR = Font(name=_FONT, size=12, bold=True)
_BODY = Font(name=_FONT, size=12)
_SMALL = Font(name=_FONT, size=10)
_NOTE = Font(name=_FONT, size=10, italic=True)
_THIN = Side(style="thin")
_HDR_BORDER = Border(top=_THIN, bottom=_THIN)
_TOP_BORDER = Border(top=_THIN)
_LEFT = Alignment(horizontal="left", vertical="center", wrap_text=True)
_RIGHT = Alignment(horizontal="right", vertical="center")
_WRAP = Alignment(wrap_text=True, vertical="center")


def open_filesystem(root: str | Path) -> tuple[Any, str]:
    """
    Open the filesystem a PPE directory lives on, locally or on S3.

    An S3 prefix is tried with the default credential chain first and
    anonymously when that fails, so public PISM-Cloud buckets work without
    AWS credentials.

    Parameters
    ----------
    root : str or Path
        Local directory or ``s3://bucket/prefix``.

    Returns
    -------
    tuple
        ``(fs, path)``: the fsspec filesystem and the path within it.
    """
    root = str(root)
    fs, path = fsspec.core.url_to_fs(root)
    if root.startswith("s3://"):
        try:
            fs.ls(path)
        except Exception as exc:  # pylint: disable=broad-exception-caught
            logger.info("Listing %s with credentials failed (%s); retrying anonymously", root, exc)
            fs, path = fsspec.core.url_to_fs(root, anon=True)
    return fs, path.rstrip("/")


def find_files(listing: Sequence[str], root: str, pattern: str | re.Pattern[str]) -> list[str]:
    """
    Pick the files of a directory listing whose path matches a regular expression.

    Parameters
    ----------
    listing : sequence of str
        Paths below ``root`` (one ``fs.find`` call, shared by every lookup).
    root : str
        Directory the listing was taken of.
    pattern : str or re.Pattern
        Regular expression matched against the path relative to ``root``.

    Returns
    -------
    list of str
        Matching paths, sorted.
    """
    regex = re.compile(pattern)
    return sorted(p for p in listing if regex.search(p[len(root) :].lstrip("/")))


def read_members(fs: Any, root: str, listing: Sequence[str]) -> pd.DataFrame:
    """
    Collect the member tables below a PPE directory into one, one row per counter.

    A PISM-Cloud ensemble has one job directory per run and every job carries
    the whole member table, so the tables are concatenated and de-duplicated
    by ``set_counter``; a counter whose rows disagree between files is an error.

    Parameters
    ----------
    fs : fsspec filesystem
        Filesystem ``root`` lives on.
    root : str
        Directory to search for :data:`MEMBERS_CSV`.
    listing : sequence of str
        Paths below ``root``.

    Returns
    -------
    pandas.DataFrame
        Member table sorted by ``set_counter``.

    Raises
    ------
    FileNotFoundError
        If no member table is found.
    ValueError
        If two files disagree about the same ``set_counter``, or a table lacks
        the identifying columns.
    """
    paths = find_files(listing, root, rf"(^|/){re.escape(MEMBERS_CSV)}$")
    if not paths:
        raise FileNotFoundError(f"no {MEMBERS_CSV} below {root}")
    logger.info("Reading %d member table(s)", len(paths))
    frames = []
    for p in paths:
        with fs.open(p, "rb") as fh:
            frames.append(pd.read_csv(fh))
    table = pd.concat(frames, ignore_index=True)
    rounded = table.copy()
    numeric = rounded.select_dtypes("number").columns
    rounded[numeric] = rounded[numeric].round(10)
    distinct = rounded.drop_duplicates()
    counts = distinct["set_counter"].value_counts()
    conflicts = sorted(counts[counts > 1].index)
    if conflicts:
        raise ValueError(f"member tables below {root} disagree about set_counter {conflicts}")
    table = table.loc[distinct.index].sort_values("set_counter").reset_index(drop=True)
    missing = [c for c in MEMBER_ID_COLUMNS if c not in table.columns]
    if missing:
        raise ValueError(f"{MEMBERS_CSV} lacks the columns {missing}")
    return table


def read_toml(fs: Any, paths: Sequence[str], what: str) -> dict[str, Any]:
    """
    Read one TOML out of the copies a PPE directory holds.

    Every job directory carries its own copy; they must agree.

    Parameters
    ----------
    fs : fsspec filesystem
        Filesystem the paths live on.
    paths : sequence of str
        Candidate files (all copies of the same TOML).
    what : str
        What the file is, for messages.

    Returns
    -------
    dict
        Parsed TOML, empty when ``paths`` is empty.

    Raises
    ------
    ValueError
        If the copies differ.
    """
    if not paths:
        logger.warning("No %s found; the workbook will not describe it", what)
        return {}
    texts = set()
    for p in paths:
        with fs.open(p, "rb") as fh:
            texts.add(fh.read().decode("utf-8"))
    if len(texts) > 1:
        raise ValueError(f"the {len(paths)} copies of the {what} differ; pass one explicitly")
    logger.info("%s: %s", what, paths[0])
    return tomllib.loads(texts.pop())


def _section_value(config: dict[str, Any], section: str, key: str, default: Any = None) -> Any:
    """
    Read ``key`` from a config table that may spell it plain or section-prefixed.

    Parameters
    ----------
    config : dict
        Parsed run config.
    section : str
        Top-level table name.
    key : str
        Key inside the table, without the ``section.`` prefix.
    default : Any, optional
        Value when the key is absent.

    Returns
    -------
    Any
        The value, or ``default``.
    """
    table = config.get(section) or {}
    for k in (key, f"{section}.{key}"):
        if k in table:
            return table[k]
    return default


def run_years(config: dict[str, Any]) -> tuple[int | None, int | None]:
    """
    Start and end year of a run from its config.

    The start is the spin-up leg's ``campaign.init_start`` when there is one,
    else ``time.start``. An end date on 1 January names the last complete
    year, so ``2101-01-01`` reports 2100.

    Parameters
    ----------
    config : dict
        Parsed run config.

    Returns
    -------
    tuple of int or None
        ``(start_year, end_year)``; ``None`` where the config does not say.
    """

    def year(value: Any, inclusive_end: bool = False) -> int | None:
        """
        Year of a date string, counting a 1 January end as the previous year.

        Parameters
        ----------
        value : Any
            ``YYYY`` or ``YYYY-MM-DD``; anything else gives ``None``.
        inclusive_end : bool, default False
            Treat the date as an exclusive end bound.

        Returns
        -------
        int or None
            The year.
        """
        if value is None:
            return None
        m = re.match(r"^\s*(\d{4})(?:-(\d{2})-(\d{2}))?", str(value))
        if m is None:
            return None
        y = int(m.group(1))
        if inclusive_end and m.group(2) == "01" and m.group(3) == "01":
            y -= 1
        return y

    start = year(_section_value(config, "campaign", "init_start")) or year(_section_value(config, "time", "start"))
    end = year(_section_value(config, "time", "end"), inclusive_end=True)
    return start, end


def physics_legend(config: dict[str, Any]) -> list[str]:
    """
    Describe the model choices every member shares, from the run config.

    Parameters
    ----------
    config : dict
        Parsed run config.

    Returns
    -------
    list of str
        One line per model component; empty when the config is unknown.
    """
    lines: list[str] = []
    for section in ("stress_balance", "energy", "hydrology", "ocean", "frontal_melt", "bed_deformation", "surface"):
        model = _section_value(config, section, "model")
        if model is None:
            continue
        lines.append(MODEL_NAMES.get(section, {}).get(str(model), f"{section}: {model}"))
    methods = str(_section_value(config, "calving", "methods") or "")
    if methods:
        parts = []
        for method in methods.split(","):
            name = CALVING_NAMES.get(method.strip(), method.strip())
            if method.strip() == "thickness_calving":
                threshold = _section_value(config, "calving", "thickness_calving.threshold")
                if threshold is not None:
                    name += f" ({threshold} m)"
            parts.append(name)
        lines.append(" plus ".join(parts))
    if _section_value(config, "campaign", "retreat_file"):
        lines.append("prescribed front retreat (CalFin, 1972–2019)")
    resolution = _section_value(config, "grid", "resolution")
    if resolution:
        lines.append(f"{resolution} horizontal grid")
    return lines


def run_status(listing: Sequence[str], root: str, members: pd.DataFrame, end_year: int | None) -> pd.DataFrame:
    """
    Whether each run ended successfully, from the files it left behind.

    Parameters
    ----------
    listing : sequence of str
        Paths below ``root``.
    root : str
        Ensemble directory.
    members : pandas.DataFrame
        Member table (one row per ``set_counter``), to map a failed PISM state,
        named by ESM and draw, back to its counter.
    end_year : int or None
        Last year a complete run writes; ``None`` when the config did not say.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``set_counter`` with :data:`STATUS_COLUMNS`: ``status`` is
        ``completed``, ``incomplete``, ``failed`` or ``no output``. A counter
        that failed once and later completed (a resubmitted job) counts as
        completed.
    """
    last_year: dict[str, int] = {}
    n_files: dict[str, int] = {}
    for p in find_files(listing, root, _ISMIP7_OUTPUT):
        m = _ISMIP7_OUTPUT.search(p[len(root) :].lstrip("/"))
        assert m is not None
        counter = m.group("counter")
        last_year[counter] = max(last_year.get(counter, 0), int(m.group("y1")))
        n_files[counter] = n_files.get(counter, 0) + 1
    by_draw = {(str(r["esm_id"]), int(r["uq_draw"])): str(r["set_counter"]) for _, r in members.iterrows()}
    failed: set[str] = set()
    for p in find_files(listing, root, _FAILED_STATE):
        m = _FAILED_STATE.search(p[len(root) :].lstrip("/"))
        assert m is not None
        counter = by_draw.get((m.group("esm"), int(m.group("draw"))))
        if counter is not None:
            failed.add(counter)
    rows = []
    for counter in members["set_counter"]:
        year = last_year.get(counter)
        if year is not None and end_year is not None and year >= end_year:
            status = "completed"
        elif counter in failed:
            status = "failed"
        elif year is not None:
            status = "incomplete" if end_year is not None else f"output through {year}"
        else:
            status = "no output"
        rows.append(
            {
                "set_counter": counter,
                "status": status,
                "last_output_year": year,
                "output_files": n_files.get(counter, 0),
            }
        )
    return pd.DataFrame(rows).set_index("set_counter")


def cdl_docs(path: str | Path) -> dict[str, tuple[str, str]]:
    """
    Parameter descriptions and units from a ``pism_config.cdl``.

    Parameters
    ----------
    path : str or Path
        PISM's master config.

    Returns
    -------
    dict
        ``{name: (doc, units)}``.
    """
    docs: dict[str, str] = {}
    units: dict[str, str] = {}
    pattern = re.compile(r'^\s*pism_config:(\S+?)_(doc|units)\s*=\s*"(.*)"\s*;\s*$')
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            m = pattern.match(line)
            if m:
                (docs if m.group(2) == "doc" else units)[m.group(1)] = m.group(3).replace('\\"', '"')
    return {name: (docs.get(name, ""), units.get(name, "")) for name in set(docs) | set(units)}


def parameter_doc(name: str, cdl: dict[str, tuple[str, str]] | None) -> tuple[str, str, str]:
    """
    Description, units and number format of a parameter.

    Parameters
    ----------
    name : str
        Dotted PISM parameter name.
    cdl : dict or None
        Output of :func:`cdl_docs`, consulted for parameters not in :data:`PARAMETER_DOCS`.

    Returns
    -------
    tuple of str
        ``(description, units, number_format)``.
    """
    if name in PARAMETER_DOCS:
        return PARAMETER_DOCS[name]
    if cdl and name in cdl:
        doc, unit = cdl[name]
        return doc, unit, "General"
    return "", "", "General"


def prior_bounds(spec: dict[str, Any]) -> tuple[float | None, float | None, str]:
    """
    Lower and upper bound of a sampled prior, when it has them.

    Parameters
    ----------
    spec : dict
        One ``DistSpec`` table of the UQ TOML.

    Returns
    -------
    tuple
        ``(lower, upper, extra)``: bounds for ``uniform``/``randint``, else
        ``None``, and the remaining parameters as text (e.g. ``"a=-1, b=1"``).
    """
    dist = str(spec.get("distribution", "")).lower()
    loc, scale = spec.get("loc"), spec.get("scale")
    if dist == "uniform" and loc is not None and scale is not None:
        return float(loc), float(loc) + float(scale), ""
    if dist == "randint" and "low" in spec and "high" in spec:
        return float(spec["low"]), float(spec["high"]) - 1, ""
    extra = ", ".join(f"{k}={v}" for k, v in spec.items() if k not in ("distribution", "loc", "scale"))
    return None, None, extra


def _short_reference(reference: str) -> str:
    """
    The author-year part of a citation, for running text.

    Parameters
    ----------
    reference : str
        Full citation, ``"Authors (year): title, journal"``.

    Returns
    -------
    str
        Everything before the first colon.
    """
    return reference.split(":")[0].strip()


def build_table(
    root: str | Path,
    output: str | Path,
    config_file: str | None = None,
    uq_file: str | None = None,
    posterior_file: str | Path | None = None,
    pism_config_cdl: str | Path | None = None,
    posterior_reference: str = DEFAULT_POSTERIOR_REFERENCE,
    spinup_note: str = DEFAULT_SPINUP_NOTE,
    completed_only: bool = False,
) -> Path:
    """
    Write the PPE appendix workbook for one ensemble directory.

    Parameters
    ----------
    root : str or Path
        Local directory or ``s3://`` prefix holding the ensemble: the runner's
        member tables, ``config/*.toml`` and ``uq/*.toml`` copies, and the
        ISMIP7 output tree.
    output : str or Path
        Workbook to write.
    config_file : str or None, optional
        Run config to describe instead of the copy found below ``root``.
    uq_file : str or None, optional
        UQ TOML (the priors) instead of the copy found below ``root``.
    posterior_file : str or Path or None, optional
        Posterior sample the non-prior parameters were drawn from; default
        :data:`DEFAULT_POSTERIOR` when it exists. Its percentiles go into the
        ``Priors`` sheet.
    pism_config_cdl : str or Path or None, optional
        PISM's ``pism_config.cdl``, for the description and units of parameters
        :data:`PARAMETER_DOCS` does not know.
    posterior_reference : str, optional
        Citation for the posterior sample.
    spinup_note : str, optional
        Sentence on the spin-up, appended to the caption and legend.
    completed_only : bool, default False
        List only the runs that ended successfully (and the members with at
        least one such run) instead of every run with its status.

    Returns
    -------
    Path
        The written workbook.

    Raises
    ------
    ValueError
        If ``completed_only`` leaves no run.
    """
    fs, path = open_filesystem(root)
    logger.info("Listing %s", root)
    listing = fs.find(path, maxdepth=_MAXDEPTH)
    members = read_members(fs, path, listing)
    if config_file is not None:
        config = tomllib.loads(Path(config_file).read_text("utf-8"))
    else:
        config = read_toml(fs, find_files(listing, path, r"(^|/)config/[^/]+\.toml$"), "run config")
    if uq_file is not None:
        uq = tomllib.loads(Path(uq_file).read_text("utf-8"))
    else:
        uq = read_toml(fs, find_files(listing, path, r"(^|/)uq/[^/]+\.toml$"), "UQ TOML")
    priors = {k: v for k, v in uq.items() if isinstance(v, dict) and "distribution" in v}
    if posterior_file is None and DEFAULT_POSTERIOR.exists():
        posterior_file = DEFAULT_POSTERIOR
    cdl = cdl_docs(pism_config_cdl) if pism_config_cdl else None

    start_year, end_year = run_years(config)
    status = run_status(listing, path, members, end_year)
    members = members.merge(status, left_on="set_counter", right_index=True, how="left")
    n_runs_all = len(members)
    if completed_only:
        members = members[members["status"] == "completed"].reset_index(drop=True)
        if members.empty:
            raise ValueError(f"no run below {root} completed (output through {end_year})")
    n_completed = int((members["status"] == "completed").sum())
    logger.info("%d of %d runs completed", n_completed, n_runs_all)

    params = [c for c in members.columns if c not in MEMBER_ID_COLUMNS + STATUS_COLUMNS]
    prior_params = [p for p in params if p in priors]
    posterior_params = [p for p in params if p not in priors]
    quantiles = None
    joint_draw = False
    n_posterior = 0
    if posterior_params and posterior_file is not None:
        logger.info("Reading posterior sample %s", posterior_file)
        posterior = pd.read_csv(posterior_file, index_col=0)
        have = [p for p in posterior_params if p in posterior.columns]
        quantiles = posterior[have].quantile([0.025, 0.5, 0.975])
        n_posterior = len(posterior)
        # one complete row per member means a joint draw; say so only when true
        joint = pd.merge(
            members[have].round(6).drop_duplicates(), posterior[have].round(6).drop_duplicates(), how="inner", on=have
        )
        joint_draw = len(joint) == members["ism_member_id"].nunique()
    ordered = posterior_params + prior_params
    short_ref = _short_reference(posterior_reference)

    per_member = members.sort_values("ism_member_id").groupby("ism_member_id").first()
    n_members = len(per_member)
    esms = list(dict.fromkeys(members["esm_id"]))
    forcing_ids = sorted(set(members["forcing_member_id"]))
    set_ids = sorted(set(members["set_id"]))
    experiments = list(dict.fromkeys(members["experiment_id"]))
    first, last = members["set_counter"].iloc[0], members["set_counter"].iloc[-1]

    wb = Workbook()

    # ------------------------------------------------------------------ PPE
    ws = wb.active
    ws.title = "PPE"
    headers = [
        "counter_id",
        "experiment_id",
        "Start year",
        "End year",
        "ESM_id",
        "ism_member_id",
        "forcing_member_id",
        "Status",
        "Last output year",
    ]
    for j, h in enumerate(headers, 1):
        c = ws.cell(2, j, h)
        c.font, c.border, c.alignment = _HDR, _HDR_BORDER, Alignment(vertical="center")
    for i, (_, r) in enumerate(members.iterrows(), start=3):
        values = [
            r["set_counter"],
            str(r["experiment_id"]).upper(),
            start_year,
            end_year,
            r["esm_id"],
            r["ism_member_id"],
            r["forcing_member_id"],
            r["status"],
            None if pd.isna(r["last_output_year"]) else int(r["last_output_year"]),
        ]
        for j, v in enumerate(values, 1):
            c = ws.cell(i, j, v)
            c.font, c.alignment = _BODY, _LEFT
            if i == 3:
                c.border = _TOP_BORDER
        ws.row_dimensions[i].height = 15.5
    cap_row = 3 + len(members)
    sampling = []
    if posterior_params:
        how = "a joint draw from" if joint_draw else "drawn from"
        sampling.append(f"the {len(posterior_params)} ice-dynamics parameters are {how} the posterior of {short_ref}")
    if prior_params:
        sampling.append(
            f"the {len(prior_params)} remaining parameters are Latin-hypercube samples of the priors in sheet 'Priors'"
        )
    if completed_only:
        completion = f"Only the {n_completed} of {n_runs_all} runs that ended successfully (output through {end_year}) are listed. "
    elif end_year is not None:
        completion = (
            f"{n_completed} of {n_runs_all} runs ended successfully (output through {end_year}); see column Status. "
        )
    else:
        completion = ""
    caption = (
        completion + f"Experiments {first}–{last} form a {n_members}-member perturbed parameter ensemble "
        f"(set_id = {', '.join(set_ids)}, ism_member_id = {per_member.index[0]}–{per_member.index[-1]}), each member run "
        f"once per ESM ({', '.join(esms)}; {', '.join(e.upper() for e in experiments)}; forcing_member_id = "
        f"{', '.join(forcing_ids)}). Every member perturbs the {len(params)} parameters listed in sheet 'Members'"
        + (": " + "; ".join(sampling) if sampling else "")
        + ".\n"
        + spinup_note
    )
    ws.merge_cells(start_row=cap_row, start_column=1, end_row=cap_row, end_column=len(headers))
    c = ws.cell(cap_row, 1, caption)
    c.font, c.alignment = _BODY, Alignment(wrap_text=True, vertical="top")
    ws.row_dimensions[cap_row].height = 125

    legend_col = len(headers) + 2
    legend = [
        (
            ", ".join(set_ids),
            f"{per_member.index[0]}–{per_member.index[-1]}",
            f"{n_members} joint parameter draws; same model physics in every member:",
        )
    ]
    legend += [("", "", line) for line in physics_legend(config)]
    legend += [("", ", ".join(forcing_ids), "forcing member(s) of these runs")]
    legend += [("", "", spinup_note)]
    for k, (a, b, text) in enumerate(legend):
        r = 3 + k
        for off, v in ((0, a), (1, b), (2, text)):
            if v:
                c = ws.cell(r, legend_col + off, v)
                c.font, c.alignment = _SMALL, Alignment(vertical="center", wrap_text=off == 2)
    if len(legend) > 1:
        ws.merge_cells(start_row=3, start_column=legend_col, end_row=2 + len(legend), end_column=legend_col)
    ws.cell(3, legend_col).alignment = Alignment(horizontal="center", vertical="center")
    for r in range(3, 3 + len(legend)):
        for off in range(3):
            ws.cell(r, legend_col + off).border = Border(
                left=_THIN if off == 0 else None,
                right=_THIN if off == 2 else None,
                top=_THIN if r == 3 else None,
                bottom=_THIN if r == 2 + len(legend) else None,
            )
    for col, w in zip("ABCDEFGHI", (12.8, 16.8, 12.8, 12.8, 16.8, 16.0, 18.0, 13.0, 16.0)):
        ws.column_dimensions[col].width = w
    ws.column_dimensions[get_column_letter(legend_col)].width = 7.3
    ws.column_dimensions[get_column_letter(legend_col + 1)].width = 12
    ws.column_dimensions[get_column_letter(legend_col + 2)].width = 95
    ws.freeze_panes = "A3"

    # -------------------------------------------------------------- Members
    wm = wb.create_sheet("Members")
    counters = members.pivot(index="ism_member_id", columns="esm_id", values="set_counter")
    completed_per_member = members[members["status"] == "completed"].groupby("ism_member_id").size()
    head = ["ism_member_id", "uq_draw"] + [f"counter_id ({esm})" for esm in esms] + ["Completed runs"] + ordered
    first_param_col = 4 + len(esms)
    if posterior_params:
        c = wm.cell(1, first_param_col, f"Ice dynamics — draw from the posterior of {short_ref}")
        c.font = _HDR
        if len(posterior_params) > 1:
            wm.merge_cells(
                start_row=1,
                start_column=first_param_col,
                end_row=1,
                end_column=first_param_col + len(posterior_params) - 1,
            )
    if prior_params:
        col0 = first_param_col + len(posterior_params)
        c = wm.cell(1, col0, "Latin-hypercube samples of the priors in sheet 'Priors'")
        c.font = _HDR
        if len(prior_params) > 1:
            wm.merge_cells(start_row=1, start_column=col0, end_row=1, end_column=col0 + len(prior_params) - 1)
    for j, h in enumerate(head, 1):
        c = wm.cell(2, j, h)
        c.font, c.border, c.alignment = _HDR, _HDR_BORDER, _WRAP
    for j, name in enumerate(ordered, start=first_param_col):
        c = wm.cell(3, j, parameter_doc(name, cdl)[1])
        c.font, c.alignment, c.border = Font(name=_FONT, size=10, italic=True), _WRAP, Border(bottom=_THIN)
    first_m = 4
    for i, (mid, r) in enumerate(per_member.iterrows(), start=first_m):
        wm.cell(i, 1, mid).font = _BODY
        wm.cell(i, 2, int(r["uq_draw"])).font = _BODY
        for k, esm in enumerate(esms):
            counter = counters.loc[mid, esm] if esm in counters.columns else None
            wm.cell(i, 3 + k, None if pd.isna(counter) else counter).font = _BODY
        wm.cell(i, 3 + len(esms), int(completed_per_member.get(mid, 0))).font = _BODY
        for j, name in enumerate(ordered, start=first_param_col):
            c = wm.cell(i, j, float(r[name]))
            c.font, c.number_format, c.alignment = _BODY, parameter_doc(name, cdl)[2], _RIGHT
    last_m = first_m + n_members - 1
    wm.row_dimensions[2].height = 80
    for j in range(1, len(head) + 1):
        wm.column_dimensions[get_column_letter(j)].width = 22 if 3 <= j < first_param_col else 16
    wm.freeze_panes = wm.cell(first_m, first_param_col).coordinate
    c = wm.cell(
        last_m + 2,
        1,
        "uq_draw is the 0-based ensemble index of the sampler. Each ism_member_id is run once per ESM; the counter_id "
        "columns give the ISMIP7 set counter of each run and 'Completed runs' how many of them ended successfully.",
    )
    c.font = _NOTE

    # --------------------------------------------------------------- Priors
    wp = wb.create_sheet("Priors")
    ph = [
        "PISM parameter",
        "Description",
        "Units",
        "Distribution",
        "loc",
        "scale",
        "Other parameters",
        "Lower",
        "Upper",
        "Posterior 2.5%",
        "Posterior median",
        "Posterior 97.5%",
        "Ensemble min",
        "Ensemble max",
    ]
    for j, h in enumerate(ph, 1):
        c = wp.cell(2, j, h)
        c.font, c.border, c.alignment = _HDR, _HDR_BORDER, _WRAP
    wp.row_dimensions[2].height = 32
    for i, name in enumerate(ordered, start=3):
        desc, unit, fmt = parameter_doc(name, cdl)
        wp.cell(i, 1, name)
        wp.cell(i, 2, desc)
        wp.cell(i, 3, unit)
        if name in priors:
            spec = priors[name]
            wp.cell(i, 4, str(spec.get("distribution")))
            if "loc" in spec:
                wp.cell(i, 5, float(spec["loc"])).number_format = fmt
            if "scale" in spec:
                wp.cell(i, 6, float(spec["scale"])).number_format = fmt
            lower, upper, extra = prior_bounds(spec)
            if extra:
                wp.cell(i, 7, extra)
            if lower is not None:
                wp.cell(i, 8, lower).number_format = fmt
                wp.cell(i, 9, upper).number_format = fmt
        else:
            wp.cell(i, 4, f"posterior ({short_ref})" if quantiles is not None else "posterior")
            if quantiles is not None and name in quantiles.columns:
                for j, qq in zip((10, 11, 12), (0.025, 0.5, 0.975)):
                    wp.cell(i, j, float(quantiles.loc[qq, name])).number_format = fmt
        wp.cell(i, 13, float(per_member[name].min())).number_format = fmt
        wp.cell(i, 14, float(per_member[name].max())).number_format = fmt
        for j in range(1, len(ph) + 1):
            c = wp.cell(i, j)
            c.font = _BODY
            c.alignment = _WRAP if j <= 4 or j == 7 else _RIGHT
            if i == 3:
                c.border = _TOP_BORDER
    if quantiles is not None:
        wp.cell(2, 10).comment = Comment(
            f"Percentiles of the posterior sample {Path(str(posterior_file)).name} ({n_posterior} rows). "
            + (
                "Each member takes one complete row of it, so these parameters are jointly sampled."
                if joint_draw
                else ""
            ),
            "pism-terra",
        )
    notes = []
    if prior_params:
        notes.append(
            "Priors: SciPy distribution parameters as in the UQ TOML; a uniform prior spans [loc, loc + scale]. "
            "Members are a Latin-hypercube sample of the priors."
        )
    if posterior_params:
        notes.append(f"Posterior parameters: {posterior_reference}.")
    notes.append("Ensemble min/max are over the members in sheet 'Members'.")
    row = 3 + len(ordered) + 1
    for k, text in enumerate(notes):
        c = wp.cell(row + k, 1, text)
        c.font, c.alignment = _NOTE, Alignment(wrap_text=True, vertical="top")
        wp.merge_cells(start_row=row + k, start_column=1, end_row=row + k, end_column=len(ph))
        wp.row_dimensions[row + k].height = 30
    for col, w in zip("ABCDEFGHIJKLMN", (52, 60, 18, 34, 12, 12, 22, 12, 12, 14, 14, 14, 14, 14)):
        wp.column_dimensions[col].width = w
    wp.freeze_panes = "B3"

    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    wb.save(output)
    logger.info("Wrote %s: %d runs, %d members, %d parameters", output, len(members), n_members, len(params))
    return output


def main(argv: Sequence[str] | None = None) -> int:
    """
    Console entry point (``pism-ismip7-ppe-table``).

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command-line arguments without the program name; ``sys.argv`` when ``None``.

    Returns
    -------
    int
        Exit code, ``0`` on success.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = ArgumentParser(
        description="Describe an ISMIP7 perturbed parameter ensemble as an Excel appendix (values only, no formulas).",
        formatter_class=ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "PATH",
        help=f"Ensemble directory, local or s3://bucket/prefix; searched for {MEMBERS_CSV}, config/*.toml, uq/*.toml "
        "and the ISMIP7 output tree.",
    )
    parser.add_argument(
        "-o", "--output", default=None, help="Workbook to write; default <PATH basename>_ppe_table.xlsx."
    )
    parser.add_argument("--config-file", default=None, help="Run config TOML, instead of the copy found under PATH.")
    parser.add_argument(
        "--uq-file", default=None, help="UQ TOML with the priors, instead of the copy found under PATH."
    )
    parser.add_argument(
        "--posterior-file",
        default=None,
        help=f"Posterior sample the non-prior parameters were drawn from; default {DEFAULT_POSTERIOR.name} when present.",
    )
    parser.add_argument(
        "--pism-config-cdl",
        default=None,
        help="pism_config.cdl, for descriptions and units of parameters not built in.",
    )
    parser.add_argument("--posterior-reference", default=DEFAULT_POSTERIOR_REFERENCE, help="Citation of the posterior.")
    parser.add_argument("--spinup-note", default=DEFAULT_SPINUP_NOTE, help="Spin-up sentence for caption and legend.")
    parser.add_argument(
        "--completed-only",
        action="store_true",
        default=False,
        help="List only runs whose ISMIP7 output reaches the config's end year, instead of every run with a Status column.",
    )
    args = parser.parse_args(list(argv) if argv is not None else None)

    output = args.output or f"{Path(args.PATH.rstrip('/')).name}_ppe_table.xlsx"
    build_table(
        args.PATH,
        output,
        config_file=args.config_file,
        uq_file=args.uq_file,
        posterior_file=args.posterior_file,
        pism_config_cdl=args.pism_config_cdl,
        posterior_reference=args.posterior_reference,
        spinup_note=args.spinup_note,
        completed_only=args.completed_only,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
