"""Run pism-s4f-report on a synthetic project, with the USGS tools stubbed."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from pism_terra.glacier import s4f_report

RGI = "RGI2000-v7.0-C-01-00001"


def _grid(n: int, step: float) -> dict:
    """
    Projected coordinates with units.

    Parameters
    ----------
    n : int
        Points per axis.
    step : float
        Spacing in metres.

    Returns
    -------
    dict
        Coordinates for :class:`xarray.Dataset`.
    """
    axis = np.arange(n) * step
    return {"x": ("x", axis, {"units": "m"}), "y": ("y", axis, {"units": "m"})}


@pytest.fixture(name="project")
def fixture_project(tmp_path: Path) -> Path:
    """
    One glacier with three dh members, observations and a uq.csv.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.

    Returns
    -------
    Path
        Project directory.
    """
    rng = np.random.default_rng(1)
    truth = -2.0 + 0.3 * rng.normal(size=(16, 16))
    (tmp_path / RGI / "input").mkdir(parents=True)
    xr.Dataset(
        {"dh": (("y", "x"), truth, {"units": "m"}), "dh_err": (("y", "x"), np.full((16, 16), 0.5), {"units": "m"})},
        coords=_grid(16, 100.0),
    ).to_netcdf(tmp_path / RGI / "input" / f"obs_{RGI}.nc")
    dh_dir = tmp_path / RGI / "output" / "dh"
    dh_dir.mkdir(parents=True)
    for member in "012":
        xr.Dataset(
            {"usurf": (("time", "y", "x"), (truth[::2, ::2] + (int(member) - 1))[None], {"units": "m"})},
            coords={"time": [np.datetime64("2010-01-01")], **_grid(8, 200.0)},
        ).to_netcdf(dh_dir / f"dh_{RGI}_id_0_uq_{member}_2000-01-01_2020-01-01.nc")
    pd.DataFrame({"uq": [0, 1, 2], "surface.pdd.factor_ice": [0.004, 0.006, 0.008]}).to_csv(
        tmp_path / RGI / "output" / "uq.csv", index=False
    )
    return tmp_path


def _stub_usgs(module, monkeypatch: pytest.MonkeyPatch, prefix: str, fail: bool = False) -> None:
    """
    Replace a USGS tool's pipeline by one that writes a figure and a skill table.

    Parameters
    ----------
    module : module
        ``usgs_benchmark_glaciers`` or ``usgs_benchmark_stakes``.
    monkeypatch : pytest.MonkeyPatch
        Fixture.
    prefix : str
        File-name prefix of the tool's outputs.
    fail : bool, optional
        Raise instead of writing.
    """

    def fake(run_dir, *, output_dir, **kwargs):  # pylint: disable=unused-argument
        """
        Write one figure and a skill table, or raise.

        Parameters
        ----------
        run_dir : Path
            Ignored.
        output_dir : Path
            Where the outputs go.
        **kwargs : Any
            Ignored.

        Returns
        -------
        pandas.DataFrame
            Empty.
        """
        if fail:
            raise RuntimeError("no scalar files")
        glacier = Path(output_dir) / RGI
        glacier.mkdir(parents=True, exist_ok=True)
        (glacier / f"{prefix}_Gulkana_{RGI}.png").write_bytes(b"\x89PNG\r\n\x1a\n")
        pd.DataFrame({"glacier": ["Gulkana"], "rmse": [0.4]}).to_csv(
            Path(output_dir) / f"{prefix}_skill.csv", index=False
        )
        return pd.DataFrame()

    monkeypatch.setattr(module, "run_pipeline", fake)


def test_report_runs_all_tools_and_renders_pages(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    All three tools run, a failing one is reported, and the pages link the figures and logo.

    Parameters
    ----------
    project : Path
        Synthetic project fixture.
    monkeypatch : pytest.MonkeyPatch
        Fixture.
    """
    _stub_usgs(s4f_report.usgs_benchmark_glaciers, monkeypatch, "usgs_benchmark")
    _stub_usgs(s4f_report.usgs_benchmark_stakes, monkeypatch, "usgs_benchmark_stakes", fail=True)
    out = project / "report"
    pages = s4f_report.main([str(project), "--output-path", str(out), "--no-bootstrap", "--fudge-factors", "3"])

    assert [p.name for p in pages] == [
        "index.html",
        "importance_sampling.html",
        "usgs_glaciers.html",
        "usgs_stakes.html",
    ]
    index = (out / "index.html").read_text(encoding="utf-8")
    assert 'src="_static/logo.svg"' in index and (out / "_static" / "logo.svg").is_file()
    assert (out / "_static" / "report.css").is_file()
    assert index.count('class="status ok"') == 2 and 'class="status failed"' in index
    assert "no scalar files" in index
    importance = (out / "importance_sampling.html").read_text(encoding="utf-8")
    assert f"importance_sampling/{RGI}/importance_usurf_ff_3.png" in importance
    assert "Summary" in importance and "<table" in importance
    glaciers = (out / "usgs_glaciers.html").read_text(encoding="utf-8")
    assert f"usgs_glaciers/{RGI}/usgs_benchmark_Gulkana_{RGI}.png" in glaciers
    stakes = (out / "usgs_stakes.html").read_text(encoding="utf-8")
    assert "This step failed" in stakes and "RuntimeError" in stakes


def test_no_run_renders_existing_outputs_and_skip_marks_tools(project: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """
    ``--no-run`` reuses outputs on disk and ``--skip`` marks a tool as skipped.

    Parameters
    ----------
    project : Path
        Synthetic project fixture.
    monkeypatch : pytest.MonkeyPatch
        Fixture.
    """
    _stub_usgs(s4f_report.usgs_benchmark_glaciers, monkeypatch, "usgs_benchmark")
    _stub_usgs(s4f_report.usgs_benchmark_stakes, monkeypatch, "usgs_benchmark_stakes")
    out = project / "report"
    s4f_report.main(
        [str(project), "--output-path", str(out), "--no-bootstrap", "--skip", "stakes", "--title", "Test report"]
    )
    index = (out / "index.html").read_text(encoding="utf-8")
    assert 'class="status skipped"' in index and "Test report" in index

    s4f_report.main([str(project), "--output-path", str(out), "--no-run"])
    index = (out / "index.html").read_text(encoding="utf-8")
    assert index.count('class="status ok"') == 2 and 'class="status skipped"' in index
    assert s4f_report.cli([str(project), "--output-path", str(out), "--no-run"]) == 0
