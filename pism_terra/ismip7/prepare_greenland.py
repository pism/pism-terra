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

"""
Prepare ISMIP7 Greenland data sets.
"""

import time
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from pathlib import Path
from typing import Any, Sequence, cast

import rioxarray
import toml
import xarray as xr
from dask.distributed import Client, as_completed
from pyfiglet import Figlet

from pism_terra.download import download_earthaccess, download_netcdf

version = "1.3"
url = f"https://g-ab4495.8c185.08cc.data.globus.org/ISMIP6/ISMIP7_Prep/Observations/Greenland/GreenlandObsISMIP7-v{version}.nc"
# ds = download_netcdf(url)


gcms = ["CESM2-WACCM"]
forcing = {"historical": {"start": 1980, "end": 2015}, "projection": {"start": 2015, "end": 2100}}


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    """
    Run MALA sampling for the PDD posterior and assemble an ArviZ InferenceData.

    This function is the programmatic entry point. It parses command-line style
    arguments, runs the sampler, and returns a dictionary containing the resulting
    `arviz.InferenceData` object (and optionally other intermediate artifacts).

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command-line arguments **excluding** the program name (i.e., like
        ``sys.argv[1:]``). If ``None`` (default), arguments are taken from the
        current process' ``sys.argv[1:]``. Passing ``argv=[]`` is recommended
        when calling from a Jupyter notebook to avoid ipykernel arguments.

    Returns
    -------
    dict[str, Any]
        Results dictionary. At minimum this includes:

        - ``"idata"``: :class:`arviz.InferenceData`
          InferenceData containing ``posterior`` and ``prior`` groups, and
          optionally a ``sample_stats`` group (e.g., log-probability, step sizes,
          acceptance indicators) if available.

        Additional keys may be included depending on the configuration and
        sampler implementation (for example, raw prior/posterior arrays, runtime
        metadata, or filenames of saved outputs).
    """

    parser = ArgumentParser()
    parser.add_argument("CONFIG_FILE", nargs=1)
    parser.add_argument("DATA_PATH", nargs=1)
    parser.add_argument("OUTPUT_PATH", nargs=1)
    args = parser.parse_args(list(argv) if argv is not None else None)

    config_file = args.CONFIG_FILE[0]
    data_path = Path(args.DATA_PATH[0])
    output_path = Path(args.OUTPUT_PATH[0])
    output_path.mkdir(parents=True, exist_ok=True)

    f = Figlet(font="standard")
    banner = f.renderText("pism-terra")
    print("=" * 80)
    print(banner)
    print("=" * 80)
    print("Preparing ISMIP7 Greenland data")
    print("-" * 80)
    print("")

    config = toml.loads(Path(config_file).read_text("utf-8"))

    climate(data_path, output_path, config)

    return {"config": config}


# def bedmachine(output_path: Path | str) -> Path | str:
#     """
#     Prepare BedMachine.
#     """

#     filter_str = "v6.nc"
#     doi = "10.5067/6B6B225B8V2D"
#     p = download_earthaccess(doi=doi, filter_str=filter_str, result_dir=result_dir)[0]

#     return Path(p)


def _process_single_forcing(
    gcm: str,
    forcing: str,
    base_path: Path,
    output_path: Path,
    epoch: str,
    version: str,
    start_year: int,
    end_year: int,
    short_hand: str,
    fields: list[str],
    ismip7_to_pism: dict[str, str],
) -> Path:
    """
    Process a single GCM/forcing combination.

    Parameters
    ----------
    gcm : str
        GCM name.
    forcing : str
        Forcing type.
    base_path : Path
        Base path to input data.
    output_path : Path
        Output directory.
    epoch : str
        Epoch name (e.g., "historical").
    version : str
        Version string (e.g., "v1").
    start_year : int
        Start year for time selection.
    end_year : int
        End year for time selection.
    short_hand : str
        Short hand identifier for forcing.
    fields : list[str]
        List of climate fields to process.
    ismip7_to_pism : dict[str, str]
        Variable name mapping from ISMIP7 to PISM conventions.

    Returns
    -------
    Path
        Path to the output NetCDF file.
    """
    dss = []
    for m_var in fields:
        p = base_path / Path(gcm) / Path(epoch) / Path(short_hand) / Path(m_var) / Path(version)
        urls = list(p.glob(f"{m_var}_{gcm}_{epoch}_{short_hand}_*.nc"))
        ds = xr.open_mfdataset(
            urls,
            parallel=True,
            engine="h5netcdf",
            chunks={"time": 12, "y": -1, "x": -1},
        ).sel({"time": slice(str(start_year), str(end_year))})
        dss.append(ds)

    ds = xr.merge(dss)
    ds = ds.rename_vars({k: v for k, v in ismip7_to_pism.items() if k in ds})
    ds.rio.write_crs("EPSG:3413", inplace=True)

    # Build encoding for chunked, compressed output
    encoding = {}
    for var in ds.data_vars:
        if var != "spatial_ref":
            encoding[var] = {
                "zlib": True,
                "complevel": 2,
            }

    output_file = output_path / Path(f"{forcing}_{epoch}_{gcm}_{version}_{start_year}_{end_year}.nc")
    ds.to_netcdf(output_file, encoding=encoding, engine="h5netcdf")

    return output_file


def climate(
    base_path: Path | str,
    output_path: Path | str,
    config: dict,
    epoch: str = "historical",
    n_workers: int = 4,
) -> list[Path | str]:
    """
    Process climate data for all GCMs and forcings in parallel.

    Parameters
    ----------
    base_path : Path or str
        Base path to input data.
    output_path : Path or str
        Output directory.
    config : dict
        Configuration dictionary.
    epoch : str, optional
        Epoch name, by default "historical".
    n_workers : int, optional
        Number of dask workers, by default 4.

    Returns
    -------
    list[Path | str]
        List of output file paths.
    """
    start_time = time.perf_counter()

    base_path = Path(base_path)
    output_path = Path(output_path)

    version = "v" + str(config[epoch]["version"])
    start_year = config[epoch]["start_year"]
    end_year = config[epoch]["end_year"]
    short_hand = config["forcing"]["climate"]["short_hand"]
    ismip7_to_pism = config["ismip7_to_pism"]
    fields = config["forcing"]["climate"]["fields"]

    # Build list of tasks
    tasks = []
    for gcm in config[epoch]["gcms"]:
        for forcing, forcing_dict in config["forcing"].items():
            tasks.append((gcm, forcing))

    # Process in parallel using dask.distributed
    with Client(n_workers=n_workers, threads_per_worker=2) as client:
        print(f"Dask dashboard: {client.dashboard_link}")

        futures = []
        for gcm, forcing in tasks:
            future = client.submit(
                _process_single_forcing,
                gcm,
                forcing,
                base_path,
                output_path,
                epoch,
                version,
                start_year,
                end_year,
                short_hand,
                fields,
                ismip7_to_pism,
            )
            futures.append(future)

        # Collect results as they complete
        o_files = []
        for future in as_completed(futures):
            output_file = future.result()
            print(f"Completed: {output_file}")
            o_files.append(output_file)

    elapsed = time.perf_counter() - start_time
    print(f"Total processing time: {elapsed:.2f} seconds")

    return o_files


def cli(argv: Sequence[str] | None = None) -> int:
    """
    Console entry point.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Command-line arguments (without the program name). If None, uses sys.argv.

    Returns
    -------
    int
        Exit code (0 for success).
    """
    _ = main(argv=argv)
    return 0


if __name__ == "__main__":
    __spec__ = None  # type: ignore
    raise SystemExit(cli())
