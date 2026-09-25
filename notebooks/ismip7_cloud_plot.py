"""
Regional mass balance of the ISMIP7 cloud runs against Mankoff, step by step.

The work lives in :mod:`pism_terra.ismip7.greenland.mass_balance`; this is
the notebook-friendly walk through it, one stage per cell. The console
script does the same in one go::

    pism-ismip7-greenland-mass-balance --project 2026_10_ismip7_core_ctrl figures/
    pism-ismip7-greenland-mass-balance --root 2026_10_ismip7_core_ctrl/output figures/

``ROOT`` is either the bucket (read anonymously) or a local copy of the same
tree; nothing else changes.
"""

from pathlib import Path

from dask.distributed import Client

from pism_terra.ismip7.greenland import mass_balance as mb

# %% Where the run is: the cloud bucket, or the same tree on disk.
ROOT = "s3://pism-cloud-data/ismip7_production/2026_10_ismip7_core_ctrl/output"
# ROOT = "2026_10_ismip7_core_ctrl/output"
OUTPUT = Path("figures")
REFERENCE_YEAR = "1985"

# %% A local cluster; the Dask progress display follows the work on it.
client = Client()
print(client.dashboard_link)

# %% Open the submission as one lazy ensemble on (gcm_id, ssp_id, time, y, x).
paths = mb.find_files(ROOT, mb.DEFAULT_VARIABLES)
ds = mb.open_submission(paths)
print(ds)

# %% Integrate the fluxes over the basins: the one pass over the data.
outline = mb.resolve_outline(ROOT, None)
regions = mb.compute_regions(ds, outline, reference_year=REFERENCE_YEAR)
OUTPUT.mkdir(parents=True, exist_ok=True)
regions.to_netcdf(OUTPUT / "regional_mass_balance.nc")
print(regions)

# %% The observations, on the same footing.
mankoff = mb.load_mankoff(f"{ROOT}/observations/{mb.DEFAULT_MANKOFF}", reference_year=REFERENCE_YEAR)

# %% One panel per basin.
mb.plot_regions(regions, mankoff, OUTPUT / "regional_mass_balance.png", sigma=1)

# %% Later sessions: plot again from the saved series without touching the ensemble.
# regions = xr.open_dataset(OUTPUT / "regional_mass_balance.nc")
