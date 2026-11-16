# USGS benchmark glaciers

The USGS Benchmark Glacier Project publishes glacier-wide winter, summer and
annual mass balances, plus the stake measurements they are built from, for a
handful of North American glaciers: Gulkana, Wolverine, Lemon Creek and Taku
in Alaska, and South Cascade and Sperry in the contiguous United States. The
ScienceBase release *Compiled Input Data and Glacier-Wide Mass Balances*
([doi:10.5066/F7HD7SRF](https://doi.org/10.5066/F7HD7SRF)) is the source for
four commands that compare it with PISM output or plot it on its own.

| Command | Reads | Compares with PISM |
| --- | --- | --- |
| `pism-glacier-usgs-benchmark-glaciers` | glacier-wide `Bw`, `Bs`, `Ba` | `scalar_G_*.nc` |
| `pism-glacier-usgs-benchmark-stakes` | stake `bw`, `ba`, sub-seasonal `db` | `spatial_*.nc` |
| `pism-glacier-usgs-benchmark-annual` | glacier-wide `Ba` | — (observations only) |
| `pism-glacier-usgs-generate-geopackage` | stakes and sites | — (export only) |

All four download the release on first use through
{py:func}`pism_terra.download.download_usgs_benchmark` into `--data-path`
(default `~/usgs_benchmark`) and reuse it afterwards; `--force-overwrite`
fetches it again. The shared reading, RGI matching, time bookkeeping and
skill statistics live in `pism_terra.glacier.usgs`.

## Matching glaciers to RGI IDs

The release names glaciers, not RGI IDs. The two model comparisons place each
glacier in the RGI v7 glacier (`-G-`) outline holding most of its stakes
({py:func}`~pism_terra.glacier.usgs.match_rgi_ids`); a glacier with no stake
inside any outline falls back to the nearest outline within 5 km, or is left
unmatched. The outlines of RGI region 01 are downloaded into
`<data-path>/rgi_archive` unless `--rgi-glacier-file` names a local file.
South Cascade and Sperry lie in RGI region 02, which Snow4Flow does not model,
so the model comparisons skip them
(`SKIPPED_GLACIERS` in `pism_terra.glacier.usgs`). Kahiltna and Kennicott
have point data only and no glacier-wide solution.

The match table (`*_rgi_match.csv`) is written to `--output-path`, and every
glacier's figures and NetCDF go to an `<rgi_id>/` sub-directory below it.

## Glacier-wide balances

```bash
pism-glacier-usgs-benchmark-glaciers RUN_DIR --output-path usgs_glaciers
```

`RUN_DIR` is searched recursively for `scalar_G_*.nc` (the per-glacier output
of `pism-postprocess-scalar`); omit it to plot the observations alone. The
modelled `tendency_of_ice_mass` of the matched glacier is divided by the
modelled ice area to give a specific balance in m w.e. per year and drawn
against the observed annual balance, with a right axis reading the same
values in Gt/yr at the mean observed area. Several runs are drawn as their
median and 5–95 % range.

When the model output is monthly, winter and summer balances are also
modelled: `tendency_of_ice_mass_due_to_surface_mass_flux` is integrated over
each year's *measured* season boundaries (`Bw_Date`, `Ba_Date` of the
release), with months straddling a boundary counted in proportion
({py:func}`~pism_terra.glacier.usgs.integrate_rate`). The surface flux rather
than the total tendency is the stake network's counterpart, since flow and
discharge move mass without a stake seeing it.

Outputs:

- `<rgi_id>/usgs_benchmark_<Glacier>_<rgi_id>.png` / `.pdf` — the time series.
- `<rgi_id>/usgs_benchmark_<Glacier>_<rgi_id>.nc` — observed and modelled
  series, with the skill as attributes.
- `usgs_benchmark_skill.csv` — Pearson `r` (from three years on), MAE and bias
  per glacier and season.

`--uncertainty` draws a constant one-sigma error bar where the release has
none, `--ntasks` sets the number of worker processes reading the model files,
and `--plot-start`/`--plot-end` limit the balance-year axis.

## Stake balances

```bash
pism-glacier-usgs-benchmark-stakes RUN_DIR --output-path usgs_stakes
```

`RUN_DIR` is searched for monthly `spatial_*.nc`. The modelled surface mass
balance (`surface_accumulation_flux` minus `surface_runoff_flux`) is sampled
at every stake (`--method nearest` or `linear`) and integrated over the
interval between the two visits of each measurement. A winter or annual
balance is measured against the previous summer surface, so by default the
modelled balance starts where the model's own surface is lowest after the
previous fall visit rather than at the visit itself
(`--start-at summer-surface`; `visit` starts at the visit).

Outputs per glacier, under `<rgi_id>/`:

- `usgs_benchmark_stakes_<Glacier>_<rgi_id>.png` — one panel per site.
- `…_scatter.png` — modelled against observed balances.
- `…_gradient.png` and `.csv` — balance against elevation per season, with
  two-piece linear fits hinged at the ELA.
- `….nc` and `….csv` — the sampled and observed balances.

and at the top level `usgs_benchmark_stakes_skill.csv` (per site and pooled
per glacier), `usgs_benchmark_stakes_gradients.csv` and
`usgs_benchmark_stakes_model_start.csv` (how far each modelled interval was
moved to the summer surface).

## Annual and cumulative balances

```bash
pism-glacier-usgs-benchmark-annual --output-path usgs_annual
```

Reproduces Fig. 2 of the USGS
[2025 Benchmark Glaciers Executive Summary](https://www.usgs.gov/programs/ecosystems-land-change-science-program/science/2025-usgs-benchmark-glaciers-executive),
with one figure per glacier rather than stacked panels: the annual balance
`Ba` as bars shaded by sign and magnitude (left axis), the cumulative balance
as a black line (right axis, 15 times the left so the zeros line up), and
the latest year's balance, its rank from highest to lowest, its percentile
and the total mass change over the record.

The percentile follows the USGS convention, `(n − rank) / (n − 1)`: the share
of the other years ranked below the latest one, so a record low is 0 and a
record high 100. No model output or RGI matching is involved, so every
glacier with a glacier-wide solution is plotted, South Cascade and Sperry
included; `--glaciers` picks a subset by the release's names (`Gulkana
LemonCreek …`). All figures share their axis ranges; `--plot-start` and
`--plot-end` override the year axis.

Outputs:

- `usgs_annual_balance_<Glacier>.png` / `.pdf`.
- `usgs_annual_balance_summary.csv` — latest year, `Ba`, rank, number of
  years, percentile, cumulative balance and first year per glacier.

## Stake GeoPackage

```bash
pism-glacier-usgs-generate-geopackage usgs_stakes.gpkg
```

Writes the release's point measurements, joined with the site coordinates, to
a GeoPackage with three layers: `sites` (the measurement sites), `stakes`
(seasonal `bw`/`ba` per site and year) and `subseasonal` (`db` between two
visits). Units stay m w.e.; the release's two date notations become proper
date fields.

## In the Snow4Flow report

`pism-s4f-report` runs `pism-glacier-usgs-benchmark-glaciers` and
`pism-glacier-usgs-benchmark-stakes` on a project and adds a page for each to
its HTML report; see {doc}`../reference/cli`.

See the {doc}`../reference/api` for the functions behind each command.
