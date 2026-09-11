"""Time one open_mfdataset configuration in a fresh process. Prints a JSON line."""

import argparse
import glob
import json
import os
import resource
import sys
import time
from collections.abc import Callable
from functools import partial

p = argparse.ArgumentParser()
p.add_argument("dir")
p.add_argument("--engine", default="netcdf4")
p.add_argument("--parallel", default="none", choices=["none", "threads", "processes", "distributed"])
p.add_argument("--preprocess", default="none", choices=["none", "full", "light"])
p.add_argument("--combine", default="by_coords", choices=["by_coords", "nested"])
p.add_argument("--fast-combine", action="store_true", help="data_vars/coords=minimal, compat/join=override")
p.add_argument("--chunks", default="{}", help="'{}', 'auto', 'none'")
p.add_argument("--no-decode", action="store_true")
p.add_argument("--n", type=int, default=0, help="use only first n files")
p.add_argument("--workers", type=int, default=8)
p.add_argument("--page-buf", type=int, default=0, help="h5netcdf page_buf_size in bytes (paged files only)")
p.add_argument("--load-var", default="", help="after open, .load() this variable and time it")
a = p.parse_args()


def main(a):
    import dask
    import xarray as xr

    files = sorted(
        glob.glob(os.path.join(a.dir, "*.nc")), key=lambda f: int(os.path.basename(f).split("_uq_")[1].split("_")[0])
    )
    if a.n:
        files = files[: a.n]

    def preprocess_light(ds):
        """What preprocess_netcdf would shrink to if uq_id were already in the file."""
        ds = ds.drop_dims("cfg", errors="ignore")  # PISM's |S1 config var
        return ds

    pre: Callable | None
    if a.preprocess == "full":
        sys.path.insert(0, "/Users/andy/base/pism-terra")
        from pism_terra.processing import preprocess_netcdf

        d0 = xr.open_dataset(files[0], engine=a.engine).dims
        if "uq_id" in d0:
            pre = partial(preprocess_netcdf, uq_dim=None)
        elif "exp_id" in d0:
            pre = partial(preprocess_netcdf, exp_dim="run_id", uq_dim=None)
        else:
            pre = preprocess_netcdf
    elif a.preprocess == "light":
        pre = preprocess_light
    else:
        pre = None

    kw = dict(engine=a.engine, combine=a.combine)
    if a.combine == "nested":
        kw["concat_dim"] = "uq_id"
    if a.fast_combine:
        kw.update(data_vars="minimal", coords="minimal", compat="override", join="override")
    if a.chunks == "auto":
        kw["chunks"] = "auto"
    elif a.chunks == "none":
        kw["chunks"] = None
    else:
        kw["chunks"] = {}
    if a.no_decode:
        kw.update(decode_cf=False)
    if a.page_buf and a.engine == "h5netcdf":
        kw["driver_kwds"] = {"page_buf_size": a.page_buf}

    client = None
    ctx = dask.config.set(scheduler="synchronous")
    if a.parallel == "threads":
        ctx = dask.config.set(scheduler="threads")
    elif a.parallel == "processes":
        ctx = dask.config.set(scheduler="processes", num_workers=a.workers)
    elif a.parallel == "distributed":
        from dask.distributed import Client, LocalCluster

        cluster = LocalCluster(
            n_workers=a.workers, threads_per_worker=1, processes=True, dashboard_address=None, silence_logs=40
        )
        client = Client(cluster)
        ctx = dask.config.set(scheduler="distributed")

    with ctx:
        t0 = time.perf_counter()
        ds = xr.open_mfdataset(files, preprocess=pre, parallel=a.parallel != "none", **kw)
        t_open = time.perf_counter() - t0
        t_load = None
        if a.load_var:
            t0 = time.perf_counter()
            ds[a.load_var].load()
            t_load = time.perf_counter() - t0
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e9  # bytes on macOS
    out = dict(
        layout=os.path.basename(a.dir.rstrip("/")),
        n=len(files),
        engine=a.engine,
        parallel=a.parallel,
        preprocess=a.preprocess,
        combine=a.combine + ("+fast" if a.fast_combine else ""),
        chunks=a.chunks,
        decode=not a.no_decode,
        t_open=round(t_open, 2),
        t_load=None if t_load is None else round(t_load, 2),
        rss_gb=round(rss, 2),
        dims={k: int(v) for k, v in ds.sizes.items()},
        ntasks=len(ds["usurf"].data.dask) if hasattr(ds["usurf"].data, "dask") else 0,
    )
    print(json.dumps(out))
    if client:
        client.close()
        cluster.close()


if __name__ == "__main__":
    main(a)
