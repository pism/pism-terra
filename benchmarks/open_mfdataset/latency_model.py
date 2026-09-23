"""Emulate a latency-bound filesystem (Lustre / S3): add a fixed delay per read() and time
xr.open_dataset (h5netcdf engine, python file object) for the first N files of each layout."""

import glob
import io
import sys
import time

import xarray as xr

DELAY = float(sys.argv[1]) if len(sys.argv) > 1 else 0.001
N = 20


class SlowFile(io.FileIO):
    def readinto(self, b):
        time.sleep(DELAY)
        return super().readinto(b)


def bench(layout, **kw):
    files = sorted(glob.glob(f"data/{layout}/*.nc"))[:N]
    t = time.perf_counter()
    for f in files:
        fo = SlowFile(f, "rb")
        ds = xr.open_dataset(fo, engine="h5netcdf", **kw)
        ds.close()
        fo.close()
    return (time.perf_counter() - t) / N


print(f"per-read delay {DELAY*1000:.1f} ms; mean open time per file over {N} files (h5netcdf engine)")
for label, layout, kw in [
    ("A  PISM today (2330 cfg attrs)", "A_pism_today", {}),
    ("F  uq_id + JSON config, no attrs", "F_uq_id_json_cfg", {}),
    ("H  A repacked paged, page_buf 4 MiB", "H_pism_today_paged1M", dict(driver_kwds={"page_buf_size": 4 << 20})),
    ("I  F repacked paged, page_buf 4 MiB", "I_uq_id_json_paged1M", dict(driver_kwds={"page_buf_size": 4 << 20})),
]:
    print(f"  {label:40s} {bench(layout, **kw)*1000:8.0f} ms/file")
