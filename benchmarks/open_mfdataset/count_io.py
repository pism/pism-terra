"""Count POSIX-level read()/seek() calls needed to open one file with xarray (h5netcdf engine,
python file object). A hardware-independent proxy for cold-cache / Lustre / S3 open latency."""

import glob
import io
import os
import sys

import xarray as xr


class CountingFile(io.FileIO):
    def __init__(self, *a, **k):
        super().__init__(*a, **k)
        self.n_read = 0
        self.n_seek = 0
        self.bytes = 0
        self.offsets = []

    def readinto(self, b):
        n = super().readinto(b)
        self.n_read += 1
        self.bytes += n or 0
        return n

    def read(self, size=-1):
        b = super().read(size)
        self.n_read += 1
        self.bytes += len(b)
        return b

    def seek(self, off, whence=0):
        self.n_seek += 1
        r = super().seek(off, whence)
        self.offsets.append(r)
        return r


def measure(path, **kw):
    f = CountingFile(path, "rb")
    ds = xr.open_dataset(f, engine="h5netcdf", **kw)
    n_open = (f.n_read, f.n_seek, f.bytes)
    # touch one 2-D variable's first record to include one data read
    ds["usurf"].isel({d: 0 for d in ds["usurf"].dims if d not in ("y", "x")}).values
    n_all = (f.n_read, f.n_seek, f.bytes)
    ds.close()
    f.close()
    return n_open, n_all


if __name__ == "__main__":
    root = sys.argv[1]
    kw = {}
    if len(sys.argv) > 2 and sys.argv[2] == "paged":
        kw = dict(driver_kwds={"page_buf_size": 4 * 1024 * 1024})
    print(
        f"{'layout':26s} {'size MB':>8s} | {'open: reads':>11s} {'seeks':>6s} {'KB':>7s} | {'+1 record: reads':>16s} {'KB':>7s}"
    )
    for d in sorted(glob.glob(os.path.join(root, "*"))):
        files = sorted(glob.glob(os.path.join(d, "*.nc")))
        if not files:
            continue
        (r0, s0, b0), (r1, s1, b1) = measure(files[0], **kw)
        print(
            f"{os.path.basename(d):26s} {os.path.getsize(files[0])/1e6:8.1f} | {r0:11d} {s0:6d} {b0/1024:7.0f} | {r1:16d} {b1/1024:7.0f}"
        )
