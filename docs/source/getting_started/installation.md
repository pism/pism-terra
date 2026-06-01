# How to install

pism-terra installs into a conda environment that already provides PISM and a
working GDAL stack.

## Just `pism-terra` conda environment

```bash
git clone https://github.com/pism/pism-terra.git
cd pism-terra
conda env create -f environment.yml
conda activate pism-terra
python -m pip install -e .
```

For the documentation build add the `docs` extras:

```bash
python -m pip install -e ".[docs]"
```

## With `pism`

You can install PISM requisites using conda
  
```bash
git clone https://github.com/pism/pism.git
cd pism
git checkout feature/inverse
conda env create -f environment.yml
conda activate pism
```

Then build PISM::
  
```bash
CMAKE_BUILD_PARALLEL_LEVEL=8 python -m pip install --no-build-isolation -v .
```

```bash
cd ..
git clone https://github.com/pism/pism-terra.git
cd pism-terra
git checkout aaschwanden/summer-school
python -m pip install .
```


## Required `LD_PRELOAD` workaround for libz

On affected conda-forge environments `liborc.so` (a transitive dependency of
`pyarrow`) re-exports zlib's deflate API as global symbols. When libgdal calls
`deflateInit_` during GeoTIFF compression — including the in-memory
`MemoryFile` writes that `dem_stitcher` uses — the dynamic loader binds it to
liborc's copy instead of `libz.so.1`. The two implementations have
incompatible heap layouts; the next unrelated `_int_malloc` then aborts with:

```
Fatal glibc error: malloc.c:4241 (_int_malloc): assertion failed
```

Workaround: preload conda's libz so it wins symbol resolution. Wire it up as
an env activation hook so you don't forget:

```bash
mkdir -p $CONDA_PREFIX/etc/conda/activate.d $CONDA_PREFIX/etc/conda/deactivate.d
cat > $CONDA_PREFIX/etc/conda/activate.d/zz-libz-preload.sh <<'EOF'
export _PISM_TERRA_OLD_LD_PRELOAD="$LD_PRELOAD"
export LD_PRELOAD="$CONDA_PREFIX/lib/libz.so.1${LD_PRELOAD:+:$LD_PRELOAD}"
EOF
cat > $CONDA_PREFIX/etc/conda/deactivate.d/zz-libz-preload.sh <<'EOF'
export LD_PRELOAD="$_PISM_TERRA_OLD_LD_PRELOAD"
unset _PISM_TERRA_OLD_LD_PRELOAD
EOF
```

Verify that the leak is still present:

```bash
nm -D --defined-only $CONDA_PREFIX/lib/liborc.so | grep -c "T deflate"
```

A non-zero result means the `LD_PRELOAD` is still required. Once it returns
`0` (after a future liborc rebuild) the preload can be removed.

Tracking issue: https://github.com/conda-forge/orc-feedstock/issues

```{admonition} TODO
- Pin minimum supported Python and PISM versions.
- Note macOS-specific quirks (if any).
- Add an "Install from PyPI" section once a release is cut.
```
