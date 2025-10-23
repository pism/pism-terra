# Copyright (C) 2025 Andy Aschwanden
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
"""
Colormaps.
"""
from importlib.resources import files
from pathlib import Path

import matplotlib.pylab as plt
import numpy as np
from matplotlib import colors


def qgis2cmap(
    filename: Path | str,
    num_levels: int = 256,  # Renamed from N to num_levels
    name: str = "my colormap",
) -> colors.LinearSegmentedColormap:
    """
    Read a colormap exported from QGIS raster layers and return a colormap.

    Parameters
    ----------
    filename : Path or str
        The path to the QGIS colormap file.
    num_levels : int, optional
        The number of RGB quantization levels, by default 256.
    name : str, optional
        The name of the colormap, by default "my colormap".

    Returns
    -------
    matplotlib.colors.LinearSegmentedColormap
        The matplotlib colormap.
    """
    m_data = np.loadtxt(filename, skiprows=2, delimiter=",")[:, :-1]
    values_scaled = (m_data[:, 0] - np.min(m_data[:, 0])) / (np.max(m_data[:, 0]) - np.min(m_data[:, 0]))
    colors_scaled = m_data[:, 1::] / 255.0
    m_colors = [(values_scaled[k], colors_scaled[k]) for k in range(len(values_scaled))]
    cmap = colors.LinearSegmentedColormap.from_list(name, m_colors, N=num_levels)

    return cmap


def register_colormaps(path: str | Path | None = None) -> None:
    """
    Register colormaps from text files.

    Parameters
    ----------
    path : str, Path, optional
        The directory where the colormap text files are located. If not provided, the
        'pism_ragis.data' directory is used.

    Examples
    --------
    >>> register_colormaps()
    >>> register_colormaps('/path/to/colormap/files')
    """
    if path is not None:
        cmap_files = Path(path).glob("*.txt")
    else:
        cmap_files = Path(str(files("pism_terra.data").joinpath("*.txt"))).parent.glob("*.txt")
    for cmap_file in cmap_files:
        name = cmap_file.name.removesuffix(".txt")
        cmap = qgis2cmap(cmap_file, name=name)
        plt.colormaps.register(cmap)
