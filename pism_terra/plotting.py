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
Plotting methods.
"""

import numpy as np


def blend_multiply(rgb: np.ndarray, intensity: np.ndarray) -> np.ndarray:
    """
    Combine an RGB image with an intensity map using "overlay" blending.

    This function combines an RGB image with an intensity map using "overlay" blending. The RGB image
    and the intensity map are combined by multiplying the RGB values by the intensity values. The resulting
    image is then scaled to have values between 0 and 1.

    Parameters
    ----------
    rgb : np.ndarray
        An (M, N, 3) RGB array of floats ranging from 0 to 1. This represents the color image.
    intensity : np.ndarray
        An (M, N, 1) array of floats ranging from 0 to 1. This represents the grayscale image.

    Returns
    -------
    np.ndarray
        An (M, N, 3) RGB array representing the combined images. The values in the array range from 0 to 1.
    """

    alpha = rgb[..., -1, np.newaxis]
    img_scaled = np.clip(rgb[..., :3] * intensity, 0.0, 1.0)
    return img_scaled * alpha + intensity * (1.0 - alpha)
