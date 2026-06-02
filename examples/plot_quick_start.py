"""
Quickstart.

Quick start
===========

Placeholder sphinx-gallery example. Replace with a self-contained pism-terra
workflow once the gallery is wired up against pinned test data.
"""

# %%
# This example is intentionally minimal so the docs build succeeds out of
# the box. A real example should:
#
# 1. Stage a small glacier (e.g. RGI2000-v7.0-C-01-04374).
# 2. Plot the boot file and one climate field.
# 3. Render a tiny time-series from a saved PISM output.

import numpy as np

x = np.linspace(0, 1, 64)
y = x**2
print("placeholder gallery example —", y[-3:])
