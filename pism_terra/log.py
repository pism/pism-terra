# Copyright (C) 2025, 2026 Andy Aschwanden
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
Centralized logging configuration.
"""

from __future__ import annotations

import logging
from pathlib import Path


def setup_logging(log_file: str | Path) -> None:
    """
    Configure logging for pism-terra CLI entry points.

    Sets up two handlers:

    - **Console** (root): INFO level with ``%(message)s`` format for clean
      terminal output.
    - **File**: INFO level with full
      ``%(asctime)s - %(name)s - %(levelname)s - %(message)s`` format for
      detailed log files.

    Parameters
    ----------
    log_file : str or Path
        Path to the log file. Parent directories must already exist.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    file_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(file_format))
    logging.getLogger("pism_terra").addHandler(file_handler)
