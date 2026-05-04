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

    - **Console** (root): WARNING level so per-step INFO chatter does not
      interleave with tqdm progress bars on the terminal.
    - **File**: INFO level with full
      ``%(asctime)s - %(name)s - %(levelname)s - %(message)s`` format for
      detailed log files.

    The ``pism_terra`` logger is set to INFO so all package-level INFO records
    are captured by the file handler (and bubble up to console only at WARNING+).

    Parameters
    ----------
    log_file : str or Path
        Path to the log file. Parent directories must already exist.
    """
    file_format = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    logging.basicConfig(level=logging.WARNING, format=file_format)
    for handler in logging.root.handlers:
        handler.setLevel(logging.WARNING)

    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter(file_format))

    pkg_logger = logging.getLogger("pism_terra")
    pkg_logger.setLevel(logging.INFO)
    pkg_logger.addHandler(file_handler)

    # Quiet noisy third-party INFO chatter (CDS API, AWS, etc.) on the console.
    for name in ("cdsapi", "datapi", "multiurl", "ecmwf", "botocore", "s3transfer", "boto3"):
        logging.getLogger(name).setLevel(logging.WARNING)
