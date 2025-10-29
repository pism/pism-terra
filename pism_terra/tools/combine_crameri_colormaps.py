#!/usr/bin/env python3
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

# pylint: disable=too-many-nested-blocks

"""
Combine QGIS color ramp style XMLs into one file.

Searches for all files matching *_QGIS.xml under a root directory, collects
all <colorramp> entries, and writes a single QGIS style XML suitable for
importing via QGIS Style Manager.
"""

import argparse
import sys
import xml.etree.ElementTree as ET
from copy import deepcopy
from pathlib import Path
from xml.dom import minidom


def _new_qgis_style_root() -> ET.Element:
    """
    Create a minimal QGIS style XML root element.

    Constructs a ``<qgis_style version="1">`` element containing empty
    ``<symbols/>`` and ``<colorramps/>`` children, which matches the
    structure produced by QGIS Style Manager exports.

    Returns
    -------
    xml.etree.ElementTree.Element
        Root element for a QGIS style XML document.

    Notes
    -----
    The returned element is intended to be populated with one or more
    ``<colorramp>`` children under the ``<colorramps>`` node.
    """
    root = ET.Element("qgis_style", {"version": "1"})
    ET.SubElement(root, "symbols")  # keep empty section for completeness
    ET.SubElement(root, "colorramps")
    return root


def _pretty_print(elem: ET.Element) -> str:
    """
    Convert an XML element to pretty-printed UTF-8 XML text.

    Parameters
    ----------
    elem : xml.etree.ElementTree.Element
        Root element to serialize.

    Returns
    -------
    str
        UTF-8 encoded, pretty-printed XML text.

    Raises
    ------
    xml.parsers.expat.ExpatError
        If the XML tree cannot be parsed by the pretty-printer.
    """
    rough = ET.tostring(elem, encoding="utf-8")
    return minidom.parseString(rough).toprettyxml(indent="  ", encoding="utf-8").decode("utf-8")


def _iter_colorramps_from_file(xml_path: Path):
    """
    Yield all ``<colorramp>`` elements from a QGIS style XML file.

    Parameters
    ----------
    xml_path : pathlib.Path
        Path to a QGIS style XML file (root tag ``qgis_style``).

    Yields
    ------
    tuple[xml.etree.ElementTree.Element, pathlib.Path]
        Pairs of (``<colorramp>`` element, source path).

    Notes
    -----
    - Files that are malformed XML, not ``qgis_style`` roots, or contain no
      ``<colorramps>`` section are skipped with a warning to ``stderr``.
    - The yielded elements are not deep-copied; callers should copy if they
      intend to modify them.
    """
    try:
        tree = ET.parse(xml_path)
    except ET.ParseError as e:
        print(f"[WARN] Skipping malformed XML: {xml_path} ({e})", file=sys.stderr)
        return

    root = tree.getroot()
    if root.tag != "qgis_style":
        print(f"[WARN] Not a QGIS style file (root tag={root.tag}): {xml_path}", file=sys.stderr)
        return

    colorramps = root.find("colorramps")
    if colorramps is None:
        # No ramps present; not an error
        return

    for ramp in colorramps.findall("colorramp"):
        yield ramp, xml_path


def _next_unique_name(base: str, used: set[str]) -> str:
    """
    Generate a unique name by suffixing ``_2``, ``_3``, … if needed.

    Parameters
    ----------
    base : str
        Proposed name to de-duplicate.
    used : set of str
        Names already in use.

    Returns
    -------
    str
        ``base`` if unused; otherwise the first available ``f"{base}_{k}"``.

    Examples
    --------
    >>> _next_unique_name("viridis", {"viridis"})
    'viridis_2'
    """
    if base not in used:
        return base
    i = 2
    while f"{base}_{i}" in used:
        i += 1
    return f"{base}_{i}"


def combine_qgis_colorramps(
    root_dir: Path,
    output_file: Path,
    dup_policy: str = "rename",
    glob_pattern: str = "*_QGIS.xml",
) -> int:
    """
    Combine QGIS color ramp style XML files into a single style file.

    Recursively searches ``root_dir`` for files matching ``glob_pattern``,
    extracts each ``<colorramp>`` element, resolves duplicate names according
    to ``dup_policy``, and writes a single QGIS style XML to ``output_file``.

    Parameters
    ----------
    root_dir : pathlib.Path
        Directory to search recursively for input XML files.
    output_file : pathlib.Path
        Destination XML file to write the combined style.
    dup_policy : {"keep-first", "overwrite", "rename"}, default "rename"
        Strategy for handling duplicate ``name`` attributes on ``<colorramp>``:
        - ``"keep-first"``: retain the first occurrence; skip subsequent ones.
        - ``"overwrite"``: keep only the last occurrence encountered.
        - ``"rename"``: append ``_2``, ``_3``, … to make names unique.
    glob_pattern : str, default "*_QGIS.xml"
        Glob pattern for matching input files (applied with ``Path.rglob``).

    Returns
    -------
    int
        Process exit code: ``0`` on success, nonzero on error
        (e.g., no files found or no color ramps extracted).

    Raises
    ------
    ValueError
        If ``dup_policy`` is not one of the supported values.
    OSError
        If writing ``output_file`` fails.

    Notes
    -----
    - The output file has one ``<qgis_style>`` root with a single
      ``<colorramps>`` section containing all merged ramps.
    - XMLs that are not QGIS style files (root tag mismatch) or malformed
      are skipped with a warning to ``stderr``.

    Examples
    --------
    >>> rc = combine_qgis_colorramps(
    ...     root_dir=Path("/data/ramps"),
    ...     output_file=Path("combined_colorramps.xml"),
    ...     dup_policy="rename",
    ... )
    >>> rc
    0
    """
    # Collect all matching files
    xml_files = sorted(root_dir.rglob(glob_pattern))
    if not xml_files:
        print(f"[ERROR] No files matching {glob_pattern!r} found under {root_dir}", file=sys.stderr)
        return 2

    out_root = _new_qgis_style_root()
    out_colorramps = out_root.find("colorramps")
    assert out_colorramps is not None

    used_names: set[str] = set()
    count_added = 0
    count_seen = 0

    for xml_path in xml_files:
        for ramp, src in _iter_colorramps_from_file(xml_path):
            count_seen += 1
            name = ramp.get("name", "").strip()
            if not name:
                # Fallback: derive from file stem + index
                name = xml_path.stem

            if name in used_names:
                if dup_policy == "keep-first":
                    print(
                        f"[INFO] Duplicate name '{name}' in {src.name}: keeping first, skipping this one.",
                        file=sys.stderr,
                    )
                    continue
                if dup_policy == "overwrite":
                    # Remove the previously added ramp with this name, then add the new one
                    for existing in list(out_colorramps.findall("colorramp")):
                        if existing.get("name") == name:
                            out_colorramps.remove(existing)
                            break
                    out_colorramps.append(deepcopy(ramp))
                    used_names.add(name)  # unchanged
                    count_added += 1
                elif dup_policy == "rename":
                    new_name = _next_unique_name(name, used_names)
                    ramp_copy = deepcopy(ramp)
                    ramp_copy.set("name", new_name)
                    out_colorramps.append(ramp_copy)
                    used_names.add(new_name)
                    count_added += 1
                else:
                    print(f"[ERROR] Unknown dup_policy: {dup_policy}", file=sys.stderr)
                    return 3
            else:
                out_colorramps.append(deepcopy(ramp))
                used_names.add(name)
                count_added += 1

    if count_added == 0:
        print("[ERROR] No colorramps found to write.", file=sys.stderr)
        return 4

    output_file.parent.mkdir(parents=True, exist_ok=True)
    xml_text = _pretty_print(out_root)
    output_file.write_text(xml_text, encoding="utf-8")
    print(f"[OK] Wrote {count_added} colorramps (seen {count_seen}) to: {output_file}")
    return 0


def main():
    """
    Command-line entry point to combine QGIS color ramp XML files.

    Parses CLI arguments, calls :func:`combine_qgis_colorramps`, and exits
    with its return code.

    Notes
    -----
    See ``--help`` for usage:
    ``python combine_qgis_colorramps.py ROOT --out OUTPUT.xml [--dup-policy ...] [--pattern ...]``.
    """
    p = argparse.ArgumentParser(description="Combine QGIS *_QGIS.xml color ramps into one style file.")
    p.add_argument("root", type=Path, help="Root directory to search (recursively).")
    p.add_argument("--out", type=Path, required=True, help="Output combined XML file.")
    p.add_argument(
        "--dup-policy",
        choices=("keep-first", "overwrite", "rename"),
        default="rename",
        help="How to handle duplicate ramp names (default: rename).",
    )
    p.add_argument(
        "--pattern",
        default="*_QGIS.xml",
        help="Glob pattern for input files (default: *_QGIS.xml).",
    )
    args = p.parse_args()

    rc = combine_qgis_colorramps(
        root_dir=args.root,
        output_file=args.out,
        dup_policy=args.dup_policy,
        glob_pattern=args.pattern,
    )
    sys.exit(rc)


if __name__ == "__main__":
    main()
