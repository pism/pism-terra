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

# pylint: disable=broad-exception-caught
"""
AWS syncing.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

# ---------------------------
# AWS CLI wrappers (simple)
# ---------------------------


def s3_sync_to_local(s3_uri: str, dest: str | Path, *, delete: bool = False) -> None:
    """
    Sync an S3 prefix to a local directory using the AWS CLI.

    Parameters
    ----------
    s3_uri : str
        The S3 URI to sync from, e.g. ``"s3://my-bucket/path/"``.
    dest : str or pathlib.Path
        Local destination directory. Created if it does not exist.
    delete : bool, default False
        If True, delete local files that are not present at the source.

    Raises
    ------
    subprocess.CalledProcessError
        If the ``aws s3 sync`` command exits with a non-zero code.

    Notes
    -----
    Requires the AWS CLI to be installed and configured (credentials, region).
    Mirrors the behavior of ``aws s3 sync`` including multipart uploads,
    built-in retries, and include/exclude semantics if you extend the command.
    """
    dest = Path(dest)
    dest.mkdir(parents=True, exist_ok=True)
    cmd = ["aws", "s3", "sync", s3_uri, str(dest)]
    if delete:
        cmd.append("--delete")
    subprocess.run(cmd, check=True)


def s3_sync_from_local(src: str | Path, s3_uri: str, *, delete: bool = False) -> None:
    """
    Sync a local directory to an S3 prefix using the AWS CLI.

    Parameters
    ----------
    src : str or pathlib.Path
        Local source directory to upload from.
    s3_uri : str
        Destination S3 URI, e.g. ``"s3://my-bucket/path/"``.
    delete : bool, default False
        If True, delete S3 objects at the destination that do not exist locally.

    Raises
    ------
    subprocess.CalledProcessError
        If the ``aws s3 sync`` command exits with a non-zero code.

    Notes
    -----
    Requires the AWS CLI to be installed and configured (credentials, region).
    This thin wrapper lets you orchestrate robust syncs from Python without
    re-implementing the CLI logic.
    """
    cmd = ["aws", "s3", "sync", str(src), s3_uri]
    if delete:
        cmd.append("--delete")
    subprocess.run(cmd, check=True)


# -----------------------------------------
# Pure boto3 implementation (one-way syncs)
# -----------------------------------------


def download_from_s3(s3_uri: str, dest: str | Path) -> Path:
    """
    Download a file from AWS S3.

    Parameters
    ----------
    s3_uri : str
        URI of S3 object to download.
    dest : str or Path
        Path to the downloaded file.

    Returns
    -------
    Path
        Path to the downloaded file.
    """
    dest = Path(dest)

    parsed_url = urlparse(s3_uri)
    bucket = parsed_url.netloc
    prefix = parsed_url.path.lstrip("/")

    s3 = boto3.client("s3")
    s3.download_file(bucket, prefix, str(dest))

    return dest


def _md5(path: Path, chunk: int = 8 * 1024 * 1024) -> str:
    """
    Compute the hexadecimal MD5 of a local file.

    Parameters
    ----------
    path : pathlib.Path
        File path to hash.
    chunk : int, default 8*1024*1024
        Read size (bytes) for streaming the file.

    Returns
    -------
    str
        Lowercase hex MD5 digest.

    Raises
    ------
    OSError
        If the file cannot be opened/read.
    """
    h = hashlib.md5()
    with path.open("rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _needs_download(local: Path, size: int, etag: str) -> bool:
    """
    Decide whether an S3 object should be downloaded to a local path.

    The test considers local existence, size equality, and (for single-part
    S3 uploads) an MD5 comparison against the object's ETag.

    Parameters
    ----------
    local : pathlib.Path
        Target local file path.
    size : int
        Content length of the S3 object in bytes.
    etag : str
        S3 object's ETag string (usually quoted). For multipart uploads,
        ETag contains a dash and is not a simple MD5.

    Returns
    -------
    bool
        True if a download is recommended (missing or differs), False if
        the local file appears up to date.

    Notes
    -----
    * ETag equals MD5 only for **single-part** uploads. If the ETag contains
      a dash (multipart), this function falls back to size comparison.
    * If a local MD5 cannot be computed (I/O error), the function returns True.
    """
    if not local.exists():
        return True
    if local.stat().st_size != size:
        return True
    etag_clean = etag.strip('"')
    # multipart uploads have ETags like "md5-nparts"
    if "-" in etag_clean:
        return False  # cannot safely compare; sizes already match
    try:
        return _md5(local) != etag_clean
    except Exception:
        return True


def s3_to_local(
    bucket: str,
    prefix: str,
    dest_dir: str | Path,
    *,
    exclude_keys: Iterable[str] = (),
    dry_run: bool = False,
    delete_extra: bool = False,
    max_concurrency: int = 8,
) -> None:
    """
    Sync objects under an S3 prefix **to** a local directory (one-way).

    Parameters
    ----------
    bucket : str
        Source S3 bucket name.
    prefix : str
        Source key prefix (acts like a folder). May be empty.
    dest_dir : str or pathlib.Path
        Local destination directory. Created if missing.
    exclude_keys : Iterable[str], optional
        Exact S3 keys to skip during sync.
    dry_run : bool, default False
        If True, only print planned actions; do not transfer or delete files.
    delete_extra : bool, default False
        If True, delete local files under ``dest_dir`` that are not present
        under ``bucket/prefix``.
    max_concurrency : int, default 8
        Maximum worker threads for concurrent transfers.

    Raises
    ------
    botocore.exceptions.BotoCoreError
        On AWS SDK client errors (e.g., credential or network issues).
    botocore.exceptions.ClientError
        On S3 API errors (e.g., permissions).
    OSError
        If local filesystem operations fail (mkdir/unlink/write).

    Notes
    -----
    * Change detection uses size and, for single-part uploads, MD5 vs ETag.
    * Multipart uploads are considered up-to-date when sizes match.
    * Listing is paginated via ``list_objects_v2``.
    """
    dest = Path(dest_dir)
    dest.mkdir(parents=True, exist_ok=True)

    s3 = boto3.client("s3", config=Config(retries={"max_attempts": 10}))
    paginator = s3.get_paginator("list_objects_v2")
    txconf = TransferConfig(
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        max_concurrency=max_concurrency,
        use_threads=True,
    )

    s3_local_abs = set()

    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.endswith("/") or key in exclude_keys:
                continue
            rel = key[len(prefix) :] if prefix and key.startswith(prefix) else key
            local_path = dest / rel.lstrip("/")
            local_path.parent.mkdir(parents=True, exist_ok=True)

            if _needs_download(local_path, obj["Size"], obj["ETag"]):
                print("DOWNLOAD", f"s3://{bucket}/{key}", "→", local_path)
                if not dry_run:
                    s3.download_file(bucket, key, str(local_path), Config=txconf)

            s3_local_abs.add(str(local_path.resolve()))

    if delete_extra:
        for p in dest.rglob("*"):
            if p.is_file() and str(p.resolve()) not in s3_local_abs:
                print("DELETE local", p)
                if not dry_run:
                    p.unlink()


def local_to_s3(
    src_dir: str | Path,
    bucket: str,
    prefix: str,
    *,
    dry_run: bool = False,
    delete_extra: bool = False,
    max_concurrency: int = 8,
) -> None:
    """
    Sync a local directory **to** an S3 prefix (one-way).

    Parameters
    ----------
    src_dir : str or pathlib.Path
        Local directory to upload from.
    bucket : str
        Destination S3 bucket name.
    prefix : str
        Destination key prefix (acts like a folder). A trailing ``/`` is added
        automatically when constructing object keys.
    dry_run : bool, default False
        If True, only print planned actions; do not transfer or delete objects.
    delete_extra : bool, default False
        If True, delete S3 objects under ``prefix`` that do not exist locally.
    max_concurrency : int, default 8
        Maximum worker threads for concurrent transfers.

    Raises
    ------
    botocore.exceptions.BotoCoreError
        On AWS SDK client errors (e.g., credential or network issues).
    botocore.exceptions.ClientError
        On S3 API errors (e.g., permissions).
    OSError
        If local filesystem operations fail (directory traversal).

    Notes
    -----
    * Change detection uses size equality and, for single-part uploads,
      MD5 comparison with the S3 object's ETag.
    * Multipart uploads are treated as up-to-date when sizes match because
      ETag is not a simple MD5 in that case.
    * Deletions are batched (up to 1000 objects per request) when enabled.
    """
    src = Path(src_dir)
    s3 = boto3.client("s3", config=Config(retries={"max_attempts": 10}))
    txconf = TransferConfig(
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        max_concurrency=max_concurrency,
        use_threads=True,
    )

    # Build local index
    local_files = {str(p.relative_to(src)).replace(os.sep, "/"): p for p in src.rglob("*") if p.is_file()}

    # Existing keys under prefix (for deletes and quick membership checks)
    s3_keys = set()
    paginator = s3.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []):
            s3_keys.add(obj["Key"])

    # Upload missing/changed
    for rel_key, path in local_files.items():
        key = f"{prefix.rstrip('/')}/{rel_key}"

        need_upload = True
        try:
            head = s3.head_object(Bucket=bucket, Key=key)
            same_size = head["ContentLength"] == path.stat().st_size
            if same_size:
                etag = head["ETag"].strip('"')
                if "-" in etag:  # multipart: cannot MD5 compare
                    need_upload = False
                else:  # single-part
                    need_upload = _md5(path) != etag
            else:
                need_upload = True
        except s3.exceptions.NoSuchKey:
            need_upload = True
        except Exception:
            # On ambiguous errors, prefer uploading.
            need_upload = True

        if need_upload:
            print("UPLOAD", path, "→", f"s3://{bucket}/{key}")
            if not dry_run:
                s3.upload_file(str(path), bucket, key, Config=txconf)

    # Delete extras on S3 if requested
    if delete_extra:
        want_keys = {f"{prefix.rstrip('/')}/{k}" for k in local_files}
        to_delete = [k for k in s3_keys if k.startswith(prefix) and k not in want_keys]
        for k in to_delete:
            print(f"DELETE s3://{bucket}/{k}")
        if not dry_run and to_delete:
            # batch in chunks of 1000
            for i in range(0, len(to_delete), 1000):
                s3.delete_objects(
                    Bucket=bucket,
                    Delete={"Objects": [{"Key": k} for k in to_delete[i : i + 1000]]},
                )
