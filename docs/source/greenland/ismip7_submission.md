# ISMIP7 Submission

UAF's ISMIP7 Greenland output, run with PISM, is published on
[Source Cooperative](https://source.coop) as the product
[`ismip/ismip7-uaf-pism`](https://source.coop/ismip/ismip7-uaf-pism). The product has two
prefixes:

`data/`
: The published data. It mirrors the submission as held in the ISMIP7 GHub collection at
  the University at Buffalo, which remains the archive of record and where the ISMIP7
  compliance checks are applied.

`submitted/`
: The modelling group's upload inbox. Files here may not have been checked yet.

This page describes how to copy a run's submission from the PISM cloud bucket into
`submitted/`. The copy runs from the PISM JupyterHub, inside AWS, so the data never passes
through a laptop.

## What gets uploaded

Only the submission tree `output/GrIS/` of a cloud project is uploaded. It holds one
directory per experiment counter, for example `GrIS/UAF/PISM/CORE/C011/`, with one file per
ISMIP7 variable. The other directories next to it in `output/` (`basins/`,
`observations/`, the per-job `Cxxx/` copies) are working products and stay in the cloud
bucket.

## Before you start

- A Source Cooperative account that is a member of the `ismip` organization with write
  access to the product. Ask the product's owners or `hello@source.coop` for access.
- The [rclone](https://rclone.org) program. The `rclone` package on PyPI is only a Python
  wrapper around it and does not include the program itself.

## 1. Install rclone

Download the release for the hub's architecture into `~/.local/bin`, which needs no
administrator rights. Check the architecture with `uname -m`: `x86_64` needs the `amd64`
build, `aarch64` the `arm64` build.

```bash
cd /tmp
curl -LO https://downloads.rclone.org/rclone-current-linux-amd64.zip
unzip -o rclone-current-linux-amd64.zip
cp rclone-*-linux-amd64/rclone ~/.local/bin/
chmod +x ~/.local/bin/rclone
rclone version
```

## 2. Configure the two remotes

rclone copies between two named remotes: the PISM cloud bucket, read anonymously, and
Source Cooperative's S3-compatible endpoint `https://data.source.coop`. Put both in
`~/.config/rclone/rclone.conf`:

```ini
[pism]
type = s3
provider = AWS
region = us-west-2

[source]
type = s3
provider = Other
endpoint = https://data.source.coop
env_auth = true
region = us-west-2
list_version = 2
list_url_encode = false
sign_accept_encoding = false
```

`pism` has no credentials, so rclone reads the bucket anonymously. `source` takes its
credentials from the environment (`env_auth = true`, see the next step).

The file does not always survive a restart of the hub server. `rclone config show` lists
the remotes that are there; these two commands write them again:

```bash
rclone config create pism s3 provider AWS region us-west-2
rclone config create source s3 provider Other endpoint https://data.source.coop \
    env_auth true region us-west-2 list_version 2 list_url_encode false \
    sign_accept_encoding false
```

The last four settings of `source` are needed because the endpoint is a proxy behind
Cloudflare rather than AWS itself. `sign_accept_encoding = false` is the essential one:
Cloudflare rewrites the `Accept-Encoding` header, so a signature that covers it no longer
matches and every request fails with `SignatureDoesNotMatch`.

## 3. Get temporary credentials

Uploads use short-lived credentials that Source Cooperative issues for the product:

1. Sign in at [source.coop](https://source.coop) and open the product page.
2. Open the lock menu and choose **View Credentials**, then **Environment Variables**.
3. Paste the `export` lines into the hub terminal.

These set the access key, secret key, session token, region and endpoint. The credentials
expire, and the page shows when. Never commit them, paste them into a notebook that is
saved, or share them.

If an AWS profile is active in the terminal, unset it so that rclone uses the exported
credentials:

```bash
unset AWS_PROFILE
```

:::{note}
Source Cooperative also offers a command-line tool,
[`source-coop`](https://github.com/source-cooperative/source-coop-cli), that fetches
credentials after a browser login. Its login waits for the browser on a port of the
machine it runs on, which a browser cannot reach on the JupyterHub, so the credentials from
the web page are the practical route there.
:::

## 4. Check access

List the product. An empty result without an error means the credentials and the `source`
remote work:

```bash
rclone lsf source:ismip/ismip7-uaf-pism/
```

## 5. Copy the submission

Do a dry run first. It lists the files rclone would copy without writing anything:

```bash
rclone copy pism:pism-cloud-data/ismip7_production/<project>/output/GrIS \
    source:ismip/ismip7-uaf-pism/submitted/GrIS/ \
    --progress --dry-run
```

`<project>` is the cloud project, for example `2026_10_core_plume`. If the list looks
right, run the copy:

```bash
rclone copy pism:pism-cloud-data/ismip7_production/<project>/output/GrIS \
    source:ismip/ismip7-uaf-pism/submitted/GrIS/ \
    --progress --s3-upload-cutoff 0 --s3-chunk-size 50Mi --s3-no-check-bucket
```

- **Check the dry run's list.** A misspelled `<project>` is not an error: rclone finds
  nothing under the prefix and reports `There was nothing to transfer`.
- **Keep the trailing slash on the destination.** Without it, rclone probes the path with a
  request that the proxy answers as if an object existed there, and stops with
  `is a file not a directory`.
- **Send every file as a multipart upload** (`--s3-upload-cutoff 0`). rclone uploads a file
  at or below the cutoff in a single request that carries a `Content-MD5` header, which the
  proxy passes on to the storage without signing it; the storage then refuses the request
  (see [Troubleshooting](#troubleshooting)). Multipart uploads are not affected, and a
  cutoff of zero makes even the smallest file one.
- **Upload in 50 MiB parts** (`--s3-chunk-size 50Mi`), so that no single request comes
  close to the request-size limits common behind Cloudflare.
- **Do not create the bucket** (`--s3-no-check-bucket`). rclone otherwise first tries to
  create the bucket `ismip`, which the proxy refuses with `400 Bad Request`.
- **Rerun after an interruption.** `rclone copy` skips files that are already in place, so
  after expired credentials or a dropped connection, export fresh credentials and run the
  same command again.

The 210 GiB (429 files) of `2026_10_core_plume` copied in about twenty minutes from the
hub.

## 6. Verify the upload

Compare the two sides file by file:

```bash
rclone check pism:pism-cloud-data/ismip7_production/<project>/output/GrIS \
    source:ismip/ismip7-uaf-pism/submitted/GrIS/ \
    --size-only --one-way
```

It should report `0 differences found` and the number of files of the submission as
`matching files`. Without a `matching files` line, both sides were empty: check
`<project>`. `--size-only` compares sizes rather than checksums, which the proxy does not
necessarily return in the same form as the cloud bucket.

## Troubleshooting

| Message | Cause | Fix |
|---|---|---|
| `SignatureDoesNotMatch: signature mismatch` | rclone signs `Accept-Encoding`, which Cloudflare rewrites | `sign_accept_encoding = false` on the `source` remote |
| `is a file not a directory` | The proxy answers rclone's probe of the destination path | Trailing slash on the destination |
| `didn't find section in config file` | `rclone.conf` is missing, for example after a restart of the hub server | Write the remotes again, see [step 2](#2-configure-the-two-remotes) |
| `AccessDenied: There were headers present in the request which were not signed` on `PutObject` | A single-request upload carries `Content-MD5`, which the proxy forwards unsigned | `--s3-upload-cutoff 0` |
| `400 Bad Request` on `PUT /ismip` (with `-vv --dump headers`) | rclone tries to create the bucket | `--s3-no-check-bucket` |
| `There was nothing to transfer`, or a check without a `matching files` line | Nothing under the source prefix | Check `<project>` |
| `AccessDenied`, `ExpiredToken`, or `Forbidden` on `HeadObject` | Credentials expired or not exported in this terminal | Export fresh credentials from the product page |
| `AccessDenied` on the first upload although listing works | The account can read but not write the product | Ask for write access to `ismip/ismip7-uaf-pism` |

To see why the storage refuses a request, print the body of its answer. It names, for
example, the header that was not signed (`<HeadersNotSigned>content-md5</HeadersNotSigned>`):

```bash
rclone copy <source file> <destination>/ --s3-no-check-bucket --retries 1 \
    -vv --dump responses 2>&1 | grep -ao "<Error>.*</Error>"
```

`--dump headers` prints the session token of the credentials in clear text; do not share
its output unredacted.

## Further reading

- [Upload your data](https://docs.source.coop/data-upload) and
  [Access data](https://docs.source.coop/data-proxy) in the Source Cooperative documentation.
- [rclone's S3 backend](https://rclone.org/s3/), for the remote settings used above.
