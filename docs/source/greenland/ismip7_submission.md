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
    --progress --s3-upload-cutoff 50Mi --s3-chunk-size 50Mi
```

- **Keep the trailing slash on the destination.** Without it, rclone probes the path with a
  request that the proxy answers as if an object existed there, and stops with
  `is a file not a directory`.
- **Upload in 50 MiB parts.** With these two flags every file above 50 MiB is sent in
  parts, so no single request comes close to the request-size limits common behind
  Cloudflare. rclone's default sends files of up to 200 MiB in one request.
- **Rerun after an interruption.** `rclone copy` skips files that are already in place, so
  after expired credentials or a dropped connection, export fresh credentials and run the
  same command again.

A submission of a few tens of GB copies in minutes from the hub.

## 6. Verify the upload

Compare the two sides file by file:

```bash
rclone check pism:pism-cloud-data/ismip7_production/<project>/output/GrIS \
    source:ismip/ismip7-uaf-pism/submitted/GrIS/ \
    --size-only --one-way
```

It should report no differences. `--size-only` compares sizes rather than checksums, which
the proxy does not necessarily return in the same form as the cloud bucket.

## Troubleshooting

| Message | Cause | Fix |
|---|---|---|
| `SignatureDoesNotMatch: signature mismatch` | rclone signs `Accept-Encoding`, which Cloudflare rewrites | `sign_accept_encoding = false` on the `source` remote |
| `is a file not a directory` | The proxy answers rclone's probe of the destination path | Trailing slash on the destination |
| `AccessDenied` or `ExpiredToken` | Credentials expired or not exported in this terminal | Export fresh credentials from the product page |
| `AccessDenied` on the first upload although listing works | The account can read but not write the product | Ask for write access to `ismip/ismip7-uaf-pism` |

## Further reading

- [Upload your data](https://docs.source.coop/data-upload) and
  [Access data](https://docs.source.coop/data-proxy) in the Source Cooperative documentation.
- [rclone's S3 backend](https://rclone.org/s3/), for the remote settings used above.
