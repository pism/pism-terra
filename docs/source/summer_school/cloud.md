# Running in the cloud

Everything on the previous pages runs on a laptop or a cluster. The same runs
can be handed to **PISM-Cloud**, a [HyP3](https://hyp3-docs.asf.alaska.edu)
deployment at ASF that stages the inputs, renders the run scripts and executes
them on AWS. You drive it from a notebook; nothing is installed on your machine
beyond the notebook itself.

## What you need

- An [Earthdata Login](https://urs.earthdata.nasa.gov/) (EDL) account — this is
  what authenticates you to the PISM-Cloud server.
- An OpenScienceLab account, so you have somewhere to run the notebook and
  analyse the results afterwards (see {doc}`getting_started`).
- AWS credentials in your environment (the default boto3 chain / `~/.aws`).
  The app uploads your config, template and UQ files straight to S3, so it
  needs write access to the bucket you name.

## The app

`notebooks/pism_cloud_app.ipynb` is the button-driven version. Run it under
[Voila](https://voila.readthedocs.io) to get the UI without the code cells:

```bash
voila notebooks/pism_cloud_app.ipynb --strip_sources=True
```

The six steps mirror the local workflow:

1. **Connect** — pick the server (PISM Cloud Production, or the Test server
   when you are experimenting) and log in with your EDL credentials. On
   success the app prints your remaining credits.
2. **Glacier & metadata** — the RGI ID, a `name` and a `project` (these become
   the S3 prefix your results land under), the bucket, and `ntasks`.
3. **Upload input files** — the PISM config (`.toml`) and the Jinja2 template
   (`.j2`) are required, a UQ file (`.toml`) is optional. Each is copied to
   `s3://{bucket}/glacier/{name}/{kind}/`.
4. **Run type & submit** — `forward` or `inverse`, submitted as a
   `PISM_TERRA_RUN_FORWARD` or `PISM_TERRA_RUN_INVERSE` job. This is the
   *prepare* step: it stages the inputs and writes the run scripts, exactly
   what `pism-glacier-run-forward` / `pism-glacier-run-inverse` do locally.
5. **Job status** — poll once, or auto-refresh every 30 s, until the prepare
   job reports `SUCCEEDED`.
6. **Execute** — the app lists the run scripts the prepare job wrote to
   `s3://{bucket}/{name}/{project}/{job_id}/{rgi_id}/run_scripts/` and submits
   one `PISM_TERRA_EXECUTE` job per script. Track those with **Refresh status**
   as well.

```{admonition} Submit prepare first, then execute
:class: note

The two stages are separate job types on purpose: the prepare job is cheap and
its output — the rendered run scripts — is worth reading before you spend
compute on it. An inverse run script holds three legs (init, inversion, main
run); see {doc}`inverse_modeling`.
```

## Doing it by hand

{doc}`../examples/pism_cloud_intro` walks the same sequence as plain
`hyp3_sdk` calls, without the widgets. Use it when you want to script a batch
of glaciers, or to see exactly which job parameters the buttons are filling in.

## Results

Outputs land under `s3://{bucket}/{name}/{project}/{job_id}/{rgi_id}/`, in the
same `output/` layout a local run produces. Analyse them from
[OpenScienceLab](https://opensciencelab.asf.alaska.edu/), next to the data, or
download them and work locally.
