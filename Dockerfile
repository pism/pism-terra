FROM condaforge/mambaforge:latest

# For opencontainers label definitions, see:
#    https://github.com/opencontainers/image-spec/blob/master/annotations.md
LABEL org.opencontainers.image.title="pism-terra"
LABEL org.opencontainers.image.description="Global PISM"
LABEL org.opencontainers.image.vendor="Geophysical Institute, University of Alaska Fairbanks"
LABEL org.opencontainers.image.authors="Andy Ascwanden <andy.aschwanden@gmail.com>, Joseph H. Kennedy <me@jhkennedy.org>"
LABEL org.opencontainers.image.licenses="BSD-3-Clause"
LABEL org.opencontainers.image.url="https://github.com/pism/pism-terra"
LABEL org.opencontainers.image.source="https://github.com/pism/pism-terra"
LABEL org.opencontainers.image.documentation="https://github.com/pism/pism-terra"

# Dynamic lables to define at build time via `docker build --label`
# LABEL org.opencontainers.image.created=""
# LABEL org.opencontainers.image.version=""
# LABEL org.opencontainers.image.revision=""

ARG DEBIAN_FRONTEND=noninteractive
ENV PYTHONDONTWRITEBYTECODE=true

RUN apt-get update && apt-get install -y --no-install-recommends unzip vim && \
    apt-get clean && rm -rf /var/lib/apt/lists/*

ARG WORKER_UID=1000
ARG WORKER_GID=1000

RUN groupadd -g "${WORKER_GID}" --system worker && \
    useradd -l -u "${WORKER_UID}" -g "${WORKER_GID}" --system -d /home/worker -m  -s /bin/bash worker && \
    chown -R worker:worker /opt

USER ${WORKER_UID}
SHELL ["/bin/bash", "-l", "-c"]
WORKDIR /home/worker

COPY --chown=${WORKER_UID}:${WORKER_GID} . /pism-terra/

RUN mamba env create -f /pism-terra/environment.yml && \
    conda clean -afy && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /home/worker/.profile && \
    echo "conda activate pism-terra" >> /home/worker/.profile

RUN conda activate pism-terra && \
    python -m pip install --no-cache-dir /pism-terra


ENTRYPOINT ["/pism-terra/pism_terra/etc/entrypoint.sh"]
CMD ["-h"]
