# FIXME: better tag
ARG PISM_TAG=cloud-test
FROM ghcr.io/pism/pism:${PISM_TAG} AS runtime

FROM runtime AS build
ARG DEBIAN_FRONTEND=noninteractive
ARG ONEAPI_VERSION=2025.3
ARG IMPI_VERSION=2021.17

USER root
RUN <<EOF
    echo "Install build tools"

    set -e
    set -u
    set -x

    apt-get update

    apt-get install -y --no-install-recommends \
    autoconf \
    automake \
    git \
    libtool \
    make \
    intel-oneapi-compiler-dpcpp-cpp-${ONEAPI_VERSION} \
    intel-oneapi-compiler-fortran-${ONEAPI_VERSION} \
    intel-oneapi-mpi-devel-${IMPI_VERSION} \
    ""

    rm -rf /var/lib/apt/lists/*
EOF

RUN <<EOF
    echo "Install NCAR/peak_memusage"

    set -e
    set -u
    set -x

    build_dir=/var/tmp/build/peak_memusage
    prefix=/opt/peak_memusage

    mkdir -p ${build_dir}
    cd ${build_dir}

    git clone --depth=1 https://github.com/NCAR/peak_memusage.git .

    ./autogen.sh

    ./configure CC=mpiicx CXX=mpiicpx FC=mpiifx \
    --disable-nvml \
    --disable-fortran \
    --disable-openmp \
    --enable-mpi \
    --prefix=${prefix} || (cat config.log && exit 1)

    make all && make install

    rm -rf ${build_dir}
EOF

FROM runtime

USER root

RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl git unzip vim \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

COPY --from=build /opt/peak_memusage/ /opt/peak_memusage

RUN chown -R worker /opt/

USER worker
WORKDIR /home/worker

ENV PATH=/opt/peak_memusage/bin:$PATH

RUN curl -L -O "https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-$(uname)-$(uname -m).sh" && \
    bash Miniforge3-$(uname)-$(uname -m).sh -b -p /opt/conda && \
    rm Miniforge3-$(uname)-$(uname -m).sh

ENV PATH=/opt/conda/bin:$PATH
SHELL ["/bin/bash", "-l", "-c"]

COPY --chown=worker . /pism-terra/

RUN mamba env create -f /pism-terra/environment.yml && \
    conda clean -afy && \
    echo ". /opt/conda/etc/profile.d/conda.sh" >> /home/worker/.profile && \
    echo "conda activate pism-terra" >> /home/worker/.profile

RUN conda activate pism-terra && \
    python -m pip install --no-cache-dir /pism-terra


ENTRYPOINT ["/pism-terra/pism_terra/etc/entrypoint.sh"]
CMD ["-h"]
