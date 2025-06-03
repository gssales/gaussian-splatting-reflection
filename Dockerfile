FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# Install base utilities
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y wget libgl-dev libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# Create a workspace directory and clone the repository
WORKDIR /workspace
COPY . .

RUN conda env create --file environment-linux.yml && conda init bash

SHELL ["conda", "run", "-n", "gs_reflection", "/bin/bash", "-c"]