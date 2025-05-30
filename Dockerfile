FROM pytorch/pytorch:1.13.1-cuda11.6-cudnn8-devel

# Install base utilities
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y build-essential wget ninja-build unzip libgl-dev ffmpeg\
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR /opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh && \
    /bin/bash ~/miniconda.sh -b -p /opt/conda

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# Install necessary packages
RUN apt update

# Create a workspace directory and clone the repository
WORKDIR /workspace
RUN git clone https://github.com/gssales/gaussian-splatting-reflection --recursive

# Create a Conda environment and activate it
WORKDIR /workspace/gaussian-splatting-reflection

RUN conda env create --file environment.yml && conda init bash && exec bash && conda activate gs_reflection

