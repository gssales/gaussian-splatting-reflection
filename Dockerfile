FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

ENV TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"

# Install base utilities
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update \
    && apt-get install -y git wget libgl-dev libglib2.0-0 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Install miniconda
ENV CONDA_DIR=/opt/conda
RUN wget --quiet https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda.sh \
    && /bin/bash ~/miniconda.sh -b -p /opt/conda \
    && conda init bash

# Put conda in path so we can use conda activate
ENV PATH=$CONDA_DIR/bin:$PATH

# Create a workspace directory and clone the repository
WORKDIR /workspace/gaussian-splatting-reflection
COPY . .

RUN conda env create --file environment-linux.yml && conda init bash

# Create a workspace directory and clone the repository
WORKDIR /workspace

RUN git clone https://github.com/graphdeco-inria/gaussian-splatting.git --recursive \
    && git clone https://github.com/gapszju/3DGS-DR.git --recursive \
    && git clone https://github.com/hbb1/2d-gaussian-splatting.git --recursive

WORKDIR /workspace/gaussian-splatting

RUN conda create --name gs_reflection --clone gaussian_splatting \
    && conda activate gaussian_splatting \
    && pip install submodules/diff-gaussian-rasterization
    
WORKDIR /workspace/3DGS-DR

RUN pip install submodules/diff-gaussian-rasterization_c3 \
    && pip install submodules/diff-gaussian-rasterization_c7 \
    && pip install submodules/cubemapencoder

WORKDIR /workspace/2d-gaussian-splatting

RUN pip install submodules/diff-surfel-rasterization

SHELL ["conda", "run", "-n", "gs_reflection", "/bin/bash", "-c"]