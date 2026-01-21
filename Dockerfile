FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

ENV TORCH_CUDA_ARCH_LIST="3.5;5.0;6.0;6.1;7.0;7.5;8.0;8.6+PTX"

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

RUN conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/main \
 && conda tos accept --override-channels --channel https://repo.anaconda.com/pkgs/r

# Create a workspace directory and clone the repository
WORKDIR /workspace
COPY . .

RUN conda create -n gs_reflection python=3.7.16 && conda init bash

SHELL ["conda", "run", "-n", "gs_reflection", "/bin/bash", "-c"]

RUN pip install -r requirements.txt --extra-index-url https://download.pytorch.org/whl/cu113 \
  && pip install submodules/diff-surfel-rasterization \
  && pip install submodules/cubemapencoder \
  && pip install submodules/fused-ssim \
  && pip install submodules/simple-knn

# ENTRYPOINT ["conda", "run", "-n", "gs_reflection", "/bin/bash", "-c"]