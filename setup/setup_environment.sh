#!/bin/bash

# -------------------------------
# 1. Install System Dependencies
# -------------------------------
apt-get update && apt-get install -y \
    build-essential \
    libgl1 \
    libglvnd0 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    libjpeg-dev \
    libpng-dev \
    libtiff-dev \
    ffmpeg \
    libglm-dev \
    clang \
    gcc-9 \
    g++-9

# Set GCC 9 as the default
update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 60 --slave /usr/bin/g++ g++ /usr/bin/g++-9
update-alternatives --set gcc /usr/bin/gcc-9

# Explicitly set CC and CXX for consistent compiler use
export CC=/usr/bin/gcc-9
export CXX=/usr/bin/g++-9

# -------------------------------
# 2. Install PyTorch with CUDA 11.8
# -------------------------------
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu118
#export CUDA_HOME=/usr/local/cuda
#export LD_LIBRARY_PATH="$CUDA_HOME/lib64:$LD_LIBRARY_PATH"

# -------------------------------
# 3. Install Main Python Requirements
# -------------------------------
pip install \
    torchmetrics \
    trimesh \
    plyfile \
    scipy \
    ninja \
    camtools \
    einops \
    lpips \
    tensorboard \
    tqdm \
    transformers \
    omegaconf \
    open_clip_torch \
    open3d \
    opencv-python-headless \
    "numpy<2" \
    pillow \
    pytorch-lightning \
    PyYAML \
    ipykernel \
    roma \
    huggingface-hub

# -------------------------------
# Final Checks
# -------------------------------
echo "Environment setup complete"
