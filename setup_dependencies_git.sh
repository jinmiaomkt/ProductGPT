#!/usr/bin/env bash
# ----------------------------------------------------------------------
# AWS EC2 Dependency Setup Script
# ----------------------------------------------------------------------
# This script updates the system, installs Python and essential tools,
# checks for an NVIDIA GPU and installs drivers/CUDA if present, then
# installs PyTorch (with CUDA support if available) and other libraries.
# ----------------------------------------------------------------------

set -euo pipefail

info() {
  echo -e "\n[INFO] $1\n"
}

# ----------------------------------------------------------------------
# 1. Update and upgrade the system
# ----------------------------------------------------------------------
info "Updating and upgrading the system..."
sudo dnf update -y
sudo dnf upgrade -y

# ----------------------------------------------------------------------
# 2. Install Python3, pip, and essential tools
# ----------------------------------------------------------------------
info "Installing Python3, pip, and essential tools..."
sudo dnf install -y python3-pip gcc-c++ make

# ----------------------------------------------------------------------
# 3. Install PyTorch with CUDA 11.8 and TorchMetrics
# ----------------------------------------------------------------------
info "Installing PyTorch (CUDA 11.8) and TorchMetrics..."
pip3 install --upgrade pip

pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install torchmetrics
pip3 install openpyxl

# ----------------------------------------------------------------------
# 4. Install additional Python libraries
# ----------------------------------------------------------------------
info "Installing additional Python libraries..."
pip3 install \
  numpy \
  torchtext \
  gensim \
  langchain \
  matplotlib \
  numba \
  tensorflow \
  thinc \
  linformer \
  reformer-pytorch \
  pytorch_lamb \
  scikit-learn \
  tokenizers \
  mpi4py

# ----------------------------------------------------------------------
# 5. Try installing DeepSpeed (if fails, skip)
# ----------------------------------------------------------------------
info "Attempting to install DeepSpeed..."
if ! pip3 install deepspeed --no-build-isolation --no-cache-dir; then
  echo "[WARNING] Failed to install DeepSpeed. You can try manually or skip if not needed."
fi

# ----------------------------------------------------------------------
# 6. Check for GPU and CUDA availability
# ----------------------------------------------------------------------
info "Verifying PyTorch installation and CUDA availability..."
python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

# ----------------------------------------------------------------------
# 7. System Info
# ----------------------------------------------------------------------
info "System information:"
uname -a
cat /etc/os-release || true
