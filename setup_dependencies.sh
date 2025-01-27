#!/usr/bin/env bash
# ----------------------------------------------------------------------
# AWS EC2 Dependency Setup Script
# ----------------------------------------------------------------------
# This script updates the system, installs Python and essential tools,
# checks for an NVIDIA GPU and installs drivers/CUDA if present, then
# installs PyTorch (with CUDA support if available) and other libraries.
# ----------------------------------------------------------------------

# Exit immediately if any command exits with a non-zero status,
# treat unset variables as an error, and prevent errors in a pipeline
# from being hidden.
set -euo pipefail

# A function to display messages in a clearer format
info() {
  echo -e "\n[INFO] $1\n"
}

# ----------------------------------------------------------------------
# 1. Update and upgrade the system
# ----------------------------------------------------------------------
info "Updating and upgrading the system..."
sudo apt-get update -y
sudo apt-get upgrade -y

# ----------------------------------------------------------------------
# 2. Install Python3, pip, and essential tools
# ----------------------------------------------------------------------
info "Installing Python3, pip, and essential tools..."
sudo apt-get install -y python3 python3-pip build-essential

# ----------------------------------------------------------------------
# 3. Install NVIDIA Drivers and CUDA (if GPU is detected)
# ----------------------------------------------------------------------
if lspci | grep -i nvidia > /dev/null; then
  info "NVIDIA GPU detected. Installing NVIDIA drivers and CUDA..."

  # Download and configure the CUDA repository pin
  curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
  sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600

  # Download and install the CUDA repository package
  wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-ubuntu2204_12.0.0-1_amd64.deb
  sudo dpkg -i cuda-repo-ubuntu2204_12.0.0-1_amd64.deb

  # Fetch the NVIDIA GPG key and update
  sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/7fa2af80.pub
  sudo apt-get update -y

  # Install CUDA
  sudo apt-get install -y cuda
fi

if lspci | grep -i nvidia > /dev/null; then
  info "NVIDIA GPU detected. Installing NVIDIA drivers and CUDA..."

  # 1. Download and install the CUDA keyring
  wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
  sudo dpkg -i cuda-keyring_1.0-1_all.deb

  # 2. Update and install CUDA (replace 'cuda' with 'cuda-12-0' if you need a specific version)
  sudo apt-get update -y
  sudo apt-get install -y cuda
fi


# ----------------------------------------------------------------------
# 4. Install PyTorch, TorchMetrics, and related libraries
# ----------------------------------------------------------------------
info "Installing PyTorch (with CUDA 11.8 wheels) and TorchMetrics..."
pip3 install --upgrade pip
pip3 install \
  torch \
  torchvision \
  torchaudio \
  --index-url https://download.pytorch.org/whl/cu118
pip3 install torchmetrics

# ----------------------------------------------------------------------
# 5. Install additional Python libraries
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
  thinc

# ----------------------------------------------------------------------
# 6. Verify the PyTorch installation and CUDA availability
# ----------------------------------------------------------------------
info "Verifying PyTorch installation..."
python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

info "All dependencies have been installed successfully!"


# Download the CUDA keyring .deb package
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb

# Install the keyring
sudo dpkg -i cuda-keyring_1.0-1_all.deb

# Update your apt repository listings
sudo apt-get update

# Finally, install CUDA
sudo apt-get -y install cuda

lsb_release -a
