#!/bin/bash
# AWS EC2 Dependency Setup Script

# Update and upgrade the system
echo "Updating and upgrading the system..."
sudo apt-get update -y && sudo apt-get upgrade -y

# Install Python3, pip, and essential tools
echo "Installing Python3, pip, and essential tools..."
sudo apt-get install -y python3 python3-pip build-essential

# Install NVIDIA Drivers and CUDA if GPU is available
if lspci | grep -i nvidia > /dev/null; then
  echo "Installing NVIDIA drivers and CUDA..."
  curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-ubuntu2204.pin
  sudo mv cuda-ubuntu2204.pin /etc/apt/preferences.d/cuda-repository-pin-600
  wget https://developer.download.nvidia.com/compute/cuda/12.0.0/local_installers/cuda-repo-ubuntu2204_12.0.0-1_amd64.deb
  sudo dpkg -i cuda-repo-ubuntu2204_12.0.0-1_amd64.deb
  sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/7fa2af80.pub
  sudo apt-get update -y
  sudo apt-get install -y cuda
fi

# Uninstall any existing PyTorch and related libraries
echo "Removing existing PyTorch libraries..."
pip3 uninstall -y torch torchvision torchaudio

# Install PyTorch and TorchMetrics
echo "Installing PyTorch and TorchMetrics..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip3 install torchmetrics

# Install additional Python libraries
echo "Installing additional Python libraries..."
pip3 install numpy torchtext gensim langchain matplotlib numba tensorflow thinc

# Verify installations
echo "Verifying PyTorch installation..."
python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available())"

echo "All dependencies have been installed successfully!"
