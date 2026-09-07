#!/usr/bin/env bash
# ────────────────────────────────────────────────────────────────────────────
# AWS EC2 Dependency Setup Script  (revised)
# ────────────────────────────────────────────────────────────────────────────
# • Updates the instance.
# • Installs Python 3.x *with* development headers, build tools, Open MPI,
#   NVIDIA drivers / CUDA (if a GPU is present), PyTorch (CUDA when possible),
#   and the libraries you use for ProductGPT.
#
# Tested on:
#   • Amazon Linux 2023  (dnf)
#   • Amazon Linux 2     (yum)  – the commands auto-switch.
# ---------------------------------------------------------------------------

set -euo pipefail

log() { printf '\n[INFO] %s\n\n' "$*"; }
warn() { printf '\n[WARN] %s\n\n' "$*" >&2; }

# ────────────────────────── 0. Detect package manager ──────────────────────
if command -v dnf >/dev/null 2>&1;   then PM="dnf"
elif command -v yum >/dev/null 2>&1; then PM="yum"
else
  echo "[ERROR] Neither dnf nor yum found – unsupported AMI?" >&2
  exit 1
fi

# ────────────────────────── 1. System update / tools ───────────────────────
log "Updating the system and installing base development tools…"
sudo "$PM" -y update
# AL2023 already ships gcc/make, but ensure full toolchain is present
sudo "$PM" -y groupinstall "Development Tools" || true
sudo "$PM" -y install \
  python3 python3-devel python3-pip \
  git wget unzip tar \
  openmpi openmpi-devel \
  gcc-c++ make

# ────────────────────────── 2. Open MPI env vars (one-time) ────────────────
# Add Open MPI to PATH/LIB so mpi4py can find it at runtime.
if [ ! -f /etc/profile.d/openmpi.sh ]; then
  log "Adding Open MPI to system-wide PATH/LD_LIBRARY_PATH…"
  sudo tee /etc/profile.d/openmpi.sh >/dev/null <<'EOF'
export PATH=/opt/amazon/openmpi/bin:$PATH
export LD_LIBRARY_PATH=/opt/amazon/openmpi/lib:$LD_LIBRARY_PATH
EOF
  sudo chmod +x /etc/profile.d/openmpi.sh
fi
# shellcheck disable=SC1091
source /etc/profile.d/openmpi.sh

# ────────────────────────── 3. Optional NVIDIA driver + CUDA ───────────────
if lspci | grep -qi nvidia; then
  log "NVIDIA GPU detected – installing driver & CUDA toolkit…"
  # Amazon Linux helper – this installs the latest DKMS driver + CUDA
  if command -v nvidia-smi >/dev/null 2>&1; then
    log "NVIDIA driver already present – skipping driver install."
  else
    curl -s https://raw.githubusercontent.com/awslabs/nvidia-driver-toolkit/master/nvidia-driver-toolkit \
      | sudo bash -s -- -n  # “-n” = non-interactive
  fi
else
  warn "No NVIDIA GPU detected – PyTorch will fall back to CPU build."
fi

# ────────────────────────── 4. Upgrade pip + core wheels ───────────────────
log "Upgrading pip and wheel…"
python3 -m pip install --upgrade pip wheel setuptools

# ────────────────────────── 5. Install PyTorch (+CUDA if available) ────────
log "Installing PyTorch…"
if command -v nvidia-smi >/dev/null 2>&1; then
  # CUDA 11.8 wheels (PyTorch’s default for AWS GPU instances)
  python3 -m pip install torch torchvision torchaudio \
    --index-url https://download.pytorch.org/whl/cu118
else
  python3 -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
fi

# TorchMetrics & data I/O helpers
python3 -m pip install torchmetrics openpyxl

# ────────────────────────── 6. Scientific / ML stack ───────────────────────
log "Installing additional Python libraries…"
python3 -m pip install \
  numpy numba matplotlib scikit-learn \
  tokenizers gensim langchain torchtext \
  tensorflow thinc linformer reformer-pytorch \
  pytorch_lamb mpi4py

# ────────────────────────── 7. DeepSpeed (best-effort) ─────────────────────
log "Attempting DeepSpeed install (this may take several minutes)…"
if ! python3 -m pip install deepspeed --no-build-isolation --no-cache-dir; then
  warn "DeepSpeed build failed – skip for now (you can retry later)."
fi

# ────────────────────────── 8. Quick sanity checks ─────────────────────────
log "Verifying installations…"
python3 - <<'PY'
import sys, torch, mpi4py
print("Python:", sys.version.split()[0])
print("PyTorch:", torch.__version__, "| CUDA available:", torch.cuda.is_available())
print("mpi4py:", mpi4py.__version__)
PY

log "Setup complete – you are ready to train ProductGPT on this instance! 🚀"
