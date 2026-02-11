#!/bin/bash
###############################################################################
# Digital Ocean GPU Droplet (H200) Startup Script for picoGPT
# 
# Target: Ubuntu 22.04 GPU Droplet with NVIDIA H200
# 
# Preinstalled on droplet (DO NOT reinstall):
#   - nvidia-container-toolkit 1.17.8
#   - cuda-keyring 1.1-1
#   - cuda-drivers-575
#   - cuda-toolkit-12-9
#   - bzip2 (8-GPU only)
#   - MLNX_OFED (8-GPU only)
#   - nvidia-fabricmanager-575 (8-GPU only)
#
# This script installs:
#   - System dependencies (build tools, git, etc.)
#   - Python 3.11 + pip + venv
#   - PyTorch with CUDA 12.9 support
#   - Rust toolchain + maturin (for unitokenizer)
#   - All Python project dependencies
#   - Builds the Rust-based unitokenizer extension
#
# Usage:
#   1. Paste this script into Digital Ocean "User Data" when creating a droplet
#   2. Or run manually: chmod +x startup.sh && sudo ./startup.sh
#
###############################################################################

set -euo pipefail

# ─── Configuration ───────────────────────────────────────────────────────────
PROJECT_DIR="/root/picoGPT"
PYTHON_VERSION="3.11"
CUDA_VERSION="12.9"
LOG_FILE="/var/log/picogpt-startup.log"

# ─── Logging ─────────────────────────────────────────────────────────────────
exec > >(tee -a "$LOG_FILE") 2>&1
echo "=========================================="
echo "picoGPT Startup Script"
echo "Started at: $(date)"
echo "=========================================="

# ─── 1. System update & base packages ────────────────────────────────────────
echo "[1/8] Updating system and installing base packages..."
export DEBIAN_FRONTEND=noninteractive

apt-get update -y
# Skip full upgrade to preserve preinstalled NVIDIA/CUDA versions
apt-get upgrade -y -o Dpkg::Options::="--force-confold" \
    --exclude=cuda* --exclude=nvidia* --exclude=libnvidia* 2>/dev/null || apt-get upgrade -y
apt-get install -y \
    build-essential \
    cmake \
    git \
    curl \
    wget \
    unzip \
    htop \
    tmux \
    screen \
    pkg-config \
    libssl-dev \
    libffi-dev \
    zlib1g-dev \
    libbz2-dev \
    libreadline-dev \
    libsqlite3-dev \
    liblzma-dev \
    libncurses5-dev \
    libncursesw5-dev \
    software-properties-common \
    ca-certificates \
    gnupg \
    lsb-release

# ─── 2. Python 3.11 ──────────────────────────────────────────────────────────
echo "[2/8] Installing Python ${PYTHON_VERSION}..."
add-apt-repository ppa:deadsnakes/ppa -y
apt-get update -y
apt-get install -y \
    python${PYTHON_VERSION} \
    python${PYTHON_VERSION}-venv \
    python${PYTHON_VERSION}-dev \
    python${PYTHON_VERSION}-distutils

# Set Python 3.11 as default
update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1
update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION}

# Install pip
curl -sS https://bootstrap.pypa.io/get-pip.py | python${PYTHON_VERSION}
python${PYTHON_VERSION} -m pip install --upgrade pip setuptools wheel

# ─── 3. CUDA environment (preinstalled on DO GPU Droplet) ─────────────────────
echo "[3/8] Configuring CUDA environment (drivers & toolkit preinstalled)..."

# Verify preinstalled NVIDIA stack
echo "Verifying preinstalled NVIDIA H200 stack:"
nvidia-smi
nvcc --version

# Ensure CUDA environment variables are set persistently
if ! grep -q 'CUDA_HOME' /etc/environment 2>/dev/null; then
    cat >> /etc/environment << 'EOF'
CUDA_HOME=/usr/local/cuda
EOF
fi

if [ ! -f /etc/profile.d/cuda.sh ]; then
    cat > /etc/profile.d/cuda.sh << 'EOF'
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}
EOF
fi

source /etc/profile.d/cuda.sh 2>/dev/null || true
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:${LD_LIBRARY_PATH:-}

# ─── 4. Rust toolchain ───────────────────────────────────────────────────────
echo "[4/8] Installing Rust toolchain..."
if ! command -v rustc &> /dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
fi
source "$HOME/.cargo/env"
rustc --version
cargo --version

# ─── 5. Clone / prepare project ──────────────────────────────────────────────
echo "[5/8] Setting up project directory..."
mkdir -p "$PROJECT_DIR"

# If the project is hosted on git, uncomment and adjust:
# git clone https://github.com/YOUR_USER/picoGPT.git "$PROJECT_DIR"

# If uploading via scp/rsync, this directory will be the target.
# For now, create essential subdirectories:
mkdir -p "$PROJECT_DIR/checkpoints"
mkdir -p "$PROJECT_DIR/dataset"
mkdir -p "$PROJECT_DIR/data_cache"
mkdir -p "$PROJECT_DIR/logs"
mkdir -p "$PROJECT_DIR/tensorboard"

echo "Project directory ready at $PROJECT_DIR"
echo "Upload your project files with:"
echo "  scp -r ./picoGPT/* root@<droplet-ip>:$PROJECT_DIR/"

# ─── 6. Python virtual environment & dependencies ────────────────────────────
echo "[6/8] Creating Python virtual environment and installing dependencies..."
cd "$PROJECT_DIR"

python${PYTHON_VERSION} -m venv venv
source venv/bin/activate

pip install --upgrade pip setuptools wheel

# PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129

# Project dependencies
pip install \
    numpy \
    typing_extensions \
    datasets \
    tqdm \
    sentencepiece \
    wandb \
    tensorboard \
    matplotlib \
    torchviz \
    graphviz \
    requests

# Install maturin for building Rust Python extensions
pip install maturin

# If requirements.txt exists, install from it too
if [ -f requirements.txt ]; then
    pip install -r requirements.txt
fi

# ─── 7. Build unitokenizer (Rust extension) ──────────────────────────────────
echo "[7/8] Building unitokenizer Rust extension..."
if [ -d "$PROJECT_DIR/tokenizer" ] && [ -f "$PROJECT_DIR/tokenizer/Cargo.toml" ]; then
    cd "$PROJECT_DIR/tokenizer"
    maturin develop --release
    cd "$PROJECT_DIR"
    echo "unitokenizer built successfully."
else
    echo "WARNING: tokenizer/ directory not found. Skipping Rust build."
    echo "After uploading project files, run:"
    echo "  source /root/picoGPT/venv/bin/activate"
    echo "  cd /root/picoGPT/tokenizer && maturin develop --release"
fi

# ─── 8. Verification ─────────────────────────────────────────────────────────
echo "[8/8] Verifying installation..."

echo ""
echo "=== Python ==="
python --version
pip --version

echo ""
echo "=== PyTorch ==="
python -c "
import torch
print(f'PyTorch version: {torch.__version__}')
print(f'CUDA available:  {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'CUDA version:    {torch.version.cuda}')
    print(f'GPU count:       {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)}')
        mem = torch.cuda.get_device_properties(i).total_mem / (1024**3)
        print(f'         Memory: {mem:.1f} GB')
print(f'cuDNN available: {torch.backends.cudnn.is_available()}')
if torch.backends.cudnn.is_available():
    print(f'cuDNN version:   {torch.backends.cudnn.version()}')
"

echo ""
echo "=== Rust ==="
rustc --version
cargo --version

echo ""
echo "=== NVIDIA ==="
nvidia-smi 2>/dev/null || echo "nvidia-smi not available (reboot may be needed)"

echo ""
echo "=== unitokenizer ==="
python -c "from unitokenizer import UnigramTokenizer; print('unitokenizer: OK')" 2>/dev/null || \
    echo "unitokenizer: NOT BUILT (build after uploading project files)"

# ─── Setup convenience aliases ───────────────────────────────────────────────
cat >> /root/.bashrc << 'ALIASES'

# picoGPT aliases
alias activate='source /root/picoGPT/venv/bin/activate'
alias train='cd /root/picoGPT && source venv/bin/activate && python train.py'
alias tb='tensorboard --logdir /root/picoGPT/tensorboard --bind_all'
alias gpu='nvidia-smi'
alias gpuwatch='watch -n 1 nvidia-smi'
ALIASES

# ─── Setup tmux config for training sessions ─────────────────────────────────
cat > /root/.tmux.conf << 'TMUX'
set -g mouse on
set -g history-limit 50000
set -g status-interval 5
set -g status-right '#(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits)%% GPU | %H:%M'
TMUX

# ─── Done ─────────────────────────────────────────────────────────────────────
echo ""
echo "=========================================="
echo "picoGPT setup complete!"
echo "Finished at: $(date)"
echo "=========================================="
echo ""
echo "Next steps:"
echo "  1. Upload project files:"
echo "     scp -r ./picoGPT/* root@<droplet-ip>:/root/picoGPT/"
echo ""
echo "  2. SSH into the droplet and activate env:"
echo "     ssh root@<droplet-ip>"
echo "     source /root/picoGPT/venv/bin/activate"
echo ""
echo "  3. Build tokenizer (if not already built):"
echo "     cd /root/picoGPT/tokenizer && maturin develop --release"
echo ""
echo "  4. Start training in tmux:"
echo "     tmux new -s train"
echo "     cd /root/picoGPT && python train.py"
echo ""
echo "  5. Monitor GPU:"
echo "     watch -n 1 nvidia-smi"
echo ""
echo "  6. TensorBoard (optional):"
echo "     tensorboard --logdir /root/picoGPT/tensorboard --bind_all --port 6006"
echo ""
echo "Log file: $LOG_FILE"
