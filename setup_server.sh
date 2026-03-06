#!/bin/bash
# ============================================================
# TT-VLA Server Setup Script
# Target: 8xRTX 5090, Ubuntu 24.04, CUDA 13.1
# ============================================================
set -e

echo "============================================"
echo "TT-VLA Server Environment Setup"
echo "============================================"

# --- 1. Setup directories ---
echo "[1/7] Setting up directories..."
WORK_DIR="/data/liuyue/tt-vla"
mkdir -p "$WORK_DIR"
mkdir -p "$WORK_DIR/logs"
mkdir -p "$WORK_DIR/checkpoints"
mkdir -p "$WORK_DIR/data"

# --- 2. Initialize conda ---
echo "[2/7] Setting up conda environment..."
export PATH="$HOME/miniconda3/bin:$PATH"
eval "$(conda shell.bash hook)"

# Create environment
conda create -n ttvla python=3.10 -y 2>/dev/null || true
conda activate ttvla

# --- 3. Install PyTorch (CUDA 12.4 for RTX 5090) ---
echo "[3/7] Installing PyTorch..."
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# --- 4. Install core dependencies ---
echo "[4/7] Installing core dependencies..."
pip install peft>=0.7.0 \
    transformers>=4.40.0 \
    accelerate>=0.27.0 \
    sentencepiece \
    protobuf \
    pyyaml \
    pillow \
    tqdm \
    matplotlib \
    tensorboard

# --- 5. Install ManiSkill3 ---
echo "[5/7] Installing ManiSkill3..."
pip install mani-skill
# ManiSkill3 requires additional assets
# python -m mani_skill.utils.download_asset "PickCube-v1"

# --- 6. Install OpenVLA dependencies ---
echo "[6/7] Installing VLA model dependencies..."
pip install timm
# Pre-download OpenVLA model (7B, ~14GB)
echo "To pre-download OpenVLA model, run:"
echo "  python -c \"from transformers import AutoModelForVision2Seq; AutoModelForVision2Seq.from_pretrained('openvla/openvla-7b')\""

# --- 7. Verify installation ---
echo "[7/7] Verifying installation..."
python -c "
import torch
print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
if torch.cuda.is_available():
    print(f'GPU count: {torch.cuda.device_count()}')
    for i in range(torch.cuda.device_count()):
        print(f'  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_mem / 1e9:.1f}GB)')

import peft
print(f'PEFT: {peft.__version__}')

import transformers
print(f'Transformers: {transformers.__version__}')

try:
    import mani_skill
    print(f'ManiSkill: {mani_skill.__version__}')
except:
    print('ManiSkill: not installed (optional for first test)')

print()
print('Environment ready!')
"

echo ""
echo "============================================"
echo "Setup complete!"
echo "Work directory: $WORK_DIR"
echo ""
echo "Quick test:"
echo "  conda activate ttvla"
echo "  cd $WORK_DIR"
echo "  python scripts/run_ttvla.py --mode test"
echo ""
echo "Full simulation:"
echo "  python scripts/run_ttvla.py --mode sim --vla openvla --gpu 0"
echo "============================================"
