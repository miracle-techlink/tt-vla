#!/bin/bash
# ============================================================
# TT-VLA One-Click Launcher
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# --- Defaults ---
MODE="test"
GPU="0"
EPISODES=""
VLA=""
CONFIG="configs/default.yaml"
EXTRA_ARGS=""

usage() {
    cat <<EOF
TT-VLA One-Click Launcher

Usage: ./run.sh [MODE] [OPTIONS]

Modes:
  test      Quick smoke test with MockVLA (default, ~10s)
  train     Run TT-VLA with real/mock VLA
  warmup    Warm-up fine-tuning on ManiSkill3
  eval      Evaluate a checkpoint
  monitor   Real-time training monitor

Options:
  --gpu ID          GPU device(s), e.g. '0' or '0,1' (default: 0)
  --episodes N      Number of episodes
  --vla MODEL       VLA model: mock, openvla (default: from config)
  --config PATH     Config file (default: configs/default.yaml)
  --mock            Force mock VLA mode
  --checkpoint PATH Checkpoint for eval mode
  -h, --help        Show this help

Examples:
  ./run.sh                              # Quick smoke test
  ./run.sh test                         # Same as above
  ./run.sh train --mock --episodes 10   # Train with mock VLA
  ./run.sh train --vla openvla --gpu 0  # Train with OpenVLA
  ./run.sh monitor                      # Watch training progress
EOF
    exit 0
}

# --- Parse args ---
if [[ $# -gt 0 && ! "$1" =~ ^-- ]]; then
    MODE="$1"
    shift
fi

while [[ $# -gt 0 ]]; do
    case "$1" in
        --gpu)        GPU="$2"; shift 2 ;;
        --episodes)   EPISODES="$2"; shift 2 ;;
        --vla)        VLA="$2"; shift 2 ;;
        --config)     CONFIG="$2"; shift 2 ;;
        --mock)       VLA="mock"; shift ;;
        --checkpoint) CHECKPOINT="$2"; shift 2 ;;
        -h|--help)    usage ;;
        *)            EXTRA_ARGS="$EXTRA_ARGS $1"; shift ;;
    esac
done

# --- Pre-flight checks ---
preflight() {
    echo "Pre-flight checks..."

    # Python
    if ! command -v python &>/dev/null && ! command -v python3 &>/dev/null; then
        echo "ERROR: python not found. Install Python 3.10+" >&2
        exit 1
    fi
    PYTHON=$(command -v python3 || command -v python)

    # PyTorch + CUDA
    if ! $PYTHON -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
        echo "WARNING: CUDA not available. Running on CPU."
    fi

    # Core deps
    for pkg in torch peft yaml numpy; do
        if ! $PYTHON -c "import $pkg" 2>/dev/null; then
            echo "ERROR: Missing dependency: $pkg" >&2
            echo "  Run: pip install -r requirements.txt" >&2
            exit 1
        fi
    done

    # Config file
    if [[ ! -f "$CONFIG" ]]; then
        echo "ERROR: Config not found: $CONFIG" >&2
        exit 1
    fi

    # GPU info
    if command -v nvidia-smi &>/dev/null; then
        GPU_COUNT=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1)
        echo "  GPU: ${GPU_COUNT}x ${GPU_NAME}"
    fi

    echo "  Python: $($PYTHON --version 2>&1)"
    echo "  Config: $CONFIG"
    echo "  Mode: $MODE"
    echo ""
}

PYTHON=$(command -v python3 || command -v python)

# --- Run ---
case "$MODE" in
    test)
        preflight
        EPS=${EPISODES:-2}
        echo "Running smoke test ($EPS episodes, MockVLA)..."
        echo "================================================"
        CUDA_VISIBLE_DEVICES="$GPU" $PYTHON -m ttvla test --gpu 0 --episodes "$EPS" --config "$CONFIG" $EXTRA_ARGS
        ;;
    train)
        preflight
        EPS=${EPISODES:-80}
        CMD="CUDA_VISIBLE_DEVICES=$GPU $PYTHON -m ttvla train --gpu 0 --episodes $EPS --config $CONFIG"
        [[ "$VLA" == "mock" ]] && CMD="$CMD --mock"
        [[ -n "$VLA" && "$VLA" != "mock" ]] && CMD="$CMD --overrides vla.model_name=$VLA"
        CMD="$CMD $EXTRA_ARGS"
        echo "Starting TT-VLA training..."
        echo "  Episodes: $EPS"
        echo "  GPU: $GPU"
        echo "  VLA: ${VLA:-from config}"
        echo "================================================"
        eval $CMD
        ;;
    warmup)
        preflight
        STEPS=${EPISODES:-10000}
        MODEL=${VLA:-openvla/openvla-7b}
        echo "Starting warm-up fine-tuning..."
        echo "  Steps: $STEPS"
        echo "  Model: $MODEL"
        echo "================================================"
        CUDA_VISIBLE_DEVICES="$GPU" $PYTHON -m ttvla warmup --gpu 0 --steps "$STEPS" --model "$MODEL" $EXTRA_ARGS
        ;;
    eval)
        preflight
        if [[ -z "$CHECKPOINT" ]]; then
            echo "ERROR: --checkpoint required for eval mode" >&2
            exit 1
        fi
        EPS=${EPISODES:-20}
        CUDA_VISIBLE_DEVICES="$GPU" $PYTHON -m ttvla eval --gpu 0 --episodes "$EPS" --checkpoint "$CHECKPOINT" --config "$CONFIG" $EXTRA_ARGS
        ;;
    monitor)
        $PYTHON -m ttvla monitor $EXTRA_ARGS
        ;;
    *)
        echo "Unknown mode: $MODE"
        usage
        ;;
esac
