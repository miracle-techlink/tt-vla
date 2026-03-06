# TT-VLA: Test-Time Reinforcement Learning for VLA Adaptation

Reproduction of ["On-the-Fly Adaptation of Foundation Model-Based Robotic Manipulation Agents via Test-Time Reinforcement Learning"](https://arxiv.org/abs/2601.06748) (Zhai et al., 2025).

## Overview

TT-VLA adapts pre-trained Vision-Language-Action (VLA) models at test time using value-free PPO with dense progress-based rewards. No value function is needed — one-step advantage (γ=0, λ=0) simplifies the update to:

```
L(θ) = -E_t[min(r_t(θ)·r_t, clip(r_t(θ), 1-ε, 1+ε)·r_t)]
```

where `r_t = p_t - p_{t-1}` is the progress-based dense reward.

## Architecture

- **VLA Models**: OpenVLA-7B (7.5B params, Llama-2-7B + DINOv2 + SigLIP)
- **Adaptation**: LoRA (rank 32) on attention layers, 79.9M trainable params (1.05%), reset per episode
- **Environment**: ManiSkill3 PickCube-v1, `pd_ee_delta_pose` control (7-dim EE actions)
- **Progress Estimator**: Environment oracle reward (grasped=0.4, placed=0.85, success=1.0)

## Experiment Results

### Warm-up Fine-tuning (Phase 1)

| Parameter | Value |
|-----------|-------|
| Training Method | LoRA (rank=32) + DDP (8 GPUs) |
| Training Data | 200 scripted demos (~17.8k transitions) |
| Training Steps | 3,000 (early-stopped from 10,000, loss < 0.30) |
| Final Loss | 0.216 |
| Training Speed | 3.3 steps/s (8× RTX 5090) |

### Evaluation Results

| Method | Episodes | Success Rate | Avg Reward | Avg Steps |
|--------|----------|-------------|------------|-----------|
| OpenVLA (no fine-tuning) | 5 | 0% | ~0 | 160 |
| **Warm-up only (step 3000)** | **20** | **0%** | **35.0** | **160** |
| Scripted Expert | 200 | 100% | - | ~100 |
| TT-VLA (warm-up + PPO) | - | Pending | - | - |

**Key Findings:**
- Warm-up achieved non-zero avg reward (35.0), confirming learned manipulation behavior
- Bimodal distribution: 40% episodes reach reward >50 (approach + grasp attempt)
- 0% task completion — motivates TT-VLA test-time adaptation
- See `results/warmup_eval_analysis.pdf` for detailed analysis

## Project Structure

```
tt-vla/
├── configs/default.yaml              # All hyperparameters from paper
├── results/
│   ├── warmup_eval_results.json      # Warm-up evaluation (20 episodes)
│   ├── warmup_eval_results.csv       # Per-episode breakdown
│   ├── warmup_eval_analysis.pdf      # Analysis report with charts
│   └── warmup_eval_analysis.md       # Markdown version of report
├── src/
│   ├── tt_vla.py                     # Core Algorithm 1: value-free PPO loop
│   ├── vla_wrapper.py                # OpenVLA + Mock VLA wrappers
│   ├── lora_utils.py                 # LoRA apply/reset/optimizer
│   ├── reward.py                     # Progress estimators (oracle, heuristic)
│   ├── env_wrapper.py                # ManiSkill3 environment wrapper
│   └── progress_vlac_proxy.py        # Learned VLAC substitute
├── scripts/
│   ├── run_ttvla.py                  # Main entry point
│   ├── run_all_experiments.py        # Full Table 1 experiments
│   ├── eval_warmup.py                # Warm-up checkpoint evaluation
│   ├── warmup_finetune.py            # DDP warm-up fine-tuning (8-GPU)
│   ├── train_monitor.py              # Live web dashboard (port 8899)
│   ├── test_maniskill.py             # ManiSkill3 integration test
│   ├── test_openvla_ttvla.py         # End-to-end OpenVLA + TT-VLA test
│   ├── generate_analysis_pdf.py      # Regenerate PDF report from CSV
│   └── download_openvla.py           # OpenVLA model download utility
├── warmup_ckpt/                      # Training checkpoints (not in git)
│   ├── step_1000.pt ... step_3000.pt
│   └── loss_curve.json
├── requirements.txt
└── setup_server.sh
```

## Key Hyperparameters (Paper Section 4.2)

| Parameter | Value |
|-----------|-------|
| LoRA rank | 32 |
| Learning rate | 1e-4 (TT-VLA), 2e-5 (warm-up) |
| Clip epsilon (ε) | 0.2 |
| Update interval (K) | 8 steps |
| Max horizon (T) | 160 steps |
| GAE γ, λ | 0, 0 (one-step advantage) |
| Warm-up steps | 3,000 (early-stopped) |
| Action space | 7-dim EE delta pose + gripper |

## Quick Start

```bash
# 1. Setup environment
bash setup_server.sh

# 2. Warm-up fine-tuning (8-GPU DDP)
torchrun --nproc_per_node=8 scripts/warmup_finetune.py --steps 3000 --batch-size 2 --num-demos 200

# 3. Evaluate warm-up checkpoint (no TT-VLA)
CUDA_VISIBLE_DEVICES=0,1 python scripts/eval_warmup.py --checkpoint warmup_ckpt/step_3000.pt --episodes 20

# 4. Monitor training progress (web dashboard)
python scripts/train_monitor.py  # http://localhost:8899

# 5. Run TT-VLA (warm-up + test-time PPO)
CUDA_VISIBLE_DEVICES=0,1 python scripts/test_openvla_ttvla.py --episodes 20 --resume warmup_ckpt/step_3000.pt
```

## Hardware

- **GPU**: 8× NVIDIA GeForce RTX 5090 (32GB each)
- **Training**: DDP across all 8 GPUs, ~16.8GB peak per GPU
- **Inference**: 2 GPUs with `device_map="auto"`
- **Model Size**: OpenVLA-7B ~14GB in BF16

## References

- TT-VLA: Zhai et al., "On-the-Fly Adaptation of Foundation Model-Based Robotic Manipulation Agents via Test-Time Reinforcement Learning", 2025
- OpenVLA: Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model", 2024
- ManiSkill3: Gu et al., 2023
