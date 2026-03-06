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
- **Adaptation**: LoRA (rank 32) on attention layers, 33.5M trainable params (0.44%), reset per episode
- **Environment**: ManiSkill3 PickCube-v1, `pd_ee_delta_pose` control (7-dim EE actions)
- **Progress Estimator**: Environment oracle reward (grasped=0.4, placed=0.85, success=1.0)

## Project Structure

```
tt-vla/
├── configs/default.yaml              # All hyperparameters from paper
├── results/
│   ├── baseline_results.csv          # Summary: mock + OpenVLA baselines
│   ├── episode_details.csv           # Per-episode breakdown (15 episodes)
│   └── baseline_analysis.pdf         # 3-page analysis report
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
│   ├── test_maniskill.py             # ManiSkill3 integration test
│   ├── test_openvla_ttvla.py         # End-to-end OpenVLA + TT-VLA test
│   ├── warmup_finetune.py            # 10k-step warm-up fine-tuning
│   ├── generate_analysis_pdf.py      # Regenerate PDF report from CSV
│   └── download_openvla.py           # OpenVLA model download utility
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
| Warm-up steps | 10,000 |
| Action space | 7-dim EE delta pose + gripper |

## Baseline Results

| Model | Episodes | Success Rate | Avg Reward | Time/Episode |
|-------|----------|-------------|------------|-------------|
| Mock VLA (random) | 10 | 0% | 0.0 | 6.6s |
| OpenVLA-7B (no warm-up) | 5 | 0% | 0.0 | 74.4s |
| Scripted Expert | 10 | 100% | - | ~0.5s |

Both baselines show 0% SR as expected — the paper states warm-up fine-tuning is required before TT-VLA can improve performance. Paper reports ~15% SR after warm-up (no TT-VLA) and ~45% SR with TT-VLA on PickCube-v1.

## Quick Start

```bash
# 1. Setup environment
bash setup_server.sh

# 2. Test pipeline with mock VLA (no model download)
CUDA_VISIBLE_DEVICES=0 python scripts/test_openvla_ttvla.py --model mock --episodes 5

# 3. Test with OpenVLA-7B baseline
CUDA_VISIBLE_DEVICES=0,1 python scripts/test_openvla_ttvla.py --episodes 5 --rank 32

# 4. Warm-up fine-tuning (10k steps)
CUDA_VISIBLE_DEVICES=0,1 python scripts/warmup_finetune.py --steps 10000 --num-demos 500

# 5. Evaluate with TT-VLA after warm-up
CUDA_VISIBLE_DEVICES=0,1 python scripts/test_openvla_ttvla.py --episodes 20 --resume warmup_ckpt/warmup_final.pt
```

## Warm-up Fine-tuning

Per paper Section 4.1: "Consistent with (Liu et al., 2025a), we apply a warm-up fine-tuning phase of 10k steps prior to evaluation."

The warm-up script (`scripts/warmup_finetune.py`):
1. Collects demonstrations using a scripted heuristic policy (100% success rate)
2. Fine-tunes OpenVLA-7B with LoRA using supervised learning on demo data
3. Saves checkpoints every 1000 steps for resumable training

## Hardware

Tested on:
- **GPU**: NVIDIA GeForce RTX 5090 (32GB) × 2
- **Model loading**: `device_map="auto"` for multi-GPU sharding
- **Memory**: OpenVLA-7B requires ~14GB in BF16

## References

- TT-VLA: Zhai et al., "On-the-Fly Adaptation of Foundation Model-Based Robotic Manipulation Agents via Test-Time Reinforcement Learning", 2025
- OpenVLA: Kim et al., "OpenVLA: An Open-Source Vision-Language-Action Model", 2024
- ManiSkill3: Gu et al., 2023
