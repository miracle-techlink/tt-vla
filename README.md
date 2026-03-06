# TT-VLA: Test-Time Reinforcement Learning for VLA Adaptation

Reproduction of ["On-the-Fly Adaptation of Foundation Model-Based Robotic Manipulation Agents via Test-Time Reinforcement Learning"](https://arxiv.org/abs/2601.06748) (Zhai et al., 2025).

## Overview

TT-VLA adapts pre-trained Vision-Language-Action (VLA) models at test time using value-free PPO with dense progress-based rewards. No value function is needed — one-step advantage (γ=0, λ=0) simplifies the update to:

```
L(θ) = -E_t[min(r_t(θ)·r_t, clip(r_t(θ), 1-ε, 1+ε)·r_t)]
```

where `r_t = p_t - p_{t-1}` is the progress-based dense reward.

## Architecture

- **VLA Models**: OpenVLA-7B, OpenVLA-RL, Nora
- **Adaptation**: LoRA (rank 16/32) on attention layers, reset per episode
- **Environment**: ManiSkill3 (PickCube-v1, WidowX-250S)
- **Progress Estimator**: VLAC proxy (ResNet-18 based) or env oracle reward

## Project Structure

```
tt-vla/
├── configs/default.yaml          # All hyperparameters from paper
├── src/
│   ├── tt_vla.py                 # Core Algorithm 1: value-free PPO loop
│   ├── vla_wrapper.py            # OpenVLA + Mock VLA wrappers
│   ├── lora_utils.py             # LoRA apply/reset/optimizer
│   ├── reward.py                 # Progress estimators (oracle, heuristic)
│   ├── env_wrapper.py            # ManiSkill3 environment wrapper
│   └── progress_vlac_proxy.py    # Learned VLAC substitute
├── scripts/
│   ├── run_ttvla.py              # Main entry point (test/sim/sweep modes)
│   ├── run_all_experiments.py    # Full Table 1 experiments (15 variants)
│   ├── test_maniskill.py         # ManiSkill3 integration test
│   └── test_openvla_ttvla.py     # End-to-end OpenVLA + TT-VLA test
├── requirements.txt
└── setup_server.sh
```

## Key Hyperparameters (Paper Section 4.2)

| Parameter | Value |
|-----------|-------|
| LoRA rank | 16 or 32 |
| Learning rate | {1e-5, 5e-5, 1e-4} |
| Clip epsilon (ε) | 0.2 |
| Update interval (K) | 8 |
| Horizon (T) | 160 |
| GAE γ, λ | 0, 0 (one-step advantage) |

## Quick Start

```bash
# Test with mock VLA (no model download needed)
python scripts/run_ttvla.py --mode test

# Test with ManiSkill3 + mock VLA
python scripts/test_maniskill.py

# Full experiment with OpenVLA
CUDA_VISIBLE_DEVICES=0 python scripts/test_openvla_ttvla.py --episodes 5 --rank 32

# Run all Table 1 experiments
python scripts/run_all_experiments.py --gpu 0 --vla openvla
```

## Server Setup

```bash
bash setup_server.sh
```

## References

- TT-VLA: Zhai et al., "On-the-Fly Adaptation of Foundation Model-Based Robotic Manipulation Agents via Test-Time Reinforcement Learning", 2025
- OpenVLA: Kim et al., 2024
- VLAC: Zhai et al., 2025
- ManiSkill3: Gu et al., 2023
