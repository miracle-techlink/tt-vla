#!/usr/bin/env python3
"""
Main entry point for TT-VLA experiments.

Usage:
  # Quick test with mock VLA (no GPU needed for model)
  python run_ttvla.py --mode test

  # Run with OpenVLA on ManiSkill3
  python run_ttvla.py --mode sim --vla openvla --gpu 0

  # Run baseline (no adaptation)
  python run_ttvla.py --mode baseline --vla openvla --gpu 0

  # Run specific task variant
  python run_ttvla.py --mode sim --vla openvla --variant unseen_obj_pos

  # Hyperparameter sweep
  python run_ttvla.py --mode sweep --vla openvla --gpu 0
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import json
import time
import torch
import numpy as np
from pathlib import Path


def load_config(config_path: str = None) -> dict:
    if config_path is None:
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "configs", "default.yaml"
        )
    with open(config_path) as f:
        return yaml.safe_load(f)


def run_test(config: dict):
    """Quick test with mock VLA and dummy environment."""
    from ttvla.vla_wrapper import SimpleMockVLA
    from ttvla.env_wrapper import DummyEnvWrapper
    from ttvla.reward import EnvOracleProgress
    from ttvla.tt_vla import TTVLA

    print("=" * 60)
    print("[TEST MODE] Running TT-VLA with mock VLA + dummy env")
    print("=" * 60)

    device = config["experiment"]["device"]
    if not torch.cuda.is_available():
        device = "cpu"
        config["experiment"]["device"] = "cpu"

    # Build components
    vla = SimpleMockVLA(device=device, action_dim=7)
    env = DummyEnvWrapper(config["env"])
    env.make()

    progress_est = EnvOracleProgress(env)

    # Build TT-VLA
    ttvla = TTVLA(vla, progress_est, config)

    # Apply LoRA to mock model
    from ttvla.lora_utils import apply_lora_to_vla, get_lora_optimizer
    # For mock model, we skip LoRA and use direct parameter updates
    ttvla.optimizer = torch.optim.AdamW(
        [p for p in vla.model.parameters() if p.requires_grad],
        lr=config["ttvla"]["learning_rate"],
        weight_decay=0.01,
    )

    # Run a few episodes
    print("\nRunning 3 test episodes...")
    for ep in range(3):
        result = ttvla.run_episode(env, seed=ep, verbose=True)
        print(f"\nEpisode {ep+1}: success={result.success}, "
              f"progress={result.final_progress:.3f}, "
              f"reward={result.total_reward:.4f}, "
              f"updates={result.num_updates}, "
              f"time={result.wall_time:.2f}s")

    print("\n[TEST] Pipeline test completed successfully!")


def run_sim(config: dict, args):
    """Run TT-VLA with real VLA model on ManiSkill3."""
    from ttvla.vla_wrapper import build_vla
    from ttvla.env_wrapper import build_env
    from ttvla.reward import build_progress_estimator
    from ttvla.tt_vla import TTVLA

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    config["experiment"]["device"] = device

    print("=" * 60)
    print(f"[SIM MODE] TT-VLA on ManiSkill3")
    print(f"  VLA: {args.vla}")
    print(f"  Device: {device}")
    print(f"  Variant: {args.variant}")
    print(f"  LR: {config['ttvla']['learning_rate']}")
    print(f"  LoRA rank: {config['lora']['rank']}")
    print(f"  Update interval K: {config['ttvla']['update_interval']}")
    print("=" * 60)

    # Override VLA model
    if args.vla == "openvla":
        config["vla"]["model_name"] = "openvla/openvla-7b"
    elif args.vla == "openvla-rl":
        config["vla"]["model_name"] = "openvla/openvla-7b-rl"
    elif args.vla == "nora":
        config["vla"]["model_name"] = "Embodied-CoT/Nora-FT-ManiSkill3"

    # Build components
    print("\n[1/4] Loading VLA model...")
    vla = build_vla(config["vla"], device=device)

    print("[2/4] Creating environment...")
    env = build_env(config["env"])
    env.make(task_variant=args.variant, seed=42)

    print("[3/4] Setting up progress estimator...")
    progress_est = build_progress_estimator(
        {**config["progress"], "device": device}, env=env
    )

    print("[4/4] Initializing TT-VLA...")
    ttvla = TTVLA(vla, progress_est, config)
    ttvla.setup_lora()

    # Run experiment
    log_dir = os.path.join(config["experiment"]["log_dir"], args.vla, args.variant)
    summary = ttvla.run_experiment(
        env,
        num_episodes=config["experiment"]["num_episodes"],
        seeds=config["experiment"]["seeds"],
        task_variant=args.variant,
        log_dir=log_dir,
        verbose=True,
    )

    env.close()
    return summary


def run_baseline(config: dict, args):
    """Run baseline without TT-VLA adaptation (for comparison)."""
    from ttvla.vla_wrapper import build_vla
    from ttvla.env_wrapper import build_env
    from ttvla.reward import build_progress_estimator

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"

    if args.vla == "openvla":
        config["vla"]["model_name"] = "openvla/openvla-7b"

    print("=" * 60)
    print(f"[BASELINE] No adaptation - {args.vla} on {args.variant}")
    print("=" * 60)

    vla = build_vla(config["vla"], device=device)
    env = build_env(config["env"])
    env.make(task_variant=args.variant, seed=42)
    progress_est = build_progress_estimator(
        {**config["progress"], "device": device}, env=env
    )

    results = []
    seeds = config["experiment"]["seeds"]
    num_per_seed = config["experiment"]["num_episodes"] // len(seeds)

    for seed in seeds:
        for ep in range(num_per_seed):
            episode_seed = seed * 10000 + ep
            image, instruction = env.reset(seed=episode_seed)
            progress_est.reset()

            done = False
            total_reward = 0
            steps = 0

            while not done:
                action, _ = vla.sample_action(image, instruction)
                image, _, done, info = env.step(action)
                reward = progress_est.compute_reward(image, instruction)
                total_reward += reward
                steps += 1

            eval_info = env.evaluate()
            success = bool(eval_info.get("success", False))
            results.append({"success": success, "reward": total_reward, "steps": steps})
            sr = sum(r["success"] for r in results) / len(results) * 100
            print(f"  Episode {len(results)}: {'OK' if success else 'FAIL'} | SR={sr:.1f}%")

    success_rate = sum(r["success"] for r in results) / len(results) * 100
    print(f"\n[BASELINE RESULT] {args.vla} on {args.variant}: SR={success_rate:.1f}%")

    log_dir = os.path.join(config["experiment"]["log_dir"], "baseline", args.vla, args.variant)
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "results.json"), "w") as f:
        json.dump({"success_rate": success_rate, "results": results}, f, indent=2)

    env.close()


def run_sweep(config: dict, args):
    """
    Hyperparameter sweep matching paper's search space.

    Paper Section 4.1:
      - LoRA ranks: {16, 32}
      - Learning rates: {1e-5, 5e-5, 1e-4}
      - Update intervals: {1, 4, 8, 16} (Table 2)
    """
    from ttvla.vla_wrapper import build_vla
    from ttvla.env_wrapper import build_env
    from ttvla.reward import build_progress_estimator
    from ttvla.tt_vla import TTVLA

    device = f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu"
    config["experiment"]["device"] = device

    sweep_params = {
        "lora_rank": [16, 32],
        "learning_rate": [1e-5, 5e-5, 1e-4],
        "update_interval": [1, 4, 8, 16],
    }

    print("=" * 60)
    print("[SWEEP MODE] Hyperparameter search")
    print(f"  Ranks: {sweep_params['lora_rank']}")
    print(f"  LRs: {sweep_params['learning_rate']}")
    print(f"  Update intervals: {sweep_params['update_interval']}")
    total = np.prod([len(v) for v in sweep_params.values()])
    print(f"  Total combinations: {total}")
    print("=" * 60)

    sweep_results = []

    for rank in sweep_params["lora_rank"]:
        for lr in sweep_params["learning_rate"]:
            for K in sweep_params["update_interval"]:
                config["lora"]["rank"] = rank
                config["lora"]["alpha"] = rank * 2
                config["ttvla"]["learning_rate"] = lr
                config["ttvla"]["update_interval"] = K

                # Use fewer episodes for sweep
                config["experiment"]["num_episodes"] = 30

                print(f"\n--- rank={rank}, lr={lr}, K={K} ---")

                vla = build_vla(config["vla"], device=device)
                env = build_env(config["env"])
                env.make(task_variant=args.variant, seed=42)
                progress_est = build_progress_estimator(
                    {**config["progress"], "device": device}, env=env
                )

                ttvla = TTVLA(vla, progress_est, config)
                ttvla.setup_lora()

                summary = ttvla.run_experiment(
                    env, num_episodes=30, seeds=[42],
                    task_variant=args.variant,
                    log_dir=f"logs/sweep/r{rank}_lr{lr}_K{K}",
                    verbose=False,
                )

                sweep_results.append({
                    "rank": rank, "lr": lr, "K": K,
                    "success_rate": summary["success_rate"],
                    "avg_reward": summary["avg_reward"],
                })

                print(f"  SR={summary['success_rate']:.1f}%")
                env.close()
                del vla, ttvla

    # Find best
    best = max(sweep_results, key=lambda x: x["success_rate"])
    print(f"\n[BEST] rank={best['rank']}, lr={best['lr']}, K={best['K']}, SR={best['success_rate']:.1f}%")

    with open("logs/sweep_results.json", "w") as f:
        json.dump(sweep_results, f, indent=2)


def main():
    parser = argparse.ArgumentParser(description="TT-VLA: Test-Time RL for VLAs")
    parser.add_argument("--mode", choices=["test", "sim", "baseline", "sweep"], default="test")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--vla", choices=["openvla", "openvla-rl", "nora", "tracevla"], default="openvla")
    parser.add_argument("--variant", type=str, default="default", help="Task variant for eval")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--rank", type=int, default=None)
    parser.add_argument("--update-interval", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=None)
    args = parser.parse_args()

    config = load_config(args.config)

    # Override from CLI
    if args.lr is not None:
        config["ttvla"]["learning_rate"] = args.lr
    if args.rank is not None:
        config["lora"]["rank"] = args.rank
        config["lora"]["alpha"] = args.rank * 2
    if args.update_interval is not None:
        config["ttvla"]["update_interval"] = args.update_interval
    if args.episodes is not None:
        config["experiment"]["num_episodes"] = args.episodes

    if args.mode == "test":
        run_test(config)
    elif args.mode == "sim":
        run_sim(config, args)
    elif args.mode == "baseline":
        run_baseline(config, args)
    elif args.mode == "sweep":
        run_sweep(config, args)


if __name__ == "__main__":
    main()
