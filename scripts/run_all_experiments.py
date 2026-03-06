#!/usr/bin/env python3
"""
Run all TT-VLA experiments from the paper (Table 1).

This script runs TT-VLA across all evaluation dimensions:
  - Execution: 3 task variants
  - Vision: 5 task variants
  - Semantics: 7 task variants

For each variant, runs both baseline (no adaptation) and TT-VLA,
then computes relative improvement (Δ and ↑ from Table 1).

Usage:
  # Run all experiments on GPU 0
  python run_all_experiments.py --gpu 0 --vla openvla

  # Run only execution dimension
  python run_all_experiments.py --gpu 0 --vla openvla --dimension execution

  # Quick test (fewer episodes)
  python run_all_experiments.py --gpu 0 --vla openvla --quick
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import yaml
import json
import torch
from pathlib import Path


# Paper Table 1: all evaluation task variants
EVAL_TASKS = {
    "execution": [
        "unseen_obj_pos",      # Obj. Pos.
        "unseen_robot_pose",   # Robot Pose
        "obj_reposition",      # Obj. Rep.
    ],
    "vision": [
        "unseen_table",        # Table
        "texture_weak",        # Texture-w
        "texture_strong",      # Texture-s
        "noise_weak",          # Noise-w
        "noise_strong",        # Noise-s
    ],
    "semantics": [
        "multi_obj_ood",       # M-Obj. OOD
        "unseen_instruct",     # Instruct
        "dist_recep",          # Dist Recep.
        "unseen_recep",        # Recep.
        "multi_recep",         # M Recep.
        "unseen_object",       # Object
        "multi_obj_ind",       # M-Obj. IND
    ],
}


def run_experiment_pair(config, vla_name, variant, gpu, num_episodes):
    """Run both baseline and TT-VLA for a single task variant."""
    from src.vla_wrapper import build_vla
    from src.env_wrapper import build_env
    from src.reward import build_progress_estimator
    from src.tt_vla import TTVLA

    device = f"cuda:{gpu}"
    config["experiment"]["device"] = device
    config["experiment"]["num_episodes"] = num_episodes

    # Map VLA name
    vla_models = {
        "openvla": "openvla/openvla-7b",
        "openvla-rl": "openvla/openvla-7b-rl",
        "nora": "Embodied-CoT/Nora-FT-ManiSkill3",
    }
    config["vla"]["model_name"] = vla_models.get(vla_name, vla_name)

    print(f"\n{'='*60}")
    print(f"Task: {variant} | VLA: {vla_name}")
    print(f"{'='*60}")

    # --- Baseline (no adaptation) ---
    print("\n[Baseline] Running without adaptation...")
    vla = build_vla(config["vla"], device=device)
    env = build_env(config["env"])
    env.make(task_variant=variant, seed=42)
    progress_est = build_progress_estimator(
        {**config["progress"], "device": device}, env=env
    )

    baseline_successes = 0
    baseline_results = []
    seeds = config["experiment"]["seeds"]

    for seed in seeds:
        for ep in range(num_episodes // len(seeds)):
            ep_seed = seed * 10000 + ep
            image, instruction = env.reset(seed=ep_seed)
            progress_est.reset()
            done = False
            while not done:
                action, _ = vla.sample_action(image, instruction)
                image, _, done, info = env.step(action)
            eval_info = env.evaluate()
            success = bool(eval_info.get("success", False))
            baseline_successes += int(success)
            baseline_results.append(success)

    baseline_sr = baseline_successes / len(baseline_results) * 100
    print(f"  Baseline SR: {baseline_sr:.2f}%")

    # --- TT-VLA ---
    print("\n[TT-VLA] Running with test-time adaptation...")
    ttvla = TTVLA(vla, progress_est, config)
    ttvla.setup_lora()

    log_dir = f"logs/{vla_name}/{variant}"
    summary = ttvla.run_experiment(
        env,
        num_episodes=num_episodes,
        seeds=seeds,
        task_variant=variant,
        log_dir=log_dir,
        verbose=False,
    )

    ttvla_sr = summary["success_rate"]
    delta = ttvla_sr - baseline_sr
    relative_gain = delta / max(baseline_sr, 0.01) * 100

    print(f"\n  Baseline: {baseline_sr:.2f}%")
    print(f"  TT-VLA:  {ttvla_sr:.2f}%")
    print(f"  Δ:       {'+' if delta >= 0 else ''}{delta:.2f}%")
    print(f"  ↑:       {'+' if relative_gain >= 0 else ''}{relative_gain:.1f}%")

    env.close()
    del vla, ttvla

    return {
        "variant": variant,
        "baseline_sr": baseline_sr,
        "ttvla_sr": ttvla_sr,
        "delta": delta,
        "relative_gain": relative_gain,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--vla", default="openvla")
    parser.add_argument("--dimension", choices=["execution", "vision", "semantics", "all"], default="all")
    parser.add_argument("--quick", action="store_true", help="Run fewer episodes")
    parser.add_argument("--config", default=None)
    args = parser.parse_args()

    config_path = args.config or os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        "configs", "default.yaml"
    )
    with open(config_path) as f:
        config = yaml.safe_load(f)

    num_episodes = 12 if args.quick else 80

    # Select dimensions
    dimensions = list(EVAL_TASKS.keys()) if args.dimension == "all" else [args.dimension]

    all_results = {}
    for dim in dimensions:
        print(f"\n\n{'#'*60}")
        print(f"# Dimension: {dim.upper()}")
        print(f"{'#'*60}")

        dim_results = []
        for variant in EVAL_TASKS[dim]:
            result = run_experiment_pair(config, args.vla, variant, args.gpu, num_episodes)
            dim_results.append(result)

        # Dimension average
        avg_baseline = sum(r["baseline_sr"] for r in dim_results) / len(dim_results)
        avg_ttvla = sum(r["ttvla_sr"] for r in dim_results) / len(dim_results)
        avg_delta = avg_ttvla - avg_baseline

        print(f"\n{'='*60}")
        print(f"[{dim.upper()} SUMMARY]")
        print(f"  Avg Baseline: {avg_baseline:.2f}%")
        print(f"  Avg TT-VLA:   {avg_ttvla:.2f}%")
        print(f"  Avg Δ:        {'+' if avg_delta >= 0 else ''}{avg_delta:.2f}%")
        print(f"{'='*60}")

        all_results[dim] = {
            "tasks": dim_results,
            "avg_baseline": avg_baseline,
            "avg_ttvla": avg_ttvla,
            "avg_delta": avg_delta,
        }

    # Save all results
    os.makedirs("logs", exist_ok=True)
    with open(f"logs/full_results_{args.vla}.json", "w") as f:
        json.dump(all_results, f, indent=2)

    # Print final table
    print(f"\n\n{'='*80}")
    print(f"FINAL RESULTS: {args.vla} + TT-VLA")
    print(f"{'='*80}")
    print(f"{'Variant':<25} {'Baseline':>10} {'TT-VLA':>10} {'Δ':>10} {'↑':>10}")
    print("-" * 65)
    for dim in all_results:
        for r in all_results[dim]["tasks"]:
            print(f"  {r['variant']:<23} {r['baseline_sr']:>9.2f}% {r['ttvla_sr']:>9.2f}% "
                  f"{r['delta']:>+9.2f}% {r['relative_gain']:>+9.1f}%")
        avg = all_results[dim]
        print(f"  {'Avg ' + dim:<23} {avg['avg_baseline']:>9.2f}% {avg['avg_ttvla']:>9.2f}% "
              f"{avg['avg_delta']:>+9.2f}%")
        print()


if __name__ == "__main__":
    main()
