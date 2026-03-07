"""
Unified TT-VLA Trainer.

Consolidates the training logic from test_openvla_ttvla.py and run_ttvla.py
into a single, clean interface.
"""

import os
import json
import time
import numpy as np
import torch
from pathlib import Path
from typing import Optional, Dict

from .tt_vla import TTVLA
from .reward import build_progress_estimator
from .lora_utils import apply_lora_to_vla


class TTVLATrainer:
    """Unified trainer for TT-VLA experiments."""

    def __init__(self, config: dict):
        self.config = config
        requested_device = config.get("experiment", {}).get("device", "cuda:0")
        if "cuda" in requested_device and not torch.cuda.is_available():
            print("[Trainer] CUDA not available, falling back to CPU")
            requested_device = "cpu"
        self.device = requested_device
        self.log_dir = Path(config.get("experiment", {}).get("log_dir", "logs/"))
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.vla = None
        self.env = None
        self.ttvla = None
        self._jsonl_path = self.log_dir / "ttvla_train.jsonl"

    def _build_vla(self):
        """Build VLA model based on config."""
        vla_cfg = self.config.get("vla", {})
        model_name = vla_cfg.get("model_name", "openvla/openvla-7b")
        action_dim = vla_cfg.get("action_dim", 7)

        if model_name == "mock":
            from .vla_wrapper import SimpleMockVLA
            vla = SimpleMockVLA(device=self.device, action_dim=action_dim)
            vla.peft_model = None
            print(f"[Trainer] Using MockVLA (action_dim={action_dim})")
        else:
            os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
            from .vla_wrapper import OpenVLAWrapper
            vla = OpenVLAWrapper(model_name, device=self.device, action_dim=action_dim)
            vla.load()

            lora_cfg = self.config.get("lora", {})
            rank = lora_cfg.get("rank", 32)
            alpha = lora_cfg.get("alpha", 64)
            targets = lora_cfg.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"])

            print(f"[Trainer] Applying LoRA (rank={rank})...")
            vla.peft_model = apply_lora_to_vla(vla.model, rank=rank, alpha=alpha, target_modules=targets)
            vla.model = vla.peft_model

        return vla

    def _build_env(self):
        """Build ManiSkill3 environment."""
        env_cfg = self.config.get("env", {})
        from .env_wrapper import ManiSkillEnvWrapper, DummyEnvWrapper

        vla_name = self.config.get("vla", {}).get("model_name", "")
        if vla_name == "mock":
            env = DummyEnvWrapper(env_cfg)
            env.make()
            return env

        env = ManiSkillEnvWrapper(env_cfg)
        env.make()
        return env

    def run(self, num_episodes: int = 80, verbose: bool = True) -> Dict:
        """Run TT-VLA experiment."""
        self.vla = self._build_vla()
        self.env = self._build_env()

        # Build progress estimator
        progress_cfg = self.config.get("progress", {})
        progress_est = build_progress_estimator(progress_cfg, env=self.env)

        # Build TT-VLA
        self.ttvla = TTVLA(self.vla, progress_est, self.config)
        is_mock = self.config.get("vla", {}).get("model_name", "") == "mock"
        if not is_mock:
            self.ttvla.setup_lora()

        seeds = self.config.get("experiment", {}).get("seeds", [42, 123, 456])

        results = []
        successes = 0

        print(f"\n{'='*60}")
        print(f"TT-VLA Experiment: {num_episodes} episodes")
        print(f"  Model: {self.config.get('vla', {}).get('model_name', '?')}")
        print(f"  LoRA rank: {self.config.get('lora', {}).get('rank', '?')}")
        ttvla_cfg = self.config.get("ttvla", {})
        print(f"  LR: {ttvla_cfg.get('learning_rate', '?')}, "
              f"K: {ttvla_cfg.get('update_interval', '?')}, "
              f"eps: {ttvla_cfg.get('clip_epsilon', '?')}")
        print(f"{'='*60}\n")

        for ep in range(num_episodes):
            seed = seeds[ep % len(seeds)] * 10000 + ep

            if verbose:
                print(f"[Episode {ep+1}/{num_episodes}]")

            start = time.time()
            result = self.ttvla.run_episode(self.env, seed=seed, verbose=verbose)
            elapsed = time.time() - start

            if result.success:
                successes += 1

            ep_data = {
                "episode": ep + 1,
                "seed": seed,
                "success": result.success,
                "total_reward": float(result.total_reward),
                "final_progress": float(result.final_progress),
                "num_steps": result.num_steps,
                "num_updates": result.num_updates,
                "avg_loss": float(np.mean(result.policy_losses)) if result.policy_losses else 0.0,
                "wall_time": elapsed,
            }
            results.append(ep_data)

            # Write JSONL for real-time monitoring
            with open(self._jsonl_path, "a") as f:
                f.write(json.dumps(ep_data) + "\n")

            if verbose:
                sr = successes / len(results) * 100
                print(f"  -> {'SUCCESS' if result.success else 'FAIL'} | "
                      f"progress={result.final_progress:.3f} | "
                      f"reward={result.total_reward:.4f} | "
                      f"loss={ep_data['avg_loss']:.4f} | "
                      f"time={elapsed:.1f}s | "
                      f"SR={sr:.1f}%\n")

        # Summary
        success_rate = successes / len(results) * 100
        avg_reward = np.mean([r["total_reward"] for r in results])
        avg_progress = np.mean([r["final_progress"] for r in results])

        summary = {
            "success_rate": success_rate,
            "avg_reward": float(avg_reward),
            "avg_progress": float(avg_progress),
            "num_episodes": len(results),
            "results": results,
        }

        print(f"{'='*60}")
        print(f"RESULTS")
        print(f"  Success Rate: {success_rate:.1f}%")
        print(f"  Avg Reward: {avg_reward:.4f}")
        print(f"  Avg Progress: {avg_progress:.3f}")
        print(f"{'='*60}")

        # Save results
        result_path = self.log_dir / "ttvla_results.json"
        with open(result_path, "w") as f:
            json.dump(summary, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else o)
        print(f"Results saved to {result_path}")

        if self.env:
            self.env.close()

        return summary

    def evaluate(self, checkpoint: str, num_episodes: int = 20):
        """Evaluate a saved LoRA checkpoint."""
        self.vla = self._build_vla()

        # Load LoRA weights
        if checkpoint and os.path.exists(checkpoint):
            print(f"[Trainer] Loading checkpoint: {checkpoint}")
            state = torch.load(checkpoint, map_location=self.device, weights_only=False)
            self.vla.model.load_state_dict(state, strict=False)

        self.env = self._build_env()
        progress_cfg = self.config.get("progress", {})
        progress_est = build_progress_estimator(progress_cfg, env=self.env)

        self.ttvla = TTVLA(self.vla, progress_est, self.config)
        # Don't setup LoRA again — weights already loaded

        results = []
        successes = 0

        for ep in range(num_episodes):
            result = self.ttvla.run_episode(self.env, seed=ep * 42, verbose=True)
            if result.success:
                successes += 1
            results.append({"success": result.success, "progress": result.final_progress})
            sr = successes / len(results) * 100
            print(f"  Episode {ep+1}: {'SUCCESS' if result.success else 'FAIL'} | "
                  f"progress={result.final_progress:.3f} | SR={sr:.1f}%")

        print(f"\nEval SR: {successes}/{num_episodes} = {successes/num_episodes*100:.1f}%")
        self.env.close()
