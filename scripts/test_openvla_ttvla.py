#!/usr/bin/env python3
"""
End-to-end test: TT-VLA with OpenVLA on ManiSkill3 PickCube.

This is the main integration test that matches the paper's experimental setup:
  - OpenVLA-7B as base VLA policy
  - ManiSkill3 PickCube-v1 environment
  - LoRA rank=32 for test-time adaptation
  - Value-free PPO with dense progress reward
  - Update interval K=8

Usage:
  CUDA_VISIBLE_DEVICES=0 python test_openvla_ttvla.py
  CUDA_VISIBLE_DEVICES=0 python test_openvla_ttvla.py --episodes 10 --rank 16 --lr 5e-5
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import torch
import numpy as np
import time
import json
import gymnasium as gym
import mani_skill.envs  # noqa


class ManiSkillVLAEnv:
    """
    ManiSkill3 environment adapter for TT-VLA.
    Handles observation extraction, action formatting, and progress tracking.
    """

    def __init__(self, env_id="PickCube-v1", image_size=(480, 640)):
        self.env = gym.make(
            env_id,
            obs_mode="rgb",
            render_mode="cameras",
            max_episode_steps=160,
        )
        self.image_size = image_size
        self._instruction = "pick up the cube and place it on the target"
        self._step_count = 0

    def reset(self, seed=None):
        self._step_count = 0
        obs, info = self.env.reset(seed=seed)
        image = self._extract_rgb(obs)
        return image, self._instruction

    def step(self, action):
        self._step_count += 1
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = np.array(action, dtype=np.float32).flatten()

        # Pad/truncate to match env action space
        env_action_dim = self.env.action_space.shape[0]
        if len(action) < env_action_dim:
            action = np.concatenate([action, np.zeros(env_action_dim - len(action))])
        elif len(action) > env_action_dim:
            action = action[:env_action_dim]

        obs, reward, term, trunc, info = self.env.step(action)
        image = self._extract_rgb(obs)
        done = bool(term) or bool(trunc)
        return image, float(reward), done, info

    def _extract_rgb(self, obs):
        """Extract RGB image from ManiSkill3 observation dict."""
        if isinstance(obs, dict) and "sensor_data" in obs:
            for cam_name in obs["sensor_data"]:
                rgb = obs["sensor_data"][cam_name]["rgb"]
                if isinstance(rgb, torch.Tensor):
                    rgb = rgb.squeeze(0).cpu().numpy()
                return rgb
        return np.zeros((*self.image_size, 3), dtype=np.uint8)

    def evaluate(self):
        return self.env.unwrapped.evaluate()

    def get_dense_reward(self):
        """Get ManiSkill3's built-in dense reward as progress proxy."""
        eval_info = self.evaluate()
        success = bool(eval_info.get("success", False))
        is_grasped = bool(eval_info.get("is_grasped", False))
        is_placed = bool(eval_info.get("is_obj_placed", False))
        if success:
            return 1.0
        elif is_placed:
            return 0.85
        elif is_grasped:
            return 0.4
        return 0.0

    def close(self):
        self.env.close()

    @property
    def instruction(self):
        return self._instruction


class EnvProgressEstimator:
    """Progress estimator using environment ground-truth signals."""

    def __init__(self, env: ManiSkillVLAEnv):
        self.env = env
        self._prev_progress = 0.0

    def reset(self):
        self._prev_progress = 0.0

    def compute_reward(self, obs, instruction):
        p_t = self.env.get_dense_reward()
        r_t = p_t - self._prev_progress
        self._prev_progress = p_t
        return r_t

    @property
    def current_progress(self):
        return self._prev_progress


def run_ttvla_episode(vla, env, progress_est, config, seed=None, verbose=True):
    """Run a single TT-VLA episode (Algorithm 1)."""
    from src.lora_utils import get_lora_optimizer, reset_lora_parameters

    K = config["update_interval"]
    eps = config["clip_epsilon"]
    lr = config["learning_rate"]

    # Reset
    progress_est.reset()
    image, instruction = env.reset(seed=seed)

    # Reset LoRA for fresh episode
    if hasattr(vla, 'peft_model') and vla.peft_model is not None:
        reset_lora_parameters(vla.peft_model)
        optimizer = get_lora_optimizer(vla.peft_model, lr=lr)
    else:
        optimizer = torch.optim.AdamW(
            [p for p in vla.get_model().parameters() if p.requires_grad],
            lr=lr, weight_decay=0.01,
        )

    buffer = []  # List of (obs, action, reward, log_prob_old)
    total_reward = 0
    num_updates = 0
    t = 0
    done = False

    while not done:
        # Sample action
        action, log_prob_old = vla.sample_action(image, instruction)

        # Execute
        next_image, env_reward, done, info = env.step(action)

        # Dense reward
        reward = progress_est.compute_reward(next_image, instruction)

        # Store in buffer
        buffer.append((image, action, reward, log_prob_old))
        total_reward += reward

        # PPO update every K steps
        if (t + 1) % K == 0 and len(buffer) > 0:
            loss = ppo_update(vla, optimizer, buffer, instruction, eps, vla.device)
            num_updates += 1
            buffer.clear()

            if verbose and (t + 1) % (K * 2) == 0:
                print(f"  Step {t+1}: progress={progress_est.current_progress:.3f}, "
                      f"reward={reward:.4f}, loss={loss:.4f}")

        image = next_image
        t += 1

    eval_info = env.evaluate()
    success = bool(eval_info.get("success", False))

    return {
        "success": success,
        "total_reward": total_reward,
        "final_progress": progress_est.current_progress,
        "num_steps": t,
        "num_updates": num_updates,
    }


def ppo_update(vla, optimizer, buffer, instruction, clip_eps, device):
    """
    Value-free PPO update (Algorithm 1, lines 10-14).

    L_i = -min(r_i(θ)·Â_i, clip(r_i(θ), 1-ε, 1+ε)·Â_i)
    where Â_i = r_i (one-step advantage, Eq. 9)
    """
    total_loss = torch.tensor(0.0, device=device, requires_grad=True)

    for obs, action, reward, log_prob_old in buffer:
        # Compute current log prob
        log_prob_new = vla.compute_log_prob(obs, instruction, action)

        if isinstance(log_prob_old, (int, float)):
            log_prob_old_t = torch.tensor(log_prob_old, dtype=torch.float32, device=device)
        else:
            log_prob_old_t = torch.tensor(float(log_prob_old), dtype=torch.float32, device=device)

        # Importance ratio
        ratio = torch.exp(log_prob_new.float() - log_prob_old_t)

        # Advantage = reward (one-step)
        advantage = torch.tensor(reward, dtype=torch.float32, device=device)

        # Clipped surrogate
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantage
        loss_i = -torch.min(surr1, surr2)

        total_loss = total_loss + loss_i

    avg_loss = total_loss / len(buffer)

    optimizer.zero_grad()
    avg_loss.backward()
    torch.nn.utils.clip_grad_norm_(
        [p for p in vla.get_model().parameters() if p.requires_grad], 1.0
    )
    optimizer.step()

    return avg_loss.item()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=5)
    parser.add_argument("--rank", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--update-interval", type=int, default=8)
    parser.add_argument("--clip-eps", type=float, default=0.2)
    parser.add_argument("--model", default="openvla/openvla-7b")
    parser.add_argument("--mock", action="store_true", help="Use mock VLA for testing")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    config = {
        "update_interval": args.update_interval,
        "clip_epsilon": args.clip_eps,
        "learning_rate": args.lr,
    }

    # Load VLA
    if args.mock:
        from src.vla_wrapper import SimpleMockVLA
        vla = SimpleMockVLA(device=device, action_dim=8)
        vla.peft_model = None
    else:
        from src.vla_wrapper import OpenVLAWrapper
        from src.lora_utils import apply_lora_to_vla

        print(f"\nLoading {args.model}...")
        vla = OpenVLAWrapper(args.model, device=device, action_dim=7)
        vla.load()

        print(f"Applying LoRA (rank={args.rank})...")
        vla.peft_model = apply_lora_to_vla(
            vla.model,
            rank=args.rank,
            alpha=args.rank * 2,
            target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        )
        vla.model = vla.peft_model

    # Create environment
    print("\nCreating ManiSkill3 PickCube-v1...")
    env = ManiSkillVLAEnv("PickCube-v1")
    progress_est = EnvProgressEstimator(env)

    # Run episodes
    print(f"\n{'='*60}")
    print(f"TT-VLA Experiment: {args.episodes} episodes")
    print(f"  Model: {args.model if not args.mock else 'MockVLA'}")
    print(f"  LoRA rank: {args.rank}")
    print(f"  LR: {args.lr}, K: {args.update_interval}, ε: {args.clip_eps}")
    print(f"{'='*60}\n")

    results = []
    for ep in range(args.episodes):
        print(f"[Episode {ep+1}/{args.episodes}]")
        start = time.time()
        result = run_ttvla_episode(vla, env, progress_est, config, seed=ep * 42, verbose=True)
        elapsed = time.time() - start

        results.append(result)
        sr = sum(r["success"] for r in results) / len(results) * 100
        print(f"  -> {'SUCCESS' if result['success'] else 'FAIL'} | "
              f"progress={result['final_progress']:.3f} | "
              f"reward={result['total_reward']:.4f} | "
              f"updates={result['num_updates']} | "
              f"time={elapsed:.1f}s | "
              f"running SR={sr:.1f}%\n")

    # Summary
    success_rate = sum(r["success"] for r in results) / len(results) * 100
    avg_reward = np.mean([r["total_reward"] for r in results])

    print(f"{'='*60}")
    print(f"RESULTS")
    print(f"  Success Rate: {success_rate:.1f}%")
    print(f"  Avg Reward: {avg_reward:.4f}")
    print(f"  Avg Progress: {np.mean([r['final_progress'] for r in results]):.3f}")
    print(f"{'='*60}")

    # Save
    os.makedirs("logs", exist_ok=True)
    with open("logs/ttvla_test_results.json", "w") as f:
        json.dump({"success_rate": success_rate, "results": results}, f, indent=2)


if __name__ == "__main__":
    main()
