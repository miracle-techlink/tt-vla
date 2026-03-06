"""
TT-VLA: Test-Time Reinforcement Learning for VLA.

Core implementation of Algorithm 1 from the paper:
  "On-the-Fly VLA Adaptation via Test-Time Reinforcement Learning"
  (arXiv:2601.06748)

Key design choices:
  1. Value-free PPO: c1=0, c2=0 (no value loss, no entropy bonus)
     => L(θ) = E_t[L_t^CLIP(θ)]                             (Eq. 8)
  2. One-step advantage: γ=0, λ=0
     => Â_t = δ_t = r_t                                     (Eq. 9)
  3. Dense progress-based reward:
     r_t = p_t - p_{t-1}                                     (Eq. 7)
  4. LoRA-only updates to preserve pretrained priors
  5. Episodic buffer cleared after each policy update
"""

import torch
import numpy as np
import time
import json
import os
from dataclasses import dataclass, field
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path

from .reward import ProgressEstimator
from .lora_utils import apply_lora_to_vla, reset_lora_parameters, get_lora_optimizer


@dataclass
class TransitionBuffer:
    """
    Experience buffer B for collecting transitions within an episode.

    Stores: (observation, action, reward, log_prob_old)
    Cleared after each policy update (Algorithm 1, line 14).
    """
    observations: List[np.ndarray] = field(default_factory=list)
    actions: List[np.ndarray] = field(default_factory=list)
    rewards: List[float] = field(default_factory=list)
    log_probs_old: List[float] = field(default_factory=list)

    def add(self, obs: np.ndarray, action: np.ndarray, reward: float, log_prob_old: float):
        self.observations.append(obs)
        self.actions.append(action)
        self.rewards.append(reward)
        self.log_probs_old.append(log_prob_old)

    def clear(self):
        self.observations.clear()
        self.actions.clear()
        self.rewards.clear()
        self.log_probs_old.clear()

    def __len__(self):
        return len(self.rewards)


@dataclass
class EpisodeResult:
    """Result of a single episode."""
    success: bool = False
    total_reward: float = 0.0
    num_steps: int = 0
    num_updates: int = 0
    final_progress: float = 0.0
    policy_losses: List[float] = field(default_factory=list)
    progress_history: List[float] = field(default_factory=list)
    reward_history: List[float] = field(default_factory=list)
    wall_time: float = 0.0


class TTVLA:
    """
    Test-Time Reinforcement Learning for VLA (TT-VLA).

    Implements the value-free PPO variant from the paper:
      - No value function V(s) needed
      - No entropy regularization
      - One-step advantage Â_t = r_t
      - Clipped surrogate objective for stable updates

    Algorithm 1 pseudocode (paper page 17):
      Input: pretrained VLA π_θ, frozen progress estimator Φ,
             instruction l, obs horizon H, update interval K,
             clip threshold ε, learning rate η
      For each episode:
        1. Load pretrained VLA π_θ; p_0 ← 0; B ← ∅
        2. For t = 0, 1, ..., T:
           a. Sample a_t ~ π_θ(a_t | o_{t-H+1:t}, l), get log π_old, execute a_t
           b. Get new observation o_{t+1}
           c. Compute p_t ← Φ(o_{0:t}, l)
           d. Compute r_t ← p_t - p_{t-1}
           e. Store (o_{t+1}, a_t, r_t, log π_old(a_t|o_t)) in B
           f. If t mod K == 0: PPO update on B, clear B
    """

    def __init__(
        self,
        vla_wrapper,
        progress_estimator: ProgressEstimator,
        config: dict,
    ):
        self.vla = vla_wrapper
        self.progress_estimator = progress_estimator

        # TT-VLA hyperparameters
        ttvla_cfg = config.get("ttvla", {})
        self.lr = ttvla_cfg.get("learning_rate", 1e-4)
        self.clip_epsilon = ttvla_cfg.get("clip_epsilon", 0.2)
        self.update_interval = ttvla_cfg.get("update_interval", 8)
        self.gamma = ttvla_cfg.get("gamma", 0.0)
        self.lam = ttvla_cfg.get("lam", 0.0)
        self.max_grad_norm = ttvla_cfg.get("max_grad_norm", 1.0)
        self.weight_decay = ttvla_cfg.get("weight_decay", 0.01)

        # LoRA config
        lora_cfg = config.get("lora", {})
        self.lora_rank = lora_cfg.get("rank", 32)
        self.lora_alpha = lora_cfg.get("alpha", 64)
        self.lora_target = lora_cfg.get("target_modules", ["q_proj", "v_proj", "k_proj", "o_proj"])

        # Device
        self.device = config.get("experiment", {}).get("device", "cuda:0")

        # State
        self.buffer = TransitionBuffer()
        self.optimizer = None
        self.peft_model = None
        self._is_lora_applied = False

    def setup_lora(self):
        """Apply LoRA to VLA model and create optimizer."""
        if self._is_lora_applied:
            return

        model = self.vla.get_model()
        if model is None:
            print("[TT-VLA] Warning: VLA model is None, skipping LoRA setup")
            return

        # Apply LoRA
        self.peft_model = apply_lora_to_vla(
            model,
            rank=self.lora_rank,
            alpha=self.lora_alpha,
            target_modules=self.lora_target,
        )

        # Replace model in wrapper
        self.vla.model = self.peft_model

        # Create optimizer for LoRA parameters
        self.optimizer = get_lora_optimizer(
            self.peft_model, lr=self.lr, weight_decay=self.weight_decay
        )

        self._is_lora_applied = True
        print(f"[TT-VLA] LoRA applied. rank={self.lora_rank}, lr={self.lr}, K={self.update_interval}, ε={self.clip_epsilon}")

    def run_episode(
        self,
        env,
        instruction: Optional[str] = None,
        seed: Optional[int] = None,
        verbose: bool = False,
    ) -> EpisodeResult:
        """
        Run a single TT-VLA episode (Algorithm 1).

        This is the core loop:
        1. Reset environment and LoRA parameters
        2. For each timestep:
           - Sample action from policy
           - Execute in environment
           - Compute progress-based reward
           - Store transition in buffer
           - Every K steps: perform PPO update
        """
        result = EpisodeResult()
        start_time = time.time()

        # === Line 2: Load pretrained VLA π_θ; p_0 ← 0; B ← ∅ ===
        if self._is_lora_applied:
            reset_lora_parameters(self.peft_model)
            # Re-create optimizer for fresh LoRA params
            self.optimizer = get_lora_optimizer(
                self.peft_model, lr=self.lr, weight_decay=self.weight_decay
            )

        self.buffer.clear()
        self.progress_estimator.reset()

        # Reset environment
        image, env_instruction = env.reset(seed=seed)
        if instruction is None:
            instruction = env_instruction

        done = False
        t = 0

        # === Line 3: For each time step t = 0, 1, 2, ..., T ===
        while not done:
            # === Line 4: Sample a_t ~ π_θ(a_t | o_{t-H+1:t}, l) ===
            # Get log π_θ_old(a_t | o_t) and execute a_t
            action, log_prob_old = self.vla.sample_action(image, instruction)

            # Execute action in environment
            next_image, env_reward, done, info = env.step(action)

            # === Line 6: Compute p_t ← Φ(o_{0:t}, l) ===
            # === Line 7: Compute r_t ← p_t - p_{t-1} ===
            reward = self.progress_estimator.compute_reward(next_image, instruction)

            # === Line 8: Store transition in buffer B ===
            self.buffer.add(
                obs=image,
                action=action,
                reward=reward,
                log_prob_old=log_prob_old,
            )

            # Track metrics
            result.total_reward += reward
            result.progress_history.append(self.progress_estimator.current_progress)
            result.reward_history.append(reward)

            # === Line 9-15: If t mod K == 0: PPO update ===
            if (t + 1) % self.update_interval == 0 and len(self.buffer) > 0:
                loss = self._ppo_update(instruction)
                result.policy_losses.append(loss)
                result.num_updates += 1

                if verbose:
                    print(f"  Step {t+1}: progress={self.progress_estimator.current_progress:.3f}, "
                          f"reward={reward:.4f}, loss={loss:.4f}")

            # Move to next step
            image = next_image
            t += 1

        # Check success
        eval_info = env.evaluate()
        result.success = bool(eval_info.get("success", False))
        result.num_steps = t
        result.final_progress = self.progress_estimator.current_progress
        result.wall_time = time.time() - start_time

        return result

    def _ppo_update(self, instruction: str) -> float:
        """
        Perform value-free PPO update on buffer (Algorithm 1, lines 10-14).

        For each transition (o_i, a_i, r_i, log π_old) in B:
          1. Compute ratio: r_i(θ) = exp(log π_θ(a_i|o_i) - log π_old(a_i|o_i))
          2. Advantage: Â_i = r_i  (one-step, Eq. 9)
          3. Clipped loss: L_i = min(r_i(θ)·Â_i, clip(r_i(θ), 1-ε, 1+ε)·Â_i)
          4. Update: θ ← θ + η·∇_θ Σ_i L_i
          5. Clear buffer B
        """
        if self.optimizer is None or len(self.buffer) == 0:
            return 0.0

        total_loss = 0.0

        # === Line 10: for each (o_i, a_i, r_i, log π_old(a_i|o_i)) ∈ B ===
        for i in range(len(self.buffer)):
            obs = self.buffer.observations[i]
            action = self.buffer.actions[i]
            reward = self.buffer.rewards[i]
            log_prob_old = self.buffer.log_probs_old[i]

            # === Line 11: Compute ratio ===
            # r_i(θ) = exp(log π_θ(a_i|o_i) - log π_θ_old(a_i|o_i))
            log_prob_new = self.vla.compute_log_prob(obs, instruction, action)

            if isinstance(log_prob_old, (int, float)):
                log_prob_old_tensor = torch.tensor(log_prob_old, dtype=torch.float32, device=self.device)
            else:
                log_prob_old_tensor = log_prob_old.detach()

            ratio = torch.exp(log_prob_new - log_prob_old_tensor)

            # === Advantage: Â_i = r_i (Eq. 9, one-step) ===
            advantage = torch.tensor(reward, dtype=torch.float32, device=self.device)

            # === Line 12: Clipped surrogate loss (Eq. 4) ===
            # L_i = min(r_i(θ)·Â_i, clip(r_i(θ), 1-ε, 1+ε)·Â_i)
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1.0 - self.clip_epsilon, 1.0 + self.clip_epsilon) * advantage
            loss_i = -torch.min(surr1, surr2)  # Negative because we maximize

            total_loss += loss_i

        # === Line 13: Update policy parameters ===
        # θ ← θ + η·∇_θ Σ_i L_i
        avg_loss = total_loss / len(self.buffer)
        self.optimizer.zero_grad()
        avg_loss.backward()

        # Gradient clipping for stability
        if self._is_lora_applied and self.peft_model is not None:
            torch.nn.utils.clip_grad_norm_(
                [p for p in self.peft_model.parameters() if p.requires_grad],
                self.max_grad_norm,
            )

        self.optimizer.step()

        # === Line 14: Clear buffer B ===
        self.buffer.clear()

        return avg_loss.item()

    def run_experiment(
        self,
        env,
        num_episodes: int = 80,
        seeds: List[int] = None,
        task_variant: str = "default",
        log_dir: str = "logs/",
        verbose: bool = True,
    ) -> Dict:
        """
        Run full experiment: multiple episodes with different seeds.

        Paper Section 4.1:
        "Each task is executed for 80 trials across three random seeds
         using a 640×480 RGB image as input."
        """
        if seeds is None:
            seeds = [42, 123, 456]

        os.makedirs(log_dir, exist_ok=True)

        all_results = []
        successes = 0

        for seed in seeds:
            for ep in range(num_episodes // len(seeds)):
                episode_seed = seed * 10000 + ep

                if verbose:
                    print(f"\n[Episode {len(all_results)+1}/{num_episodes}] seed={episode_seed}, variant={task_variant}")

                result = self.run_episode(env, seed=episode_seed, verbose=verbose)

                if result.success:
                    successes += 1

                all_results.append({
                    "episode": len(all_results),
                    "seed": episode_seed,
                    "success": result.success,
                    "total_reward": result.total_reward,
                    "num_steps": result.num_steps,
                    "num_updates": result.num_updates,
                    "final_progress": result.final_progress,
                    "wall_time": result.wall_time,
                    "avg_loss": np.mean(result.policy_losses) if result.policy_losses else 0,
                })

                if verbose:
                    sr = successes / len(all_results) * 100
                    print(f"  → {'SUCCESS' if result.success else 'FAIL'} | "
                          f"progress={result.final_progress:.3f} | "
                          f"updates={result.num_updates} | "
                          f"time={result.wall_time:.1f}s | "
                          f"running SR={sr:.1f}%")

        # Summary
        success_rate = successes / len(all_results) * 100
        summary = {
            "task_variant": task_variant,
            "num_episodes": len(all_results),
            "success_rate": success_rate,
            "avg_reward": np.mean([r["total_reward"] for r in all_results]),
            "avg_steps": np.mean([r["num_steps"] for r in all_results]),
            "avg_wall_time": np.mean([r["wall_time"] for r in all_results]),
            "results": all_results,
        }

        # Save results
        result_path = os.path.join(log_dir, f"ttvla_{task_variant}_results.json")
        with open(result_path, "w") as f:
            json.dump(summary, f, indent=2)

        if verbose:
            print(f"\n{'='*60}")
            print(f"[RESULTS] Task: {task_variant}")
            print(f"  Success Rate: {success_rate:.1f}%")
            print(f"  Avg Reward: {summary['avg_reward']:.4f}")
            print(f"  Saved to: {result_path}")
            print(f"{'='*60}")

        return summary
