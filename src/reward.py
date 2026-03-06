"""
Dense Progress-Based Reward for TT-VLA.

Paper Eq. 6-7:
  p_t = Φ(o_{0:t+1}, l)    -- task progress ∈ [0,1]
  r_t = p_t - p_{t-1}       -- dense reward (temporal difference in progress)

Two modes:
  1. env_oracle: use ManiSkill3 ground-truth task progress (for development)
  2. vlac: use a pretrained VLAC model as progress estimator Φ
"""

import torch
import numpy as np
from typing import Optional, Dict, Any


class ProgressEstimator:
    """Base class for task progress estimation."""

    def __init__(self):
        self._prev_progress = 0.0

    def reset(self):
        self._prev_progress = 0.0

    def compute_progress(self, obs: Any, instruction: str) -> float:
        raise NotImplementedError

    def compute_reward(self, obs: Any, instruction: str) -> float:
        """Eq. 7: r_t = p_t - p_{t-1}"""
        p_t = self.compute_progress(obs, instruction)
        r_t = p_t - self._prev_progress
        self._prev_progress = p_t
        return r_t

    @property
    def current_progress(self) -> float:
        return self._prev_progress


class EnvOracleProgress(ProgressEstimator):
    """
    Use environment's ground-truth success/progress signal.
    ManiSkill3 environments provide evaluate() with 'success' flag
    and some provide continuous progress metrics.
    """

    def __init__(self, env):
        super().__init__()
        self.env = env

    def compute_progress(self, obs: Any, instruction: str) -> float:
        """Extract progress from environment info."""
        # ManiSkill3 provides env.evaluate() -> dict with 'success' and
        # potentially intermediate progress signals
        info = self.env.evaluate() if hasattr(self.env, 'evaluate') else {}

        if 'progress' in info:
            return float(info['progress'])

        # For pick-and-place: compute progress from sub-goals
        # Typical sub-goals: reach -> grasp -> lift -> move -> place
        if hasattr(self.env, 'get_task_progress'):
            return float(self.env.get_task_progress())

        # Fallback: binary success
        success = info.get('success', False)
        if isinstance(success, (torch.Tensor, np.ndarray)):
            success = float(success)
        return float(success)


class VLACProgress(ProgressEstimator):
    """
    VLAC-based progress estimator (paper's primary method).

    VLAC (Zhai et al., 2025): A Vision-Language-Action-Critic model
    that takes observation history + language instruction and outputs
    a scalar task progress estimate p_t ∈ [0,1].

    The model is kept FROZEN during TT-VLA (only the VLA policy is updated).
    """

    def __init__(self, model_path: str, device: str = "cuda"):
        super().__init__()
        self.device = device
        self.model = self._load_vlac(model_path)
        self.obs_history = []

    def _load_vlac(self, model_path: str):
        """Load pretrained VLAC model."""
        # VLAC architecture: multimodal model that regresses task progress
        # Input: observation sequence + language instruction
        # Output: scalar progress ∈ [0,1]
        from transformers import AutoModel, AutoProcessor
        print(f"[VLAC] Loading progress estimator from {model_path}")
        model = AutoModel.from_pretrained(model_path, trust_remote_code=True)
        model = model.to(self.device).eval()
        for p in model.parameters():
            p.requires_grad = False
        self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)
        return model

    def reset(self):
        super().reset()
        self.obs_history = []

    def compute_progress(self, obs: Any, instruction: str) -> float:
        """
        Eq. 6: p_t = Φ(o_{0:t+1}, l)
        """
        self.obs_history.append(obs)
        with torch.no_grad():
            inputs = self.processor(
                images=self.obs_history,
                text=instruction,
                return_tensors="pt"
            ).to(self.device)
            progress = self.model(**inputs).logits.squeeze().clamp(0, 1)
        return progress.item()


class ManiSkillPickPlaceProgress(ProgressEstimator):
    """
    Hand-crafted progress estimator for ManiSkill3 pick-and-place.

    Decomposes task into sub-goals with continuous progress:
      0.0 - 0.2: reaching the object
      0.2 - 0.4: grasping the object
      0.4 - 0.7: lifting and moving to target
      0.7 - 1.0: placing at target location
    """

    def __init__(self, env):
        super().__init__()
        self.env = env

    def compute_progress(self, obs: Any, instruction: str) -> float:
        info = self.env.get_info() if hasattr(self.env, 'get_info') else {}

        # Try to get structured progress from ManiSkill3
        if hasattr(self.env.unwrapped, 'evaluate'):
            eval_info = self.env.unwrapped.evaluate()
            is_grasped = eval_info.get('is_grasped', False)
            is_obj_placed = eval_info.get('is_obj_placed', False)
            success = eval_info.get('success', False)

            if isinstance(is_grasped, (torch.Tensor, np.ndarray)):
                is_grasped = bool(is_grasped)
            if isinstance(is_obj_placed, (torch.Tensor, np.ndarray)):
                is_obj_placed = bool(is_obj_placed)
            if isinstance(success, (torch.Tensor, np.ndarray)):
                success = bool(success)

            if success:
                return 1.0
            elif is_obj_placed:
                return 0.85
            elif is_grasped:
                return 0.4
            else:
                return 0.0

        return 0.0


def build_progress_estimator(config: dict, env=None) -> ProgressEstimator:
    """Factory function to build progress estimator based on config."""
    mode = config.get("mode", "env_oracle")

    if mode == "env_oracle":
        assert env is not None, "env_oracle mode requires environment"
        return EnvOracleProgress(env)
    elif mode == "vlac":
        model_path = config["vlac_model"]
        assert model_path is not None, "VLAC mode requires model path"
        device = config.get("device", "cuda")
        return VLACProgress(model_path, device)
    elif mode == "maniskill_heuristic":
        assert env is not None
        return ManiSkillPickPlaceProgress(env)
    else:
        raise ValueError(f"Unknown progress estimator mode: {mode}")
