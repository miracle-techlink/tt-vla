"""
ManiSkill3 Environment Wrapper for TT-VLA.

Paper Section 4.1:
  - All simulation experiments conducted in ManiSkill 3
  - WidowX-250S robotic arm
  - Standard pick-and-place manipulation scenario
  - 640x480 RGB image input
  - Cartesian end-effector delta + binary gripper command
  - Fixed horizon of 160 steps

  Generalization dimensions (Section S3):
    - Execution: obj_pos, robot_pose, obj_reposition
    - Vision: table, texture-w, texture-s, noise-w, noise-s
    - Semantics: multi-obj, instruct, dist_recep, multi_recep, etc.
"""

import numpy as np
import torch
from typing import Dict, Any, Optional, Tuple
from PIL import Image


class ManiSkillEnvWrapper:
    """
    Wrapper for ManiSkill3 pick-and-place environment.

    Provides a consistent interface for TT-VLA:
      - reset() -> (image, instruction)
      - step(action) -> (image, reward, done, info)
      - get_task_progress() -> float
    """

    def __init__(self, config: dict):
        self.config = config
        self.env_name = config.get("name", "PickAndPlace-v1")
        self.image_size = tuple(config.get("image_size", [480, 640]))
        self.max_steps = config.get("max_episode_steps", 160)
        self.current_step = 0
        self.env = None
        self._instruction = ""

    def make(self, task_variant: str = "default", seed: int = 0):
        """
        Create the ManiSkill3 environment with specified task variant.

        Task variants map to the paper's generalization dimensions (Table 1, S3).
        """
        import mani_skill.envs  # noqa: register envs
        import gymnasium as gym

        env_kwargs = {
            "obs_mode": self.config.get("obs_mode", "rgb"),
            "control_mode": self.config.get("control_mode", "pd_ee_delta_pose"),
            "render_mode": self.config.get("render_mode", "cameras"),
            "max_episode_steps": self.max_steps,
        }

        sim_backend = self.config.get("sim_backend", "gpu")
        if sim_backend == "gpu":
            env_kwargs["sim_backend"] = "gpu"

        # Map task variant to ManiSkill3 environment config
        # These follow RL4VLA's (Liu et al., 2025a) setup
        env_id = self._resolve_env_id(task_variant)

        self.env = gym.make(env_id, **env_kwargs)
        self.env.reset(seed=seed)
        return self

    def _resolve_env_id(self, variant: str) -> str:
        """Map paper's task variant names to ManiSkill3 env IDs."""
        # The paper follows RL4VLA setup for pick-and-place in ManiSkill3
        variant_map = {
            "default": "PickCube-v1",
            "pick_and_place": "PickCube-v1",
            "stack_cube": "StackCube-v1",
            "peg_insertion": "PegInsertionSide-v1",
        }
        return variant_map.get(variant, "PickCube-v1")

    def reset(self, seed: Optional[int] = None) -> Tuple[np.ndarray, str]:
        """
        Reset environment. Returns (image, instruction).

        Paper: "At the beginning of each episode, the pretrained VLA receives
        an initial observation o_0 and instruction l."
        """
        self.current_step = 0
        obs, info = self.env.reset(seed=seed)
        image = self._extract_image(obs)
        self._instruction = self._get_instruction(info)
        return image, self._instruction

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action in environment.

        Args:
            action: [dx, dy, dz, droll, dpitch, dyaw, gripper]
                    Cartesian end-effector delta + binary gripper

        Returns:
            image, env_reward, done, info
        """
        self.current_step += 1

        # Ensure action is correct shape
        if isinstance(action, torch.Tensor):
            action = action.detach().cpu().numpy()
        action = np.array(action, dtype=np.float32).flatten()

        obs, reward, terminated, truncated, info = self.env.step(action)
        image = self._extract_image(obs)
        done = terminated or truncated or (self.current_step >= self.max_steps)

        return image, float(reward), done, info

    def _extract_image(self, obs) -> np.ndarray:
        """Extract RGB image from ManiSkill3 observation."""
        if isinstance(obs, dict):
            # ManiSkill3 returns dict with sensor data
            if "image" in obs:
                img = obs["image"]
                if isinstance(img, dict):
                    # Multiple cameras - use the first/main one
                    key = list(img.keys())[0]
                    img = img[key]
                    if isinstance(img, dict) and "rgb" in img:
                        img = img["rgb"]
            elif "sensor_data" in obs:
                sensor = obs["sensor_data"]
                key = list(sensor.keys())[0]
                img = sensor[key]["rgb"]
            elif "rgb" in obs:
                img = obs["rgb"]
            else:
                raise ValueError(f"Cannot extract image from obs keys: {obs.keys()}")
        else:
            img = obs

        if isinstance(img, torch.Tensor):
            img = img.detach().cpu().numpy()

        # Ensure uint8 [H, W, 3]
        if img.dtype == np.float32 or img.dtype == np.float64:
            img = (img * 255).clip(0, 255).astype(np.uint8)

        # Handle batch dimension
        if img.ndim == 4:
            img = img[0]

        # Handle channel-first
        if img.ndim == 3 and img.shape[0] in [3, 4]:
            img = np.transpose(img, (1, 2, 0))

        # Drop alpha channel
        if img.shape[-1] == 4:
            img = img[..., :3]

        return img

    def _get_instruction(self, info: dict) -> str:
        """Get language instruction for current task."""
        if "instruction" in info:
            return info["instruction"]
        # Default template from paper S3: "put O on R"
        return "pick up the cube and place it on the target"

    def get_task_progress(self) -> float:
        """Get current task progress from environment."""
        if self.env is None:
            return 0.0
        eval_info = self.env.unwrapped.evaluate() if hasattr(self.env.unwrapped, 'evaluate') else {}
        return float(eval_info.get('success', 0.0))

    def evaluate(self) -> Dict:
        """Get evaluation info from environment."""
        if hasattr(self.env.unwrapped, 'evaluate'):
            return self.env.unwrapped.evaluate()
        return {}

    def close(self):
        if self.env is not None:
            self.env.close()

    @property
    def instruction(self) -> str:
        return self._instruction


class DummyEnvWrapper:
    """
    Dummy environment for testing the TT-VLA pipeline without ManiSkill3.
    Generates synthetic observations and simulates pick-and-place progress.
    """

    def __init__(self, config: dict):
        self.image_size = tuple(config.get("image_size", [480, 640]))
        self.max_steps = config.get("max_episode_steps", 160)
        self.current_step = 0
        self._instruction = "pick up the red cube and place it on the yellow plate"
        self._progress = 0.0
        self._success = False

    def make(self, task_variant="default", seed=0):
        np.random.seed(seed)
        return self

    def reset(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.current_step = 0
        self._progress = 0.0
        self._success = False
        image = np.random.randint(0, 255, (*self.image_size, 3), dtype=np.uint8)
        return image, self._instruction

    def step(self, action):
        self.current_step += 1
        # Simulate gradual progress
        if not self._success:
            delta = np.random.uniform(0.002, 0.01)
            self._progress = min(1.0, self._progress + delta)
            if self._progress > 0.95:
                self._success = True
                self._progress = 1.0

        image = np.random.randint(0, 255, (*self.image_size, 3), dtype=np.uint8)
        done = self._success or self.current_step >= self.max_steps
        info = {"success": self._success, "progress": self._progress}
        return image, self._progress - (self._progress - delta if not self._success else 0), done, info

    def get_task_progress(self):
        return self._progress

    def evaluate(self):
        return {"success": self._success, "progress": self._progress}

    def close(self):
        pass

    @property
    def instruction(self):
        return self._instruction


def build_env(config: dict, use_dummy: bool = False) -> Any:
    """Factory function to build environment."""
    if use_dummy:
        return DummyEnvWrapper(config)
    return ManiSkillEnvWrapper(config)
