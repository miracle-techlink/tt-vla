#!/usr/bin/env python3
"""
Test TT-VLA pipeline with real ManiSkill3 env + mock VLA.
Validates the full pipeline before OpenVLA model download completes.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import time

# Set GPU
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def test_env_and_progress():
    """Test ManiSkill3 environment + progress-based reward."""
    import gymnasium as gym
    import mani_skill.envs

    print("=" * 60)
    print("[Test 1] ManiSkill3 Environment + Progress Reward")
    print("=" * 60)

    env = gym.make(
        "PickCube-v1",
        obs_mode="rgb",
        render_mode="cameras",
        max_episode_steps=160,
    )

    obs, info = env.reset(seed=42)

    # Extract RGB image
    rgb = obs["sensor_data"]["base_camera"]["rgb"]
    print(f"RGB shape: {rgb.shape}, dtype: {rgb.dtype}")

    # Check progress signals
    eval_info = env.unwrapped.evaluate()
    print(f"Initial evaluate: {eval_info}")

    # Run random actions
    total_reward = 0
    prev_progress = 0.0
    for step in range(50):
        action = env.action_space.sample()
        obs, reward, term, trunc, info = env.step(action)

        eval_info = env.unwrapped.evaluate()
        is_grasped = bool(eval_info.get("is_grasped", False))
        success = bool(eval_info.get("success", False))

        # Compute heuristic progress
        if success:
            progress = 1.0
        elif is_grasped:
            progress = 0.4
        else:
            progress = 0.0

        # Dense reward = progress difference (Eq. 7)
        dense_reward = progress - prev_progress
        prev_progress = progress
        total_reward += dense_reward

        if step % 10 == 0:
            print(f"  Step {step}: grasped={is_grasped}, success={success}, "
                  f"progress={progress:.2f}, dense_r={dense_reward:.3f}")

    env.close()
    print(f"Total dense reward: {total_reward:.4f}")
    print("[Test 1] PASSED\n")


def test_mock_ttvla_on_maniskill():
    """Test full TT-VLA pipeline with mock VLA on real ManiSkill3."""
    import gymnasium as gym
    import mani_skill.envs
    from ttvla.vla_wrapper import SimpleMockVLA
    from ttvla.reward import ManiSkillPickPlaceProgress
    from ttvla.tt_vla import TTVLA

    print("=" * 60)
    print("[Test 2] TT-VLA (Mock VLA) on ManiSkill3")
    print("=" * 60)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Create environment
    env = gym.make(
        "PickCube-v1",
        obs_mode="rgb",
        render_mode="cameras",
        max_episode_steps=160,
    )

    # Create mock VLA (action_dim=8 for ManiSkill3 PickCube)
    vla = SimpleMockVLA(device=device, action_dim=8)

    # Create progress estimator
    progress_est = ManiSkillPickPlaceProgress(env)

    # Config
    config = {
        "ttvla": {
            "learning_rate": 1e-4,
            "clip_epsilon": 0.2,
            "update_interval": 8,
            "gamma": 0.0,
            "lam": 0.0,
            "max_grad_norm": 1.0,
            "weight_decay": 0.01,
        },
        "lora": {
            "rank": 16,
            "alpha": 32,
            "target_modules": ["0", "1"],  # Mock model layers
        },
        "experiment": {"device": device},
    }

    # Create TT-VLA
    ttvla = TTVLA(vla, progress_est, config)

    # Setup optimizer directly (skip LoRA for mock model)
    ttvla.optimizer = torch.optim.AdamW(
        [p for p in vla.model.parameters() if p.requires_grad],
        lr=1e-4, weight_decay=0.01,
    )

    # Wrapper for env to match our interface
    class ManiSkillAdapter:
        def __init__(self, gym_env):
            self.env = gym_env
            self._instruction = "pick up the red cube and place it on the target"

        def reset(self, seed=None):
            obs, info = self.env.reset(seed=seed)
            img = obs["sensor_data"]["base_camera"]["rgb"]
            if isinstance(img, torch.Tensor):
                img = img.squeeze(0).cpu().numpy()
            return img, self._instruction

        def step(self, action):
            if isinstance(action, torch.Tensor):
                action = action.detach().cpu().numpy()
            action = np.array(action, dtype=np.float32).flatten()[:8]
            # Pad if needed
            if len(action) < 8:
                action = np.concatenate([action, np.zeros(8 - len(action))])
            obs, reward, term, trunc, info = self.env.step(action)
            img = obs["sensor_data"]["base_camera"]["rgb"]
            if isinstance(img, torch.Tensor):
                img = img.squeeze(0).cpu().numpy()
            done = bool(term) or bool(trunc)
            return img, float(reward), done, info

        def evaluate(self):
            return self.env.unwrapped.evaluate()

        def get_task_progress(self):
            eval_info = self.evaluate()
            if bool(eval_info.get("success", False)):
                return 1.0
            if bool(eval_info.get("is_grasped", False)):
                return 0.4
            return 0.0

        def close(self):
            self.env.close()

        @property
        def instruction(self):
            return self._instruction

    adapter = ManiSkillAdapter(env)

    # Run 3 episodes
    for ep in range(3):
        start = time.time()
        result = ttvla.run_episode(adapter, seed=ep * 100, verbose=True)
        elapsed = time.time() - start
        print(f"\n  Episode {ep+1}: success={result.success}, "
              f"progress={result.final_progress:.3f}, "
              f"reward={result.total_reward:.4f}, "
              f"updates={result.num_updates}, time={elapsed:.1f}s\n")

    adapter.close()
    print("[Test 2] PASSED\n")


def test_gpu_performance():
    """Quick GPU performance check."""
    print("=" * 60)
    print("[Test 3] GPU Performance Check")
    print("=" * 60)

    if not torch.cuda.is_available():
        print("No GPU available, skipping.")
        return

    device = torch.device("cuda:0")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f}GB")

    # Quick matmul benchmark
    x = torch.randn(4096, 4096, device=device, dtype=torch.bfloat16)
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(100):
        y = x @ x
    torch.cuda.synchronize()
    elapsed = time.time() - start
    tflops = 100 * 2 * 4096**3 / elapsed / 1e12
    print(f"BF16 matmul: {tflops:.1f} TFLOPS")

    # Memory test for ~7B parameter model
    estimated_mem = 7e9 * 2 / 1e9  # BF16
    print(f"Estimated OpenVLA-7B memory (BF16): {estimated_mem:.1f}GB")
    print(f"Available VRAM: {torch.cuda.mem_get_info(0)[0] / 1e9:.1f}GB")
    print(f"Fits on single GPU: {'YES' if torch.cuda.mem_get_info(0)[0] / 1e9 > estimated_mem else 'NO'}")

    print("[Test 3] PASSED\n")


if __name__ == "__main__":
    test_env_and_progress()
    test_mock_ttvla_on_maniskill()
    test_gpu_performance()
    print("=" * 60)
    print("ALL TESTS PASSED!")
    print("=" * 60)
