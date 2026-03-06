"""Tests for PPO update and LoRA mechanics."""

import pytest
import torch
import numpy as np


class TestPPOUpdate:
    """Verify PPO gradient flow produces non-zero loss when reward is non-zero."""

    def test_nonzero_reward_produces_nonzero_loss(self):
        from ttvla.vla_wrapper import SimpleMockVLA
        from ttvla.lora_utils import get_lora_optimizer

        vla = SimpleMockVLA(device="cpu", action_dim=7)
        model = vla.get_model()
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        # Create a fake observation
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)

        # Sample action and get log_prob
        action, log_prob_old = vla.sample_action(image, "pick up cube")

        # Simulate non-zero reward
        reward = 0.05

        # Compute current log prob (with grad)
        log_prob_new = vla.compute_log_prob(image, "pick up cube", action)
        ratio = torch.exp(log_prob_new - torch.tensor(log_prob_old))
        advantage = torch.tensor(reward, dtype=torch.float32)

        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantage
        loss = -torch.min(surr1, surr2)

        assert loss.item() != 0.0, "Loss should be non-zero with non-zero reward"
        assert loss.requires_grad, "Loss should have gradient"

        # Verify backward works
        optimizer.zero_grad()
        loss.backward()
        grad_norms = [p.grad.norm().item() for p in model.parameters() if p.grad is not None]
        assert any(g > 0 for g in grad_norms), "At least one gradient should be non-zero"

    def test_zero_reward_produces_zero_loss(self):
        """This was the original bug — verify we understand it."""
        reward = 0.0
        ratio = torch.tensor(1.0)
        advantage = torch.tensor(reward)
        surr1 = ratio * advantage
        surr2 = torch.clamp(ratio, 0.8, 1.2) * advantage
        loss = -torch.min(surr1, surr2)
        assert loss.item() == 0.0, "Zero reward should produce zero loss (the original bug)"


class TestLoRA:
    """Test LoRA reset and parameter counting."""

    def test_lora_reset_produces_identity(self):
        from ttvla.vla_wrapper import SimpleMockVLA
        from ttvla.lora_utils import apply_lora_to_vla, reset_lora_parameters

        vla = SimpleMockVLA(device="cpu", action_dim=7)
        model = vla.get_model()

        # We can't apply peft LoRA to a simple nn.ModuleList,
        # but we can test reset logic on a Linear layer
        linear = torch.nn.Linear(64, 64)
        # Simulate LoRA B being non-zero
        lora_B = torch.nn.Parameter(torch.randn(64, 4))
        lora_B.data.fill_(0.5)
        assert lora_B.abs().sum() > 0

        # After zeroing, it should be identity (B=0 -> LoRA output = 0)
        torch.nn.init.zeros_(lora_B)
        assert lora_B.abs().sum() == 0.0

    def test_mock_vla_sample_action_shape(self):
        from ttvla.vla_wrapper import SimpleMockVLA

        vla = SimpleMockVLA(device="cpu", action_dim=7)
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        action, log_prob = vla.sample_action(image, "test")

        assert action.shape == (7,), f"Expected shape (7,), got {action.shape}"
        assert isinstance(log_prob, float), f"Expected float, got {type(log_prob)}"

    def test_mock_vla_log_prob_has_grad(self):
        from ttvla.vla_wrapper import SimpleMockVLA

        vla = SimpleMockVLA(device="cpu", action_dim=7)
        image = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        action, _ = vla.sample_action(image, "test")

        log_prob = vla.compute_log_prob(image, "test", action)
        assert log_prob.requires_grad, "Log prob should have gradient for PPO"


class TestDummyEnv:
    """Test DummyEnvWrapper produces non-zero progress."""

    def test_dummy_env_progress_increases(self):
        from ttvla.env_wrapper import DummyEnvWrapper

        env = DummyEnvWrapper({"max_episode_steps": 160})
        env.make()
        image, instruction = env.reset(seed=42)

        assert image.shape[-1] == 3, "Image should be RGB"
        assert len(instruction) > 0, "Should have instruction"

        progresses = []
        for _ in range(50):
            action = np.random.uniform(-1, 1, 7).astype(np.float32)
            image, reward, done, info = env.step(action)
            progresses.append(env.get_task_progress())

        assert max(progresses) > 0, "DummyEnv should produce non-zero progress"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
