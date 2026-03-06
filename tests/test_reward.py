"""Tests for progress estimator and reward computation."""

import pytest
import numpy as np


class TestProgressEstimator:
    """Test the base ProgressEstimator contract."""

    def test_reset_clears_progress(self):
        from ttvla.reward import ProgressEstimator

        class DummyProgress(ProgressEstimator):
            def compute_progress(self, obs, instruction):
                return 0.5

        p = DummyProgress()
        p.compute_reward(None, "test")
        assert p.current_progress == 0.5
        p.reset()
        assert p.current_progress == 0.0

    def test_reward_is_progress_delta(self):
        from ttvla.reward import ProgressEstimator

        class IncreasingProgress(ProgressEstimator):
            def __init__(self):
                super().__init__()
                self._step = 0

            def compute_progress(self, obs, instruction):
                self._step += 1
                return self._step * 0.1

        p = IncreasingProgress()
        r1 = p.compute_reward(None, "test")  # p=0.1, prev=0 -> r=0.1
        assert abs(r1 - 0.1) < 1e-6

        r2 = p.compute_reward(None, "test")  # p=0.2, prev=0.1 -> r=0.1
        assert abs(r2 - 0.1) < 1e-6

    def test_negative_reward_on_regression(self):
        from ttvla.reward import ProgressEstimator

        class RegressingProgress(ProgressEstimator):
            def __init__(self):
                super().__init__()
                self._call = 0

            def compute_progress(self, obs, instruction):
                self._call += 1
                return 0.5 if self._call == 1 else 0.3

        p = RegressingProgress()
        r1 = p.compute_reward(None, "test")  # 0.5 - 0 = 0.5
        assert r1 > 0
        r2 = p.compute_reward(None, "test")  # 0.3 - 0.5 = -0.2
        assert r2 < 0


class TestEnvOracleProgress:
    """Test EnvOracleProgress with mock env."""

    def test_returns_success_as_progress(self):
        from ttvla.reward import EnvOracleProgress

        class FakeEnv:
            def evaluate(self):
                return {"success": True}

        p = EnvOracleProgress(FakeEnv())
        assert p.compute_progress(None, "test") == 1.0

    def test_returns_zero_on_failure(self):
        from ttvla.reward import EnvOracleProgress

        class FakeEnv:
            def evaluate(self):
                return {"success": False}

        p = EnvOracleProgress(FakeEnv())
        assert p.compute_progress(None, "test") == 0.0


class TestConfigLoader:
    """Test config loading."""

    def test_load_default_config(self):
        from ttvla.config import load_config
        config = load_config("configs/default.yaml")
        assert "env" in config
        assert "vla" in config
        assert "ttvla" in config
        assert "lora" in config

    def test_override(self):
        from ttvla.config import load_config
        config = load_config("configs/default.yaml", overrides=["ttvla.learning_rate=5e-5"])
        assert config["ttvla"]["learning_rate"] == 5e-5

    def test_override_int(self):
        from ttvla.config import load_config
        config = load_config("configs/default.yaml", overrides=["lora.rank=16"])
        assert config["lora"]["rank"] == 16


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
