"""
VLA Model Wrappers for TT-VLA.

Provides a unified interface for different VLA models:
  - OpenVLA (Kim et al., 2024): Llama-2-7B backbone, OpenVLAForActionPrediction
  - OpenVLA-RL (Liu et al., 2025a): RL-finetuned OpenVLA
  - Nora (Hung et al., 2025): Qwen-2.5-VL-3B + FAST+ tokenizer

Paper Section 4.1:
  - Agent receives RGB observation + natural-language instruction
  - Outputs Cartesian end-effector delta + binary gripper command
  - Policy is fine-tuned using LoRA at test time
"""

import os
import torch
import torch.nn as nn
import numpy as np
from PIL import Image
from typing import Tuple, Optional, Dict, Any


class VLAWrapper:
    """
    Base wrapper providing unified interface for VLA models.

    The key interface needed by TT-VLA (Algorithm 1):
      1. sample_action(obs, instruction) -> action, log_prob
      2. compute_log_prob(obs, action) -> log_prob  (for importance ratio)
    """

    def __init__(self, model_name: str, device: str = "cuda", **kwargs):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.processor = None
        self._action_dim = kwargs.get("action_dim", 7)

    def load(self):
        raise NotImplementedError

    def sample_action(
        self, image: np.ndarray, instruction: str
    ) -> Tuple[np.ndarray, float]:
        raise NotImplementedError

    def compute_log_prob(
        self, image: np.ndarray, instruction: str, action: np.ndarray
    ) -> float:
        raise NotImplementedError

    def get_model(self):
        return self.model


class OpenVLAWrapper(VLAWrapper):
    """
    OpenVLA wrapper (Kim et al., 2024).

    Architecture: OpenVLAForActionPrediction (custom, trust_remote_code=True)
      - Vision backbone: DINOv2 + SigLIP (fused)
      - LLM backbone: Llama-2-7B
      - Action head: discretized into 256 bins per dimension

    OpenVLA generates actions as token sequences:
      - 7 action dimensions (or 8 for some envs)
      - Each dimension discretized into 256 bins
      - Action tokens are generated autoregressively

    Paper: "OpenVLA is one of the most widely used open-source VLA models.
    Consistent with (Liu et al., 2025a), we apply a warm-up fine-tuning
    phase of 10k steps prior to evaluation."
    """

    # OpenVLA uses special action tokens starting at 32000
    ACTION_TOKEN_OFFSET = 32000
    N_ACTION_BINS = 256

    def __init__(self, model_name: str = "openvla/openvla-7b", device: str = "cuda", **kwargs):
        super().__init__(model_name, device, **kwargs)
        self._norm_stats = None

    def load(self):
        os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
        from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig

        print(f"[OpenVLA] Loading model: {self.model_name}")

        # Load config first to get norm_stats
        config = AutoConfig.from_pretrained(self.model_name, trust_remote_code=True)
        self._norm_stats = getattr(config, "norm_stats", None)
        self._n_action_bins = getattr(config, "n_action_bins", 256)

        # Load processor
        self.processor = AutoProcessor.from_pretrained(
            self.model_name, trust_remote_code=True
        )

        # Load model in BF16 for memory efficiency
        self.model = AutoModelForVision2Seq.from_pretrained(
            self.model_name,
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
            low_cpu_mem_usage=True,
            device_map="auto",
        )
        self.device = next(self.model.parameters()).device

        n_params = sum(p.numel() for p in self.model.parameters())
        print(f"[OpenVLA] Model loaded. Parameters: {n_params:,}")
        print(f"[OpenVLA] Action bins: {self._n_action_bins}")
        return self

    def sample_action(
        self, image: np.ndarray, instruction: str
    ) -> Tuple[np.ndarray, float]:
        """
        Generate action from OpenVLA via autoregressive token generation.

        OpenVLA prompt format: "In: What action should the robot take to {instruction}?\nOut:"
        Output: 7 action tokens, each representing a discretized action bin.
        """
        pil_image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        # Resize to 224x224 as expected by OpenVLA
        pil_image = pil_image.resize((224, 224), Image.BILINEAR)

        prompt = f"In: What action should the robot take to {instruction}?\nOut:"

        # Process inputs
        inputs = self.processor(prompt, pil_image).to(self.device, dtype=torch.bfloat16)

        # Generate action tokens
        with torch.no_grad():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self._action_dim,
                do_sample=False,
            )

        # Extract generated action token IDs
        input_len = inputs["input_ids"].shape[1]
        action_token_ids = output[0, input_len:input_len + self._action_dim]

        # Decode to continuous action values
        action = self._decode_action_tokens(action_token_ids)

        # Compute log probability of the generated sequence
        log_prob = self._compute_sequence_log_prob(inputs, action_token_ids)

        return action, log_prob

    def compute_log_prob(
        self, image: np.ndarray, instruction: str, action: np.ndarray
    ) -> torch.Tensor:
        """
        Compute log π_θ(a | o, l) for PPO importance ratio.

        Teacher-forces the action tokens and computes the log probability
        of the action sequence under the current policy.
        """
        pil_image = Image.fromarray(image) if isinstance(image, np.ndarray) else image
        pil_image = pil_image.resize((224, 224), Image.BILINEAR)

        prompt = f"In: What action should the robot take to {instruction}?\nOut:"
        inputs = self.processor(prompt, pil_image).to(self.device, dtype=torch.bfloat16)

        # Encode action to tokens
        action_token_ids = self._encode_action(action)

        # Concatenate input + action tokens for teacher forcing
        input_ids = inputs["input_ids"]
        full_ids = torch.cat([input_ids, action_token_ids.unsqueeze(0).to(input_ids.device)], dim=1)

        # Forward pass with gradients enabled (for PPO update)
        outputs = self.model(
            input_ids=full_ids,
            attention_mask=torch.ones_like(full_ids),
            pixel_values=inputs.get("pixel_values"),
        )

        # Extract logits for action token positions
        # logits shape: [1, seq_len, vocab_size]
        action_logits = outputs.logits[0, input_ids.shape[1] - 1:-1, :]  # Shifted by 1

        # Get log probs for the specific action tokens
        log_probs = torch.log_softmax(action_logits.float(), dim=-1)
        token_log_probs = log_probs.gather(1, action_token_ids.unsqueeze(1)).squeeze(1)

        # Sum across action dimensions for joint log probability
        return token_log_probs.sum()

    def _decode_action_tokens(self, token_ids: torch.Tensor) -> np.ndarray:
        """
        Decode action token IDs to continuous action values.

        OpenVLA maps each token to a bin index, then to [-1, 1] range.
        If norm_stats are available, denormalize to actual action range.
        """
        action = []
        for tid in token_ids[:self._action_dim]:
            bin_idx = tid.item() - self.ACTION_TOKEN_OFFSET
            bin_idx = max(0, min(self.N_ACTION_BINS - 1, bin_idx))
            value = (bin_idx / (self.N_ACTION_BINS - 1)) * 2 - 1
            action.append(value)
        return np.array(action, dtype=np.float32)

    def _encode_action(self, action: np.ndarray) -> torch.Tensor:
        """Encode continuous action to token IDs."""
        token_ids = []
        for val in action[:self._action_dim]:
            # Normalize to [0, 1] from [-1, 1]
            normalized = (float(val) + 1) / 2
            normalized = max(0.0, min(1.0, normalized))
            bin_idx = int(normalized * (self.N_ACTION_BINS - 1))
            token_id = bin_idx + self.ACTION_TOKEN_OFFSET
            token_ids.append(token_id)
        return torch.tensor(token_ids, dtype=torch.long, device=self.device)

    def _compute_sequence_log_prob(self, inputs, action_token_ids) -> float:
        """Compute log prob of action sequence (for initial log_prob_old)."""
        with torch.no_grad():
            input_ids = inputs["input_ids"]
            full_ids = torch.cat([input_ids, action_token_ids.unsqueeze(0)], dim=1)

            outputs = self.model(
                input_ids=full_ids,
                attention_mask=torch.ones_like(full_ids),
                pixel_values=inputs.get("pixel_values"),
            )

            action_logits = outputs.logits[0, input_ids.shape[1] - 1:-1, :]
            log_probs = torch.log_softmax(action_logits.float(), dim=-1)
            token_log_probs = log_probs.gather(
                1, action_token_ids.unsqueeze(1)
            ).squeeze(1)

            return token_log_probs.sum().item()


class SimpleMockVLA(VLAWrapper):
    """
    Lightweight mock VLA for testing the TT-VLA pipeline.
    Uses a simple CNN+MLP to map image -> actions with Gaussian policy.
    Supports proper log_prob computation for PPO.
    """

    def __init__(self, device: str = "cuda", action_dim: int = 7, **kwargs):
        super().__init__("mock_vla", device, action_dim=action_dim)
        self._action_dim = action_dim
        self._build_model(action_dim)

    def _build_model(self, action_dim: int):
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2), nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(4),
            nn.Flatten(),
            nn.Linear(64 * 4 * 4, 256), nn.ReLU(),
        ).to(self.device)

        self.action_head = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, action_dim * 2),
        ).to(self.device)

        self.model = nn.ModuleList([self.vision_encoder, self.action_head])

    def load(self):
        return self

    def sample_action(self, image, instruction):
        img_tensor = self._preprocess_image(image)
        with torch.no_grad():
            features = self.vision_encoder(img_tensor)
            output = self.action_head(features)
        mean, log_std = output.chunk(2, dim=-1)
        std = log_std.clamp(-5, 2).exp()
        dist = torch.distributions.Normal(mean, std)
        action = dist.sample()
        log_prob = dist.log_prob(action).sum(dim=-1)
        return action.squeeze(0).cpu().numpy(), log_prob.item()

    def compute_log_prob(self, image, instruction, action):
        img_tensor = self._preprocess_image(image)
        action_tensor = torch.tensor(action, dtype=torch.float32, device=self.device).unsqueeze(0)
        features = self.vision_encoder(img_tensor)
        output = self.action_head(features)
        mean, log_std = output.chunk(2, dim=-1)
        std = log_std.clamp(-5, 2).exp()
        dist = torch.distributions.Normal(mean, std)
        log_prob = dist.log_prob(action_tensor).sum(dim=-1)
        return log_prob.squeeze()

    def _preprocess_image(self, image):
        if isinstance(image, np.ndarray):
            img = torch.tensor(image, dtype=torch.float32).permute(2, 0, 1) / 255.0
        elif isinstance(image, torch.Tensor):
            img = image.float()
            if img.ndim == 3 and img.shape[-1] == 3:
                img = img.permute(2, 0, 1) / 255.0
            elif img.ndim == 4:
                img = img.squeeze(0).permute(2, 0, 1) / 255.0
        else:
            img = image
        img = nn.functional.interpolate(img.unsqueeze(0), size=(64, 64), mode="bilinear")
        return img.to(self.device)


def build_vla(config: dict, device: str = "cuda") -> VLAWrapper:
    """Factory function to build VLA model."""
    model_name = config.get("model_name", "openvla/openvla-7b")

    if model_name == "mock":
        return SimpleMockVLA(device, action_dim=config.get("action_dim", 7))
    elif "openvla" in model_name.lower():
        wrapper = OpenVLAWrapper(model_name, device, action_dim=config.get("action_dim", 7))
        return wrapper.load()
    else:
        # Default to OpenVLA-style loading
        wrapper = OpenVLAWrapper(model_name, device, action_dim=config.get("action_dim", 7))
        return wrapper.load()
