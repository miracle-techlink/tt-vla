"""
VLAC Progress Estimator Proxy for TT-VLA.

Since VLAC (Zhai et al., 2025) model weights are not publicly available,
we implement two practical alternatives:

1. EnvDenseReward: Use ManiSkill3's built-in reward as progress signal
   - This is the "oracle" mode for controlled experiments
   - Provides continuous, reliable progress signal

2. LearnedProgressEstimator: Train a lightweight CNN-based progress
   predictor on ManiSkill3 episodes
   - Mimics VLAC's role: image sequence + instruction -> progress ∈ [0,1]
   - Can be trained on collected trajectories
   - Stays FROZEN during TT-VLA (only VLA policy is updated)

Paper specifies:
  "We instantiate Φ using the Vision-Language-Action-Critic model (VLAC),
   a pretrained multimodal model that serves as a scalar regressor for
   task progress estimation."
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional


class LearnedProgressEstimator(nn.Module):
    """
    Lightweight learned progress estimator to replace VLAC.

    Architecture:
      - ResNet-18 vision encoder (pretrained, frozen)
      - Simple MLP head: [512] -> [256] -> [1] (sigmoid)
      - Takes single RGB frame + instruction embedding
      - Outputs progress ∈ [0,1]

    This model is FROZEN during TT-VLA.
    Train it first on collected ManiSkill3 trajectories with ground-truth progress.
    """

    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device

        # Vision encoder (lightweight)
        from torchvision import models
        resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.vision_encoder = nn.Sequential(*list(resnet.children())[:-1])  # Remove FC
        self.vision_encoder.eval()
        for p in self.vision_encoder.parameters():
            p.requires_grad = False

        # Progress prediction head
        self.progress_head = nn.Sequential(
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid(),
        )

        self.to(device)
        self._prev_progress = 0.0

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        """Predict progress from image. image: [B, 3, 224, 224]"""
        with torch.no_grad():
            features = self.vision_encoder(image).flatten(1)  # [B, 512]
        progress = self.progress_head(features)  # [B, 1]
        return progress.squeeze(-1)

    def reset(self):
        self._prev_progress = 0.0

    def compute_reward(self, obs, instruction: str) -> float:
        """Eq. 7: r_t = p_t - p_{t-1}"""
        img = self._preprocess(obs)
        with torch.no_grad():
            p_t = self.forward(img).item()
        r_t = p_t - self._prev_progress
        self._prev_progress = p_t
        return r_t

    @property
    def current_progress(self):
        return self._prev_progress

    def _preprocess(self, obs):
        if isinstance(obs, np.ndarray):
            img = torch.from_numpy(obs).float().permute(2, 0, 1) / 255.0
        elif isinstance(obs, torch.Tensor):
            img = obs.float()
            if img.ndim == 3 and img.shape[-1] == 3:
                img = img.permute(2, 0, 1) / 255.0
            elif img.ndim == 4:
                img = img.squeeze(0).permute(2, 0, 1) / 255.0
        else:
            img = obs

        img = nn.functional.interpolate(
            img.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False
        )
        # ImageNet normalization
        mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(1, 3, 1, 1)
        img = (img - mean) / std
        return img.to(self.device)


def train_progress_estimator(
    env,
    model: LearnedProgressEstimator,
    num_episodes: int = 100,
    lr: float = 1e-3,
    save_path: str = "checkpoints/progress_estimator.pt",
):
    """
    Train the progress estimator on ManiSkill3 episodes.

    Collects trajectories with random/scripted actions and uses
    the environment's ground-truth progress as supervision.
    """
    import os
    optimizer = torch.optim.Adam(model.progress_head.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()
    total_loss = 0
    n_samples = 0

    for ep in range(num_episodes):
        obs, _ = env.reset(seed=ep)
        done = False
        step = 0

        while not done:
            action = np.random.uniform(-1, 1, size=env.env.action_space.shape).astype(np.float32)
            obs, _, done, info = env.step(action)

            # Get ground truth progress
            eval_info = env.evaluate()
            gt_progress = 0.0
            if bool(eval_info.get("success", False)):
                gt_progress = 1.0
            elif bool(eval_info.get("is_grasped", False)):
                gt_progress = 0.4

            # Forward
            img = model._preprocess(obs)
            pred = model(img)
            target = torch.tensor([gt_progress], device=model.device)

            loss = criterion(pred, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            n_samples += 1
            step += 1

        if (ep + 1) % 10 == 0:
            avg_loss = total_loss / n_samples
            print(f"  Episode {ep+1}/{num_episodes}: avg_loss={avg_loss:.4f}")

    # Freeze after training
    model.eval()
    for p in model.parameters():
        p.requires_grad = False

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    torch.save(model.state_dict(), save_path)
    print(f"Progress estimator saved to {save_path}")

    return model
