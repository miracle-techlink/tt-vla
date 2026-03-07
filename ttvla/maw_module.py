"""
Memory-as-Weights (MaW) Module for OpenVLA-7B.

Implements the core MaW mechanism from:
  "Memory-as-Weights: Fast-Weight Memory for Real-Time Vision-Language-Action Models"

The key idea: inject SwiGLU fast-weight networks after selected transformer layers.
These fast weights are updated via test-time training (TTT) on a self-supervised
reconstruction objective, enabling the model to accumulate episodic context
without growing the KV cache.

Architecture:
  - SwiGLUFastWeight: A SwiGLU MLP used as a fast-weight memory store.
    Trained online via reconstruction loss to memorize hidden-state patterns.
  - MaWLayer: Residual wrapper that applies the fast-weight output with a
    learnable (or fixed) mixing coefficient alpha.
  - MaWVLAWrapper: Extends OpenVLAWrapper; inserts MaW layers via hooks
    after every 4th transformer layer (8 total), implements LaCT (lazy
    chunked TTT) and cross-episode warm-start.
  - MaWTrainer: Extends TTVLATrainer with MaW-specific logic.

Memory budget:
  d_model=4096, d_inner=10922 => ~268MB per layer (bf16).
  8 layers => ~2.1GB total, fits alongside the 14GB model on a 32GB GPU.
"""

import math
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import time
import json
import os
from typing import Optional, Dict, List, Tuple, Any
from pathlib import Path
from dataclasses import dataclass, field

from .vla_wrapper import OpenVLAWrapper, VLAWrapper
from .tt_vla import TTVLA, TransitionBuffer, EpisodeResult
from .reward import build_progress_estimator
from .lora_utils import apply_lora_to_vla, get_lora_optimizer


# ---------------------------------------------------------------------------
# 1. SwiGLU Fast-Weight Network
# ---------------------------------------------------------------------------

class SwiGLUFastWeight(nn.Module):
    """
    SwiGLU fast-weight network for episodic memory storage.

    This MLP is trained online (TTT) to reconstruct hidden states, effectively
    memorizing the input distribution. The weights themselves become the memory.

    Architecture (matches Llama-2 MLP structure):
      gate = Silu(x @ W_gate)
      up   = x @ W_up
      down = (gate * up) @ W_down
      out  = down

    The reconstruction objective is: L = ||SwiGLU(x) - x||^2

    Args:
        d_model: Hidden dimension of the transformer (4096 for Llama-2-7B).
        d_inner: Inner dimension of the SwiGLU MLP. Default: floor(2.667 * d_model).
    """

    def __init__(self, d_model: int, d_inner: Optional[int] = None):
        super().__init__()
        self.d_model = d_model
        self.d_inner = d_inner if d_inner is not None else math.floor(2.667 * d_model)

        # SwiGLU projections (no bias, matching Llama-2 convention)
        self.w_gate = nn.Linear(d_model, self.d_inner, bias=False)
        self.w_up = nn.Linear(d_model, self.d_inner, bias=False)
        self.w_down = nn.Linear(self.d_inner, d_model, bias=False)

        self._init_weights()

    def _init_weights(self):
        """Initialize to near-identity: small random weights so initial output ~ 0."""
        # Small init so fast-weight output starts near zero
        nn.init.normal_(self.w_gate.weight, std=0.01)
        nn.init.normal_(self.w_up.weight, std=0.01)
        nn.init.zeros_(self.w_down.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: SwiGLU reconstruction.

        Args:
            x: Input tensor of shape (..., d_model).

        Returns:
            Reconstructed tensor of shape (..., d_model).
        """
        gate = F.silu(self.w_gate(x))
        up = self.w_up(x)
        return self.w_down(gate * up)

    def get_state(self) -> Dict[str, torch.Tensor]:
        """Get fast-weight state for cross-episode transfer."""
        return {k: v.clone() for k, v in self.state_dict().items()}

    def load_state(self, state: Dict[str, torch.Tensor]):
        """Load fast-weight state from a previous episode."""
        self.load_state_dict(state)

    def reset(self):
        """Reset fast weights to initial state (no memory)."""
        self._init_weights()


# ---------------------------------------------------------------------------
# 2. MaW Layer (residual wrapper)
# ---------------------------------------------------------------------------

class MaWLayer(nn.Module):
    """
    MaW layer inserted after a transformer layer via forward hook.

    Applies a residual connection:
      output = hidden_states + alpha * fast_weight(hidden_states)

    The fast weights are updated via TTT (test-time training) on the
    reconstruction loss: L = ||fast_weight(h) - h||^2.

    Args:
        d_model: Hidden dimension.
        d_inner: Inner dimension for SwiGLU.
        alpha_init: Initial value of the residual mixing coefficient.
        ttt_lr: Learning rate for the TTT Adam optimizer.
    """

    def __init__(
        self,
        d_model: int,
        d_inner: Optional[int] = None,
        alpha_init: float = 0.1,
        ttt_lr: float = 1e-3,
    ):
        super().__init__()
        self.fast_weight = SwiGLUFastWeight(d_model, d_inner)

        # Learnable mixing coefficient, initialized to alpha_init.
        # Using a raw parameter (not sigmoid-gated) for simplicity;
        # clamped in forward to [0, 1].
        self.alpha = nn.Parameter(torch.tensor(alpha_init))

        # Layer norm applied before the fast-weight network for stability
        self.layer_norm = nn.LayerNorm(d_model)

        # TTT optimizer (created lazily so device placement is correct)
        self.ttt_lr = ttt_lr
        self._ttt_optimizer: Optional[torch.optim.Adam] = None

    def _ensure_ttt_optimizer(self):
        """Lazily create the TTT optimizer on the correct device."""
        if self._ttt_optimizer is None:
            self._ttt_optimizer = torch.optim.Adam(
                self.fast_weight.parameters(), lr=self.ttt_lr
            )

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        """
        Apply fast-weight memory as a residual.

        Args:
            hidden_states: Transformer hidden states, shape (batch, seq_len, d_model).

        Returns:
            Modified hidden states with fast-weight residual added.
        """
        alpha = self.alpha.clamp(0.0, 1.0)
        normed = self.layer_norm(hidden_states)
        fw_out = self.fast_weight(normed)
        return hidden_states + alpha * fw_out

    def ttt_update(self, hidden_states: torch.Tensor) -> float:
        """
        Run one TTT step: update fast weights via reconstruction loss.

        The objective is to minimize ||SwiGLU(h) - h||^2 so that the
        fast-weight network learns to reconstruct (and thus memorize)
        the hidden-state distribution.

        Args:
            hidden_states: Hidden states to memorize, shape (batch, seq_len, d_model)
                           or (seq_len, d_model).

        Returns:
            Scalar reconstruction loss (for logging).
        """
        self._ensure_ttt_optimizer()

        # Detach targets: we train the fast weights, not the backbone
        target = hidden_states.detach()
        normed = self.layer_norm(target)

        # Reconstruction loss
        reconstructed = self.fast_weight(normed)
        loss = F.mse_loss(reconstructed, target)

        # Update fast weights
        self._ttt_optimizer.zero_grad()
        loss.backward()
        self._ttt_optimizer.step()

        return loss.item()

    def get_state(self) -> Dict[str, Any]:
        """Get full state for cross-episode warm-start."""
        state = {
            "fast_weight": self.fast_weight.get_state(),
            "alpha": self.alpha.detach().clone(),
            "layer_norm": {k: v.clone() for k, v in self.layer_norm.state_dict().items()},
        }
        # Also save optimizer state for warm-start continuity
        if self._ttt_optimizer is not None:
            state["optimizer"] = copy.deepcopy(self._ttt_optimizer.state_dict())
        return state

    def load_state(self, state: Dict[str, Any]):
        """Load state from a previous episode for warm-start."""
        self.fast_weight.load_state(state["fast_weight"])
        self.alpha.data.copy_(state["alpha"])
        self.layer_norm.load_state_dict(state["layer_norm"])
        if "optimizer" in state and self._ttt_optimizer is not None:
            self._ttt_optimizer.load_state_dict(state["optimizer"])

    def reset(self):
        """Reset fast weights and optimizer (cold start)."""
        self.fast_weight.reset()
        self._ttt_optimizer = None  # Will be re-created lazily


# ---------------------------------------------------------------------------
# 3. MaWVLAWrapper
# ---------------------------------------------------------------------------

# Which Llama-2 layers get a MaW layer (0-indexed, every 4th layer)
DEFAULT_MAW_LAYER_INDICES = [3, 7, 11, 15, 19, 23, 27, 31]


class MaWVLAWrapper(OpenVLAWrapper):
    """
    OpenVLA wrapper augmented with Memory-as-Weights (MaW) layers.

    After loading the base model, this wrapper:
      1. Inserts MaWLayer modules after every 4th transformer layer
         (layers 3, 7, 11, 15, 19, 23, 27, 31) using forward hooks.
      2. During inference, accumulates hidden states in a chunk buffer.
      3. Every C tokens, runs TTT update on the fast weights (LaCT).
      4. Supports cross-episode warm-start by saving/loading fast-weight states.

    The hooks modify the output of selected transformer layers in-place,
    avoiding any changes to the base model architecture.

    Args:
        model_name: HuggingFace model ID for OpenVLA.
        device: Target device.
        maw_layer_indices: Which transformer layers to augment.
        d_inner: Inner dimension for SwiGLU (None = auto).
        alpha_init: Initial residual mixing coefficient.
        ttt_lr: Learning rate for TTT updates.
        chunk_size: Number of tokens between TTT updates (LaCT chunk size C).
    """

    def __init__(
        self,
        model_name: str = "openvla/openvla-7b",
        device: str = "cuda",
        maw_layer_indices: Optional[List[int]] = None,
        d_inner: Optional[int] = None,
        alpha_init: float = 0.1,
        ttt_lr: float = 1e-3,
        chunk_size: int = 2048,
        **kwargs,
    ):
        super().__init__(model_name, device, **kwargs)
        self.maw_layer_indices = maw_layer_indices or DEFAULT_MAW_LAYER_INDICES
        self.d_inner = d_inner
        self.alpha_init = alpha_init
        self.ttt_lr = ttt_lr
        self.chunk_size = chunk_size

        # MaW layers (created after model is loaded and d_model is known)
        self.maw_layers: Dict[int, MaWLayer] = {}
        self._hooks: List[torch.utils.hooks.RemovableHook] = []

        # LaCT chunk buffer: accumulates hidden states between TTT updates
        # Key: layer_index, Value: list of hidden-state tensors
        self._chunk_buffers: Dict[int, List[torch.Tensor]] = {}
        self._token_count: int = 0

        # Cross-episode state
        self._saved_maw_state: Optional[Dict[int, Dict]] = None

    def load(self):
        """Load base model, then attach MaW layers via hooks."""
        super().load()
        self._attach_maw_layers()
        return self

    def _get_transformer_layers(self) -> nn.ModuleList:
        """
        Navigate the model hierarchy to find the transformer layer list.

        OpenVLA uses Llama-2-7B backbone. The layers are at:
          model.language_model.model.layers  (AutoModelForVision2Seq)
        or sometimes:
          model.model.layers  (direct Llama)

        Returns:
            nn.ModuleList of transformer layers.
        """
        model = self.model

        # If wrapped in PEFT, unwrap first
        if hasattr(model, "base_model"):
            model = model.base_model
        if hasattr(model, "model"):
            model = model.model

        # Try common paths for OpenVLA / Llama-2 backbone
        candidates = [
            lambda m: m.language_model.model.layers,
            lambda m: m.model.layers,
            lambda m: m.layers,
            lambda m: m.language_model.model.model.layers,
        ]
        for getter in candidates:
            try:
                layers = getter(model)
                if isinstance(layers, nn.ModuleList) and len(layers) == 32:
                    return layers
            except (AttributeError, TypeError):
                continue

        raise RuntimeError(
            "Could not locate transformer layers in the model. "
            "Expected 32-layer Llama-2 backbone. Model type: "
            f"{type(self.model).__name__}"
        )

    def _attach_maw_layers(self):
        """Create MaW layers and register forward hooks on selected transformer layers."""
        transformer_layers = self._get_transformer_layers()
        d_model = 4096  # Llama-2-7B hidden size

        # Verify d_model from first layer if possible
        first_layer = transformer_layers[0]
        if hasattr(first_layer, "hidden_size"):
            d_model = first_layer.hidden_size
        elif hasattr(first_layer, "self_attn") and hasattr(first_layer.self_attn, "hidden_size"):
            d_model = first_layer.self_attn.hidden_size

        print(f"[MaW] Attaching {len(self.maw_layer_indices)} MaW layers "
              f"(d_model={d_model}, d_inner={self.d_inner or math.floor(2.667 * d_model)})")

        dtype = next(self.model.parameters()).dtype

        for idx in self.maw_layer_indices:
            if idx >= len(transformer_layers):
                print(f"[MaW] Warning: layer index {idx} out of range, skipping")
                continue

            # Place MaW layer on the SAME device as the transformer layer it hooks
            layer_device = next(transformer_layers[idx].parameters()).device

            maw_layer = MaWLayer(
                d_model=d_model,
                d_inner=self.d_inner,
                alpha_init=self.alpha_init,
                ttt_lr=self.ttt_lr,
            ).to(device=layer_device, dtype=dtype)

            self.maw_layers[idx] = maw_layer
            self._chunk_buffers[idx] = []

            # Register forward hook on the transformer layer
            hook = transformer_layers[idx].register_forward_hook(
                self._make_hook(idx)
            )
            self._hooks.append(hook)

        total_params = sum(
            sum(p.numel() for p in ml.parameters())
            for ml in self.maw_layers.values()
        )
        total_bytes = total_params * 2  # bf16
        print(f"[MaW] Total fast-weight parameters: {total_params:,} "
              f"({total_bytes / 1e9:.2f} GB in bf16)")

    def _make_hook(self, layer_idx: int):
        """
        Create a forward hook closure for a specific transformer layer.

        The hook:
          1. Extracts hidden states from the layer output.
          2. Applies the MaW residual transformation.
          3. Buffers hidden states for LaCT (lazy chunked TTT).

        Llama-2 decoder layers return a tuple:
          (hidden_states, present_key_value, ...)
        We modify hidden_states in-place and return the updated tuple.
        """

        def hook_fn(module, input, output):
            # Llama-2 layer output: (hidden_states, ...) or just hidden_states
            if isinstance(output, tuple):
                hidden_states = output[0]
            else:
                hidden_states = output

            maw_layer = self.maw_layers[layer_idx]

            # Apply MaW residual (no grad for inference path)
            with torch.no_grad():
                modified = maw_layer(hidden_states)

            # Buffer hidden states for LaCT updates
            self._chunk_buffers[layer_idx].append(hidden_states.detach())

            if isinstance(output, tuple):
                return (modified,) + output[1:]
            return modified

        return hook_fn

    def _run_lact_update(self):
        """
        LaCT: Lazy Chunked Test-Time Training.

        When enough tokens have been accumulated (>= chunk_size C),
        run TTT updates on all MaW layers using the buffered hidden states.

        Returns:
            Dict mapping layer index to reconstruction loss, or None if
            no update was performed.
        """
        losses = {}

        for idx, maw_layer in self.maw_layers.items():
            buffer = self._chunk_buffers[idx]
            if not buffer:
                continue

            # Concatenate buffered hidden states
            # Each entry is (batch, seq_len, d_model); concat along seq dim
            all_hidden = torch.cat(buffer, dim=1)  # (batch, total_tokens, d_model)

            # Run TTT update
            loss = maw_layer.ttt_update(all_hidden)
            losses[idx] = loss

            # Clear buffer
            self._chunk_buffers[idx] = []

        return losses if losses else None

    def sample_action(self, image, instruction):
        """
        Sample action with MaW-augmented inference.

        After each forward pass, checks if enough tokens have been
        accumulated to trigger a LaCT update.
        """
        action, log_prob = super().sample_action(image, instruction)

        # Count tokens processed (approximate: each action sample processes
        # the full input sequence once)
        # We estimate tokens from the chunk buffer size
        total_buffered = sum(
            sum(t.shape[1] for t in buf) for buf in self._chunk_buffers.values()
        )
        if total_buffered >= self.chunk_size:
            self._run_lact_update()

        return action, log_prob

    # --- Cross-episode warm-start ---

    def save_maw_state(self) -> Dict[int, Dict]:
        """Save all MaW layer states for cross-episode transfer."""
        state = {idx: ml.get_state() for idx, ml in self.maw_layers.items()}
        self._saved_maw_state = state
        return state

    def load_maw_state(self, state: Dict[int, Dict]):
        """Load MaW layer states from a previous episode."""
        for idx, layer_state in state.items():
            if idx in self.maw_layers:
                self.maw_layers[idx].load_state(layer_state)
        self._saved_maw_state = state

    def warm_start(self):
        """
        Warm-start MaW layers from the last saved state.

        Call this at the beginning of a new episode to carry over
        episodic memory from the previous episode.
        """
        if self._saved_maw_state is not None:
            self.load_maw_state(self._saved_maw_state)
            print("[MaW] Warm-started from saved state")
        else:
            print("[MaW] No saved state; starting cold")

    def reset_maw(self):
        """Reset all MaW layers to initial state (cold start)."""
        for maw_layer in self.maw_layers.values():
            maw_layer.reset()
        for idx in self._chunk_buffers:
            self._chunk_buffers[idx] = []
        self._token_count = 0

    def save_maw_checkpoint(self, path: str):
        """Save MaW states to disk."""
        state = self.save_maw_state()
        # Move tensors to CPU for storage
        cpu_state = {}
        for idx, layer_state in state.items():
            cpu_layer = {}
            for k, v in layer_state.items():
                if isinstance(v, torch.Tensor):
                    cpu_layer[k] = v.cpu()
                elif isinstance(v, dict):
                    cpu_layer[k] = {
                        kk: vv.cpu() if isinstance(vv, torch.Tensor) else vv
                        for kk, vv in v.items()
                    }
                else:
                    cpu_layer[k] = v
            cpu_state[idx] = cpu_layer
        torch.save(cpu_state, path)
        print(f"[MaW] Saved checkpoint to {path}")

    def load_maw_checkpoint(self, path: str):
        """Load MaW states from disk."""
        state = torch.load(path, map_location=self.device, weights_only=False)
        self.load_maw_state(state)
        print(f"[MaW] Loaded checkpoint from {path}")

    def remove_hooks(self):
        """Remove all forward hooks (cleanup)."""
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()

    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_hooks()


# ---------------------------------------------------------------------------
# 4. MaWTrainer
# ---------------------------------------------------------------------------

@dataclass
class MaWEpisodeResult(EpisodeResult):
    """Extended episode result with MaW-specific metrics."""
    maw_losses: Dict[int, List[float]] = field(default_factory=dict)
    num_lact_updates: int = 0
    avg_reconstruction_loss: float = 0.0


class MaWTrainer:
    """
    Trainer that integrates MaW with the TT-VLA pipeline.

    Extends the standard TTVLATrainer flow:
      - Uses MaWVLAWrapper instead of OpenVLAWrapper.
      - During inference, hidden states are accumulated in chunk buffers.
      - Every C=2048 tokens, runs TTT update on fast weights (LaCT).
      - Tracks and logs reconstruction loss alongside RL metrics.
      - Supports cross-episode warm-start of fast-weight states.

    Args:
        config: Full experiment configuration dict.
    """

    def __init__(self, config: dict):
        self.config = config

        requested_device = config.get("experiment", {}).get("device", "cuda:0")
        if "cuda" in requested_device and not torch.cuda.is_available():
            print("[MaWTrainer] CUDA not available, falling back to CPU")
            requested_device = "cpu"
        self.device = requested_device

        self.log_dir = Path(config.get("experiment", {}).get("log_dir", "logs/"))
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.vla: Optional[MaWVLAWrapper] = None
        self.env = None
        self.ttvla: Optional[TTVLA] = None

        # MaW config
        maw_cfg = config.get("maw", {})
        self.chunk_size = maw_cfg.get("chunk_size", 2048)
        self.ttt_lr = maw_cfg.get("ttt_lr", 1e-3)
        self.alpha_init = maw_cfg.get("alpha_init", 0.1)
        self.d_inner = maw_cfg.get("d_inner", None)
        self.maw_layer_indices = maw_cfg.get(
            "layer_indices", DEFAULT_MAW_LAYER_INDICES
        )
        self.warm_start_enabled = maw_cfg.get("warm_start", True)

        # Logging
        self._jsonl_path = self.log_dir / "maw_train.jsonl"
        self._lact_losses: Dict[int, List[float]] = {}

    def _build_vla(self) -> MaWVLAWrapper:
        """Build MaW-augmented VLA model."""
        vla_cfg = self.config.get("vla", {})
        model_name = vla_cfg.get("model_name", "openvla/openvla-7b")
        action_dim = vla_cfg.get("action_dim", 7)

        vla = MaWVLAWrapper(
            model_name=model_name,
            device=self.device,
            maw_layer_indices=self.maw_layer_indices,
            d_inner=self.d_inner,
            alpha_init=self.alpha_init,
            ttt_lr=self.ttt_lr,
            chunk_size=self.chunk_size,
            action_dim=action_dim,
        )
        vla.load()
        return vla

    def _build_env(self):
        """Build environment (same as TTVLATrainer)."""
        env_cfg = self.config.get("env", {})
        vla_name = self.config.get("vla", {}).get("model_name", "")

        if vla_name == "mock":
            from .env_wrapper import DummyEnvWrapper
            env = DummyEnvWrapper(env_cfg)
        else:
            from .env_wrapper import ManiSkillEnvWrapper
            env = ManiSkillEnvWrapper(env_cfg)

        env.make()
        return env

    def run(
        self,
        num_episodes: int = 80,
        verbose: bool = True,
    ) -> Dict:
        """
        Run MaW + TT-VLA experiment.

        Flow per episode:
          1. Optionally warm-start MaW from previous episode.
          2. Run TT-VLA episode (with MaW hooks active).
          3. After episode, save MaW state for warm-start.
          4. Log MaW-specific metrics (reconstruction loss, LaCT updates).
        """
        self.vla = self._build_vla()
        self.env = self._build_env()

        # Build progress estimator
        progress_cfg = self.config.get("progress", {})
        progress_est = build_progress_estimator(progress_cfg, env=self.env)

        # Build TT-VLA with MaW-augmented wrapper
        self.ttvla = TTVLA(self.vla, progress_est, self.config)
        self.ttvla.setup_lora()

        seeds = self.config.get("experiment", {}).get("seeds", [42, 123, 456])

        results = []
        successes = 0

        print(f"\n{'='*60}")
        print(f"MaW + TT-VLA Experiment: {num_episodes} episodes")
        print(f"  Model: {self.config.get('vla', {}).get('model_name', '?')}")
        print(f"  MaW layers: {self.maw_layer_indices}")
        print(f"  Chunk size (C): {self.chunk_size}")
        print(f"  TTT LR: {self.ttt_lr}")
        print(f"  Alpha init: {self.alpha_init}")
        print(f"  Warm-start: {self.warm_start_enabled}")
        print(f"{'='*60}\n")

        for ep in range(num_episodes):
            seed = seeds[ep % len(seeds)] * 10000 + ep

            if verbose:
                print(f"[Episode {ep + 1}/{num_episodes}] seed={seed}")

            # Warm-start MaW from previous episode
            if ep > 0 and self.warm_start_enabled:
                self.vla.warm_start()
            elif ep == 0:
                self.vla.reset_maw()

            # Run episode
            start = time.time()
            result = self.ttvla.run_episode(self.env, seed=seed, verbose=verbose)
            elapsed = time.time() - start

            # Force a final LaCT update with any remaining buffered states
            final_losses = self.vla._run_lact_update()

            if result.success:
                successes += 1

            # Save MaW state for next episode warm-start
            if self.warm_start_enabled:
                self.vla.save_maw_state()

            # Collect MaW metrics
            avg_recon_loss = 0.0
            if final_losses:
                avg_recon_loss = float(np.mean(list(final_losses.values())))

            ep_data = {
                "episode": ep + 1,
                "seed": seed,
                "success": result.success,
                "total_reward": float(result.total_reward),
                "final_progress": float(result.final_progress),
                "num_steps": result.num_steps,
                "num_updates": result.num_updates,
                "avg_policy_loss": float(np.mean(result.policy_losses)) if result.policy_losses else 0.0,
                "avg_reconstruction_loss": avg_recon_loss,
                "wall_time": elapsed,
            }
            results.append(ep_data)

            # Append to JSONL log
            with open(self._jsonl_path, "a") as f:
                f.write(json.dumps(ep_data) + "\n")

            if verbose:
                sr = successes / len(results) * 100
                print(
                    f"  -> {'SUCCESS' if result.success else 'FAIL'} | "
                    f"progress={result.final_progress:.3f} | "
                    f"reward={result.total_reward:.4f} | "
                    f"recon_loss={avg_recon_loss:.6f} | "
                    f"time={elapsed:.1f}s | "
                    f"SR={sr:.1f}%\n"
                )

        # Summary
        success_rate = successes / len(results) * 100
        avg_reward = float(np.mean([r["total_reward"] for r in results]))
        avg_progress = float(np.mean([r["final_progress"] for r in results]))

        summary = {
            "success_rate": success_rate,
            "avg_reward": avg_reward,
            "avg_progress": avg_progress,
            "num_episodes": len(results),
            "maw_config": {
                "layer_indices": self.maw_layer_indices,
                "chunk_size": self.chunk_size,
                "ttt_lr": self.ttt_lr,
                "alpha_init": self.alpha_init,
                "warm_start": self.warm_start_enabled,
            },
            "results": results,
        }

        print(f"{'='*60}")
        print(f"RESULTS (MaW + TT-VLA)")
        print(f"  Success Rate: {success_rate:.1f}%")
        print(f"  Avg Reward: {avg_reward:.4f}")
        print(f"  Avg Progress: {avg_progress:.3f}")
        print(f"{'='*60}")

        # Save results
        result_path = self.log_dir / "maw_results.json"
        with open(result_path, "w") as f:
            json.dump(summary, f, indent=2, default=lambda o: float(o) if isinstance(o, (np.floating,)) else o)
        print(f"Results saved to {result_path}")

        # Save final MaW checkpoint
        ckpt_path = str(self.log_dir / "maw_final.pt")
        self.vla.save_maw_checkpoint(ckpt_path)

        if self.env:
            self.env.close()

        return summary
