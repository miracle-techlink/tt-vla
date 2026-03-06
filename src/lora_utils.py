"""
LoRA utilities for TT-VLA test-time adaptation.

Paper Implementation Details (Section 4.1):
  - LoRA ranks: {16, 32}
  - Target: attention projection layers
  - Only LoRA parameters are updated at test time
  - Base model parameters remain frozen
"""

import torch
from peft import LoraConfig, get_peft_model, PeftModel
from typing import List, Optional


def apply_lora_to_vla(
    model,
    rank: int = 32,
    alpha: int = 64,
    dropout: float = 0.0,
    target_modules: Optional[List[str]] = None,
) -> PeftModel:
    """
    Apply LoRA adapters to a VLA model for test-time fine-tuning.

    The base model is frozen; only LoRA parameters (A, B matrices) are trainable.
    This is critical for TT-VLA:
      - Preserves SFT/RL-trained priors
      - Enables efficient test-time parameter updates
      - Small parameter footprint for fast adaptation

    Args:
        model: Base VLA model (e.g., OpenVLA, Nora)
        rank: LoRA rank r (paper uses 16 or 32)
        alpha: LoRA scaling factor (typically 2*rank)
        dropout: LoRA dropout
        target_modules: Which modules to apply LoRA to

    Returns:
        PeftModel with LoRA adapters
    """
    if target_modules is None:
        target_modules = ["q_proj", "v_proj", "k_proj", "o_proj"]

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        lora_dropout=dropout,
        target_modules=target_modules,
        bias="none",
        task_type="CAUSAL_LM",
    )

    peft_model = get_peft_model(model, lora_config)

    # Freeze base model, only LoRA params trainable
    for name, param in peft_model.named_parameters():
        if "lora_" not in name:
            param.requires_grad = False
        else:
            param.requires_grad = True

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in peft_model.parameters())
    print(f"[LoRA] Trainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")

    return peft_model


def reset_lora_parameters(model: PeftModel):
    """
    Reset LoRA parameters to zero (identity) for a new episode.

    At the start of each episode, we want the model to behave as the
    original pretrained VLA, then adapt during the episode.
    """
    for name, param in model.named_parameters():
        if "lora_A" in name:
            # Re-initialize A with Kaiming uniform
            torch.nn.init.kaiming_uniform_(param, a=5**0.5)
        elif "lora_B" in name:
            # Initialize B to zero so LoRA starts as identity
            torch.nn.init.zeros_(param)


def get_lora_optimizer(model: PeftModel, lr: float, weight_decay: float = 0.01):
    """Create AdamW optimizer for LoRA parameters only."""
    lora_params = [p for n, p in model.named_parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(lora_params, lr=lr, weight_decay=weight_decay)
    return optimizer
