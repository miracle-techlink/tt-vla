#!/usr/bin/env python3
"""
Warm-up fine-tuning for OpenVLA on ManiSkill3 PickCube-v1 (optimized).

Optimizations over v1:
- True batched forward pass (not sequential per-sample)
- Gradient checkpointing to save VRAM
- Larger effective batch size with gradient accumulation
- Pre-tokenized dataset for faster data loading

Usage:
  python scripts/warmup_finetune.py --steps 10000 --batch-size 8 --grad-accum 4
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import json
import time
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import gymnasium as gym
import mani_skill.envs  # noqa
from peft import LoraConfig, get_peft_model


# ── Scripted Expert Policy ──────────────────────────────────────────

class ScriptedPickCubePolicy:
    def __init__(self):
        self.phase = "reach"
        self.grasp_steps = 0

    def reset(self):
        self.phase = "reach"
        self.grasp_steps = 0

    def act(self, obs, info):
        tcp_pose = obs["extra"]["tcp_pose"].squeeze().cpu().numpy()
        obj_pose = obs["extra"]["obj_pose"].squeeze().cpu().numpy()
        goal_pos = obs["extra"]["goal_pos"].squeeze().cpu().numpy()
        is_grasped = info.get("is_grasped", False)
        if isinstance(is_grasped, torch.Tensor):
            is_grasped = is_grasped.item()
        tcp_pos, obj_pos = tcp_pose[:3], obj_pose[:3]
        action = np.zeros(7, dtype=np.float32)

        if self.phase == "reach":
            target = obj_pos.copy(); target[2] += 0.05
            delta = target - tcp_pos
            if np.linalg.norm(delta) < 0.015: self.phase = "descend"
            action[:3] = np.clip(delta * 10, -1, 1); action[6] = 1.0
        elif self.phase == "descend":
            delta = obj_pos - tcp_pos; delta[2] -= 0.005
            if np.linalg.norm(delta) < 0.012: self.phase = "grasp"; self.grasp_steps = 0
            action[:3] = np.clip(delta * 10, -1, 1); action[6] = 1.0
        elif self.phase == "grasp":
            self.grasp_steps += 1; action[6] = -1.0
            if self.grasp_steps > 15: self.phase = "lift" if is_grasped else "reach"
        elif self.phase == "lift":
            if not is_grasped: self.phase = "reach"
            else:
                target = tcp_pos.copy(); target[2] = goal_pos[2] + 0.1
                if tcp_pos[2] > goal_pos[2] + 0.05: self.phase = "move"
                action[:3] = np.clip((target - tcp_pos) * 10, -1, 1)
            action[6] = -1.0
        elif self.phase == "move":
            if not is_grasped: self.phase = "reach"
            else:
                target = goal_pos.copy(); target[2] += 0.05
                if np.linalg.norm((target - tcp_pos)[:2]) < 0.02: self.phase = "place"
                action[:3] = np.clip((target - tcp_pos) * 10, -1, 1)
            action[6] = -1.0
        elif self.phase == "place":
            delta = goal_pos - tcp_pos
            if np.linalg.norm(delta) < 0.02: action[6] = 1.0
            else: action[:3] = np.clip(delta * 10, -1, 1); action[6] = -1.0
        return action


def collect_demonstrations(num_demos=200, max_steps=200):
    env = gym.make("PickCube-v1", obs_mode="rgb+state_dict", render_mode="cameras",
                    control_mode="pd_ee_delta_pose", max_episode_steps=max_steps)
    policy = ScriptedPickCubePolicy()
    demos = []
    successes = 0
    instruction = "pick up the cube and place it on the target"
    print(f"Collecting {num_demos} demonstrations...")
    for ep in range(num_demos):
        obs, info = env.reset(seed=ep)
        policy.reset()
        episode_data = []
        for step in range(max_steps):
            rgb = obs["sensor_data"]["base_camera"]["rgb"]
            if isinstance(rgb, torch.Tensor): rgb = rgb.squeeze(0).cpu().numpy()
            action = policy.act(obs, info)
            episode_data.append((rgb, instruction, action.copy()))
            obs, reward, term, trunc, info = env.step(action)
            done = bool(term) or bool(trunc)
            success = info.get("success", False)
            if isinstance(success, torch.Tensor): success = success.item()
            if done:
                if success: successes += 1; demos.extend(episode_data)
                break
        if (ep + 1) % 20 == 0:
            print(f"  Episode {ep+1}/{num_demos}, demos: {len(demos)}, SR: {successes}/{ep+1} ({100*successes/(ep+1):.1f}%)")
    env.close()
    print(f"Done: {len(demos)} transitions, {successes}/{num_demos} success ({100*successes/num_demos:.1f}%)")
    return demos


# ── Pre-tokenized Dataset ───────────────────────────────────────────

def encode_action(action, n_bins=256, vocab_size=32000):
    """OpenVLA encoding: token_id = vocab_size - 1 - bin_idx"""
    bins = np.linspace(-1, 1, n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    tokens = []
    for val in action[:7]:
        val = float(max(-1, min(1, val)))
        bin_idx = int(np.argmin(np.abs(bin_centers - val)))
        tokens.append(vocab_size - 1 - bin_idx)
    return tokens


class PreTokenizedDataset(Dataset):
    """Pre-tokenize all demos for fast batched training."""

    def __init__(self, demos, processor, n_action_bins=256):
        self.input_ids_list = []
        self.pixel_values_list = []
        self.action_tokens_list = []
        self.attn_mask_list = []

        print(f"Pre-tokenizing {len(demos)} demos...")
        t0 = time.time()
        for i, (image, instruction, action) in enumerate(demos):
            pil_image = Image.fromarray(image).resize((224, 224), Image.BILINEAR)
            prompt = f"In: What action should the robot take to {instruction}?\nOut:"
            inputs = processor(prompt, pil_image)

            action_tokens = encode_action(action, n_action_bins, 32000)
            action_tensor = torch.tensor(action_tokens, dtype=torch.long)

            self.input_ids_list.append(inputs["input_ids"].squeeze(0))
            self.pixel_values_list.append(inputs["pixel_values"].squeeze(0))
            self.action_tokens_list.append(action_tensor)

            if (i + 1) % 2000 == 0:
                print(f"  Tokenized {i+1}/{len(demos)} ({time.time()-t0:.1f}s)")

        # All input_ids should have same length (same prompt + processor)
        self.input_len = self.input_ids_list[0].shape[0]
        print(f"Pre-tokenization done in {time.time()-t0:.1f}s, input_len={self.input_len}")

    def __len__(self):
        return len(self.input_ids_list)

    def __getitem__(self, idx):
        input_ids = self.input_ids_list[idx]
        pixel_values = self.pixel_values_list[idx]
        action_tokens = self.action_tokens_list[idx]

        # Concat input_ids + action_tokens
        full_ids = torch.cat([input_ids, action_tokens])
        labels = full_ids.clone()
        labels[:self.input_len] = -100  # Mask prompt tokens

        attn_mask = torch.ones_like(full_ids)
        return {
            "input_ids": full_ids,
            "pixel_values": pixel_values,
            "labels": labels,
            "attention_mask": attn_mask,
        }


# ── Optimized Training Loop ────────────────────────────────────────

def warmup_finetune(
    demos, model_name="openvla/openvla-7b", lora_rank=32, lr=2e-5,
    total_steps=10000, batch_size=8, grad_accum=4,
    save_dir="warmup_ckpt", save_every=2000, resume=None,
):
    from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig

    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    n_action_bins = getattr(config, "n_action_bins", 256)

    model = AutoModelForVision2Seq.from_pretrained(
        model_name, torch_dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True, device_map="auto",
    )
    device = next(model.parameters()).device

    # Enable gradient checkpointing to save VRAM
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled")
    elif hasattr(model, "base_model") and hasattr(model.base_model, "gradient_checkpointing_enable"):
        model.base_model.gradient_checkpointing_enable()
        print("Gradient checkpointing enabled (base_model)")

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_rank, lora_alpha=lora_rank,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"LoRA: {trainable:,} trainable / {total_p:,} total ({100*trainable/total_p:.2f}%)")

    # Enable gradient checkpointing after peft wrapping
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # Pre-tokenize dataset
    dataset = PreTokenizedDataset(demos, processor, n_action_bins)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True,
                            num_workers=4, pin_memory=True, drop_last=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )

    # Resume
    start_step = 0
    if resume and os.path.exists(resume):
        ckpt = torch.load(resume, map_location="cpu", weights_only=False)
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0)
        print(f"Resumed from {resume} at step {start_step}")

    # Training
    model.train()
    losses = []
    t0 = time.time()
    step = start_step
    effective_batch = batch_size * grad_accum

    print(f"\nTraining: {total_steps} steps, batch={batch_size}, grad_accum={grad_accum}, effective_batch={effective_batch}")
    print(f"Dataset: {len(dataset)} transitions")

    data_iter = iter(dataloader)
    optimizer.zero_grad()
    accum_loss = 0.0
    accum_count = 0

    while step < total_steps:
        # Get batch (cycle dataloader)
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(dataloader)
            batch = next(data_iter)

        # Move to device
        input_ids = batch["input_ids"].to(device)
        pixel_values = batch["pixel_values"].to(device, dtype=torch.bfloat16)
        labels = batch["labels"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        # Forward
        outputs = model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            pixel_values=pixel_values,
            labels=labels,
        )
        loss = outputs.loss / grad_accum
        loss.backward()

        accum_loss += outputs.loss.item()
        accum_count += 1

        if accum_count >= grad_accum:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()

            step += 1
            losses.append(accum_loss / accum_count)
            accum_loss = 0.0
            accum_count = 0

            if step % 100 == 0:
                avg_loss = np.mean(losses[-100:])
                elapsed = time.time() - t0
                sps = (step - start_step) / elapsed
                eta = (total_steps - step) / sps if sps > 0 else 0
                print(f"  Step {step}/{total_steps} | Loss: {avg_loss:.4f} | "
                      f"Speed: {sps:.1f} steps/s | ETA: {eta/60:.1f} min")

            if step % save_every == 0:
                ckpt_path = os.path.join(save_dir, f"step_{step}.pt")
                torch.save({
                    "step": step,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "loss": np.mean(losses[-min(save_every, len(losses)):]),
                }, ckpt_path)
                print(f"  Checkpoint saved: {ckpt_path}")

    # Final save
    final_path = os.path.join(save_dir, "warmup_final.pt")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": np.mean(losses[-100:]) if losses else 0,
    }, final_path)

    loss_path = os.path.join(save_dir, "loss_curve.json")
    with open(loss_path, "w") as f:
        json.dump({"losses": losses, "total_steps": step}, f)

    elapsed = time.time() - t0
    print(f"\nDone! Steps: {step}, Time: {elapsed/60:.1f} min, Final loss: {np.mean(losses[-100:]):.4f}")
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openvla/openvla-7b")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--grad-accum", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--num-demos", type=int, default=200)
    parser.add_argument("--save-dir", default="warmup_ckpt")
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    demos = collect_demonstrations(num_demos=args.num_demos, max_steps=200)
    if not demos:
        print("ERROR: No demos collected!")
        sys.exit(1)

    warmup_finetune(
        demos=demos, model_name=args.model, lora_rank=args.lora_rank,
        lr=args.lr, total_steps=args.steps, batch_size=args.batch_size,
        grad_accum=args.grad_accum, save_dir=args.save_dir,
        save_every=args.save_every, resume=args.resume,
    )


if __name__ == "__main__":
    main()
