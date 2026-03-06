#!/usr/bin/env python3
"""
Warm-up fine-tuning for OpenVLA on ManiSkill3 PickCube-v1.

Paper (Section 4.1): "Consistent with (Liu et al., 2025a), we apply a warm-up
fine-tuning phase of 10k steps prior to evaluation."

This script:
1. Collects demonstrations using a scripted heuristic policy
2. Fine-tunes OpenVLA-7B with LoRA on collected demos (supervised)
3. Saves the warm-up checkpoint for subsequent TT-VLA evaluation

Usage:
  CUDA_VISIBLE_DEVICES=0,1 python scripts/warmup_finetune.py --steps 10000
  CUDA_VISIBLE_DEVICES=0,1 python scripts/warmup_finetune.py --steps 10000 --resume warmup_ckpt/step_5000.pt
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

import argparse
import json
import time
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import gymnasium as gym
import mani_skill.envs  # noqa
from peft import LoraConfig, get_peft_model


# ── Scripted Expert Policy ──────────────────────────────────────────

class ScriptedPickCubePolicy:
    """
    Heuristic policy for PickCube-v1 using ground-truth state.
    Phases: reach -> grasp -> lift -> move -> place
    """

    def __init__(self):
        self.phase = "reach"
        self.grasp_steps = 0

    def reset(self):
        self.phase = "reach"
        self.grasp_steps = 0

    def act(self, obs: dict, info: dict) -> np.ndarray:
        tcp_pose = obs["extra"]["tcp_pose"].squeeze().cpu().numpy()
        obj_pose = obs["extra"]["obj_pose"].squeeze().cpu().numpy()
        goal_pos = obs["extra"]["goal_pos"].squeeze().cpu().numpy()
        tcp_to_obj = obs["extra"]["tcp_to_obj_pos"].squeeze().cpu().numpy()
        is_grasped = info.get("is_grasped", False)
        if isinstance(is_grasped, torch.Tensor):
            is_grasped = is_grasped.item()

        tcp_pos = tcp_pose[:3]
        obj_pos = obj_pose[:3]

        # 8-dim action: [dx, dy, dz, dax, day, daz, dgrip_width, ???]
        # For pd_ee_delta_pos control: [dx, dy, dz, dgrip] (simplified)
        # ManiSkill3 PickCube default action space is 8-dim
        action = np.zeros(8, dtype=np.float32)

        if self.phase == "reach":
            # Move above the object first
            target = obj_pos.copy()
            target[2] += 0.05  # 5cm above
            delta = target - tcp_pos
            dist = np.linalg.norm(delta)
            if dist < 0.02:
                self.phase = "descend"
            else:
                action[:3] = np.clip(delta * 5, -1, 1)
            action[6] = 1.0  # Open gripper

        elif self.phase == "descend":
            # Lower onto the object
            delta = obj_pos - tcp_pos
            delta[2] -= 0.005  # Slight offset down
            dist = np.linalg.norm(delta)
            if dist < 0.015:
                self.phase = "grasp"
                self.grasp_steps = 0
            else:
                action[:3] = np.clip(delta * 5, -1, 1)
            action[6] = 1.0  # Open gripper

        elif self.phase == "grasp":
            # Close gripper
            self.grasp_steps += 1
            action[6] = -1.0  # Close gripper
            if self.grasp_steps > 10:
                if is_grasped:
                    self.phase = "lift"
                else:
                    # Retry
                    self.phase = "reach"

        elif self.phase == "lift":
            if not is_grasped:
                self.phase = "reach"
            else:
                # Lift up
                target = tcp_pos.copy()
                target[2] = goal_pos[2] + 0.1
                delta = target - tcp_pos
                if tcp_pos[2] > goal_pos[2] + 0.05:
                    self.phase = "move"
                else:
                    action[:3] = np.clip(delta * 5, -1, 1)
            action[6] = -1.0  # Keep gripper closed

        elif self.phase == "move":
            if not is_grasped:
                self.phase = "reach"
            else:
                target = goal_pos.copy()
                target[2] += 0.05
                delta = target - tcp_pos
                dist_xy = np.linalg.norm(delta[:2])
                if dist_xy < 0.02:
                    self.phase = "place"
                else:
                    action[:3] = np.clip(delta * 5, -1, 1)
            action[6] = -1.0

        elif self.phase == "place":
            # Lower to goal
            delta = goal_pos - tcp_pos
            dist = np.linalg.norm(delta)
            if dist < 0.02:
                action[6] = 1.0  # Release
            else:
                action[:3] = np.clip(delta * 5, -1, 1)
                action[6] = -1.0

        return action


# ── Demo Collection ─────────────────────────────────────────────────

def collect_demonstrations(num_demos: int = 200, max_steps: int = 200) -> list:
    """Collect demonstrations using scripted policy, return list of (image, instruction, action) tuples."""
    env = gym.make(
        "PickCube-v1",
        obs_mode="rgb+state_dict",
        render_mode="cameras",
        max_episode_steps=max_steps,
    )
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
            # Extract RGB image
            rgb = obs["sensor_data"]["base_camera"]["rgb"]
            if isinstance(rgb, torch.Tensor):
                rgb = rgb.squeeze(0).cpu().numpy()

            # Get expert action from state
            action = policy.act(obs, info)

            # Store transition
            episode_data.append((rgb, instruction, action.copy()))

            # Step environment
            obs, reward, term, trunc, info = env.step(action)
            done = bool(term) or bool(trunc)

            success = info.get("success", False)
            if isinstance(success, torch.Tensor):
                success = success.item()

            if done:
                if success:
                    successes += 1
                    demos.extend(episode_data)
                break

        if (ep + 1) % 20 == 0:
            print(f"  Episode {ep+1}/{num_demos}, demos collected: {len(demos)}, SR: {successes}/{ep+1} ({100*successes/(ep+1):.1f}%)")

    env.close()
    print(f"Demo collection done: {len(demos)} transitions from {successes}/{num_demos} successful episodes ({100*successes/num_demos:.1f}% SR)")
    return demos


# ── Demo Dataset ────────────────────────────────────────────────────

class DemoDataset(Dataset):
    """Dataset of (image, instruction, action) demos for supervised fine-tuning."""

    def __init__(self, demos: list):
        self.demos = demos

    def __len__(self):
        return len(self.demos)

    def __getitem__(self, idx):
        image, instruction, action = self.demos[idx]
        return image, instruction, action


# ── Warm-up Training Loop ──────────────────────────────────────────

def warmup_finetune(
    demos: list,
    model_name: str = "openvla/openvla-7b",
    lora_rank: int = 32,
    lr: float = 2e-5,
    total_steps: int = 10000,
    batch_size: int = 4,
    save_dir: str = "warmup_ckpt",
    save_every: int = 1000,
    resume: str = None,
):
    from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig

    os.makedirs(save_dir, exist_ok=True)

    print(f"Loading {model_name}...")
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)

    model = AutoModelForVision2Seq.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        device_map="auto",
    )
    device = next(model.parameters()).device

    # Get action token config
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    n_action_bins = getattr(config, "n_action_bins", 256)
    ACTION_TOKEN_OFFSET = 32000

    # Apply LoRA
    lora_config = LoraConfig(
        r=lora_rank,
        lora_alpha=lora_rank,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"LoRA applied: {trainable:,} trainable / {total_p:,} total ({100*trainable/total_p:.2f}%)")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=lr, weight_decay=0.01,
    )

    # Resume if specified
    start_step = 0
    if resume and os.path.exists(resume):
        ckpt = torch.load(resume, map_location="cpu")
        model.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0)
        print(f"Resumed from {resume} at step {start_step}")

    def encode_action(action, n_bins=256, offset=32000):
        """Encode continuous action to token IDs."""
        tokens = []
        for val in action[:7]:  # OpenVLA uses 7 action dims
            normalized = (float(val) + 1) / 2
            normalized = max(0.0, min(1.0, normalized))
            bin_idx = int(normalized * (n_bins - 1))
            tokens.append(bin_idx + offset)
        return tokens

    # Training loop
    model.train()
    dataset = DemoDataset(demos)
    losses = []
    t0 = time.time()

    print(f"\nStarting warm-up fine-tuning: {total_steps} steps, lr={lr}, batch_size={batch_size}")
    print(f"Dataset: {len(dataset)} transitions")

    step = start_step
    while step < total_steps:
        # Random sample from demos
        indices = random.sample(range(len(dataset)), min(batch_size, len(dataset)))

        batch_loss = 0.0
        optimizer.zero_grad()

        for idx in indices:
            image, instruction, action = dataset[idx]
            pil_image = Image.fromarray(image).resize((224, 224), Image.BILINEAR)
            prompt = f"In: What action should the robot take to {instruction}?\nOut:"

            inputs = processor(prompt, pil_image).to(device, dtype=torch.bfloat16)
            action_tokens = torch.tensor(
                encode_action(action, n_action_bins, ACTION_TOKEN_OFFSET),
                dtype=torch.long, device=device,
            )

            # Teacher forcing: input_ids + action tokens
            input_ids = inputs["input_ids"]
            full_ids = torch.cat([input_ids, action_tokens.unsqueeze(0)], dim=1)
            labels = full_ids.clone()
            labels[0, :input_ids.shape[1]] = -100  # Mask non-action tokens

            outputs = model(
                input_ids=full_ids,
                attention_mask=torch.ones_like(full_ids),
                pixel_values=inputs.get("pixel_values"),
                labels=labels,
            )

            loss = outputs.loss / len(indices)  # Accumulate gradients
            loss.backward()
            batch_loss += loss.item()

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(batch_loss)
        step += 1

        if step % 100 == 0:
            avg_loss = np.mean(losses[-100:])
            elapsed = time.time() - t0
            steps_per_sec = step / elapsed if elapsed > 0 else 0
            eta = (total_steps - step) / steps_per_sec if steps_per_sec > 0 else 0
            print(f"  Step {step}/{total_steps} | Loss: {avg_loss:.4f} | "
                  f"Speed: {steps_per_sec:.1f} steps/s | ETA: {eta/60:.1f} min")

        if step % save_every == 0:
            ckpt_path = os.path.join(save_dir, f"step_{step}.pt")
            torch.save({
                "step": step,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": np.mean(losses[-save_every:]),
            }, ckpt_path)
            print(f"  Checkpoint saved: {ckpt_path}")

    # Save final checkpoint
    final_path = os.path.join(save_dir, "warmup_final.pt")
    torch.save({
        "step": step,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": np.mean(losses[-100:]) if losses else 0,
    }, final_path)
    print(f"\nFinal checkpoint saved: {final_path}")

    # Save loss curve
    loss_path = os.path.join(save_dir, "loss_curve.json")
    with open(loss_path, "w") as f:
        json.dump({"losses": losses, "total_steps": step}, f)
    print(f"Loss curve saved: {loss_path}")

    elapsed = time.time() - t0
    print(f"\nWarm-up fine-tuning complete!")
    print(f"  Steps: {step}, Time: {elapsed/60:.1f} min")
    print(f"  Final loss: {np.mean(losses[-100:]):.4f}")

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="openvla/openvla-7b")
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--num-demos", type=int, default=500, help="Number of demo episodes to collect")
    parser.add_argument("--save-dir", default="warmup_ckpt")
    parser.add_argument("--save-every", type=int, default=1000)
    parser.add_argument("--resume", default=None)
    parser.add_argument("--skip-collection", action="store_true", help="Load cached demos")
    args = parser.parse_args()

    demo_cache = os.path.join(args.save_dir, "demos.npz")

    if args.skip_collection and os.path.exists(demo_cache):
        print(f"Loading cached demos from {demo_cache}")
        data = np.load(demo_cache, allow_pickle=True)
        demos = list(data["demos"])
    else:
        # Step 1: Collect demonstrations
        demos = collect_demonstrations(num_demos=args.num_demos, max_steps=200)
        if not demos:
            print("ERROR: No successful demonstrations collected! Cannot fine-tune.")
            print("The scripted policy may need tuning for this environment.")
            sys.exit(1)

        # Cache demos
        os.makedirs(args.save_dir, exist_ok=True)
        np.savez(demo_cache, demos=np.array(demos, dtype=object))
        print(f"Demos cached to {demo_cache}")

    # Step 2: Fine-tune
    warmup_finetune(
        demos=demos,
        model_name=args.model,
        lora_rank=args.lora_rank,
        lr=args.lr,
        total_steps=args.steps,
        batch_size=args.batch_size,
        save_dir=args.save_dir,
        save_every=args.save_every,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
