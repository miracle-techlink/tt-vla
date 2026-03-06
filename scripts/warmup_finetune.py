#!/usr/bin/env python3
"""
Warm-up fine-tuning with DDP (8-GPU parallel).

Each GPU holds a full model copy and processes independent batches.
OpenVLA-7B BF16 = ~14GB, + LoRA + optimizer + activations ~ 22-26GB per GPU.
Gradient checkpointing keeps it under 32GB.

Usage:
  torchrun --nproc_per_node=8 scripts/warmup_finetune.py --steps 10000
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import json
import time
import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import Dataset, DataLoader, DistributedSampler
from PIL import Image
import gymnasium as gym
import mani_skill.envs  # noqa
from peft import LoraConfig, get_peft_model


def encode_action(action, n_bins=256, vocab_size=32000):
    bins = np.linspace(-1, 1, n_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    tokens = []
    for val in action[:7]:
        val = float(max(-1, min(1, val)))
        bin_idx = int(np.argmin(np.abs(bin_centers - val)))
        tokens.append(vocab_size - 1 - bin_idx)
    return tokens


# ── Scripted Policy (same as before) ───────────────────────────────

class ScriptedPickCubePolicy:
    def __init__(self):
        self.phase = "reach"; self.grasp_steps = 0
    def reset(self):
        self.phase = "reach"; self.grasp_steps = 0
    def act(self, obs, info):
        tcp = obs["extra"]["tcp_pose"].squeeze().cpu().numpy()
        obj = obs["extra"]["obj_pose"].squeeze().cpu().numpy()
        goal = obs["extra"]["goal_pos"].squeeze().cpu().numpy()
        ig = info.get("is_grasped", False)
        if isinstance(ig, torch.Tensor): ig = ig.item()
        tp, op = tcp[:3], obj[:3]
        a = np.zeros(7, dtype=np.float32)
        if self.phase == "reach":
            t = op.copy(); t[2] += 0.05; d = t - tp
            if np.linalg.norm(d) < 0.015: self.phase = "descend"
            a[:3] = np.clip(d*10,-1,1); a[6] = 1.0
        elif self.phase == "descend":
            d = op - tp; d[2] -= 0.005
            if np.linalg.norm(d) < 0.012: self.phase = "grasp"; self.grasp_steps = 0
            a[:3] = np.clip(d*10,-1,1); a[6] = 1.0
        elif self.phase == "grasp":
            self.grasp_steps += 1; a[6] = -1.0
            if self.grasp_steps > 15: self.phase = "lift" if ig else "reach"
        elif self.phase == "lift":
            if not ig: self.phase = "reach"
            else:
                t = tp.copy(); t[2] = goal[2]+0.1
                if tp[2] > goal[2]+0.05: self.phase = "move"
                a[:3] = np.clip((t-tp)*10,-1,1)
            a[6] = -1.0
        elif self.phase == "move":
            if not ig: self.phase = "reach"
            else:
                t = goal.copy(); t[2] += 0.05
                if np.linalg.norm((t-tp)[:2]) < 0.02: self.phase = "place"
                a[:3] = np.clip((t-tp)*10,-1,1)
            a[6] = -1.0
        elif self.phase == "place":
            d = goal - tp
            if np.linalg.norm(d) < 0.02: a[6] = 1.0
            else: a[:3] = np.clip(d*10,-1,1); a[6] = -1.0
        return a


def collect_demonstrations(num_demos=200, max_steps=200):
    env = gym.make("PickCube-v1", obs_mode="rgb+state_dict", render_mode="cameras",
                    control_mode="pd_ee_delta_pose", max_episode_steps=max_steps)
    policy = ScriptedPickCubePolicy()
    demos = []; successes = 0
    instr = "pick up the cube and place it on the target"
    print(f"Collecting {num_demos} demos...")
    for ep in range(num_demos):
        obs, info = env.reset(seed=ep); policy.reset(); ep_data = []
        for step in range(max_steps):
            rgb = obs["sensor_data"]["base_camera"]["rgb"]
            if isinstance(rgb, torch.Tensor): rgb = rgb.squeeze(0).cpu().numpy()
            action = policy.act(obs, info)
            ep_data.append((rgb, instr, action.copy()))
            obs, r, t, tr, info = env.step(action)
            s = info.get("success", False)
            if isinstance(s, torch.Tensor): s = s.item()
            if bool(t) or bool(tr):
                if s: successes += 1; demos.extend(ep_data)
                break
        if (ep+1) % 50 == 0: print(f"  {ep+1}/{num_demos}, demos={len(demos)}, SR={100*successes/(ep+1):.0f}%")
    env.close()
    print(f"Done: {len(demos)} transitions, SR={100*successes/num_demos:.0f}%")
    return demos


# ── Pre-tokenized Dataset ───────────────────────────────────────────

class TokenizedDemoDataset(Dataset):
    def __init__(self, demos, processor, n_bins=256):
        self.items = []
        print(f"Tokenizing {len(demos)} demos...")
        t0 = time.time()
        for i, (img, instr, act) in enumerate(demos):
            pil = Image.fromarray(img).resize((224, 224), Image.BILINEAR)
            prompt = f"In: What action should the robot take to {instr}?\nOut:"
            inp = processor(prompt, pil)
            atoks = torch.tensor(encode_action(act, n_bins, 32000), dtype=torch.long)
            ids = inp["input_ids"].squeeze(0)
            pv = inp["pixel_values"].squeeze(0)
            full = torch.cat([ids, atoks])
            lab = full.clone(); lab[:ids.shape[0]] = -100
            self.items.append((full, pv, lab))
            if (i+1) % 2000 == 0: print(f"  {i+1}/{len(demos)} ({time.time()-t0:.0f}s)")
        print(f"Tokenized in {time.time()-t0:.0f}s")

    def __len__(self): return len(self.items)
    def __getitem__(self, idx):
        ids, pv, lab = self.items[idx]
        return {"input_ids": ids, "pixel_values": pv, "labels": lab, "attention_mask": torch.ones_like(ids)}


# ── Main ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=10000)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--batch-size", type=int, default=2, help="Per-GPU batch size")
    parser.add_argument("--num-demos", type=int, default=200)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--save-dir", default="warmup_ckpt")
    parser.add_argument("--save-every", type=int, default=2000)
    parser.add_argument("--resume", default=None)
    args = parser.parse_args()

    # Init distributed
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    use_ddp = world_size > 1

    if use_ddp:
        dist.init_process_group("nccl")
        torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")
    is_main = local_rank == 0

    # Collect demos on rank 0 only
    if is_main:
        demos = collect_demonstrations(args.num_demos, 200)
    else:
        demos = None

    if use_ddp:
        # Broadcast demo count
        if is_main:
            demo_count = torch.tensor([len(demos)], device=device)
        else:
            demo_count = torch.tensor([0], device=device)
        dist.broadcast(demo_count, 0)

        # Share demos via object list
        demo_list = [demos]
        dist.broadcast_object_list(demo_list, 0)
        demos = demo_list[0]

    if is_main: print(f"All ranks have {len(demos)} demos")

    # Load model on each GPU
    from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig

    if is_main: print("Loading model...")
    processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    config = AutoConfig.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
    n_bins = getattr(config, "n_action_bins", 256)

    model = AutoModelForVision2Seq.from_pretrained(
        "openvla/openvla-7b", torch_dtype=torch.bfloat16,
        trust_remote_code=True, low_cpu_mem_usage=True,
    ).to(device)

    # Gradient checkpointing
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()

    # LoRA
    lora_cfg = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_rank,
        target_modules=["q_proj","v_proj","k_proj","o_proj","gate_proj","up_proj","down_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    if is_main:
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_p = sum(p.numel() for p in model.parameters())
        print(f"LoRA: {trainable:,}/{total_p:,} ({100*trainable/total_p:.2f}%)")

    # DDP wrap
    if use_ddp:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=False)

    # Dataset
    dataset = TokenizedDemoDataset(demos, processor, n_bins)
    sampler = DistributedSampler(dataset, shuffle=True) if use_ddp else None
    loader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler,
                        shuffle=(sampler is None), num_workers=2, pin_memory=True, drop_last=True)

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.01)

    # Resume
    start_step = 0
    if args.resume and os.path.exists(args.resume):
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=False)
        raw = model.module if use_ddp else model
        raw.load_state_dict(ckpt["model_state_dict"], strict=False)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        start_step = ckpt.get("step", 0)
        if is_main: print(f"Resumed from step {start_step}")

    # Train
    model.train()
    effective_batch = args.batch_size * world_size
    if is_main:
        print(f"\nTraining: {args.steps} steps, per-gpu batch={args.batch_size}, "
              f"world={world_size}, effective={effective_batch}")

    os.makedirs(args.save_dir, exist_ok=True)
    losses = []
    t0 = time.time()
    step = start_step
    epoch = 0

    while step < args.steps:
        if sampler: sampler.set_epoch(epoch)
        for batch in loader:
            if step >= args.steps: break

            ids = batch["input_ids"].to(device)
            pv = batch["pixel_values"].to(device, dtype=torch.bfloat16)
            lab = batch["labels"].to(device)
            am = batch["attention_mask"].to(device)

            optimizer.zero_grad()
            out = model(input_ids=ids, attention_mask=am, pixel_values=pv, labels=lab)
            out.loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            step += 1
            if is_main:
                losses.append(out.loss.item())
                if step % 100 == 0:
                    avg = np.mean(losses[-100:])
                    el = time.time() - t0
                    sps = (step - start_step) / el
                    eta = (args.steps - step) / sps if sps > 0 else 0
                    print(f"  Step {step}/{args.steps} | Loss: {avg:.4f} | "
                          f"Speed: {sps:.1f} steps/s | ETA: {eta/60:.1f} min")

                if step % args.save_every == 0:
                    raw = model.module if use_ddp else model
                    torch.save({
                        "step": step,
                        "model_state_dict": raw.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": np.mean(losses[-min(args.save_every, len(losses)):]),
                    }, os.path.join(args.save_dir, f"step_{step}.pt"))
                    print(f"  Saved step_{step}.pt")
        epoch += 1

    if is_main:
        raw = model.module if use_ddp else model
        torch.save({
            "step": step, "model_state_dict": raw.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "loss": np.mean(losses[-100:]) if losses else 0,
        }, os.path.join(args.save_dir, "warmup_final.pt"))
        with open(os.path.join(args.save_dir, "loss_curve.json"), "w") as f:
            json.dump({"losses": losses, "total_steps": step}, f)
        el = time.time() - t0
        print(f"\nDone! {step} steps, {el/60:.1f} min, final loss: {np.mean(losses[-100:]):.4f}")

    if use_ddp: dist.destroy_process_group()


if __name__ == "__main__":
    main()
