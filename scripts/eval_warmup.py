#!/usr/bin/env python3
"""
Evaluate warm-up fine-tuned OpenVLA on ManiSkill3 PickCube-v1.
Pure inference — no TT-VLA PPO updates.

Usage:
  CUDA_VISIBLE_DEVICES=0,1 python scripts/eval_warmup.py --checkpoint warmup_ckpt/step_3000.pt --episodes 20
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
import gymnasium as gym
import mani_skill.envs  # noqa
from PIL import Image
from peft import LoraConfig, get_peft_model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--model", default="openvla/openvla-7b")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--max-steps", type=int, default=160)
    parser.add_argument("--lora-rank", type=int, default=32)
    parser.add_argument("--output", default="results/warmup_eval_results.json")
    args = parser.parse_args()

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load model
    from transformers import AutoModelForVision2Seq, AutoProcessor, AutoConfig

    print(f"Loading {args.model}...")
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True)
    config = AutoConfig.from_pretrained(args.model, trust_remote_code=True)
    n_action_bins = getattr(config, "n_action_bins", 256)

    model = AutoModelForVision2Seq.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, trust_remote_code=True,
        low_cpu_mem_usage=True, device_map="auto",
    )
    model_device = next(model.parameters()).device

    # Apply LoRA (same config as training)
    lora_config = LoraConfig(
        r=args.lora_rank, lora_alpha=args.lora_rank,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)

    # Load checkpoint
    print(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model.load_state_dict(ckpt["model_state_dict"], strict=False)
    ckpt_step = ckpt.get("step", "?")
    ckpt_loss = ckpt.get("loss", "?")
    print(f"Checkpoint step={ckpt_step}, loss={ckpt_loss}")

    model.eval()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_p = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_p:,} total, {trainable:,} LoRA")

    # Action decoding (matches OpenVLA encoding)
    bins = np.linspace(-1, 1, n_action_bins)
    bin_centers = (bins[:-1] + bins[1:]) / 2.0
    VOCAB_SIZE = 32000  # text_config.vocab_size - pad_to_multiple_of

    def decode_action(token_ids):
        action = []
        for tid in token_ids[:7]:
            bin_idx = VOCAB_SIZE - 1 - int(tid)
            bin_idx = max(0, min(len(bin_centers) - 1, bin_idx))
            action.append(bin_centers[bin_idx])
        return np.array(action, dtype=np.float32)

    # Create environment
    print("Creating ManiSkill3 PickCube-v1 (pd_ee_delta_pose)...")
    env = gym.make("PickCube-v1", obs_mode="rgb", render_mode="cameras",
                    control_mode="pd_ee_delta_pose", max_episode_steps=args.max_steps)

    instruction = "pick up the cube and place it on the target"

    # Run evaluation
    print(f"\n{'='*60}")
    print(f"Warm-up Evaluation: {args.episodes} episodes")
    print(f"  Model: {args.model}")
    print(f"  Checkpoint: {args.checkpoint} (step {ckpt_step})")
    print(f"  LoRA rank: {args.lora_rank}")
    print(f"  No TT-VLA (pure inference)")
    print(f"{'='*60}\n")

    results = []
    for ep in range(args.episodes):
        obs, info = env.reset(seed=ep * 42)
        total_reward = 0.0
        success = False
        t0 = time.time()

        for step in range(args.max_steps):
            # Extract RGB
            if isinstance(obs, dict) and "sensor_data" in obs:
                for cam in obs["sensor_data"]:
                    rgb = obs["sensor_data"][cam]["rgb"]
                    if isinstance(rgb, torch.Tensor):
                        rgb = rgb.squeeze(0).cpu().numpy()
                    break
            else:
                rgb = np.zeros((128, 128, 3), dtype=np.uint8)

            # Generate action
            pil_image = Image.fromarray(rgb).resize((224, 224), Image.BILINEAR)
            prompt = f"In: What action should the robot take to {instruction}?\nOut:"
            inputs = processor(prompt, pil_image).to(model_device, dtype=torch.bfloat16)

            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=7, do_sample=False)

            input_len = inputs["input_ids"].shape[1]
            action_token_ids = output[0, input_len:input_len + 7].cpu().numpy()
            action = decode_action(action_token_ids)

            # Pad to env action dim
            env_dim = env.action_space.shape[0]
            if len(action) < env_dim:
                action = np.concatenate([action, np.zeros(env_dim - len(action))])
            elif len(action) > env_dim:
                action = action[:env_dim]

            obs, reward, term, trunc, info = env.step(action)
            total_reward += float(reward)
            done = bool(term) or bool(trunc)

            s = info.get("success", False)
            if isinstance(s, torch.Tensor): s = s.item()
            if s: success = True

            if done:
                break

        elapsed = time.time() - t0
        results.append({
            "episode": ep + 1,
            "success": success,
            "total_reward": total_reward,
            "steps": step + 1,
            "time_s": round(elapsed, 1),
        })

        sr = sum(r["success"] for r in results) / len(results) * 100
        print(f"  Ep {ep+1}/{args.episodes}: {'SUCCESS' if success else 'FAIL'} | "
              f"reward={total_reward:.4f} | steps={step+1} | "
              f"time={elapsed:.1f}s | SR={sr:.1f}%")

    env.close()

    # Summary
    success_rate = sum(r["success"] for r in results) / len(results) * 100
    avg_reward = np.mean([r["total_reward"] for r in results])
    avg_steps = np.mean([r["steps"] for r in results])
    avg_time = np.mean([r["time_s"] for r in results])

    print(f"\n{'='*60}")
    print(f"RESULTS (warm-up only, no TT-VLA)")
    print(f"  Success Rate: {success_rate:.1f}%")
    print(f"  Avg Reward: {avg_reward:.4f}")
    print(f"  Avg Steps: {avg_steps:.1f}")
    print(f"  Avg Time/Ep: {avg_time:.1f}s")
    print(f"{'='*60}")

    # Save
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    output_data = {
        "experiment": "warmup_eval",
        "model": args.model,
        "checkpoint": args.checkpoint,
        "checkpoint_step": ckpt_step,
        "lora_rank": args.lora_rank,
        "episodes": args.episodes,
        "success_rate": success_rate,
        "avg_reward": avg_reward,
        "avg_steps": avg_steps,
        "avg_time_per_ep": avg_time,
        "results": results,
    }
    with open(args.output, "w") as f:
        json.dump(output_data, f, indent=2)
    print(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
