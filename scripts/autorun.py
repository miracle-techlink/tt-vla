#!/usr/bin/env python3
"""
TT-VLA Experiment Automation Skill — 自动跑实验 + 记录 + 可视化

Usage:
  python scripts/autorun.py warmup --steps 5000 --lr 2e-5        # 跑warm-up训练
  python scripts/autorun.py eval --checkpoint warmup_ckpt/step_3000.pt  # 跑评估
  python scripts/autorun.py ttvla --checkpoint warmup_ckpt/step_3000.pt  # 跑TT-VLA
  python scripts/autorun.py status                                # 查看当前任务状态
  python scripts/autorun.py queue                                 # 查看任务队列
  python scripts/autorun.py add warmup --steps 5000               # 添加到队列
  python scripts/autorun.py daemon                                # 守护进程：自动填充GPU
"""

import argparse
import json
import os
import re
import signal
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
HISTORY_FILE = EXPERIMENTS_DIR / "history.jsonl"
QUEUE_FILE = EXPERIMENTS_DIR / "task_queue.json"
STATUS_FILE = EXPERIMENTS_DIR / "current_status.json"
VIZ_DIR = EXPERIMENTS_DIR / "viz"

LOG_DIR = Path("/tmp")


# ── GPU Utilities ─────────────────────────────────────────────────────

def get_gpu_status():
    """Get GPU utilization and memory."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True, timeout=5)
        gpus = []
        for line in out.strip().split("\n"):
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 4:
                gpus.append({
                    "id": int(parts[0]), "mem_used": int(parts[1]),
                    "mem_total": int(parts[2]), "util": int(parts[3])
                })
        return gpus
    except Exception:
        return []


def get_idle_gpus(threshold=10):
    """Get list of idle GPU IDs."""
    gpus = get_gpu_status()
    return [g["id"] for g in gpus if g["util"] < threshold and g["mem_used"] < 1000]


def get_running_tasks():
    """Detect what's currently running."""
    tasks = []
    try:
        out = subprocess.check_output(["ps", "aux"], text=True, timeout=5)
        for line in out.split("\n"):
            if "warmup_finetune" in line and "python" in line and "grep" not in line:
                tasks.append({"type": "warmup", "pid": line.split()[1], "cmd": line})
            if "eval_warmup" in line and "python" in line and "grep" not in line:
                tasks.append({"type": "eval", "pid": line.split()[1], "cmd": line})
            if "train.py" in line and "acot" in line.lower() and "grep" not in line:
                tasks.append({"type": "acot", "pid": line.split()[1], "cmd": line})
            if "run_ttvla" in line and "python" in line and "grep" not in line:
                tasks.append({"type": "ttvla", "pid": line.split()[1], "cmd": line})
    except Exception:
        pass
    return tasks


# ── Task Queue ────────────────────────────────────────────────────────

def load_queue():
    if QUEUE_FILE.exists():
        return json.loads(QUEUE_FILE.read_text())
    return []


def save_queue(queue):
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    QUEUE_FILE.write_text(json.dumps(queue, indent=2, ensure_ascii=False))


def add_to_queue(task_type, config):
    queue = load_queue()
    task = {
        "id": f"{task_type}_{int(time.time())}",
        "type": task_type,
        "config": config,
        "status": "queued",
        "created": datetime.now().isoformat(),
    }
    queue.append(task)
    save_queue(queue)
    return task


# ── Experiment Runners ────────────────────────────────────────────────

def run_warmup(config):
    """Run warm-up fine-tuning experiment."""
    steps = config.get("steps", 3000)
    lr = config.get("lr", 2e-5)
    batch_size = config.get("batch_size", 2)
    num_demos = config.get("num_demos", 200)
    lora_rank = config.get("lora_rank", 32)
    save_every = config.get("save_every", 1000)
    resume = config.get("resume", None)

    idle = get_idle_gpus()
    n_gpus = min(len(idle), config.get("max_gpus", 8))

    if n_gpus == 0:
        print("ERROR: No idle GPUs available")
        return None

    log_file = LOG_DIR / "warmup_finetune.log"
    exp_id = f"warmup_{int(time.time())}"

    if n_gpus > 1:
        gpu_str = ",".join(str(g) for g in idle[:n_gpus])
        cmd = (
            f"CUDA_VISIBLE_DEVICES={gpu_str} "
            f"HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 "
            f"torchrun --nproc_per_node={n_gpus} "
            f"scripts/warmup_finetune.py "
            f"--steps {steps} --lr {lr} --batch-size {batch_size} "
            f"--num-demos {num_demos} --lora-rank {lora_rank} "
            f"--save-every {save_every}"
        )
    else:
        cmd = (
            f"CUDA_VISIBLE_DEVICES={idle[0]} "
            f"HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 "
            f"python scripts/warmup_finetune.py "
            f"--steps {steps} --lr {lr} --batch-size {batch_size} "
            f"--num-demos {num_demos} --lora-rank {lora_rank} "
            f"--save-every {save_every}"
        )

    if resume:
        cmd += f" --resume {resume}"

    print(f"[{exp_id}] Launching warm-up training...")
    print(f"  GPUs: {n_gpus} ({','.join(str(g) for g in idle[:n_gpus])})")
    print(f"  Config: steps={steps}, lr={lr}, bs={batch_size}, demos={num_demos}")
    print(f"  Log: {log_file}")

    # Save status
    update_status(exp_id, "warmup", "running", config)

    # Launch
    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            cmd, shell=True, cwd=str(PROJECT_ROOT),
            stdout=lf, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid
        )

    print(f"  PID: {proc.pid}")
    return {"exp_id": exp_id, "pid": proc.pid, "log": str(log_file)}


def run_eval(config):
    """Run evaluation experiment."""
    checkpoint = config.get("checkpoint", "warmup_ckpt/step_3000.pt")
    episodes = config.get("episodes", 20)
    max_steps = config.get("max_steps", 160)

    idle = get_idle_gpus()
    n_gpus = min(len(idle), 2)

    if n_gpus == 0:
        print("ERROR: No idle GPUs available")
        return None

    gpu_str = ",".join(str(g) for g in idle[:n_gpus])
    log_file = LOG_DIR / "eval_warmup.log"
    exp_id = f"eval_{int(time.time())}"

    # Determine output filename
    ckpt_name = Path(checkpoint).stem
    output = f"results/eval_{ckpt_name}_{exp_id}.json"

    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu_str} "
        f"HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 "
        f"python scripts/eval_warmup.py "
        f"--checkpoint {checkpoint} --episodes {episodes} "
        f"--max-steps {max_steps} --output {output}"
    )

    print(f"[{exp_id}] Launching evaluation...")
    print(f"  GPUs: {gpu_str}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Episodes: {episodes}")
    print(f"  Output: {output}")

    update_status(exp_id, "eval", "running", config)

    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            cmd, shell=True, cwd=str(PROJECT_ROOT),
            stdout=lf, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid
        )

    print(f"  PID: {proc.pid}")
    return {"exp_id": exp_id, "pid": proc.pid, "log": str(log_file), "output": output}


def run_ttvla(config):
    """Run TT-VLA test-time adaptation."""
    checkpoint = config.get("checkpoint", "warmup_ckpt/step_3000.pt")
    episodes = config.get("episodes", 20)
    lr = config.get("lr", 1e-4)
    update_k = config.get("update_interval", 8)
    clip_eps = config.get("clip_epsilon", 0.2)

    idle = get_idle_gpus()
    n_gpus = min(len(idle), 2)

    if n_gpus == 0:
        print("ERROR: No idle GPUs available")
        return None

    gpu_str = ",".join(str(g) for g in idle[:n_gpus])
    log_file = LOG_DIR / "ttvla_run.log"
    exp_id = f"ttvla_{int(time.time())}"

    cmd = (
        f"CUDA_VISIBLE_DEVICES={gpu_str} "
        f"HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 "
        f"python scripts/run_ttvla.py "
        f"--episodes {episodes} --resume {checkpoint} "
        f"--rank 32 --lr {lr}"
    )

    print(f"[{exp_id}] Launching TT-VLA...")
    print(f"  GPUs: {gpu_str}")
    print(f"  Checkpoint: {checkpoint}")
    print(f"  Episodes: {episodes}, lr={lr}, K={update_k}")

    update_status(exp_id, "ttvla", "running", config)

    with open(log_file, "w") as lf:
        proc = subprocess.Popen(
            cmd, shell=True, cwd=str(PROJECT_ROOT),
            stdout=lf, stderr=subprocess.STDOUT,
            preexec_fn=os.setsid
        )

    print(f"  PID: {proc.pid}")
    return {"exp_id": exp_id, "pid": proc.pid, "log": str(log_file)}


# ── Status Management ─────────────────────────────────────────────────

def update_status(exp_id, task_type, status, config=None, result=None):
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "exp_id": exp_id,
        "type": task_type,
        "status": status,
        "config": config,
        "result": result,
        "updated": datetime.now().isoformat(),
    }
    STATUS_FILE.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def show_status():
    """Show comprehensive status."""
    print(f"\n{'='*60}")
    print(f"  TT-VLA Experiment Status")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # GPU status
    gpus = get_gpu_status()
    if gpus:
        print("GPUs:")
        for g in gpus:
            bar_len = int(g["util"] / 5)
            bar = "█" * bar_len + "░" * (20 - bar_len)
            status = "BUSY" if g["util"] > 50 else "IDLE" if g["util"] < 10 else "LOW"
            print(f"  GPU{g['id']}: [{bar}] {g['util']:>3}% | {g['mem_used']:>5}/{g['mem_total']}MB | {status}")
        print()

    # Running tasks
    tasks = get_running_tasks()
    if tasks:
        print("Running Tasks:")
        for t in tasks:
            print(f"  [{t['type']:>7}] PID={t['pid']}")
    else:
        print("Running Tasks: None")
    print()

    # Check training logs
    for name, logpath in [
        ("Warm-up", LOG_DIR / "warmup_finetune.log"),
        ("ACoT", LOG_DIR / "acot_train.log"),
        ("Eval", LOG_DIR / "eval_warmup.log"),
        ("TT-VLA", LOG_DIR / "ttvla_run.log"),
    ]:
        if logpath.exists():
            lines = logpath.read_text().splitlines()
            if lines:
                # Get last meaningful line
                last = ""
                for line in reversed(lines):
                    if line.strip() and not line.strip().startswith("Warning") and "UserWarning" not in line:
                        last = line.strip()
                        break
                if last:
                    print(f"  [{name:>7}] {last[:80]}")
    print()

    # Queue
    queue = load_queue()
    queued = [t for t in queue if t["status"] == "queued"]
    if queued:
        print(f"Queue ({len(queued)} pending):")
        for t in queued:
            print(f"  [{t['type']:>7}] {json.dumps(t['config'])}")
    print()

    # Experiment history
    if HISTORY_FILE.exists():
        entries = [json.loads(l) for l in HISTORY_FILE.read_text().strip().split("\n") if l.strip()]
        print(f"Experiment History: {len(entries)} total")
        for e in entries[-3:]:
            r = e.get("result", {})
            c = e.get("config", {})
            method = c.get("method", "?")
            sr = r.get("success_rate", "?")
            rw = r.get("avg_reward", "?")
            if isinstance(sr, (int, float)):
                sr = f"{sr:.1f}%"
            if isinstance(rw, (int, float)):
                rw = f"{rw:.2f}"
            print(f"  {e.get('timestamp', '?')[:16]} | {method:>12} | SR={sr:>6} | R={rw:>8}")
    print()


# ── Auto-log after experiment ─────────────────────────────────────────

def auto_log_eval(output_file, config):
    """Automatically log evaluation results to experiment history."""
    output_path = PROJECT_ROOT / output_file
    if not output_path.exists():
        print(f"Output file not found: {output_path}")
        return

    data = json.loads(output_path.read_text())
    result = {
        "success_rate": data.get("success_rate", 0),
        "avg_reward": data.get("avg_reward", 0),
        "episodes": data.get("episodes", 0),
        "avg_steps": data.get("avg_steps", 0),
    }

    entry = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "result": result,
        "notes": f"Auto-logged from {output_file}",
    }

    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)
    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Auto-logged: SR={result['success_rate']:.1f}%, reward={result['avg_reward']:.2f}")
    return entry


# ── Visualization ─────────────────────────────────────────────────────

def generate_comparison_chart():
    """Generate experiment comparison chart."""
    if not HISTORY_FILE.exists():
        print("No experiments to visualize")
        return

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt

    entries = [json.loads(l) for l in HISTORY_FILE.read_text().strip().split("\n") if l.strip()]
    if len(entries) < 1:
        print("Not enough experiments to compare")
        return

    VIZ_DIR.mkdir(parents=True, exist_ok=True)

    # Bar chart: SR and reward across experiments
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("Experiment Comparison", fontsize=14, fontweight='bold')

    labels = []
    srs = []
    rewards = []
    for i, e in enumerate(entries):
        c = e.get("config", {})
        r = e.get("result", {})
        label = c.get("method", f"exp_{i+1}")
        if len(label) > 15:
            label = label[:15]
        labels.append(label)
        srs.append(r.get("success_rate", 0))
        rewards.append(r.get("avg_reward", 0))

    x = np.arange(len(labels))

    ax1.bar(x, srs, color='#4ecca3', edgecolor='white')
    ax1.set_ylabel('Success Rate (%)')
    ax1.set_title('Success Rate')
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)
    ax1.set_ylim(0, max(max(srs) * 1.2, 10))

    ax2.bar(x, rewards, color='#e94560', edgecolor='white')
    ax2.set_ylabel('Avg Reward')
    ax2.set_title('Average Reward')
    ax2.set_xticks(x)
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)

    plt.tight_layout()
    out = VIZ_DIR / "experiment_comparison.png"
    plt.savefig(out, dpi=150)
    plt.close()
    print(f"Chart saved to {out}")


# ── Daemon Mode ───────────────────────────────────────────────────────

def daemon_loop(check_interval=60):
    """Daemon: monitor GPU and auto-run queued tasks."""
    print(f"Autorun daemon started (check every {check_interval}s)")
    print(f"Queue file: {QUEUE_FILE}")
    print("Press Ctrl+C to stop\n")

    while True:
        try:
            queue = load_queue()
            pending = [t for t in queue if t["status"] == "queued"]

            if not pending:
                time.sleep(check_interval)
                continue

            idle = get_idle_gpus()
            running = get_running_tasks()

            if len(idle) < 2:
                time.sleep(check_interval)
                continue

            # Pick next task
            task = pending[0]
            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Auto-starting: {task['type']} (idle GPUs: {len(idle)})")

            result = None
            if task["type"] == "warmup":
                result = run_warmup(task["config"])
            elif task["type"] == "eval":
                result = run_eval(task["config"])
            elif task["type"] == "ttvla":
                result = run_ttvla(task["config"])

            if result:
                task["status"] = "running"
                task["started"] = datetime.now().isoformat()
                task["pid"] = result.get("pid")
                save_queue(queue)
            else:
                task["status"] = "failed"
                task["error"] = "Launch failed"
                save_queue(queue)

            time.sleep(check_interval)

        except KeyboardInterrupt:
            print("\nDaemon stopped.")
            break
        except Exception as e:
            print(f"Daemon error: {e}")
            time.sleep(check_interval)


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TT-VLA Experiment Automation")
    sub = parser.add_subparsers(dest="command")

    # warmup
    wp = sub.add_parser("warmup", help="Run warm-up fine-tuning")
    wp.add_argument("--steps", type=int, default=3000)
    wp.add_argument("--lr", type=float, default=2e-5)
    wp.add_argument("--batch-size", type=int, default=2)
    wp.add_argument("--num-demos", type=int, default=200)
    wp.add_argument("--lora-rank", type=int, default=32)
    wp.add_argument("--save-every", type=int, default=1000)
    wp.add_argument("--resume", default=None)
    wp.add_argument("--max-gpus", type=int, default=8)

    # eval
    ev = sub.add_parser("eval", help="Run evaluation")
    ev.add_argument("--checkpoint", default="warmup_ckpt/step_3000.pt")
    ev.add_argument("--episodes", type=int, default=20)
    ev.add_argument("--max-steps", type=int, default=160)

    # ttvla
    tt = sub.add_parser("ttvla", help="Run TT-VLA test-time adaptation")
    tt.add_argument("--checkpoint", default="warmup_ckpt/step_3000.pt")
    tt.add_argument("--episodes", type=int, default=20)
    tt.add_argument("--lr", type=float, default=1e-4)

    # status
    sub.add_parser("status", help="Show current status")

    # queue management
    aq = sub.add_parser("add", help="Add task to queue")
    aq.add_argument("task_type", choices=["warmup", "eval", "ttvla"])
    aq.add_argument("--config", default="{}", help="JSON config")

    sub.add_parser("queue", help="Show task queue")

    # viz
    sub.add_parser("viz", help="Generate comparison charts")

    # daemon
    dm = sub.add_parser("daemon", help="Auto-run daemon")
    dm.add_argument("--interval", type=int, default=60)

    args = parser.parse_args()

    if args.command == "warmup":
        config = {
            "method": "warmup", "steps": args.steps, "lr": args.lr,
            "batch_size": args.batch_size, "num_demos": args.num_demos,
            "lora_rank": args.lora_rank,
        }
        run_warmup(config)

    elif args.command == "eval":
        config = {
            "method": "eval", "checkpoint": args.checkpoint,
            "episodes": args.episodes, "max_steps": args.max_steps,
        }
        run_eval(config)

    elif args.command == "ttvla":
        config = {
            "method": "ttvla", "checkpoint": args.checkpoint,
            "episodes": args.episodes, "lr": args.lr,
        }
        run_ttvla(config)

    elif args.command == "status":
        show_status()

    elif args.command == "add":
        config = json.loads(args.config)
        config["type"] = args.task_type
        task = add_to_queue(args.task_type, config)
        print(f"Added to queue: {task['id']}")

    elif args.command == "queue":
        queue = load_queue()
        if not queue:
            print("Queue is empty")
        else:
            for t in queue:
                print(f"  [{t['status']:>8}] {t['type']} | {json.dumps(t['config'])}")

    elif args.command == "viz":
        generate_comparison_chart()

    elif args.command == "daemon":
        daemon_loop(args.interval)

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
