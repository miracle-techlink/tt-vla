#!/usr/bin/env python3
"""
TT-VLA Algorithm Optimization Skill — 诊断 + 方案 + 记录 闭环

Usage:
  python scripts/optimize.py diagnose          # 诊断当前状态
  python scripts/optimize.py propose           # 基于诊断提出优化方案
  python scripts/optimize.py log --config '{...}' --result '{...}'  # 记录实验
  python scripts/optimize.py history           # 查看实验历史
  python scripts/optimize.py compare           # 对比所有实验结果
"""

import argparse
import json
import os
import re
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np

# ── Paths ─────────────────────────────────────────────────────────────

PROJECT_ROOT = Path(__file__).parent.parent
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"
HISTORY_FILE = EXPERIMENTS_DIR / "history.jsonl"
BEST_CONFIG_FILE = EXPERIMENTS_DIR / "best_configs.yaml"
RESULTS_DIR = PROJECT_ROOT / "results"
WARMUP_CKPT_DIR = PROJECT_ROOT / "warmup_ckpt"
LOG_FILES = {
    "warmup": Path("/tmp/warmup_finetune.log"),
    "acot": Path("/tmp/acot_train.log"),
    "eval": Path("/tmp/eval_warmup.log"),
}


# ── 1. Diagnose ──────────────────────────────────────────────────────

class Diagnostics:
    """Run diagnostic rules on training logs and eval results."""

    def __init__(self):
        self.issues = []
        self.info = []

    def run_all(self):
        self._check_warmup_training()
        self._check_eval_results()
        self._check_acot_training()
        self._check_gpu_status()
        self._check_loss_curve()
        return self.issues, self.info

    def _check_warmup_training(self):
        """Analyze warm-up training log."""
        log = LOG_FILES["warmup"]
        if not log.exists():
            self.info.append(("warmup", "No warm-up training log found"))
            return

        lines = log.read_text().splitlines()
        entries = []
        for line in lines:
            m = re.search(r"Step (\d+)/(\d+) \| Loss: ([\d.]+) \| Speed: ([\d.]+) steps/s", line)
            if m:
                entries.append({
                    "step": int(m.group(1)), "total": int(m.group(2)),
                    "loss": float(m.group(3)), "speed": float(m.group(4)),
                })

        if not entries:
            self.info.append(("warmup", "Log exists but no training entries parsed"))
            return

        latest = entries[-1]
        self.info.append(("warmup", f"Step {latest['step']}/{latest['total']}, loss={latest['loss']:.4f}, speed={latest['speed']:.1f} steps/s"))

        # Rule: 训练停滞
        if len(entries) >= 10:
            recent_losses = [e["loss"] for e in entries[-10:]]
            older_losses = [e["loss"] for e in entries[-20:-10]] if len(entries) >= 20 else recent_losses
            if np.mean(recent_losses) >= np.mean(older_losses) * 0.98:
                self.issues.append({
                    "severity": "warning",
                    "component": "warmup",
                    "issue": "训练停滞",
                    "detail": f"最近10条loss均值 {np.mean(recent_losses):.4f} vs 之前 {np.mean(older_losses):.4f}，无明显下降",
                    "suggestion": "考虑：(1) 降低lr (2) 增加数据量 (3) 触发早停",
                })

            # Rule: loss 震荡
            loss_std = np.std(recent_losses)
            loss_mean = np.mean(recent_losses)
            if loss_std > 0.3 * loss_mean and loss_mean > 0.1:
                self.issues.append({
                    "severity": "warning",
                    "component": "warmup",
                    "issue": "Loss 震荡剧烈",
                    "detail": f"std/mean = {loss_std/loss_mean:.2f}",
                    "suggestion": "lr 可能过大，尝试降低 lr 或增大 batch size",
                })

        # Rule: 速度过慢
        if latest["speed"] < 1.0:
            self.issues.append({
                "severity": "info",
                "component": "warmup",
                "issue": "训练速度偏低",
                "detail": f"{latest['speed']:.1f} steps/s",
                "suggestion": "检查是否用了DDP多卡，增加num_workers，或减小模型加载开销",
            })

    def _check_eval_results(self):
        """Analyze evaluation results."""
        eval_json = RESULTS_DIR / "warmup_eval_results.json"
        if not eval_json.exists():
            self.info.append(("eval", "No evaluation results found"))
            return

        data = json.loads(eval_json.read_text())
        sr = data.get("success_rate", 0)
        avg_reward = data.get("avg_reward", 0)
        episodes = data.get("episodes", 0)
        results = data.get("results", [])

        self.info.append(("eval", f"SR={sr:.1f}%, avg_reward={avg_reward:.2f}, episodes={episodes}"))

        rewards = [r["total_reward"] for r in results]

        # Rule: SR=0 但 reward>0
        if sr == 0 and avg_reward > 10:
            self.issues.append({
                "severity": "info",
                "component": "eval",
                "issue": "部分行为已学会但无法完成任务",
                "detail": f"SR=0% 但 avg_reward={avg_reward:.1f}",
                "suggestion": "模型学到了接近/抓取动作但无法完成全流程。下一步：(1) 增加warm-up步数 (2) 启用TT-VLA在线适配 (3) 增加demo数量",
            })

        # Rule: reward 方差大
        if len(rewards) > 5:
            cv = np.std(rewards) / (np.mean(rewards) + 1e-8)
            if cv > 0.8:
                self.issues.append({
                    "severity": "warning",
                    "component": "eval",
                    "issue": "策略不稳定，reward方差大",
                    "detail": f"CV(reward) = {cv:.2f}, range=[{min(rewards):.1f}, {max(rewards):.1f}]",
                    "suggestion": "增大训练batch size或降低lr以稳定策略",
                })

        # Rule: 所有 episode 都到 max steps
        if results and all(r["steps"] == results[0]["steps"] for r in results):
            self.issues.append({
                "severity": "info",
                "component": "eval",
                "issue": "所有episode均达到最大步数",
                "detail": f"全部 {len(results)} 个 episode 都跑满 {results[0]['steps']} 步",
                "suggestion": "模型未学会终止条件，考虑在训练数据中加入终止信号",
            })

    def _check_acot_training(self):
        """Analyze ACoT training log."""
        log = LOG_FILES["acot"]
        if not log.exists():
            self.info.append(("acot", "No ACoT training log found"))
            return

        lines = log.read_text().splitlines()
        progress_lines = [l for l in lines if "Progress on:" in l]
        if not progress_lines:
            self.info.append(("acot", "Log exists but no progress entries"))
            return

        last = progress_lines[-1]
        m = re.search(r"Progress on: ([\d.]+)kit/([\d.]+)kit rate:([\d.]+)s/it remaining:([\d:]+)", last)
        if m:
            current = float(m.group(1)) * 1000
            total = float(m.group(2)) * 1000
            rate = float(m.group(3))
            remaining = m.group(4)
            self.info.append(("acot", f"Step {current:.0f}/{total:.0f}, rate={rate:.1f}s/step, ETA={remaining}"))

    def _check_gpu_status(self):
        """Check GPU utilization."""
        try:
            import subprocess
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

            idle = [g for g in gpus if g["util"] < 10]
            busy = [g for g in gpus if g["util"] >= 70]
            self.info.append(("gpu", f"{len(gpus)} GPUs: {len(busy)} busy, {len(idle)} idle"))

            if idle and not busy:
                self.issues.append({
                    "severity": "warning",
                    "component": "gpu",
                    "issue": "所有GPU空闲",
                    "detail": f"{len(idle)} 张卡无任务",
                    "suggestion": "可以启动训练任务（ACoT/warm-up/eval）",
                })

            low_util = [g for g in gpus if 10 <= g["util"] < 70]
            if low_util:
                self.issues.append({
                    "severity": "info",
                    "component": "gpu",
                    "issue": f"{len(low_util)} 张卡利用率偏低",
                    "detail": ", ".join(f"GPU{g['id']}={g['util']}%" for g in low_util),
                    "suggestion": "可能存在数据加载瓶颈，检查 num_workers 和 prefetch 设置",
                })
        except Exception:
            self.info.append(("gpu", "nvidia-smi not available"))

    def _check_loss_curve(self):
        """Analyze loss curve from checkpoint dir."""
        loss_file = WARMUP_CKPT_DIR / "loss_curve.json"
        if not loss_file.exists():
            return

        data = json.loads(loss_file.read_text())
        losses = data.get("losses", [])
        if len(losses) < 100:
            return

        # Rule: 过拟合检测 — loss 极低但 eval SR=0
        final_loss = np.mean(losses[-100:])
        eval_json = RESULTS_DIR / "warmup_eval_results.json"
        if eval_json.exists():
            eval_data = json.loads(eval_json.read_text())
            if final_loss < 0.1 and eval_data.get("success_rate", 0) == 0:
                self.issues.append({
                    "severity": "critical",
                    "component": "warmup",
                    "issue": "疑似过拟合",
                    "detail": f"训练loss={final_loss:.4f}极低，但eval SR=0%",
                    "suggestion": "增大lora_dropout，减少训练步数，或增加数据多样性",
                })


# ── 2. Propose ────────────────────────────────────────────────────────

SEARCH_SPACE = {
    "lr": [1e-5, 2e-5, 5e-5],
    "lora_rank": [16, 32, 64],
    "batch_size": [2, 4, 8],
    "warmup_steps": [1000, 3000, 5000, 10000],
    "grad_accum": [1, 2, 4],
    "num_demos": [100, 200, 500],
    "early_stop_loss": [0.10, 0.15, 0.20, 0.30],
    "lora_dropout": [0.0, 0.05, 0.10],
}


def load_history():
    """Load experiment history."""
    if not HISTORY_FILE.exists():
        return []
    entries = []
    for line in HISTORY_FILE.read_text().strip().split("\n"):
        if line.strip():
            entries.append(json.loads(line))
    return entries


def propose_optimizations(issues, history):
    """Generate optimization proposals based on diagnostics and history."""
    proposals = []
    tried_configs = {json.dumps(h.get("config", {}), sort_keys=True) for h in history}

    for issue in issues:
        comp = issue["component"]
        iss = issue["issue"]

        if "停滞" in iss:
            # 尝试降lr
            for lr in [1e-5, 5e-6]:
                cfg = {"change": "lr", "value": lr, "reason": "训练停滞，降低学习率"}
                if json.dumps(cfg, sort_keys=True) not in tried_configs:
                    proposals.append({
                        "priority": "high",
                        "action": f"降低 lr → {lr}",
                        "config_change": cfg,
                        "expected": "loss继续下降",
                        "risk": "收敛更慢",
                        "gpu_cost": "同等",
                    })
                    break

        if "震荡" in iss:
            proposals.append({
                "priority": "high",
                "action": "增大 effective batch size (grad_accum 翻倍)",
                "config_change": {"change": "grad_accum", "value": 8, "reason": "loss震荡"},
                "expected": "梯度更稳定，loss平滑下降",
                "risk": "单步更慢",
                "gpu_cost": "同等",
            })

        if "部分行为已学会" in iss:
            proposals.append({
                "priority": "medium",
                "action": "增加warm-up步数到 5000-10000",
                "config_change": {"change": "warmup_steps", "value": 5000, "reason": "部分学会需要更多训练"},
                "expected": "SR从0%提升到5-15%",
                "risk": "可能过拟合",
                "gpu_cost": "多~30min",
            })
            proposals.append({
                "priority": "high",
                "action": "启用TT-VLA在线PPO适配",
                "config_change": {"change": "enable_ttvla", "value": True, "reason": "warm-up已收敛，需test-time适配"},
                "expected": "SR提升到30-45%（论文目标）",
                "risk": "需要调PPO超参",
                "gpu_cost": "2-4 GPU",
            })

        if "过拟合" in iss:
            proposals.append({
                "priority": "critical",
                "action": "增大 lora_dropout 到 0.10，减少训练步数",
                "config_change": {"change": "lora_dropout", "value": 0.10, "reason": "过拟合"},
                "expected": "eval表现改善",
                "risk": "训练loss上升",
                "gpu_cost": "同等",
            })
            proposals.append({
                "priority": "high",
                "action": "增加demo数量到 500",
                "config_change": {"change": "num_demos", "value": 500, "reason": "数据不足导致过拟合"},
                "expected": "泛化能力提升",
                "risk": "数据收集时间增加",
                "gpu_cost": "同等",
            })

        if "不稳定" in iss:
            proposals.append({
                "priority": "medium",
                "action": "增大 batch_size 或降低 lr",
                "config_change": {"change": "batch_size", "value": 4, "reason": "策略不稳定"},
                "expected": "reward方差降低",
                "risk": "显存增加",
                "gpu_cost": "可能需更多GPU",
            })

    # Default: 如果没有具体建议
    if not proposals:
        proposals.append({
            "priority": "info",
            "action": "当前无明显问题，可以继续当前配置",
            "config_change": {},
            "expected": "按计划完成训练",
            "risk": "无",
            "gpu_cost": "无变化",
        })

    # Sort by priority
    priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3, "info": 4}
    proposals.sort(key=lambda p: priority_order.get(p["priority"], 5))
    return proposals


# ── 3. Log ────────────────────────────────────────────────────────────

def log_experiment(config: dict, result: dict, notes: str = ""):
    """Append an experiment record to history."""
    EXPERIMENTS_DIR.mkdir(parents=True, exist_ok=True)

    entry = {
        "timestamp": datetime.now().isoformat(),
        "config": config,
        "result": result,
        "notes": notes,
    }

    with open(HISTORY_FILE, "a") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    # Update best config if this is the best result
    _update_best_config(entry)
    return entry


def _update_best_config(entry):
    """Update best_configs.yaml if this result is the best so far."""
    history = load_history()
    if not history:
        return

    # Find best by success_rate, then by avg_reward
    best = max(history, key=lambda h: (
        h.get("result", {}).get("success_rate", 0),
        h.get("result", {}).get("avg_reward", 0),
    ))

    best_yaml = (
        f"# Auto-updated: {datetime.now().isoformat()}\n"
        f"# Best experiment result\n"
        f"best_success_rate: {best['result'].get('success_rate', 0)}\n"
        f"best_avg_reward: {best['result'].get('avg_reward', 0)}\n"
        f"config:\n"
    )
    for k, v in best.get("config", {}).items():
        best_yaml += f"  {k}: {v}\n"

    BEST_CONFIG_FILE.write_text(best_yaml)


# ── 4. Compare ────────────────────────────────────────────────────────

def compare_experiments():
    """Compare all experiments in a table."""
    history = load_history()
    if not history:
        print("No experiments recorded yet.")
        return

    print(f"\n{'='*80}")
    print(f"{'#':>3} | {'Date':>10} | {'SR':>6} | {'Reward':>8} | {'Steps':>6} | Config Changes")
    print(f"{'-'*80}")

    for i, h in enumerate(history):
        ts = h.get("timestamp", "?")[:10]
        r = h.get("result", {})
        sr = r.get("success_rate", "?")
        rw = r.get("avg_reward", "?")
        steps = r.get("steps", "?")
        cfg = h.get("config", {})
        cfg_str = ", ".join(f"{k}={v}" for k, v in cfg.items() if k != "reason")
        if isinstance(sr, (int, float)):
            sr = f"{sr:.1f}%"
        if isinstance(rw, (int, float)):
            rw = f"{rw:.2f}"
        print(f"{i+1:>3} | {ts} | {sr:>6} | {rw:>8} | {steps:>6} | {cfg_str}")

    print(f"{'='*80}")
    print(f"Total: {len(history)} experiments\n")


# ── CLI ───────────────────────────────────────────────────────────────

def format_issues(issues, info):
    """Pretty-print diagnostic results."""
    print(f"\n{'='*60}")
    print(f"  TT-VLA Algorithm Diagnostics")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    # Status info
    print("Status:")
    for component, msg in info:
        print(f"  [{component:>7}] {msg}")

    if not issues:
        print(f"\n  No issues detected.\n")
        return

    # Issues by severity
    print(f"\nIssues ({len(issues)}):")
    severity_icon = {"critical": "!!!", "warning": " ! ", "info": " i "}
    for issue in sorted(issues, key=lambda x: {"critical": 0, "warning": 1, "info": 2}.get(x["severity"], 3)):
        icon = severity_icon.get(issue["severity"], " ? ")
        print(f"\n  [{icon}] {issue['issue']} ({issue['component']})")
        print(f"        {issue['detail']}")
        print(f"     -> {issue['suggestion']}")

    print()


def main():
    parser = argparse.ArgumentParser(description="TT-VLA Algorithm Optimization Skill")
    sub = parser.add_subparsers(dest="command")

    sub.add_parser("diagnose", help="Diagnose current training/eval status")
    sub.add_parser("propose", help="Propose optimization actions")

    log_parser = sub.add_parser("log", help="Log an experiment")
    log_parser.add_argument("--config", required=True, help="JSON config dict")
    log_parser.add_argument("--result", required=True, help="JSON result dict")
    log_parser.add_argument("--notes", default="", help="Free-form notes")

    sub.add_parser("history", help="Show experiment history")
    sub.add_parser("compare", help="Compare all experiments")

    args = parser.parse_args()

    if args.command == "diagnose":
        diag = Diagnostics()
        issues, info = diag.run_all()
        format_issues(issues, info)

    elif args.command == "propose":
        diag = Diagnostics()
        issues, info = diag.run_all()
        format_issues(issues, info)

        history = load_history()
        proposals = propose_optimizations(issues, history)

        print(f"{'='*60}")
        print(f"  Optimization Proposals")
        print(f"{'='*60}\n")
        for i, p in enumerate(proposals):
            print(f"  [{p['priority'].upper():>8}] #{i+1}: {p['action']}")
            print(f"           Expected: {p['expected']}")
            print(f"           Risk: {p['risk']} | GPU: {p['gpu_cost']}")
            if p.get("config_change"):
                print(f"           Config: {json.dumps(p['config_change'], ensure_ascii=False)}")
            print()

    elif args.command == "log":
        config = json.loads(args.config)
        result = json.loads(args.result)
        entry = log_experiment(config, result, args.notes)
        print(f"Logged experiment: {json.dumps(entry, indent=2, ensure_ascii=False)}")

    elif args.command in ("history", "compare"):
        compare_experiments()

    else:
        parser.print_help()


if __name__ == "__main__":
    main()
