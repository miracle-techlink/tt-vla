#!/usr/bin/env python3
"""
TT-VLA Code Review Skill — 论文一致性 + 代码质量 + 数据可信度

Usage:
  python scripts/review.py paper          # 论文一致性检查
  python scripts/review.py code           # 代码质量审查
  python scripts/review.py data           # 实验数据可信度校验
  python scripts/review.py all            # 全部检查
"""

import argparse
import ast
import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent

# ── Paper ground truth (TT-VLA, Zhai et al. 2025) ────────────────────

PAPER_HYPERPARAMS = {
    "lora_rank": {"value": 32, "section": "4.2", "note": "LoRA rank r=32"},
    "lora_alpha": {"value": 32, "section": "4.2", "note": "alpha=r"},
    "lr_ttvla": {"value": 1e-4, "section": "4.2", "note": "TT-VLA PPO learning rate"},
    "lr_warmup": {"value": 2e-5, "section": "4.2", "note": "Warm-up fine-tuning LR"},
    "clip_epsilon": {"value": 0.2, "section": "4.2", "note": "PPO clip parameter"},
    "update_interval_K": {"value": 8, "section": "4.2", "note": "PPO update every K steps"},
    "max_horizon_T": {"value": 160, "section": "4.2", "note": "Max episode length"},
    "gamma": {"value": 0, "section": "3.2", "note": "GAE gamma=0 (one-step)"},
    "lam": {"value": 0, "section": "3.2", "note": "GAE lambda=0 (one-step)"},
    "warmup_steps": {"value": 10000, "section": "4.1", "note": "Warm-up fine-tuning steps"},
    "action_dim": {"value": 7, "section": "4.1", "note": "EE delta pose (6 pose + 1 gripper)"},
    "n_action_bins": {"value": 256, "section": "OpenVLA", "note": "Action discretization bins"},
}

PAPER_FORMULAS = {
    "ppo_loss": {
        "description": "Value-free PPO: L = -E[min(r(θ)·A, clip(r(θ),1-ε,1+ε)·A)]",
        "section": "Eq. 1 / Section 3.2",
        "key_properties": [
            "No value function (value-free)",
            "One-step advantage: A_t = r_t = p_t - p_{t-1} (progress-based)",
            "γ=0, λ=0 eliminates GAE computation",
            "Importance ratio r(θ) = π_new(a|s) / π_old(a|s)",
        ],
    },
    "action_encoding": {
        "description": "OpenVLA action tokens: token_id = vocab_size - 1 - bin_idx",
        "section": "OpenVLA (Kim et al. 2024)",
        "key_properties": [
            "256 bins over [-1, 1]",
            "bin_centers = (bins[:-1] + bins[1:]) / 2",
            "vocab_size = text_vocab_size - pad_to_multiple = 32064 - 64 = 32000",
            "Token range: [31745, 31999]",
        ],
    },
    "reward": {
        "description": "Dense progress-based reward: r_t = p_t - p_{t-1}",
        "section": "Section 3.1",
        "key_properties": [
            "Progress estimator p(s) ∈ [0, 1]",
            "Oracle: grasped→0.4, placed→0.85, success→1.0",
            "Alternative: learned VLAC proxy",
            "Reward is progress DELTA, not absolute progress",
        ],
    },
    "lora_reset": {
        "description": "LoRA weights reset per episode",
        "section": "Section 3.3",
        "key_properties": [
            "Reset LoRA to warm-up checkpoint at start of each episode",
            "Only LoRA params updated during test-time (frozen base model)",
            "Target modules: q/k/v/o_proj + gate/up/down_proj (all linear layers in LLM)",
        ],
    },
}


# ── 1. Paper Consistency Check ────────────────────────────────────────

class PaperChecker:
    def __init__(self):
        self.results = []

    def check_all(self):
        self._check_config_yaml()
        self._check_tt_vla_implementation()
        self._check_action_encoding()
        self._check_reward_implementation()
        self._check_lora_config()
        return self.results

    def _add(self, status, item, detail, file=None, line=None):
        self.results.append({
            "status": status,  # "pass", "fail", "warn", "info"
            "item": item,
            "detail": detail,
            "file": str(file) if file else None,
            "line": line,
        })

    def _read_file(self, relpath):
        p = PROJECT_ROOT / relpath
        if p.exists():
            return p.read_text(), p
        return None, None

    def _check_config_yaml(self):
        """Check configs/default.yaml against paper hyperparams."""
        text, path = self._read_file("configs/default.yaml")
        if not text:
            self._add("warn", "config", "configs/default.yaml not found")
            return

        import yaml
        try:
            cfg = yaml.safe_load(text)
        except Exception:
            # Fallback: regex parse
            cfg = {}
            for line in text.split("\n"):
                m = re.match(r"\s*(\w+)\s*:\s*(.+)", line)
                if m:
                    key, val = m.group(1), m.group(2).strip()
                    try:
                        cfg[key] = eval(val)
                    except Exception:
                        cfg[key] = val

        # Flatten nested config
        flat = {}
        def flatten(d, prefix=""):
            for k, v in d.items():
                key = f"{prefix}{k}" if not prefix else f"{prefix}.{k}"
                if isinstance(v, dict):
                    flatten(v, key)
                else:
                    flat[key] = v
        if isinstance(cfg, dict):
            flatten(cfg)

        for param, expected in PAPER_HYPERPARAMS.items():
            # Search in flat config
            found = False
            for k, v in flat.items():
                if param in k.lower().replace("-", "_").replace(".", "_"):
                    found = True
                    if isinstance(v, (int, float)) and isinstance(expected["value"], (int, float)):
                        if abs(v - expected["value"]) < 1e-10 or v == expected["value"]:
                            self._add("pass", f"config.{param}", f"{v} == {expected['value']} (Section {expected['section']})", path)
                        else:
                            self._add("fail", f"config.{param}", f"Config has {v}, paper says {expected['value']} (Section {expected['section']})", path)
                    break
            if not found:
                self._add("info", f"config.{param}", f"Not found in config (paper: {expected['value']}, Section {expected['section']})")

    def _check_tt_vla_implementation(self):
        """Check tt_vla.py for correct PPO implementation."""
        text, path = self._read_file("src/tt_vla.py")
        if not text:
            self._add("warn", "tt_vla", "src/tt_vla.py not found")
            return

        # Check: value-free (no value function/critic)
        if "value_function" in text.lower() or "critic" in text.lower():
            # Check if it's actually used or just commented
            lines = text.split("\n")
            for i, line in enumerate(lines):
                if ("value_function" in line.lower() or "critic" in line.lower()) and not line.strip().startswith("#"):
                    self._add("fail", "ppo.value_free", f"Found value function/critic reference — paper uses VALUE-FREE PPO", path, i+1)
                    break
        else:
            self._add("pass", "ppo.value_free", "No value function/critic found (correct: value-free PPO)")

        # Check: clip epsilon
        clip_matches = re.findall(r"clip.*?(\d+\.\d+)", text)
        if clip_matches:
            for val in clip_matches:
                if abs(float(val) - 0.2) < 0.01:
                    self._add("pass", "ppo.clip_eps", f"Clip epsilon = {val} matches paper (0.2)")
                    break
            else:
                self._add("warn", "ppo.clip_eps", f"Clip values found: {clip_matches}, expected 0.2")

        # Check: gamma=0, lambda=0
        if "gamma" in text:
            gamma_vals = re.findall(r"gamma\s*[=:]\s*([\d.]+)", text)
            for g in gamma_vals:
                if float(g) == 0:
                    self._add("pass", "ppo.gamma", "gamma=0 (one-step advantage)")
                    break
            else:
                if gamma_vals:
                    self._add("fail", "ppo.gamma", f"gamma={gamma_vals}, paper requires gamma=0 for one-step advantage")

        # Check: importance ratio
        if "ratio" in text.lower() or "importance" in text.lower() or "old_log_prob" in text.lower():
            self._add("pass", "ppo.importance_ratio", "Importance ratio computation found")
        else:
            self._add("warn", "ppo.importance_ratio", "No clear importance ratio computation found — PPO requires π_new/π_old")

    def _check_action_encoding(self):
        """Check action token encoding matches OpenVLA."""
        for relpath in ["src/vla_wrapper.py", "scripts/warmup_finetune.py", "scripts/eval_warmup.py"]:
            text, path = self._read_file(relpath)
            if not text:
                continue

            # Check vocab_size = 32000
            if "32000" in text:
                self._add("pass", f"action.vocab_size ({relpath})", "vocab_size=32000 found")
            elif "32064" in text:
                self._add("warn", f"action.vocab_size ({relpath})", "Found 32064 — should use 32000 (= 32064 - 64 pad)")

            # Check encoding direction: token = vocab_size - 1 - bin_idx
            if "vocab_size - 1 - bin_idx" in text or "VOCAB_SIZE - 1 -" in text or "32000 - 1 -" in text:
                self._add("pass", f"action.encoding ({relpath})", "Correct reverse encoding: token = vocab_size - 1 - bin_idx")
            elif re.search(r"\d+ - 1 - bin", text):
                self._add("pass", f"action.encoding ({relpath})", "Reverse encoding pattern found")

            # Check 256 bins
            if "256" in text and ("linspace" in text or "n_bins" in text or "n_action_bins" in text):
                self._add("pass", f"action.bins ({relpath})", "256 action bins found")

    def _check_reward_implementation(self):
        """Check reward/progress estimator."""
        text, path = self._read_file("src/reward.py")
        if not text:
            self._add("warn", "reward", "src/reward.py not found")
            return

        # Check progress-based (delta) reward
        if "delta" in text.lower() or "p_t - p_" in text or "progress" in text.lower():
            self._add("pass", "reward.progress", "Progress-based reward found")
        else:
            self._add("warn", "reward.progress", "No clear progress-delta reward — paper uses r_t = p_t - p_{t-1}")

        # Check oracle milestones
        if "0.4" in text or "grasped" in text.lower():
            self._add("pass", "reward.oracle", "Grasp milestone found")
        if "0.85" in text or "placed" in text.lower():
            self._add("pass", "reward.oracle", "Place milestone found")

    def _check_lora_config(self):
        """Check LoRA configuration."""
        for relpath in ["src/lora_utils.py", "scripts/warmup_finetune.py"]:
            text, path = self._read_file(relpath)
            if not text:
                continue

            # Check target modules
            expected_modules = {"q_proj", "v_proj", "k_proj", "o_proj", "gate_proj", "up_proj", "down_proj"}
            found_modules = set(re.findall(r'"(\w+_proj)"', text))
            if found_modules:
                missing = expected_modules - found_modules
                extra = found_modules - expected_modules
                if not missing:
                    self._add("pass", f"lora.modules ({relpath})", f"All target modules present: {sorted(found_modules)}")
                else:
                    self._add("fail", f"lora.modules ({relpath})", f"Missing: {missing}, Extra: {extra}")

            # Check rank
            rank_matches = re.findall(r"r\s*=\s*(\d+)", text)
            if "32" in rank_matches:
                self._add("pass", f"lora.rank ({relpath})", "rank=32 matches paper")


# ── 2. Code Quality Check ────────────────────────────────────────────

class CodeChecker:
    def __init__(self):
        self.results = []

    def check_all(self):
        py_files = list(PROJECT_ROOT.rglob("*.py"))
        py_files = [f for f in py_files if "__pycache__" not in str(f) and ".egg" not in str(f)]

        for f in py_files:
            self._check_file(f)
        return self.results

    def _add(self, severity, item, detail, file=None, line=None):
        self.results.append({
            "severity": severity,
            "item": item,
            "detail": detail,
            "file": str(file) if file else None,
            "line": line,
        })

    def _check_file(self, filepath):
        try:
            text = filepath.read_text()
        except Exception:
            return

        lines = text.split("\n")
        relpath = filepath.relative_to(PROJECT_ROOT)

        for i, line in enumerate(lines, 1):
            stripped = line.strip()
            if stripped.startswith("#"):
                continue

            # Rule: magic numbers in tensor operations
            if re.search(r"\.reshape\(.*\d{3,}.*\)|\.view\(.*\d{3,}.*\)", stripped):
                if not re.search(r"#.*\b(vocab|dim|hidden|embed)", stripped):
                    self._add("warn", "magic_number", f"Large magic number in reshape/view", relpath, i)

            # Rule: device mismatch risk — .to(device) after creation
            if ".cpu()" in stripped and ".cuda()" in stripped:
                self._add("warn", "device_mismatch", "Both .cpu() and .cuda() on same line", relpath, i)

            # Rule: detach/no_grad misuse
            if "loss.backward()" in stripped:
                # Check if there's a no_grad block nearby that shouldn't be there
                context = "\n".join(lines[max(0,i-5):i])
                if "torch.no_grad" in context and "with" in context:
                    self._add("fail", "grad_blocked", "loss.backward() inside torch.no_grad() — gradients will be zero!", relpath, i)

            # Rule: silent broadcasting
            if re.search(r"[\+\-\*\/]\s*\w+", stripped) and "shape" not in stripped:
                # Heuristic: tensor ops without shape comments in complex expressions
                pass  # Too noisy, skip

            # Rule: hardcoded paths
            if re.search(r'["\']/home/\w+/', stripped) and "HF_HOME" not in stripped:
                if not stripped.startswith("#"):
                    self._add("info", "hardcoded_path", f"Hardcoded absolute path", relpath, i)

            # Rule: missing weights_only in torch.load
            if "torch.load(" in stripped and "weights_only" not in stripped:
                self._add("warn", "torch_load", "torch.load() without weights_only param (PyTorch 2.6+ default changed)", relpath, i)

            # Rule: bare except
            if re.match(r"\s*except\s*:", stripped):
                self._add("warn", "bare_except", "Bare except clause — may hide bugs", relpath, i)

            # Rule: TODO/FIXME/HACK
            if re.search(r"#\s*(TODO|FIXME|HACK|XXX|BROKEN)", stripped, re.IGNORECASE):
                self._add("info", "todo", stripped.strip(), relpath, i)

        # Rule: file-level — no __all__ in module files
        # (skip, too noisy)

        # Rule: unused imports (basic check)
        try:
            tree = ast.parse(text)
            imports = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        name = alias.asname or alias.name.split(".")[0]
                        imports.append((name, node.lineno))
                elif isinstance(node, ast.ImportFrom):
                    for alias in node.names:
                        name = alias.asname or alias.name
                        imports.append((name, node.lineno))

            for name, lineno in imports:
                if name == "*" or name.startswith("_"):
                    continue
                # Simple: check if name appears elsewhere in file
                occurrences = len(re.findall(r'\b' + re.escape(name) + r'\b', text))
                if occurrences <= 1:
                    self._add("info", "unused_import", f"Possibly unused import: {name}", relpath, lineno)
        except SyntaxError:
            pass


# ── 3. Data Validity Check ───────────────────────────────────────────

class DataChecker:
    def __init__(self):
        self.results = []

    def check_all(self):
        self._check_eval_results()
        self._check_loss_curve()
        self._check_experiment_history()
        return self.results

    def _add(self, severity, item, detail):
        self.results.append({"severity": severity, "item": item, "detail": detail})

    def _check_eval_results(self):
        """Validate evaluation result files."""
        results_dir = PROJECT_ROOT / "results"
        if not results_dir.exists():
            self._add("warn", "results_dir", "results/ directory not found")
            return

        for f in results_dir.glob("*.json"):
            try:
                data = json.loads(f.read_text())
            except json.JSONDecodeError:
                self._add("fail", f"json_parse ({f.name})", "Invalid JSON")
                continue

            results = data.get("results", [])
            if not results:
                self._add("warn", f"empty ({f.name})", "No episode results")
                continue

            rewards = [r.get("total_reward", 0) for r in results]
            successes = [r.get("success", False) for r in results]
            steps_list = [r.get("steps", 0) for r in results]

            # Rule: all rewards exactly zero
            if all(r == 0 for r in rewards):
                self._add("fail", f"zero_rewards ({f.name})", "ALL rewards are exactly 0.0 — likely a bug (broken action encoding, wrong control mode, or env issue)")

            # Rule: all rewards identical (policy collapsed)
            if len(set(f"{r:.6f}" for r in rewards)) == 1 and len(rewards) > 3:
                self._add("fail", f"collapsed ({f.name})", f"All {len(rewards)} episodes have identical reward={rewards[0]:.4f} — policy may have collapsed")

            # Rule: negative rewards
            neg = [r for r in rewards if r < 0]
            if neg:
                self._add("warn", f"negative_rewards ({f.name})", f"{len(neg)}/{len(rewards)} episodes have negative reward (min={min(neg):.4f})")

            # Rule: success rate vs reward consistency
            sr = sum(successes) / len(successes)
            avg_r = np.mean(rewards)
            if sr > 0.5 and avg_r < 10:
                self._add("warn", f"sr_reward_mismatch ({f.name})", f"SR={sr*100:.0f}% but avg_reward={avg_r:.2f} — suspicious")
            if sr == 0 and avg_r > 80:
                self._add("warn", f"high_reward_no_success ({f.name})", f"SR=0% but avg_reward={avg_r:.2f} — success detection may be broken")

            # Rule: reward variance
            if len(rewards) > 5:
                cv = np.std(rewards) / (np.mean(rewards) + 1e-8)
                if cv > 1.5:
                    self._add("warn", f"high_variance ({f.name})", f"Extreme reward variance CV={cv:.2f} — results may not be statistically meaningful")

            # Rule: all same step count
            if len(set(steps_list)) == 1 and steps_list[0] > 100:
                self._add("info", f"max_steps ({f.name})", f"All episodes reached max steps ({steps_list[0]}) — no early termination")

            # Rule: too few episodes
            if len(results) < 10:
                self._add("warn", f"few_episodes ({f.name})", f"Only {len(results)} episodes — consider running at least 20 for statistical significance")

            self._add("pass", f"basic_checks ({f.name})", f"{len(results)} episodes, SR={sr*100:.1f}%, avg_reward={avg_r:.2f}")

    def _check_loss_curve(self):
        """Validate training loss curve."""
        loss_file = PROJECT_ROOT / "warmup_ckpt" / "loss_curve.json"
        if not loss_file.exists():
            self._add("info", "loss_curve", "loss_curve.json not found")
            return

        data = json.loads(loss_file.read_text())
        losses = data.get("losses", [])
        if not losses:
            self._add("warn", "loss_empty", "Loss curve has no entries")
            return

        # Rule: loss never decreases
        if len(losses) > 100:
            first_100 = np.mean(losses[:100])
            last_100 = np.mean(losses[-100:])
            if last_100 >= first_100 * 0.95:
                self._add("fail", "loss_no_decrease", f"Loss didn't decrease: start={first_100:.4f}, end={last_100:.4f} — training may not be working")
            else:
                self._add("pass", "loss_decrease", f"Loss decreased: {first_100:.4f} → {last_100:.4f}")

        # Rule: NaN/Inf in loss
        bad = [i for i, l in enumerate(losses) if not np.isfinite(l)]
        if bad:
            self._add("fail", "loss_nan", f"NaN/Inf found at {len(bad)} positions (first at step {bad[0]})")

        # Rule: loss spikes
        if len(losses) > 50:
            rolling_mean = np.convolve(losses, np.ones(50)/50, mode='valid')
            for i in range(len(rolling_mean) - 1):
                if rolling_mean[i+1] > rolling_mean[i] * 3:
                    self._add("warn", "loss_spike", f"Loss spike detected around step {i+50}")
                    break

    def _check_experiment_history(self):
        """Validate experiment history."""
        hist_file = PROJECT_ROOT / "experiments" / "history.jsonl"
        if not hist_file.exists():
            self._add("info", "history", "No experiment history yet")
            return

        entries = []
        for line in hist_file.read_text().strip().split("\n"):
            if line.strip():
                try:
                    entries.append(json.loads(line))
                except json.JSONDecodeError:
                    self._add("fail", "history_parse", f"Invalid JSON in history: {line[:50]}...")

        # Rule: duplicate configs
        configs = [json.dumps(e.get("config", {}), sort_keys=True) for e in entries]
        seen = {}
        for i, c in enumerate(configs):
            if c in seen:
                self._add("warn", "duplicate_experiment", f"Experiment #{i+1} has same config as #{seen[c]+1}")
            seen[c] = i

        self._add("pass", "history", f"{len(entries)} experiments recorded")


# ── Output ────────────────────────────────────────────────────────────

def print_results(title, results):
    icon = {"pass": " ✓ ", "fail": " ✗ ", "warn": " ! ", "info": " i "}
    color = {"pass": "\033[92m", "fail": "\033[91m", "warn": "\033[93m", "info": "\033[94m"}
    reset = "\033[0m"

    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")

    passes = sum(1 for r in results if r.get("status", r.get("severity")) == "pass")
    fails = sum(1 for r in results if r.get("status", r.get("severity")) == "fail")
    warns = sum(1 for r in results if r.get("status", r.get("severity")) == "warn")

    # Group by status
    for status in ["fail", "warn", "info", "pass"]:
        items = [r for r in results if r.get("status", r.get("severity")) == status]
        if not items:
            continue

        for r in items:
            s = r.get("status", r.get("severity"))
            ic = icon.get(s, " ? ")
            c = color.get(s, "")
            loc = ""
            if r.get("file"):
                loc = f" [{r['file']}"
                if r.get("line"):
                    loc += f":{r['line']}"
                loc += "]"
            print(f"  {c}{ic}{reset} {r.get('item', '?')}: {r.get('detail', '')}{loc}")

    print(f"\n  Summary: {passes} pass, {warns} warn, {fails} fail")
    print()

    return fails


# ── CLI ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="TT-VLA Code Review Skill")
    parser.add_argument("mode", choices=["paper", "code", "data", "all"], help="Review mode")
    parser.add_argument("--json", action="store_true", help="Output as JSON")
    args = parser.parse_args()

    total_fails = 0

    if args.mode in ("paper", "all"):
        checker = PaperChecker()
        results = checker.check_all()
        if args.json:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            total_fails += print_results("Paper Consistency Check", results)

    if args.mode in ("code", "all"):
        checker = CodeChecker()
        results = checker.check_all()
        if args.json:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            total_fails += print_results("Code Quality Check", results)

    if args.mode in ("data", "all"):
        checker = DataChecker()
        results = checker.check_all()
        if args.json:
            print(json.dumps(results, indent=2, ensure_ascii=False))
        else:
            total_fails += print_results("Data Validity Check", results)

    if total_fails > 0:
        print(f"  *** {total_fails} FAILURES found — review required ***\n")
        sys.exit(1)
    else:
        print(f"  All checks passed.\n")


if __name__ == "__main__":
    main()
