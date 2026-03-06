"""
Unified CLI for TT-VLA.

Usage:
    python -m ttvla train   [options]    # Run TT-VLA training
    python -m ttvla eval    [options]    # Evaluate checkpoint
    python -m ttvla warmup  [options]    # Warm-up fine-tuning
    python -m ttvla test    [options]    # Quick smoke test (mock VLA)
    python -m ttvla monitor [options]    # Real-time training monitor
"""

import argparse
import sys


def add_common_args(parser):
    parser.add_argument("--config", default="configs/default.yaml", help="Config file path")
    parser.add_argument("--gpu", default="0", help="GPU device(s), e.g. '0' or '0,1'")
    parser.add_argument("--seed", type=int, default=42)


def cmd_train(args):
    """Run TT-VLA test-time RL training."""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from ttvla.config import load_config
    from ttvla.trainer import TTVLATrainer

    config = load_config(args.config, overrides=args.overrides)
    config["experiment"]["device"] = f"cuda:0"

    if args.mock:
        config["vla"]["model_name"] = "mock"

    trainer = TTVLATrainer(config)
    trainer.run(
        num_episodes=args.episodes,
        verbose=True,
    )


def cmd_eval(args):
    """Evaluate a saved checkpoint."""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from ttvla.config import load_config
    from ttvla.trainer import TTVLATrainer

    config = load_config(args.config, overrides=args.overrides)
    trainer = TTVLATrainer(config)
    trainer.evaluate(checkpoint=args.checkpoint, num_episodes=args.episodes)


def cmd_warmup(args):
    """Run warm-up fine-tuning on ManiSkill3."""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

    # Delegate to existing warmup script
    sys.argv = ["warmup_finetune.py",
                "--steps", str(args.steps),
                "--lr", str(args.lr),
                "--model", args.model]
    from scripts.warmup_finetune import main
    main()


def cmd_test(args):
    """Quick smoke test with mock VLA."""
    import os
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    from ttvla.config import load_config
    from ttvla.trainer import TTVLATrainer

    config = load_config(args.config, overrides=args.overrides)
    config["vla"]["model_name"] = "mock"

    trainer = TTVLATrainer(config)
    results = trainer.run(num_episodes=args.episodes, verbose=True)

    # Validate non-zero signals
    avg_progress = results["avg_progress"]
    if avg_progress > 0:
        print(f"\nSMOKE TEST PASSED: avg_progress={avg_progress:.4f} > 0")
    else:
        print(f"\nSMOKE TEST FAILED: avg_progress={avg_progress:.4f} == 0")
        sys.exit(1)


def cmd_monitor(args):
    """Real-time training monitor."""
    from ttvla.monitor import run_monitor
    run_monitor(log_file=args.log_file, refresh=args.refresh)


def main():
    parser = argparse.ArgumentParser(
        prog="ttvla",
        description="TT-VLA: Test-Time RL for Vision-Language-Action Models",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # --- train ---
    p_train = subparsers.add_parser("train", help="Run TT-VLA training")
    add_common_args(p_train)
    p_train.add_argument("--episodes", type=int, default=80)
    p_train.add_argument("--mock", action="store_true", help="Use mock VLA")
    p_train.add_argument("--overrides", nargs="*", default=[], help="Config overrides: key=value")

    # --- eval ---
    p_eval = subparsers.add_parser("eval", help="Evaluate checkpoint")
    add_common_args(p_eval)
    p_eval.add_argument("--checkpoint", required=True, help="Path to LoRA checkpoint")
    p_eval.add_argument("--episodes", type=int, default=20)
    p_eval.add_argument("--overrides", nargs="*", default=[], help="Config overrides")

    # --- warmup ---
    p_warmup = subparsers.add_parser("warmup", help="Warm-up fine-tuning")
    add_common_args(p_warmup)
    p_warmup.add_argument("--steps", type=int, default=10000)
    p_warmup.add_argument("--lr", type=float, default=2e-5)
    p_warmup.add_argument("--model", default="openvla/openvla-7b")

    # --- test ---
    p_test = subparsers.add_parser("test", help="Quick smoke test (mock VLA)")
    add_common_args(p_test)
    p_test.add_argument("--episodes", type=int, default=2)
    p_test.add_argument("--overrides", nargs="*", default=[], help="Config overrides")

    # --- monitor ---
    p_mon = subparsers.add_parser("monitor", help="Real-time training monitor")
    p_mon.add_argument("--log-file", default="logs/ttvla_train.jsonl")
    p_mon.add_argument("--refresh", type=float, default=2.0, help="Refresh interval (seconds)")

    args = parser.parse_args()
    if args.command is None:
        parser.print_help()
        sys.exit(1)

    commands = {
        "train": cmd_train,
        "eval": cmd_eval,
        "warmup": cmd_warmup,
        "test": cmd_test,
        "monitor": cmd_monitor,
    }
    commands[args.command](args)


if __name__ == "__main__":
    main()
