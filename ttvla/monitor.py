"""Real-time training monitor. Reads JSONL log and displays live progress."""

import json
import time
import sys
from pathlib import Path


def run_monitor(log_file: str = "logs/ttvla_train.jsonl", refresh: float = 2.0):
    """Watch JSONL log file and display live training progress."""
    path = Path(log_file)
    print(f"Monitoring: {path}")
    print(f"Refresh: {refresh}s (Ctrl+C to stop)\n")

    last_size = 0

    try:
        while True:
            if not path.exists():
                print(f"\rWaiting for {path}...", end="", flush=True)
                time.sleep(refresh)
                continue

            current_size = path.stat().st_size
            if current_size == last_size:
                time.sleep(refresh)
                continue
            last_size = current_size

            # Read all episodes
            episodes = []
            with open(path) as f:
                for line in f:
                    line = line.strip()
                    if line:
                        try:
                            episodes.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue

            if not episodes:
                time.sleep(refresh)
                continue

            _render(episodes)
            time.sleep(refresh)
    except KeyboardInterrupt:
        print("\nMonitor stopped.")


def _render(episodes):
    """Render training dashboard."""
    # Clear screen
    sys.stdout.write("\033[2J\033[H")

    n = len(episodes)
    successes = sum(1 for e in episodes if e.get("success", False))
    sr = successes / n * 100

    rewards = [e["total_reward"] for e in episodes]
    progresses = [e["final_progress"] for e in episodes]
    losses = [e["avg_loss"] for e in episodes]

    latest = episodes[-1]

    print("=" * 60)
    print("  TT-VLA Training Monitor")
    print("=" * 60)

    # Progress bar
    target = 80  # default target episodes
    pct = min(n / target * 100, 100)
    bar_w = 40
    filled = int(bar_w * n / target)
    bar = "\u2588" * filled + "\u2591" * (bar_w - filled)
    print(f"  [{bar}] {pct:.0f}%  ({n}/{target} episodes)")
    print()

    # Key metrics
    print(f"  Success Rate:  {sr:.1f}% ({successes}/{n})")
    print(f"  Avg Reward:    {sum(rewards)/n:.4f}")
    print(f"  Avg Progress:  {sum(progresses)/n:.3f}")
    print(f"  Avg Loss:      {sum(losses)/n:.4f}")
    print()

    # Latest episode
    print(f"  Latest (ep {latest['episode']}):")
    print(f"    reward={latest['total_reward']:.4f}  "
          f"progress={latest['final_progress']:.3f}  "
          f"loss={latest['avg_loss']:.4f}  "
          f"time={latest['wall_time']:.1f}s")
    print()

    # Reward trend (last 10 episodes as ASCII spark)
    window = episodes[-min(20, n):]
    if len(window) >= 2:
        r_vals = [e["total_reward"] for e in window]
        print(f"  Reward trend (last {len(window)} eps): {_sparkline(r_vals)}")
        p_vals = [e["final_progress"] for e in window]
        print(f"  Progress trend:                        {_sparkline(p_vals)}")

    print()
    print("=" * 60)
    print("  Press Ctrl+C to stop")
    print("=" * 60)


def _sparkline(values):
    """Render values as ASCII sparkline."""
    if not values:
        return ""
    blocks = " _.-=^*#"
    mn, mx = min(values), max(values)
    rng = mx - mn if mx > mn else 1.0
    return "".join(blocks[min(len(blocks)-1, int((v - mn) / rng * (len(blocks)-1)))] for v in values)
