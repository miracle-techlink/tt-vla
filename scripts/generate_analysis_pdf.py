#!/usr/bin/env python3
"""Generate baseline experiment analysis PDF report."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import csv
import os

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")

def load_csv(name):
    with open(os.path.join(RESULTS_DIR, name)) as f:
        return list(csv.DictReader(f))

def main():
    baseline = load_csv("baseline_results.csv")
    episodes = load_csv("episode_details.csv")
    
    pdf_path = os.path.join(RESULTS_DIR, "baseline_analysis.pdf")
    with PdfPages(pdf_path) as pdf:
        # Page 1: Title & Summary
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        
        title_text = "TT-VLA Baseline Experiment Report"
        ax.text(0.5, 0.92, title_text, transform=ax.transAxes,
                fontsize=20, fontweight="bold", ha="center", va="top")
        ax.text(0.5, 0.87, "ManiSkill3 PickCube-v1 | OpenVLA-7B",
                transform=ax.transAxes, fontsize=13, ha="center", va="top", color="gray")
        
        summary = (
            "Experiment Overview\n"
            "=" * 50 + "\n\n"
            "This report presents baseline results for the TT-VLA\n"
            "(Test-Time Reinforcement Learning for VLA) pipeline\n"
            "reproduction on ManiSkill3 PickCube-v1.\n\n"
            "Two baseline experiments were conducted:\n"
            "  1. Mock VLA (random policy): Pipeline validation\n"
            "  2. OpenVLA-7B (no warm-up): Pre-trained model baseline\n\n"
            "Key Findings:\n"
            "  - Both baselines achieve 0% success rate, as expected\n"
            "  - OpenVLA without domain-specific warm-up cannot solve\n"
            "    the task, confirming the paper\'s claim that warm-up\n"
            "    fine-tuning is essential before TT-VLA adaptation\n"
            "  - Pipeline is fully functional end-to-end\n"
            "  - PPO updates execute correctly (20 updates/episode)\n\n"
            "Next Steps:\n"
            "  - 10k-step warm-up fine-tuning on ManiSkill3 demos\n"
            "  - Re-evaluate with TT-VLA after warm-up\n"
            "  - Compare with paper Table 1 results\n"
        )
        ax.text(0.08, 0.78, summary, transform=ax.transAxes,
                fontsize=10, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close()
        
        # Page 2: Baseline Comparison Table
        fig, axes = plt.subplots(2, 1, figsize=(8.5, 11), gridspec_kw={"height_ratios": [1, 2]})
        
        # Table
        ax = axes[0]
        ax.axis("off")
        ax.set_title("Baseline Results Summary", fontsize=14, fontweight="bold", pad=20)
        
        table_data = []
        col_labels = ["Model", "Episodes", "SR (%)", "Avg Reward", "Avg Time (s)", "Trainable %"]
        for row in baseline:
            table_data.append([
                row["model"].split("/")[-1],
                row["episodes"],
                row["success_rate"],
                row["avg_reward"],
                row["avg_time_per_ep_s"],
                row["trainable_pct"] if row["trainable_pct"] != "N/A" else "-",
            ])
        
        table = ax.table(cellText=table_data, colLabels=col_labels,
                        loc="center", cellLoc="center")
        table.auto_set_font_size(False)
        table.set_fontsize(9)
        table.scale(1.0, 1.8)
        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_facecolor("#4472C4")
                cell.set_text_props(color="white", fontweight="bold")
        
        # Episode time comparison bar chart
        ax2 = axes[1]
        mock_times = [float(e["time_s"]) for e in episodes if e["experiment"] == "mock_baseline"]
        ovla_times = [float(e["time_s"]) for e in episodes if e["experiment"] == "openvla_baseline"]
        
        x1 = range(len(mock_times))
        x2 = range(len(ovla_times))
        
        ax2.bar([i - 0.2 for i in x1], mock_times, 0.35, label="Mock VLA", color="#4472C4", alpha=0.8)
        ax2.bar([i + 0.2 for i in x2], ovla_times, 0.35, label="OpenVLA-7B", color="#ED7D31", alpha=0.8)
        ax2.set_xlabel("Episode")
        ax2.set_ylabel("Time (seconds)")
        ax2.set_title("Per-Episode Execution Time", fontsize=12, fontweight="bold")
        ax2.legend()
        ax2.set_xticks(range(max(len(mock_times), len(ovla_times))))
        ax2.set_xticklabels([str(i+1) for i in range(max(len(mock_times), len(ovla_times)))])
        
        plt.tight_layout()
        pdf.savefig(fig)
        plt.close()
        
        # Page 3: Configuration & Paper Comparison
        fig, ax = plt.subplots(figsize=(8.5, 11))
        ax.axis("off")
        ax.set_title("Experiment Configuration & Paper Comparison", fontsize=14, fontweight="bold")
        
        config_text = (
            "Experiment Configuration (Algorithm 1, TT-VLA Paper)\n"
            "=" * 55 + "\n\n"
            "Hardware:\n"
            "  GPU: NVIDIA GeForce RTX 5090 (2x)\n"
            "  Server: 172.30.1.53\n\n"
            "Model Configuration:\n"
            "  Base Model:     OpenVLA-7B (openvla/openvla-7b)\n"
            "  Architecture:   Llama-2-7B backbone + DinoV2 + SigLIP\n"
            "  Total Params:   7,541,237,184 (7.5B)\n"
            "  LoRA Rank:      32\n"
            "  Trainable:      33,554,432 (0.44%)\n"
            "  Action Bins:    256 (tokenized action space)\n\n"
            "TT-VLA Hyperparameters:\n"
            "  Learning Rate:  1e-4\n"
            "  PPO K (update interval): 8 steps\n"
            "  Clip Epsilon:   0.2\n"
            "  Max Steps/Ep:   160\n"
            "  PPO Updates/Ep: 20\n\n"
            "Environment:\n"
            "  Task:          ManiSkill3 PickCube-v1\n"
            "  Obs Mode:      RGB (480x640)\n"
            "  Action Space:  Continuous (padded to env dim)\n"
            "  Reward:        Dense progress-based\n"
            "    - Grasp:     0.4 progress\n"
            "    - Place:     0.85 progress\n"
            "    - Success:   1.0 progress\n\n"
            "Paper Comparison (Table 1, arXiv:2601.06748):\n"
            "=" * 55 + "\n\n"
            "  Paper reports for PickCube-v1:\n"
            "    OpenVLA (no TT-VLA):    ~15% SR (after warm-up)\n"
            "    OpenVLA + TT-VLA:       ~45% SR (after warm-up + TT-RL)\n\n"
            "  Our baseline (no warm-up): 0% SR\n"
            "    -> Expected! Paper requires warm-up fine-tuning\n"
            "       on domain-specific demonstrations before\n"
            "       TT-VLA can improve performance.\n\n"
            "  Next: 10k-step warm-up fine-tuning, then re-evaluate.\n"
        )
        ax.text(0.05, 0.92, config_text, transform=ax.transAxes,
                fontsize=9.5, va="top", family="monospace")
        pdf.savefig(fig)
        plt.close()
    
    print(f"PDF saved to: {pdf_path}")

if __name__ == "__main__":
    main()
