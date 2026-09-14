"""Plot a compact summary of the Project 5 bonus results."""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"


def load(name: str) -> dict:
    return json.loads((RESULTS / name).read_text(encoding="utf-8"))


def main() -> None:
    entries = [
        ("Multimodal\n(two targets)", load("eval_multimodal.json"), "#2ca02c"),
        ("Moving\ndistractors", load("eval_moving.json"), "#ff7f0e"),
        ("ResNet18\n5k steps", load("eval_resnet18.json"), "#9467bd"),
        ("ResNet18\n15k steps", load("eval_resnet18_long.json"), "#d62728"),
    ]
    labels = [e[0] for e in entries]
    success = [100 * e[1]["success_rate"] for e in entries]
    collision = [100 * e[1]["collision_rate"] for e in entries]
    timeout = [100 * e[1]["timeout_rate"] for e in entries]

    fig, (ax, ax_mode) = plt.subplots(1, 2, figsize=(11, 4.8),
                                      gridspec_kw={"width_ratios": [1.6, 1]})
    x = list(range(len(labels)))
    ax.bar(x, success, label="success", color="#2ca02c")
    ax.bar(x, collision, bottom=success, label="collision", color="#d62728")
    ax.bar(x, timeout, bottom=[s + c for s, c in zip(success, collision)],
           label="timeout", color="#7f7f7f")
    ax.set_ylim(0, 100)
    ax.set_ylabel("Episodes (%)")
    ax.set_xticks(x, labels)
    ax.set_title("Bonus closed-loop outcomes (100 episodes)")
    ax.legend(loc="upper right", frameon=False)
    for i, value in enumerate(success):
        ax.text(i, value + 2, f"{value:.0f}%", ha="center", va="bottom", fontsize=10)

    ax_mode.set_xlim(-1, 1)
    ax_mode.set_ylim(-1, 1)
    ax_mode.scatter([-0.65, 0.65], [0.55, 0.55], s=550, c=["#e41a1c", "#377eb8"],
                    edgecolors="black", linewidths=1.2, label="target modes")
    ax_mode.scatter([0], [-0.55], s=420, c="#1f77b4", edgecolors="black", label="agent")
    ax_mode.annotate("mode 0\n96.2%", (-0.65, 0.55), xytext=(-0.95, 0.82),
                     arrowprops={"arrowstyle": "->"}, ha="center")
    ax_mode.annotate("mode 1\n97.9%", (0.65, 0.55), xytext=(0.95, 0.82),
                     arrowprops={"arrowstyle": "->"}, ha="center")
    ax_mode.set_title("Two target modes")
    ax_mode.set_xlabel("Environment x")
    ax_mode.set_ylabel("Environment y")
    ax_mode.grid(alpha=0.2)
    fig.tight_layout()
    output = RESULTS / "bonus_summary.png"
    fig.savefig(output, dpi=180, bbox_inches="tight")
    print(output)


if __name__ == "__main__":
    main()
