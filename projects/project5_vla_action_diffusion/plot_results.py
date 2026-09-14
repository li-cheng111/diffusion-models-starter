"""Create the compact training-loss figure used by report.md."""
from pathlib import Path
import json

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
METRICS = ROOT / "results" / "metrics"
OUT = ROOT / "results" / "loss_curve.png"

series = {
    "vision DDPM (GAP)": METRICS / "vision_ddpm.jsonl",
    "vision DDPM (spatial, final)": METRICS / "vision_ddpm_spatial.jsonl",
    "state DDPM": METRICS / "state_ddpm.jsonl",
    "vision FM": METRICS / "vision_fm.jsonl",
}

plt.figure(figsize=(8.4, 4.6), dpi=160)
for label, path in series.items():
    if not path.exists():
        continue
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    plt.plot([row["step"] for row in rows], [row["avg100"] for row in rows], label=label)
plt.xlabel("training step")
plt.ylabel("moving-average loss (100 steps)")
plt.title("Project 5 AutoDL training curves")
plt.grid(alpha=0.25)
plt.legend(fontsize=8)
plt.tight_layout()
OUT.parent.mkdir(parents=True, exist_ok=True)
plt.savefig(OUT)
print(OUT)
