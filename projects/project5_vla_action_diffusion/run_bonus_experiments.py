"""Reproduce the Project 5 bonus experiments.

The long-running commands are intentionally explicit and sequential so that
they can be launched inside a tmux session while ``monitor_dashboard.py``
follows their status. Existing checkpoints are reused for the moving-obstacle
evaluation; no demo pickle is copied into the repository.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def run(args: list[str]) -> None:
    print("$ " + " ".join(args), flush=True)
    subprocess.run(args, cwd=ROOT, check=True)


def train_and_eval(config: str, run_dir: str, output: str, seed: int,
                   method: str = "ddpm", sample_steps: int = 20) -> None:
    config_path = ROOT / "configs" / config
    run_path = ROOT / run_dir
    run([sys.executable, "train.py", "--config", str(config_path),
         "--method", method, "--run-dir", str(run_path), "--seed", "42",
         "--collect"])
    run([sys.executable, "eval.py", "--config", str(config_path),
         "--ckpt", str(run_path / "ckpts" / "model_final.pt"),
         "--method", method, "--n_episodes", "100", "--seed-start", str(seed),
         "--exec_steps", "4", "--n_sample_steps", str(sample_steps),
         "--output", str(ROOT / output), "--rollout-dir",
         str(ROOT / "results" / "bonus_rollouts"), "--max-success-plots", "6"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preset", choices=("multimodal", "resnet18",
                                               "resnet18-long", "moving", "all"),
                        default="all")
    args = parser.parse_args()
    if args.preset in ("multimodal", "all"):
        train_and_eval("reach2d_multimodal.yaml", "runs/project5/bonus_multimodal",
                       "results/eval_multimodal.json", 20000)
    if args.preset in ("resnet18", "all"):
        train_and_eval("reach2d_resnet18.yaml", "runs/project5/bonus_resnet18",
                       "results/eval_resnet18.json", 30000)
    if args.preset in ("resnet18-long", "all"):
        train_and_eval("reach2d_resnet18_long.yaml", "runs/project5/bonus_resnet18_long",
                       "results/eval_resnet18_long.json", 30000)
    if args.preset in ("moving", "all"):
        static_ckpt = ROOT / "ckpts" / "model_final.pt"
        run([sys.executable, "eval.py", "--config",
             str(ROOT / "configs" / "reach2d_moving.yaml"), "--ckpt",
             str(static_ckpt), "--method", "ddpm", "--n_episodes", "100",
             "--seed-start", "40000", "--exec_steps", "4", "--n_sample_steps",
             "20", "--output", str(ROOT / "results/eval_moving.json"),
             "--rollout-dir", str(ROOT / "results/bonus_rollouts"),
             "--max-success-plots", "6"])


if __name__ == "__main__":
    main()
