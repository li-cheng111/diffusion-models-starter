"""Run the reproducible Project 5 experiment matrix.

The script is intentionally sequential: the dashboard can show one active
training process and every finished result while the tmux session persists.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def run(command):
    print("$ " + " ".join(map(str, command)), flush=True)
    subprocess.run(command, cwd=ROOT, check=True)


def train_eval(label, config, method, run_root, episodes=100, seed_start=10000,
               sample_steps=20, exec_steps=4, collect=False):
    run_dir = run_root / label
    train = [sys.executable, "train.py", "--config", str(config), "--method", method,
             "--run-dir", str(run_dir), "--seed", "42"]
    if collect:
        train.append("--collect")
    run(train)
    output = run_dir / f"eval_{label}.json"
    rollouts = run_dir / "rollouts"
    run([sys.executable, "eval.py", "--config", str(config),
         "--ckpt", str(run_dir / "ckpts" / "model_final.pt"),
         "--method", method, "--n_episodes", str(episodes),
         "--seed-start", str(seed_start), "--exec_steps", str(exec_steps),
         "--n_sample_steps", str(sample_steps), "--output", str(output),
         "--rollout-dir", str(rollouts), "--max-success-plots", "3"])
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", choices=("smoke", "full"), default="smoke")
    parser.add_argument("--run-root", default="runs/project5")
    parser.add_argument("--resume", action="store_true", help="Reserved for future matrix resume")
    args = parser.parse_args()
    run_root = (ROOT / args.run_root).resolve()
    run_root.mkdir(parents=True, exist_ok=True)
    if args.preset == "smoke":
        config = ROOT / "configs" / "reach2d_debug.yaml"
        train_eval("smoke_state", config, "ddpm", run_root, episodes=10,
                   seed_start=10000, sample_steps=4, exec_steps=4, collect=True)
        print(json.dumps({"state": "smoke_complete", "run_root": str(run_root)}, indent=2))
        return

    config16 = ROOT / "configs" / "reach2d_16.yaml"
    state16 = ROOT / "configs" / "reach2d_state_16.yaml"
    config32 = ROOT / "configs" / "reach2d.yaml"
    results_dir = ROOT / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    results["vision_ddpm"] = train_eval("vision_ddpm", config16, "ddpm", run_root,
                                         collect=True, sample_steps=20, exec_steps=4)
    results["state_ddpm"] = train_eval("state_ddpm", state16, "ddpm", run_root,
                                       collect=False, sample_steps=20, exec_steps=4)
    results["vision_bc"] = train_eval("vision_bc", config16, "bc", run_root,
                                      collect=False, sample_steps=1, exec_steps=4)
    results["vision_fm"] = train_eval("vision_fm", config16, "fm", run_root,
                                      collect=False, sample_steps=10, exec_steps=4)
    # H=32 is the starter default and is retained as a padding/chunk ablation.
    results["vision_ddpm_h32"] = train_eval("vision_ddpm_h32", config32, "ddpm", run_root,
                                             collect=True, sample_steps=20, exec_steps=4)
    aliases = {"state_ddpm": "eval_state.json", "vision_ddpm": "eval_vision.json",
               "vision_bc": "eval_bc.json", "vision_fm": "eval_fm.json"}
    for key, filename in aliases.items():
        shutil.copy2(results[key], results_dir / filename)
    rollout_source = run_root / "vision_ddpm" / "rollouts"
    rollout_target = results_dir / "rollouts"
    rollout_target.mkdir(parents=True, exist_ok=True)
    for path in rollout_source.glob("success_*.png"):
        shutil.copy2(path, rollout_target / path.name)
    final_ckpt = ROOT / "ckpts" / "model_final.pt"
    final_ckpt.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(run_root / "vision_ddpm" / "ckpts" / "model_final.pt", final_ckpt)
    summary = {"state": "full_complete", "run_root": str(run_root),
               "results": {k: str(v) for k, v in results.items()},
               "final_checkpoint": str(final_ckpt)}
    (run_root / "matrix_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
