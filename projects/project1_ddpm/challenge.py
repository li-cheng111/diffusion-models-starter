"""Reproducible runner and aggregator for the Project 1 challenge track.

The challenge compares linear and cosine beta schedules over the same model,
optimizer, data pipeline, and three fixed random seeds. Importing this module
never starts training. The ``run`` subcommand starts jobs only when explicitly
invoked; ``--dry_run`` prints commands without executing them.
"""

from __future__ import annotations

import argparse
import csv
import re
import subprocess
import sys
from pathlib import Path
from statistics import mean, stdev
from typing import Iterable

import yaml


PROJECT_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_ROOT.parents[1]
SCHEDULE_CONFIGS = {
    "linear": PROJECT_ROOT / "configs" / "cifar10_linear.yaml",
    "cosine": PROJECT_ROOT / "configs" / "cifar10_cosine.yaml",
}
DEFAULT_SEEDS = (42, 43, 44)
FID_PATTERN = re.compile(
    r"FID:\s*([-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def _resolve_project_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def _format_command(command: list[str]) -> str:
    return " ".join(subprocess.list2cmdline([part]) for part in command)


def _run(command: list[str], dry_run: bool) -> None:
    print(f"$ {_format_command(command)}")
    if not dry_run:
        subprocess.run(command, cwd=REPO_ROOT, check=True)


def _read_fid(path: Path) -> float:
    text = path.read_text(encoding="utf-8")
    match = FID_PATTERN.search(text)
    if match is None:
        raise ValueError(f"Could not parse FID from {path}")
    return float(match.group(1))


def _experiment_dir(output_root: Path, schedule: str, epochs: int, seed: int) -> Path:
    return output_root / f"cifar10_{schedule}_{epochs}ep_seed{seed}"


def collect_results(
    output_root: Path,
    schedules: Iterable[str],
    epoch_budgets: Iterable[int],
    seeds: Iterable[int],
    num_samples: int,
    require_complete: bool = True,
) -> list[dict[str, object]]:
    """Read per-run FID files and return one record per schedule/seed pair."""

    schedules = tuple(schedules)
    epoch_budgets = tuple(epoch_budgets)
    seeds = tuple(seeds)
    rows: list[dict[str, object]] = []
    missing: list[str] = []
    for schedule in schedules:
        for epochs in epoch_budgets:
            for seed in seeds:
                run_dir = _experiment_dir(output_root, schedule, epochs, seed)
                ckpt_dir = run_dir / "ckpt"
                ema_path = ckpt_dir / f"fid_{num_samples}_EMA.txt"
                raw_path = ckpt_dir / f"fid_{num_samples}_raw.txt"
                if not ema_path.exists() or not raw_path.exists():
                    missing.append(f"{schedule}/{epochs}ep/seed{seed}")
                    continue
                rows.append(
                    {
                        "schedule": schedule,
                        "epochs": epochs,
                        "seed": seed,
                        "fid_ema": _read_fid(ema_path),
                        "fid_raw": _read_fid(raw_path),
                        "checkpoint": str(run_dir / "ckpt" / "final.pt"),
                    }
                )

    if missing and require_complete:
        expected = len(schedules) * len(epoch_budgets) * len(seeds)
        raise FileNotFoundError(
            f"Missing {len(missing)} of {expected} challenge results: "
            + ", ".join(missing)
            + ". Finish all runs or pass --allow_incomplete to summarize."
        )
    return rows


def _stat_text(values: list[float]) -> str:
    if not values:
        return "n/a"
    spread = stdev(values) if len(values) > 1 else 0.0
    return f"{mean(values):.4f} ± {spread:.4f}"


def write_summary(
    output_root: Path,
    rows: list[dict[str, object]],
    num_samples: int,
) -> tuple[Path, Path]:
    """Write machine-readable per-run results and a Markdown summary."""

    output_root.mkdir(parents=True, exist_ok=True)
    csv_path = output_root / "results.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "schedule",
                "epochs",
                "seed",
                "fid_ema",
                "fid_raw",
                "checkpoint",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    grouped: dict[tuple[str, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["schedule"]), int(row["epochs"]))
        grouped.setdefault(key, []).append(row)
    available_seeds = sorted({int(row["seed"]) for row in rows})
    seed_text = ", ".join(str(seed) for seed in available_seeds) or "none"
    available_budgets = sorted({int(row["epochs"]) for row in rows})
    budget_text = ", ".join(str(epochs) for epochs in available_budgets) or "none"

    lines = [
        "# Project 1 challenge-track summary",
        "",
        "- Real images: 5,000 CIFAR-10 training images (no random augmentation)",
        f"- Generated images per FID run: {num_samples:,}",
        f"- Seeds present in this summary: {seed_text}",
        f"- Epoch budgets present in this summary: {budget_text}",
        "- Reported spread: sample standard deviation across seeds",
        "",
        "## Per-seed results",
        "",
        "| Schedule | Epochs | Seed | EMA FID | Raw FID | Raw - EMA |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted(
        rows,
        key=lambda item: (
            str(item["schedule"]),
            int(item["epochs"]),
            int(item["seed"]),
        ),
    ):
        delta = float(row["fid_raw"]) - float(row["fid_ema"])
        lines.append(
            f"| {row['schedule']} | {row['epochs']} | {row['seed']} | "
            f"{float(row['fid_ema']):.4f} | {float(row['fid_raw']):.4f} | "
            f"{delta:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Mean ± std across seeds",
            "",
            "| Schedule | Epochs | EMA FID | Raw FID |",
            "|---|---:|---:|---:|",
        ]
    )
    for schedule, epochs in sorted(grouped):
        schedule_rows = grouped[(schedule, epochs)]
        lines.append(
            f"| {schedule} | {epochs} | "
            f"{_stat_text([float(r['fid_ema']) for r in schedule_rows])} | "
            f"{_stat_text([float(r['fid_raw']) for r in schedule_rows])} |"
        )
    lines.extend(
        [
            "",
            "The causal interpretation and failure analysis belong in "
            "`challenge_report.md`; this file records measurements only and "
            "must not be filled with estimated values.",
            "",
        ]
    )
    summary_path = output_root / "summary.md"
    summary_path.write_text("\n".join(lines), encoding="utf-8")
    return csv_path, summary_path


def run_challenge(args: argparse.Namespace) -> None:
    output_root = _resolve_project_path(args.output_root)
    epoch_budgets = tuple(args.epoch_budgets)
    seeds = tuple(args.seeds)
    schedules = tuple(args.schedules)

    for schedule in schedules:
        config_path = SCHEDULE_CONFIGS[schedule]
        if not config_path.exists():
            raise FileNotFoundError(config_path)
        with config_path.open(encoding="utf-8") as handle:
            config = yaml.safe_load(handle)
        configured_schedule = config["diffusion"]["beta_schedule"]
        if configured_schedule != schedule:
            raise ValueError(
                f"{config_path} declares {configured_schedule!r}, expected {schedule!r}"
            )

        for epochs in epoch_budgets:
            if epochs <= 0:
                raise ValueError("epoch budgets must be positive")
            for seed in seeds:
                run_dir = _experiment_dir(output_root, schedule, epochs, seed)
                final_ckpt = run_dir / "ckpt" / "final.pt"
                if final_ckpt.exists() and not args.allow_existing:
                    raise FileExistsError(
                        f"{final_ckpt} already exists; use --allow_existing only after "
                        "verifying that overwriting this run is intended."
                    )

                _run(
                    [
                        args.python,
                        "-m",
                        "projects.project1_ddpm.train",
                        "--config",
                        str(config_path),
                        "--output_dir",
                        str(run_dir),
                        "--seed",
                        str(seed),
                        "--num_epochs",
                        str(epochs),
                    ],
                    args.dry_run,
                )
                if not args.skip_samples:
                    _run(
                        [
                            args.python,
                        "-m",
                        "projects.project1_ddpm.sample",
                            "--ckpt",
                            str(final_ckpt),
                            "--num_samples",
                            "64",
                            "--batch_size",
                            str(args.sample_batch_size),
                            "--output_dir",
                            str(run_dir / "samples_challenge_ema"),
                            "--seed",
                            str(seed),
                            "--save_grid",
                        ],
                        args.dry_run,
                    )
                _run(
                    [
                        args.python,
                        "-m",
                        "projects.project1_ddpm.evaluate",
                        "--ckpt",
                        str(final_ckpt),
                        "--num_samples",
                        str(args.num_samples),
                        "--batch_size",
                        str(args.batch_size),
                        "--data_root",
                        args.data_root,
                        "--real_split",
                        "train",
                        "--compare_ema",
                        "--seed",
                        str(seed),
                    ],
                    args.dry_run,
                )

    if args.dry_run:
        print("Dry run complete: no training, sampling, or evaluation was executed.")
        return
    rows = collect_results(
        output_root, schedules, epoch_budgets, seeds, args.num_samples
    )
    csv_path, summary_path = write_summary(output_root, rows, args.num_samples)
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")


def summarize_challenge(args: argparse.Namespace) -> None:
    output_root = _resolve_project_path(args.output_root)
    rows = collect_results(
        output_root,
        tuple(args.schedules),
        tuple(args.epoch_budgets),
        tuple(args.seeds),
        args.num_samples,
        require_complete=not args.allow_incomplete,
    )
    csv_path, summary_path = write_summary(output_root, rows, args.num_samples)
    print(f"Wrote {csv_path}")
    print(f"Wrote {summary_path}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    run_parser = subparsers.add_parser("run", help="run all requested challenge jobs")
    run_parser.add_argument(
        "--schedules",
        nargs="+",
        choices=sorted(SCHEDULE_CONFIGS),
        default=list(SCHEDULE_CONFIGS),
    )
    run_parser.add_argument(
        "--epoch_budgets",
        nargs="+",
        type=int,
        default=[200],
        help="Epoch budgets to compare; use 50 200 to answer the self-check question.",
    )
    run_parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    run_parser.add_argument("--output_root", default="runs/challenge")
    run_parser.add_argument("--data_root", default="./data")
    run_parser.add_argument("--num_samples", type=int, default=5000)
    run_parser.add_argument("--batch_size", type=int, default=64)
    run_parser.add_argument("--sample_batch_size", type=int, default=64)
    run_parser.add_argument("--python", default=sys.executable)
    run_parser.add_argument("--skip_samples", action="store_true")
    run_parser.add_argument("--allow_existing", action="store_true")
    run_parser.add_argument("--dry_run", action="store_true")
    run_parser.set_defaults(handler=run_challenge)

    summary_parser = subparsers.add_parser(
        "summarize", help="aggregate completed FID files"
    )
    summary_parser.add_argument(
        "--schedules",
        nargs="+",
        choices=sorted(SCHEDULE_CONFIGS),
        default=list(SCHEDULE_CONFIGS),
    )
    summary_parser.add_argument(
        "--epoch_budgets",
        nargs="+",
        type=int,
        default=[200],
    )
    summary_parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    summary_parser.add_argument("--output_root", default="runs/challenge")
    summary_parser.add_argument("--num_samples", type=int, default=5000)
    summary_parser.add_argument("--allow_incomplete", action="store_true")
    summary_parser.set_defaults(handler=summarize_challenge)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
