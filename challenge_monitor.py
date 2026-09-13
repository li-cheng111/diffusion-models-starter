"""Read-only live dashboard for the multi-run Project 1 challenge matrix."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import urlparse


STEP_RE = re.compile(r"step_(\d+)")
LOG_STEP_RE = re.compile(r"\[epoch\s+\d+\s+step\s+(\d+)\]")
LOSS_RE = re.compile(r"loss=([0-9.eE+-]+)")

PAGE = r'''<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>DDPM Challenge Monitor</title>
<style>
:root{color-scheme:light dark;--bg:light-dark(#f6f8fb,#15171b);--card:light-dark(#fff,#20242c);--text:light-dark(#202633,#eef2f7);--muted:light-dark(#667085,#a7afbd);--line:light-dark(#dfe4ec,#363d49);--blue:light-dark(#356ae6,#8eafff);--amber:light-dark(#c87517,#f1b56e);}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--text);font-family:system-ui,-apple-system,"Segoe UI",sans-serif}main{max-width:1180px;margin:auto;padding:26px 20px 42px}h1{margin:0;font-size:25px}.sub{color:var(--muted);font-size:13px;margin:7px 0 22px}.bar{height:18px;background:color-mix(in srgb,var(--blue) 16%,transparent);border-radius:99px;overflow:hidden}.fill{height:100%;width:0;background:var(--blue);transition:width .5s}.meta{display:flex;justify-content:space-between;margin:7px 0;font-size:14px}.cards{display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin:20px 0}.card,.panel{background:var(--card);border:1px solid var(--line);border-radius:12px;padding:15px}.label{color:var(--muted);font-size:12px}.value{font-size:20px;font-weight:650;margin-top:5px}.panel{margin-top:16px;overflow:auto}h2{font-size:16px;margin:0 0 12px}table{width:100%;border-collapse:collapse;font-size:13px}th,td{text-align:left;padding:9px 8px;border-bottom:1px solid var(--line);white-space:nowrap}th{color:var(--muted);font-weight:600}.running,.evaluating,.sampling{color:var(--amber);font-weight:650}.completed{color:var(--blue);font-weight:650}.pending{color:var(--muted)}pre{white-space:pre-wrap;word-break:break-word;max-height:230px;overflow:auto;color:var(--muted);font-size:12px;line-height:1.5;margin:0}.foot{color:var(--muted);font-size:12px;margin-top:16px}@media(max-width:760px){main{padding:20px 12px}.cards{grid-template-columns:repeat(2,1fr)}.value{font-size:17px}}
</style></head><body><main><h1>DDPM 挑战档实时监控</h1><div class="sub">只读监控：linear/cosine × seed 42/43/44；每 2 秒刷新，不会干扰训练。</div>
<div class="meta"><span id="overall">等待数据</span><span id="percent">—</span></div><div class="bar"><div id="fill" class="fill"></div></div>
<div class="cards"><div class="card"><div class="label">状态</div><div id="status" class="value">—</div></div><div class="card"><div class="label">当前实验</div><div id="current" class="value">—</div></div><div class="card"><div class="label">已完成组数</div><div id="done" class="value">—</div></div><div class="card"><div class="label">最新 loss</div><div id="loss" class="value">—</div></div></div>
<section class="panel"><h2>实验矩阵</h2><table><thead><tr><th>Schedule</th><th>Epochs</th><th>Seed</th><th>状态</th><th>Step</th><th>进度</th><th>Loss</th></tr></thead><tbody id="runs"></tbody></table></section>
<section class="panel"><h2>Runner 日志（末尾）</h2><pre id="log">—</pre></section><div id="foot" class="foot">最后更新：—</div></main>
<script>
const $=id=>document.getElementById(id); const esc=s=>String(s??'').replace(/[&<>'"]/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));
function render(d){$('overall').textContent=`${d.completed_steps.toLocaleString()} / ${d.total_steps.toLocaleString()} steps`;$('percent').textContent=`${d.percent.toFixed(1)}%`;$('fill').style.width=`${d.percent}%`;$('status').textContent=d.status_label;$('status').className=`value ${d.status}`;$('current').textContent=d.current||'—';$('done').textContent=`${d.completed_runs} / ${d.total_runs}`;$('loss').textContent=d.latest_loss??'—';$('runs').innerHTML=d.runs.map(r=>`<tr><td>${esc(r.schedule)}</td><td>${r.epochs}</td><td>${r.seed}</td><td class="${r.status}">${esc(r.status_label)}</td><td>${r.step.toLocaleString()} / ${r.total_steps.toLocaleString()}</td><td>${r.percent.toFixed(1)}%</td><td>${r.loss??'—'}</td></tr>`).join('');$('log').textContent=d.log||'暂无日志';$('foot').textContent=`最后更新：${d.observed_at} · 自动刷新：2 秒`}
async function refresh(){try{const r=await fetch('/api/progress',{cache:'no-store'});if(!r.ok)throw Error(`HTTP ${r.status}`);render(await r.json())}catch(e){$('status').textContent='监控服务连接失败';$('log').textContent=e.message}}refresh();setInterval(refresh,2000);
</script></body></html>'''


def _step(path: Path) -> int:
    match = STEP_RE.search(path.stem)
    return int(match.group(1)) if match else 0


def _latest_loss(path: Path) -> str | None:
    if not path.exists():
        return None
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
        return f"{float(rows[-1]['loss']):.5f}" if rows else None
    except (OSError, KeyError, ValueError):
        return None


def _runner_alive() -> bool:
    """Return whether a challenge runner or its active training process exists."""

    proc_root = Path("/proc")
    if not proc_root.exists():
        return True
    for entry in proc_root.iterdir():
        if not entry.name.isdigit() or int(entry.name) == os.getpid():
            continue
        try:
            command = (entry / "cmdline").read_text(errors="replace").replace("\x00", " ")
        except OSError:
            continue
        if "challenge.py" in command or " train.py" in f" {command}":
            return True
    return False


class ChallengeMonitor:
    def __init__(self, args: argparse.Namespace) -> None:
        self.root = args.root.resolve()
        self.schedules = tuple(args.schedules)
        self.seeds = tuple(args.seeds)
        self.epochs = args.epochs
        self.total_steps = args.total_steps
        self.log_path = args.log_path.resolve()

    def snapshot(self) -> dict[str, Any]:
        try:
            log = "\n".join(
                self.log_path.read_text(
                    encoding="utf-8", errors="replace"
                ).splitlines()[-80:]
            )
        except OSError:
            log = ""
        command_lines = [line for line in log.splitlines() if line.startswith("$ ")]
        last_command = command_lines[-1] if command_lines else ""
        log_steps = [int(match) for match in LOG_STEP_RE.findall(log)]
        latest_log_step = max(log_steps, default=0)
        phase = None
        if "train.py" in last_command:
            phase = "running"
        elif "sample.py" in last_command:
            phase = "sampling"
        elif "evaluate.py" in last_command:
            phase = "evaluating"
        output_dirs = re.findall(r"--output_dir\s+(\S+)", last_command)
        checkpoint_paths = re.findall(r"--ckpt\s+(\S+)", last_command)
        if output_dirs:
            active_dir = Path(output_dirs[-1]).resolve()
        elif checkpoint_paths:
            active_dir = Path(checkpoint_paths[-1]).resolve().parent.parent
        else:
            active_dir = None
        runner_alive = _runner_alive()

        runs: list[dict[str, Any]] = []
        for schedule in self.schedules:
            for seed in self.seeds:
                run_dir = self.root / f"cifar10_{schedule}_{self.epochs}ep_seed{seed}"
                ckpt_dir = run_dir / "ckpt"
                final = ckpt_dir / "final.pt"
                step_paths = list(ckpt_dir.glob("step_*.pt")) + list((run_dir / "samples").glob("step_*.png"))
                step = self.total_steps if final.exists() else max((_step(path) for path in step_paths), default=0)
                is_active = active_dir == run_dir.resolve()
                if is_active:
                    step = max(step, latest_log_step)
                status = "completed" if final.exists() else (
                    "running" if (step > 0 or is_active) and runner_alive else (
                        "stopped" if step > 0 or is_active else "pending"
                    )
                )
                if final.exists() and is_active and phase in {"sampling", "evaluating"} and runner_alive:
                    status = phase
                loss = _latest_loss(run_dir / "loss_history.csv")
                runs.append({
                    "schedule": schedule,
                    "epochs": self.epochs,
                    "seed": seed,
                    "step": step,
                    "total_steps": self.total_steps,
                    "percent": min(100.0, step / max(self.total_steps, 1) * 100.0),
                    "status": status,
                    "status_label": {
                        "completed": "已完成",
                        "running": "训练中",
                        "sampling": "采样中",
                        "evaluating": "FID 评估中",
                        "stopped": "已中断",
                        "pending": "等待中",
                    }[status],
                    "loss": loss,
                })
        completed = sum(row["status"] == "completed" for row in runs)
        current = next(
            (
                row
                for row in runs
                if row["status"] in {"running", "sampling", "evaluating", "stopped"}
            ),
            None,
        )
        if current is None:
            current = next((row for row in runs if row["status"] == "pending"), None)
        total_steps = len(runs) * self.total_steps
        completed_steps = sum(int(row["step"]) for row in runs)
        all_completed = completed == len(runs) and bool(runs)
        status = current["status"] if current else ("completed" if all_completed else "pending")
        log = "\n".join(log.splitlines()[-18:])
        latest_losses = [row["loss"] for row in runs if row["loss"] is not None]
        return {
            "status": status,
            "status_label": {
                "completed": "全部完成",
                "running": "训练中",
                "sampling": "采样中",
                "evaluating": "FID 评估中",
                "stopped": "已中断",
                "pending": "等待启动",
            }[status],
            "current": f"{current['schedule']} / {current['epochs']}ep / seed {current['seed']}" if current else None,
            "completed_runs": completed,
            "total_runs": len(runs),
            "completed_steps": completed_steps,
            "total_steps": total_steps,
            "percent": min(100.0, completed_steps / max(total_steps, 1) * 100.0),
            "latest_loss": latest_losses[-1] if latest_losses else None,
            "runs": runs,
            "log": log,
            "observed_at": __import__("datetime").datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z"),
        }


def make_handler(monitor: ChallengeMonitor):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if urlparse(self.path).path == "/api/progress":
                body = json.dumps(monitor.snapshot(), ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
            else:
                body = PAGE.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("runs/challenge"))
    parser.add_argument("--log-path", type=Path, default=Path("logs/challenge_runner.log"))
    parser.add_argument("--schedules", nargs="+", choices=("linear", "cosine"), default=("linear", "cosine"))
    parser.add_argument("--seeds", nargs="+", type=int, default=(42, 43, 44))
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--total-steps", type=int, default=78000)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    monitor = ChallengeMonitor(args)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(monitor))
    print(f"Challenge monitor: http://{args.host}:{args.port}/", flush=True)
    print(f"Watching: {monitor.root}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
