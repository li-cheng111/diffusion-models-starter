"""Local live dashboard for a running Project 1 DDPM experiment.

The monitor uses only the Python standard library. It never changes the
training process; each browser refresh reads the latest files in the run
directory and returns a small JSON snapshot.

Example:
    python monitor.py --run-dir runs/exp_mnist_baseline \
        --total-steps 23400 --train-pid 10144 --port 8765
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from urllib.parse import unquote, urlparse


STEP_RE = re.compile(r"step_(\d+)")


PAGE = r'''<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>MNIST DDPM Live Monitor</title>
  <style>
    :root {
      color-scheme: light dark;
      --bg: light-dark(#f7f8fb, #15171b);
      --surface: light-dark(#ffffff, #1e2128);
      --text: light-dark(#1f2430, #eef1f6);
      --muted: light-dark(#667085, #a7afbd);
      --border: light-dark(#dfe3eb, #343a46);
      --primary: light-dark(#356ae6, #8eafff);
      --secondary: light-dark(#d07820, #f0b46c);
      --track: color-mix(in srgb, var(--primary) 15%, transparent);
    }

    * { box-sizing: border-box; }
    body {
      margin: 0;
      min-width: 320px;
      background: var(--bg);
      color: var(--text);
      font-family: system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
    }
    main { max-width: 1120px; margin: 0 auto; padding: 28px 22px 42px; }
    h1 { margin: 0; font-size: 25px; line-height: 1.25; }
    h2 { margin: 0 0 12px; font-size: 16px; }
    .muted { color: var(--muted); }
    .subtitle { margin: 7px 0 22px; color: var(--muted); font-size: 13px; }
    .status { display: flex; flex-wrap: wrap; gap: 8px 20px; margin-bottom: 18px; font-size: 13px; }
    .status strong { color: var(--text); font-weight: 650; }
    .status .running { color: var(--secondary); }
    .status .done { color: var(--primary); }
    .progress-wrap { margin: 4px 0 26px; }
    .progress-meta { display: flex; justify-content: space-between; gap: 12px; margin-bottom: 7px; font-size: 13px; }
    .progress-track { height: 16px; overflow: hidden; border-radius: 999px; background: var(--track); }
    .progress-fill { height: 100%; width: 0; border-radius: inherit; background: var(--primary); transition: width 350ms ease; }
    .layout { display: grid; grid-template-columns: minmax(0, 1.3fr) minmax(280px, .7fr); gap: 18px; }
    .panel { min-width: 0; padding: 18px; border: 1px solid var(--border); border-radius: 12px; background: var(--surface); }
    .panel.full { grid-column: 1 / -1; }
    .chart { width: 100%; height: 235px; display: block; }
    .chart text { fill: var(--muted); font-size: 12px; }
    .chart .axis { stroke: var(--border); stroke-width: 1; }
    .chart .grid { stroke: var(--border); stroke-width: 1; stroke-dasharray: 3 5; }
    .chart .line { fill: none; stroke: var(--primary); stroke-width: 2.5; stroke-linejoin: round; stroke-linecap: round; }
    .chart .dot { fill: var(--primary); }
    .samples { display: grid; grid-template-columns: repeat(2, minmax(0, 1fr)); gap: 12px; }
    .sample { min-width: 0; }
    .sample img { display: block; width: 100%; aspect-ratio: 1; object-fit: contain; border: 1px solid var(--border); background: var(--bg); }
    .sample-label { margin-top: 6px; font-size: 12px; color: var(--muted); text-align: center; }
    .file-list { margin: 0; padding-left: 18px; color: var(--muted); font-size: 13px; line-height: 1.8; }
    .empty { color: var(--muted); font-size: 13px; }
    .footer { margin-top: 18px; color: var(--muted); font-size: 12px; }
    @media (max-width: 760px) {
      main { padding: 22px 14px 34px; }
      .layout { grid-template-columns: 1fr; }
      .panel.full { grid-column: auto; }
      .samples { grid-template-columns: repeat(2, minmax(0, 1fr)); }
    }
  </style>
</head>
<body>
  <main aria-labelledby="title">
    <h1 id="title">MNIST DDPM 实时训练监控</h1>
    <p class="subtitle">每 2 秒读取一次训练目录；页面只读，不会干扰训练进程。</p>
    <div id="status" class="status" aria-live="polite">正在连接训练监控服务…</div>

    <section class="progress-wrap" aria-label="总体训练进度">
      <div class="progress-meta"><span id="step-label">等待数据</span><span id="percent-label">—</span></div>
      <div class="progress-track" role="progressbar" aria-label="DDPM training progress" aria-valuemin="0" aria-valuemax="100" aria-valuenow="0">
        <div id="progress-fill" class="progress-fill"></div>
      </div>
    </section>

    <div class="layout">
      <section class="panel full" aria-labelledby="loss-title">
        <h2 id="loss-title">Loss 曲线</h2>
        <div id="loss-empty" class="empty">训练脚本尚未写出 loss_history.csv；文件出现后会自动显示曲线。</div>
        <svg id="loss-chart" class="chart" viewBox="0 0 900 235" role="img" aria-label="DDPM loss curve" hidden>
          <line class="axis" x1="55" y1="205" x2="875" y2="205" />
          <line class="axis" x1="55" y1="20" x2="55" y2="205" />
          <g id="loss-grid"></g><polyline id="loss-line" class="line" points=""></polyline><g id="loss-dots"></g>
          <text x="465" y="230" text-anchor="middle">training step</text>
          <text x="15" y="112" text-anchor="middle" transform="rotate(-90 15 112)">MSE loss</text>
        </svg>
      </section>

      <section class="panel" aria-labelledby="samples-title">
        <h2 id="samples-title">样本生成里程碑</h2>
        <div id="samples" class="samples"></div>
      </section>

      <section class="panel" aria-labelledby="ckpt-title">
        <h2 id="ckpt-title">Checkpoint</h2>
        <ul id="checkpoints" class="file-list"></ul>
      </section>
    </div>
    <div id="footer" class="footer">最后更新：—</div>
  </main>

  <script>
    const $ = (id) => document.getElementById(id);
    const fmt = (value) => Number(value || 0).toLocaleString('en-US');
    const esc = (value) => String(value).replace(/[&<>'"]/g, (c) => ({'&':'&amp;','<':'&lt;','>':'&gt;',"'":'&#39;','"':'&quot;'}[c]));

    function drawLoss(rows) {
      const chart = $('loss-chart');
      const empty = $('loss-empty');
      if (!rows || rows.length < 2) {
        chart.hidden = true;
        empty.hidden = false;
        return;
      }
      chart.hidden = false;
      empty.hidden = true;
      const left = 55, right = 875, top = 20, bottom = 205;
      const steps = rows.map((r) => Number(r.step));
      const losses = rows.map((r) => Number(r.loss));
      const minStep = Math.min(...steps), maxStep = Math.max(...steps);
      const maxLoss = Math.max(...losses), minLoss = Math.min(...losses);
      const spread = Math.max(maxLoss - minLoss, 1e-8);
      const x = (v) => left + (v - minStep) / Math.max(maxStep - minStep, 1) * (right - left);
      const y = (v) => bottom - (v - minLoss) / spread * (bottom - top);
      $('loss-line').setAttribute('points', rows.map((r) => `${x(Number(r.step)).toFixed(1)},${y(Number(r.loss)).toFixed(1)}`).join(' '));
      $('loss-dots').innerHTML = '';
      const latest = rows[rows.length - 1];
      const dot = document.createElementNS('http://www.w3.org/2000/svg', 'circle');
      dot.setAttribute('class', 'dot'); dot.setAttribute('cx', x(Number(latest.step))); dot.setAttribute('cy', y(Number(latest.loss))); dot.setAttribute('r', '4');
      $('loss-dots').appendChild(dot);
      $('loss-grid').innerHTML = `<text x="${left - 8}" y="${top + 4}" text-anchor="end">${Number(maxLoss).toFixed(3)}</text><text x="${left - 8}" y="${bottom + 4}" text-anchor="end">${Number(minLoss).toFixed(3)}</text><text x="${left}" y="220">${fmt(minStep)}</text><text x="${right}" y="220" text-anchor="end">${fmt(maxStep)}</text>`;
    }

    function render(data) {
      const percent = Math.max(0, Math.min(100, data.percent));
      const runningClass = data.status === 'running' ? 'running' : (data.status === 'completed' ? 'done' : '');
      $('status').innerHTML = `<span>状态：<strong class="${runningClass}">${esc(data.status_label)}</strong></span><span>当前记录步数：<strong>${fmt(data.current_step)} / ${fmt(data.total_steps)}</strong></span><span>GPU：<strong>${esc(data.gpu.label)}</strong></span><span>样本文件：<strong>${fmt(data.samples.length)}</strong></span>`;
      $('step-label').textContent = `${fmt(data.current_step)} / ${fmt(data.total_steps)} steps`;
      $('percent-label').textContent = `${percent.toFixed(1)}%`;
      $('progress-fill').style.width = `${percent}%`;
      document.querySelector('[role="progressbar"]').setAttribute('aria-valuenow', percent.toFixed(1));
      $('samples').innerHTML = data.samples.length ? data.samples.map((sample) => `<div class="sample"><img src="${esc(sample.url)}?v=${sample.mtime}" alt="DDPM step ${sample.step} generated samples"><div class="sample-label">step ${fmt(sample.step)}</div></div>`).join('') : '<div class="empty">尚未生成样本文件。</div>';
      $('checkpoints').innerHTML = data.checkpoints.length ? data.checkpoints.map((name) => `<li>${esc(name)}</li>`).join('') : '<li class="empty">尚未生成 checkpoint。</li>';
      drawLoss(data.loss);
      $('footer').textContent = `最后更新：${data.observed_at} · 自动刷新：2 秒 · 最新持久化文件：${data.latest_file || '—'}`;
    }

    async function refresh() {
      try {
        const response = await fetch('/api/progress', { cache: 'no-store' });
        if (!response.ok) throw new Error(`HTTP ${response.status}`);
        render(await response.json());
      } catch (error) {
        $('status').innerHTML = `<span>状态：<strong>监控服务连接失败</strong></span><span class="muted">${esc(error.message)}</span>`;
      }
    }
    refresh();
    setInterval(refresh, 2000);
  </script>
</body>
</html>'''


def _step_from_name(path: Path) -> int:
    match = STEP_RE.search(path.stem)
    return int(match.group(1)) if match else 0


def _pid_alive(pid: int | None) -> bool:
    if not pid:
        return False
    try:
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError, PermissionError):
        # Some Windows sandbox configurations reject os.kill(pid, 0) even
        # when the process is alive. Fall back to a read-only process query.
        if os.name != "nt":
            return False
        try:
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-NonInteractive",
                    "-Command",
                    f"$p = Get-Process -Id {int(pid)} -ErrorAction SilentlyContinue; if ($p) {{ 'alive' }}",
                ],
                capture_output=True,
                text=True,
                timeout=2,
                check=False,
            )
            return result.returncode == 0 and bool(result.stdout.strip())
        except (OSError, subprocess.TimeoutExpired):
            return False


def _gpu_status() -> dict[str, str]:
    command = [
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total",
        "--format=csv,noheader,nounits",
    ]
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=2, check=False)
        line = result.stdout.strip().splitlines()[0]
        usage, used, total = [part.strip() for part in line.split(",")[:3]]
        return {"usage": usage, "used_mib": used, "total_mib": total, "label": f"{usage}% · {used}/{total} MiB"}
    except (OSError, IndexError, ValueError, subprocess.TimeoutExpired):
        return {"usage": "—", "used_mib": "—", "total_mib": "—", "label": "unavailable"}


def _read_loss(path: Path, limit: int = 800) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows: list[dict[str, float]] = []
    try:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                try:
                    rows.append({"step": float(row["step"]), "loss": float(row["loss"])})
                except (KeyError, TypeError, ValueError):
                    continue
    except OSError:
        return []
    return rows[-limit:]


class Monitor:
    def __init__(self, run_dir: Path, total_steps: int, train_pid: int | None) -> None:
        self.run_dir = run_dir.resolve()
        self.total_steps = max(1, total_steps)
        self.train_pid = train_pid

    def snapshot(self) -> dict[str, Any]:
        samples_dir = self.run_dir / "samples"
        ckpt_dir = self.run_dir / "ckpt"
        samples = []
        for path in sorted(samples_dir.glob("step_*.png"), key=_step_from_name):
            try:
                mtime = int(path.stat().st_mtime_ns)
            except OSError:
                continue
            samples.append({"step": _step_from_name(path), "url": f"/samples/{path.name}", "mtime": mtime})
        checkpoints = sorted(path.name for path in ckpt_dir.glob("*.pt"))
        loss = _read_loss(self.run_dir / "loss_history.csv")
        persisted_steps = [_step_from_name(path) for path in ckpt_dir.glob("step_*.pt")]
        persisted_steps.extend(int(row["step"]) for row in loss)
        persisted_steps.extend(item["step"] for item in samples)
        current_step = int(max(persisted_steps, default=0))
        final_exists = (ckpt_dir / "final.pt").exists()
        active = _pid_alive(self.train_pid)
        status = "completed" if final_exists and loss else ("running" if active else ("stopped" if current_step else "unknown"))
        status_label = {
            "completed": "已完成",
            "running": "训练中",
            "stopped": "已停止",
            "unknown": "未能确认",
        }[status]
        files = [*samples, *[{"step": _step_from_name(path)} for path in ckpt_dir.glob("step_*.pt")]]
        latest_file = "—"
        if files:
            latest_file = max(files, key=lambda item: item.get("step", 0)).get("step", 0)
            latest_file = f"step {int(latest_file):,}"
        return {
            "status": status,
            "status_label": status_label,
            "current_step": current_step,
            "total_steps": self.total_steps,
            "percent": 100.0 * current_step / self.total_steps,
            "samples": samples,
            "checkpoints": checkpoints,
            "loss": loss,
            "gpu": _gpu_status(),
            "latest_file": latest_file,
            "observed_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
        }

    def sample_path(self, name: str) -> Path | None:
        candidate = (self.run_dir / "samples" / unquote(name)).resolve()
        samples_root = (self.run_dir / "samples").resolve()
        if candidate.parent != samples_root or candidate.suffix.lower() != ".png" or not candidate.is_file():
            return None
        return candidate


def make_handler(monitor: Monitor):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            route = urlparse(self.path).path
            if route == "/":
                body = PAGE.encode("utf-8")
                self._send(200, "text/html; charset=utf-8", body)
                return
            if route == "/api/progress":
                body = json.dumps(monitor.snapshot(), ensure_ascii=False).encode("utf-8")
                self._send(200, "application/json; charset=utf-8", body, cache_control="no-store")
                return
            if route.startswith("/samples/"):
                path = monitor.sample_path(route.removeprefix("/samples/"))
                if path is not None:
                    try:
                        self._send(200, "image/png", path.read_bytes(), cache_control="no-cache")
                    except OSError:
                        pass
                    return
            self._send(404, "text/plain; charset=utf-8", b"Not found")

        def _send(self, status: int, content_type: str, body: bytes, cache_control: str = "no-cache") -> None:
            self.send_response(status)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", cache_control)
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args: Any) -> None:
            return

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path(__file__).resolve().parent / "runs/exp_mnist_baseline",
    )
    parser.add_argument("--total-steps", type=int, default=23400)
    parser.add_argument("--train-pid", type=int, default=None)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()

    monitor = Monitor(args.run_dir, args.total_steps, args.train_pid)
    server = ThreadingHTTPServer((args.host, args.port), make_handler(monitor))
    print(f"Live DDPM monitor: http://{args.host}:{args.port}/", flush=True)
    print(f"Watching: {monitor.run_dir}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
