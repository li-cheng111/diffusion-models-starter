"""Small read-only web dashboard for the AutoDL Project 2 experiments.

The server intentionally has no mutation endpoints.  It only reads local
experiment files and a few process/GPU counters, making it safe to leave
running while the benchmark is in a detached screen session.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse


PROJECT_ROOT = Path(__file__).resolve().parent


HTML = r"""<!doctype html>
<html lang="zh-CN">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Project 2 · AutoDL 实时监控</title>
<style>
:root{color-scheme:dark;--bg:#0b1020;--panel:#141b2e;--line:#283450;--text:#eaf0ff;--muted:#8e9bb8;--blue:#55a7ff;--green:#45d19a;--amber:#f4bd65;--red:#ff7183}
*{box-sizing:border-box}body{margin:0;background:radial-gradient(circle at 10% 0,#17254b 0,#0b1020 43%);color:var(--text);font:14px/1.45 system-ui,-apple-system,Segoe UI,sans-serif}
main{max-width:1240px;margin:0 auto;padding:28px 24px 44px}header{display:flex;justify-content:space-between;gap:20px;align-items:flex-start;margin-bottom:22px}h1{font-size:25px;margin:0 0 5px;letter-spacing:.2px}h2{font-size:15px;margin:0 0 12px}.sub{color:var(--muted)}.pill{border:1px solid var(--line);border-radius:99px;padding:7px 12px;color:var(--green);white-space:nowrap}.pill.warn{color:var(--amber)}.pill.bad{color:var(--red)}
.grid{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:13px;margin-bottom:15px}.card{background:linear-gradient(145deg,#18213a,#11182a);border:1px solid var(--line);border-radius:13px;padding:16px;box-shadow:0 10px 28px #0002}.label{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.7px}.value{font-size:25px;font-weight:700;margin-top:6px}.small{font-size:12px;color:var(--muted);margin-top:3px;word-break:break-word}.wide{grid-column:span 2}.full{grid-column:1/-1}.progress{height:10px;background:#0b1121;border-radius:20px;overflow:hidden;border:1px solid var(--line);margin:14px 0 8px}.bar{height:100%;width:0;background:linear-gradient(90deg,var(--blue),var(--green));transition:width .5s}.row{display:flex;justify-content:space-between;gap:10px}.metric{display:flex;justify-content:space-between;align-items:center;border-bottom:1px solid #26314a;padding:7px 0}.metric:last-child{border-bottom:0}.metric b{font-variant-numeric:tabular-nums}.ok{color:var(--green)}.warn{color:var(--amber)}.bad{color:var(--red)}
table{width:100%;border-collapse:collapse;font-variant-numeric:tabular-nums}th,td{text-align:left;padding:8px 7px;border-bottom:1px solid #26314a}th{color:var(--muted);font-size:12px;font-weight:500}td.num{text-align:right}pre{max-height:190px;overflow:auto;white-space:pre-wrap;color:#b9c6e4;background:#0b1121;border-radius:9px;border:1px solid var(--line);padding:10px;font:12px/1.4 ui-monospace,SFMono-Regular,Consolas,monospace;margin:0}.foot{margin-top:18px;color:var(--muted);font-size:12px;text-align:right}@media(max-width:900px){.grid{grid-template-columns:repeat(2,minmax(0,1fr))}.wide{grid-column:span 2}}@media(max-width:560px){main{padding:18px 12px}.grid{grid-template-columns:1fr}.wide,.full{grid-column:span 1}header{display:block}.pill{display:inline-block;margin-top:12px}}
</style>
</head>
<body><main>
<header><div><h1>Project 2 · AutoDL 实时监控</h1><div class="sub" id="subtitle">正在连接远端监控服务…</div></div><div id="status" class="pill warn">CONNECTING</div></header>
<section class="grid">
  <div class="card"><div class="label">实验阶段</div><div id="stage" class="value">—</div><div id="updated" class="small">—</div></div>
  <div class="card"><div class="label">配置进度</div><div id="progress" class="value">0 / 15</div><div id="progressText" class="small">等待数据</div></div>
  <div class="card"><div class="label">运行时间</div><div id="elapsed" class="value">—</div><div id="proc" class="small">—</div></div>
  <div class="card"><div class="label">GPU</div><div id="gpu" class="value">—</div><div id="gpuSmall" class="small">—</div></div>
  <div class="card wide"><h2>实验进度</h2><div class="row"><span id="progressLabel">已发现的配置</span><b id="percent">0%</b></div><div class="progress"><div id="bar" class="bar"></div></div><div class="small">完整矩阵：DDPM 1 + DDIM 5 + Euler 5 + DPM-Solver-2 4 = 15 个配置。生成预览文件后即可显示进度，最终 FID 写入 JSON。</div></div>
  <div class="card wide"><h2>GPU / 系统</h2><div id="system"><div class="metric"><span>—</span><b>—</b></div></div></div>
  <div class="card full"><h2>已完成 / 最近结果</h2><div id="results"><div class="small">尚未写出 benchmark_all.json；当前阶段可能仍在生成 DDPM 1000 步样本。</div></div></div>
  <div class="card wide"><h2>benchmark 日志尾部</h2><pre id="log">—</pre></div>
  <div class="card wide"><h2>进度监视器尾部</h2><pre id="monitor">—</pre></div>
</section><div class="foot">只读监控 · 自动刷新间隔 2 秒 · 页面不会发送实验控制命令</div>
</main>
<script>
const $=id=>document.getElementById(id);
function esc(s){return String(s??'').replace(/[&<>"']/g,c=>({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]))}
function fmt(n,d=1){return n==null?'—':Number(n).toFixed(d)}
function render(d){
  const p=d.progress||{}, g=d.gpu||{}, proc=d.process||{}; const state=d.state||'unknown';
  $('subtitle').textContent=`${d.host||'AutoDL'} · ${d.project_dir||''}`;
  $('status').textContent=state.toUpperCase(); $('status').className='pill '+(state==='running'?'':'warn');
  $('stage').textContent=d.stage||'—'; $('updated').textContent=`更新于 ${new Date().toLocaleTimeString()}`;
  $('progress').textContent=`${p.done||0} / ${p.total||15}`; $('percent').textContent=`${p.percent||0}%`;
  $('bar').style.width=`${p.percent||0}%`; $('progressText').textContent=p.detail||'等待数据';
  $('elapsed').textContent=proc.elapsed||'—'; $('proc').textContent=proc.pid?`PID ${proc.pid} · ${proc.cpu||0}% CPU · ${proc.stat||''}`:'没有检测到主进程';
  $('gpu').textContent=g.utilization==null?'—':`${fmt(g.utilization,0)}%`;
  $('gpuSmall').textContent=g.memory_used!=null?`${fmt(g.memory_used,0)} / ${fmt(g.memory_total,0)} MiB · ${fmt(g.temperature,0)}°C`:'';
  $('system').innerHTML=[['GPU 利用率',g.utilization==null?'—':`${fmt(g.utilization,0)} %`],['显存',g.memory_used==null?'—':`${fmt(g.memory_used,0)} / ${fmt(g.memory_total,0)} MiB`],['温度',g.temperature==null?'—':`${fmt(g.temperature,0)} °C`],['功耗',g.power==null?'—':`${fmt(g.power,0)} W`],['采样预览',`${p.previews||0} 个`],['JSON 结果',`${p.results||0} 个`]].map(x=>`<div class="metric"><span>${esc(x[0])}</span><b>${esc(x[1])}</b></div>`).join('');
  const rows=(d.results||[]).map(r=>`<tr><td>${esc(r.sampler)}</td><td class="num">${r.num_steps}</td><td class="num">${r.nfe}</td><td class="num">${r.fid==null?'—':fmt(r.fid,3)}</td><td class="num">${r.sample_seconds==null?'—':fmt(r.sample_seconds,1)+'s'}</td></tr>`).join('');
  $('results').innerHTML=rows?`<table><thead><tr><th>Sampler</th><th>Steps</th><th>NFE</th><th>FID</th><th>Sampling</th></tr></thead><tbody>${rows}</tbody></table>`:`<div class="small">尚未写出 benchmark_all.json；当前阶段可能仍在生成 DDPM 1000 步样本。</div>`;
  $('log').textContent=d.log||'（暂无输出；Python 缓冲可能会在配置完成时刷新）'; $('monitor').textContent=d.monitor||'—';
}
async function refresh(){try{const r=await fetch('/api/status?t='+Date.now(),{cache:'no-store'});if(!r.ok)throw Error(r.status);render(await r.json())}catch(e){$('status').textContent='OFFLINE';$('status').className='pill bad';$('subtitle').textContent='无法读取监控服务：'+e}}
refresh();setInterval(refresh,2000);
</script></body></html>"""


def run_text(command: list[str]) -> str:
    try:
        return subprocess.check_output(command, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def tail(path: Path, lines: int = 12) -> str:
    try:
        return "\n".join(path.read_text(encoding="utf-8", errors="replace").splitlines()[-lines:])
    except OSError:
        return ""


def process_info() -> dict:
    raw = run_text(["ps", "-eo", "pid=,etime=,pcpu=,stat=,args="])
    candidates = []
    for line in raw.splitlines():
        if "bash -lc" in line or "SCREEN" in line:
            continue
        # The detached screen/bash wrappers contain the Python filename in
        # their command text too.  Only accept the actual interpreter command
        # so the dashboard reports the real worker rather than the watcher.
        is_worker = bool(
            re.search(
                r"(?:^|/)(?:python|python3)(?:\d+(?:\.\d+)?)?\s+.*(?:benchmark|evaluate_inversion)\.py(?:\s|$)",
                line,
            )
        )
        if is_worker:
            match = re.match(r"\s*(\d+)\s+(\S+)\s+(\S+)\s+(\S+)\s+(.*)", line)
            if match:
                candidates.append(match.groups())
    if not candidates:
        return {}
    pid, elapsed, cpu, stat, args = candidates[-1]
    return {"pid": int(pid), "elapsed": elapsed, "cpu": cpu, "stat": stat, "args": args}


def gpu_info() -> dict:
    raw = run_text([
        "nvidia-smi",
        "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw",
        "--format=csv,noheader,nounits",
    ])
    if not raw:
        return {}
    values = [x.strip() for x in raw.splitlines()[0].split(",")]
    keys = ["utilization", "memory_used", "memory_total", "temperature", "power"]
    result = {}
    for key, value in zip(keys, values):
        try:
            result[key] = float(value)
        except ValueError:
            result[key] = value
    return result


def benchmark_state(project_dir: Path) -> dict:
    runs = project_dir / "runs"
    samples = project_dir / "samples"
    benchmark_path = runs / "benchmark_all.json"
    results = []
    metadata = {}
    if benchmark_path.exists():
        try:
            payload = json.loads(benchmark_path.read_text(encoding="utf-8"))
            results = payload.get("results", [])
            metadata = payload.get("metadata", {})
        except (OSError, json.JSONDecodeError):
            pass
    previews = [p for p in samples.glob("*.png") if p.name != ".gitkeep"] if samples.exists() else []
    total = 15
    done = max(len(results), len(previews))
    process = process_info()
    if process:
        stage = "DDIM inversion" if "evaluate_inversion.py" in process.get("args", "") else ("FID benchmark" if done else "DDPM 1000 步生成")
        state = "running"
    elif results:
        stage, state = "Completed", "complete"
    else:
        stage, state = "Waiting", "waiting"
    return {
        "state": state,
        "stage": stage,
        "host": os.uname().nodename if hasattr(os, "uname") else os.environ.get("COMPUTERNAME", "AutoDL"),
        "project_dir": str(project_dir),
        "process": process,
        "gpu": gpu_info(),
        "progress": {"done": done, "total": total, "results": len(results), "previews": len(previews), "percent": round(100 * done / total), "detail": f"{len(results)} 个已写入 JSON，{len(previews)} 个配置已有采样预览"},
        "results": results,
        "metadata": metadata,
        "log": tail(project_dir / "logs" / "benchmark_all.log"),
        "monitor": tail(project_dir / "logs" / "progress_monitor.log"),
        "timestamp": time.time(),
    }


def make_handler(project_dir: Path):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            path = urlparse(self.path).path
            if path == "/api/status":
                payload = json.dumps(benchmark_state(project_dir), ensure_ascii=False).encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "application/json; charset=utf-8")
                self.send_header("Cache-Control", "no-store")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            if path in {"/", "/index.html"}:
                payload = HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(payload)))
                self.end_headers()
                self.wfile.write(payload)
                return
            self.send_error(404)

        def log_message(self, *_args):
            return

    return Handler


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-dir", default=str(PROJECT_ROOT))
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    args = parser.parse_args()
    project_dir = Path(args.project_dir).expanduser().resolve()
    server = ThreadingHTTPServer((args.host, args.port), make_handler(project_dir))
    print(f"Project 2 dashboard: http://{args.host}:{args.port}/", flush=True)
    print(f"Reading: {project_dir}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
