"""Live sensor state-machine builder & SSE web viewer for SeqSM.

Run:
    PYTHONPATH=src venv/bin/python examples/live_sensor_viewer.py
Then open http://127.0.0.1:8090 in a browser.
"""

from __future__ import annotations

import argparse
import json
import math
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from seqsm.streaming import StreamingSeqSM

HTML = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8"/>
<title>SeqSM Live Sensor State Machine</title>
<style>
  body { font-family: system-ui, sans-serif; margin: 1.5rem; background: #0f172a; color: #e2e8f0; }
  .grid { display: grid; grid-template-columns: 2fr 1fr; gap: 1.5rem; }
  canvas { background: #1e293b; border-radius: 8px; width: 100%; height: 520px; }
  .card { background: #1e293b; padding: 1rem; border-radius: 8px; }
  .bar { height: 10px; background: #38bdf8; border-radius: 4px; }
  table { width: 100%; border-collapse: collapse; font-size: 0.9rem; }
  td, th { padding: 6px 4px; border-bottom: 1px solid #334155; text-align: left; }
</style>
</head>
<body>
<h2>SeqSM Live Sensor State Machine</h2>
<p id="status">Connecting...</p>
<div class="grid">
  <canvas id="graph" width="800" height="520"></canvas>
  <div class="card">
    <h3>Current State</h3>
    <pre id="current">-</pre>
    <h3>Next Predicted Point</h3>
    <table><thead><tr><th>State</th><th>Prob</th><th></th></tr></thead><tbody id="preds"></tbody></table>
  </div>
</div>
<script>
const canvas = document.getElementById('graph');
const ctx = canvas.getContext('2d');
function drawGraph(g) {
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  const nodes = g.nodes || [], edges = g.edges || [];
  if (!nodes.length) return;
  const xs = nodes.map(n => n.x), ys = nodes.map(n => n.y);
  const minX = Math.min(...xs), maxX = Math.max(...xs) + 1e-6;
  const minY = Math.min(...ys), maxY = Math.max(...ys) + 1e-6;
  const pad = 60;
  const mapX = x => pad + ((x - minX) / (maxX - minX || 1)) * (canvas.width - 2 * pad);
  const mapY = y => pad + ((y - minY) / (maxY - minY || 1)) * (canvas.height - 2 * pad);
  ctx.strokeStyle = '#64748b';
  for (const e of edges) {
    ctx.lineWidth = Math.max(1, e.probability * 4);
    ctx.beginPath();
    ctx.moveTo(mapX(e.source_pos[0]), mapY(e.source_pos[1]));
    ctx.lineTo(mapX(e.target_pos[0]), mapY(e.target_pos[1]));
    ctx.stroke();
  }
  for (const n of nodes) {
    const cx = mapX(n.x), cy = mapY(n.y);
    ctx.fillStyle = '#38bdf8';
    ctx.beginPath();
    ctx.arc(cx, cy, 14, 0, Math.PI * 2);
    ctx.fill();
    ctx.fillStyle = '#f8fafc';
    ctx.font = '12px monospace';
    ctx.fillText(n.label, cx + 18, cy + 4);
  }
}
const es = new EventSource('/events');
es.onmessage = (ev) => {
  const d = JSON.parse(ev.data);
  document.getElementById('status').textContent = `Event #${d.event} | Nodes: ${d.graph.nodes.length} | Edges: ${d.graph.edges.length}`;
  document.getElementById('current').textContent = JSON.stringify(d.current, null, 2);
  const tbody = document.getElementById('preds');
  tbody.innerHTML = (d.predictions || []).map(p =>
    `<tr><td>${JSON.stringify(p.state)}</td><td>${(p.probability*100).toFixed(1)}%</td><td><div class="bar" style="width:${Math.round(p.probability*100)}px"></div></td></tr>`
  ).join('');
  drawGraph(d.graph);
};
</script>
</body>
</html>"""


def _read_linux_sensors(prev_stat: list[int] | None) -> tuple[dict[str, float], list[int] | None]:
    stat_line = Path("/proc/stat").read_text(encoding="utf-8").splitlines()[0].split()[1:]
    vals = [int(v) for v in stat_line]
    cpu_pct = 0.0
    if prev_stat is not None:
        total_d = max(1, sum(vals) - sum(prev_stat))
        idle_d = vals[3] - prev_stat[3]
        cpu_pct = 100.0 * (1.0 - idle_d / total_d)

    mem_info: dict[str, int] = {}
    for line in Path("/proc/meminfo").read_text(encoding="utf-8").splitlines()[:5]:
        k, v = line.split(":", 1)
        mem_info[k] = int(v.strip().split()[0])
    mem_pct = 100.0 * (1.0 - mem_info["MemAvailable"] / max(1, mem_info["MemTotal"]))
    return {"cpu": round(cpu_pct, 1), "mem": round(mem_pct, 1)}, vals


def make_handler(stream: StreamingSeqSM, interval: float, simulate: bool):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            if self.path == "/":
                body = HTML.encode("utf-8")
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return

            if self.path == "/events":
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()
                prev_stat: list[int] | None = None
                step = 0
                try:
                    while True:
                        if simulate or not Path("/proc/stat").exists():
                            sample = {
                                "cpu": 50.0 + 40.0 * math.sin(step * 0.4),
                                "mem": 60.0 + 20.0 * math.cos(step * 0.25),
                            }
                        else:
                            sample, prev_stat = _read_linux_sensors(prev_stat)
                        payload = stream.update(sample)
                        self.wfile.write(f"data: {json.dumps(payload)}\n\n".encode())
                        self.wfile.flush()
                        step += 1
                        time.sleep(interval)
                except (BrokenPipeError, ConnectionResetError):
                    return

            self.send_error(404)

    return Handler


def main() -> None:
    parser = argparse.ArgumentParser(description="Live sensor SeqSM viewer.")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8090)
    parser.add_argument("--interval", type=float, default=0.5)
    parser.add_argument("--simulate", action="store_true")
    args = parser.parse_args()

    stream = StreamingSeqSM(
        k=2,
        mode="clock",
        decay=0.98,
        bins={"cpu": [0, 20, 50, 80, 100], "mem": [0, 40, 70, 90, 100]},
    )
    server = ThreadingHTTPServer((args.host, args.port), make_handler(stream, args.interval, args.simulate))
    print(f"Serving SeqSM live viewer at http://{args.host}:{args.port}")
    server.serve_forever()


if __name__ == "__main__":
    main()
