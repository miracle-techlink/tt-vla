#!/usr/bin/env python3
"""Live training dashboard - serves a web page with real-time loss curves."""

import http.server
import json
import re
import os

PORT = 8899
LOG_FILE = "/tmp/warmup_finetune.log"

HTML = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>TT-VLA Training Monitor</title>
<script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
<style>
  body { font-family: -apple-system, sans-serif; background: #1a1a2e; color: #eee; margin: 0; padding: 20px; }
  .header { text-align: center; margin-bottom: 20px; }
  .header h1 { margin: 0; font-size: 24px; color: #e94560; }
  .header .sub { color: #888; font-size: 14px; }
  .grid { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; max-width: 1200px; margin: 0 auto; }
  .card { background: #16213e; border-radius: 12px; padding: 20px; box-shadow: 0 4px 15px rgba(0,0,0,0.3); }
  .card h2 { margin: 0 0 15px 0; font-size: 16px; color: #e94560; }
  .stat-row { display: flex; justify-content: space-between; padding: 8px 0; border-bottom: 1px solid #1a1a3e; }
  .stat-label { color: #888; }
  .stat-value { font-weight: bold; font-size: 18px; }
  .stat-value.good { color: #4ecca3; }
  .stat-value.warn { color: #f0a500; }
  .progress-bar { background: #1a1a2e; border-radius: 8px; height: 24px; margin: 10px 0; overflow: hidden; }
  .progress-fill { height: 100%; border-radius: 8px; background: linear-gradient(90deg, #e94560, #4ecca3);
                    transition: width 0.5s; display: flex; align-items: center; justify-content: center; font-size: 12px; }
  .full-width { grid-column: 1 / -1; }
  .log-box { background: #0f0f23; border-radius: 8px; padding: 12px; font-family: monospace; font-size: 12px;
             max-height: 150px; overflow-y: auto; color: #4ecca3; white-space: pre-wrap; }
  .update-info { text-align: center; color: #555; font-size: 12px; margin-top: 10px; }
  canvas { max-height: 300px; }
</style>
</head><body>
<div class="header">
  <h1>TT-VLA Warm-up Fine-tuning</h1>
  <div class="sub">OpenVLA-7B + LoRA | ManiSkill3 PickCube-v1 | 8x RTX 5090 DDP</div>
</div>
<div class="grid">
  <div class="card">
    <h2>Training Progress</h2>
    <div class="progress-bar"><div class="progress-fill" id="pbar" style="width:0%">0%</div></div>
    <div class="stat-row"><span class="stat-label">Step</span><span class="stat-value" id="step">-</span></div>
    <div class="stat-row"><span class="stat-label">Loss</span><span class="stat-value good" id="loss">-</span></div>
    <div class="stat-row"><span class="stat-label">Speed</span><span class="stat-value" id="speed">-</span></div>
    <div class="stat-row"><span class="stat-label">ETA</span><span class="stat-value warn" id="eta">-</span></div>
  </div>
  <div class="card">
    <h2>GPU Status</h2>
    <div id="gpu-stats"></div>
  </div>
  <div class="card full-width">
    <h2>Loss Curve</h2>
    <canvas id="lossChart"></canvas>
  </div>
  <div class="card full-width">
    <h2>Recent Log</h2>
    <div class="log-box" id="logbox">Loading...</div>
  </div>
</div>
<div class="update-info">Auto-refresh every 5s | <span id="last-update"></span></div>

<script>
const ctx = document.getElementById('lossChart').getContext('2d');
const chart = new Chart(ctx, {
  type: 'line',
  data: { labels: [], datasets: [{
    label: 'Loss', data: [], borderColor: '#e94560', backgroundColor: 'rgba(233,69,96,0.1)',
    fill: true, tension: 0.3, pointRadius: 0, borderWidth: 2
  }]},
  options: {
    responsive: true, animation: { duration: 300 },
    scales: {
      x: { title: { display: true, text: 'Step', color: '#888' }, ticks: { color: '#888' }, grid: { color: '#1a1a3e' }},
      y: { title: { display: true, text: 'Loss', color: '#888' }, ticks: { color: '#888' }, grid: { color: '#1a1a3e' },
           min: 0 }
    },
    plugins: { legend: { labels: { color: '#eee' }}}
  }
});

async function update() {
  try {
    const r = await fetch('/api/status');
    const d = await r.json();

    // Stats
    const pct = d.total_steps > 0 ? (100 * d.step / d.total_steps).toFixed(1) : 0;
    document.getElementById('step').textContent = d.step + ' / ' + d.total_steps;
    document.getElementById('loss').textContent = d.loss ? d.loss.toFixed(4) : '-';
    document.getElementById('speed').textContent = d.speed ? d.speed.toFixed(1) + ' steps/s' : '-';
    document.getElementById('eta').textContent = d.eta || '-';
    document.getElementById('pbar').style.width = pct + '%';
    document.getElementById('pbar').textContent = pct + '%';

    // GPU
    let gh = '';
    if (d.gpus) {
      d.gpus.forEach(g => {
        const pct = (g.mem_used / g.mem_total * 100).toFixed(0);
        const color = g.util > 80 ? '#4ecca3' : g.util > 50 ? '#f0a500' : '#e94560';
        gh += '<div class="stat-row"><span class="stat-label">GPU ' + g.id + '</span>' +
              '<span style="color:' + color + '">' + g.mem_used + '/' + g.mem_total + 'MB | ' + g.util + '%</span></div>';
      });
    }
    document.getElementById('gpu-stats').innerHTML = gh;

    // Loss curve
    if (d.loss_history && d.loss_history.length > 0) {
      chart.data.labels = d.loss_history.map(x => x[0]);
      chart.data.datasets[0].data = d.loss_history.map(x => x[1]);
      chart.update('none');
    }

    // Log
    if (d.recent_log) document.getElementById('logbox').textContent = d.recent_log;

    document.getElementById('last-update').textContent = 'Updated: ' + new Date().toLocaleTimeString();
  } catch(e) { console.error(e); }
}

update();
setInterval(update, 5000);
</script>
</body></html>"""


class Handler(http.server.BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass  # Suppress request logs

    def do_GET(self):
        if self.path == "/api/status":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Access-Control-Allow-Origin", "*")
            self.end_headers()
            self.wfile.write(json.dumps(get_status()).encode())
        else:
            self.send_response(200)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(HTML.encode())


def get_status():
    data = {"step": 0, "total_steps": 10000, "loss": None, "speed": None,
            "eta": None, "loss_history": [], "gpus": [], "recent_log": ""}

    # Parse log
    try:
        with open(LOG_FILE) as f:
            lines = f.readlines()
        data["recent_log"] = "".join(lines[-20:])

        for line in lines:
            m = re.search(r"Step (\d+)/(\d+) \| Loss: ([\d.]+) \| Speed: ([\d.]+) steps/s \| ETA: ([\d.]+) min", line)
            if m:
                step, total, loss, speed, eta = int(m.group(1)), int(m.group(2)), float(m.group(3)), float(m.group(4)), float(m.group(5))
                data["step"] = step
                data["total_steps"] = total
                data["loss"] = loss
                data["speed"] = speed
                data["eta"] = f"{eta:.0f} min"
                data["loss_history"].append((step, loss))
    except FileNotFoundError:
        data["recent_log"] = "Log file not found"

    # GPU stats
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True, timeout=5)
        for line in out.strip().split("\n"):
            parts = [x.strip() for x in line.split(",")]
            if len(parts) >= 4:
                data["gpus"].append({
                    "id": int(parts[0]), "mem_used": int(parts[1]),
                    "mem_total": int(parts[2]), "util": int(parts[3])
                })
    except Exception:
        pass

    return data


if __name__ == "__main__":
    print(f"Training monitor: http://0.0.0.0:{PORT}")
    server = http.server.HTTPServer(("0.0.0.0", PORT), Handler)
    server.serve_forever()
