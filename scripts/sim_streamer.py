#!/usr/bin/env python3
"""
ManiSkill3 / Isaac Sim Real-time Streaming Server

Runs simulation headless on GPU, encodes frames as JPEG,
streams via WebSocket to browser for real-time visualization.

Architecture:
  GPU: ManiSkill3 env (headless) → RGB numpy → JPEG encode (OpenCV) → WebSocket
  Browser: WebSocket → decode → <canvas> render + control panel

Usage:
  python scripts/sim_streamer.py                          # Default: random actions
  python scripts/sim_streamer.py --policy scripted        # Scripted expert
  python scripts/sim_streamer.py --policy model --checkpoint warmup_ckpt/step_3000.pt  # Model inference
  python scripts/sim_streamer.py --port 8900 --fps 15     # Custom port/fps
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import argparse
import asyncio
import json
import io
import time
import threading
import http.server
import numpy as np
import torch
import cv2
import gymnasium as gym
import mani_skill.envs  # noqa
import websockets
from PIL import Image


# ── Scripted Policy ───────────────────────────────────────────────────

class ScriptedPickCubePolicy:
    def __init__(self):
        self.phase = "reach"
        self.grasp_steps = 0

    def reset(self):
        self.phase = "reach"
        self.grasp_steps = 0

    def act(self, obs, info):
        tcp = obs["extra"]["tcp_pose"].squeeze().cpu().numpy()
        obj = obs["extra"]["obj_pose"].squeeze().cpu().numpy()
        goal = obs["extra"]["goal_pos"].squeeze().cpu().numpy()
        ig = info.get("is_grasped", False)
        if isinstance(ig, torch.Tensor):
            ig = ig.item()
        tp, op = tcp[:3], obj[:3]
        a = np.zeros(7, dtype=np.float32)

        if self.phase == "reach":
            t = op.copy(); t[2] += 0.05; d = t - tp
            if np.linalg.norm(d) < 0.015: self.phase = "descend"
            a[:3] = np.clip(d * 10, -1, 1); a[6] = 1.0
        elif self.phase == "descend":
            d = op - tp; d[2] -= 0.005
            if np.linalg.norm(d) < 0.012: self.phase = "grasp"; self.grasp_steps = 0
            a[:3] = np.clip(d * 10, -1, 1); a[6] = 1.0
        elif self.phase == "grasp":
            self.grasp_steps += 1; a[6] = -1.0
            if self.grasp_steps > 15: self.phase = "lift" if ig else "reach"
        elif self.phase == "lift":
            if not ig: self.phase = "reach"
            else:
                t = tp.copy(); t[2] = goal[2] + 0.1
                if tp[2] > goal[2] + 0.05: self.phase = "move"
                a[:3] = np.clip((t - tp) * 10, -1, 1)
            a[6] = -1.0
        elif self.phase == "move":
            if not ig: self.phase = "reach"
            else:
                t = goal.copy(); t[2] += 0.05
                if np.linalg.norm((t - tp)[:2]) < 0.02: self.phase = "place"
                a[:3] = np.clip((t - tp) * 10, -1, 1)
            a[6] = -1.0
        elif self.phase == "place":
            d = goal - tp
            if np.linalg.norm(d) < 0.02: a[6] = 1.0
            else: a[:3] = np.clip(d * 10, -1, 1); a[6] = -1.0
        return a


# ── Simulation Thread ─────────────────────────────────────────────────

class SimulationEngine:
    """Runs simulation in a separate thread, provides frames to streamer."""

    def __init__(self, args):
        self.args = args
        self.frame = None  # Latest JPEG bytes
        self.frame_lock = threading.Lock()
        self.stats = {
            "episode": 0, "step": 0, "reward": 0.0, "total_reward": 0.0,
            "success": False, "fps": 0.0, "phase": "init",
            "policy": args.policy, "running": False, "paused": False,
        }
        self.stats_lock = threading.Lock()
        self.command_queue = []
        self.command_lock = threading.Lock()
        self._running = True

    def send_command(self, cmd):
        with self.command_lock:
            self.command_queue.append(cmd)

    def _process_commands(self):
        with self.command_lock:
            cmds = self.command_queue[:]
            self.command_queue.clear()
        for cmd in cmds:
            if cmd == "reset":
                return "reset"
            elif cmd == "pause":
                with self.stats_lock:
                    self.stats["paused"] = not self.stats["paused"]
            elif cmd == "stop":
                self._running = False
        return None

    def run(self):
        args = self.args
        obs_mode = "rgb+state_dict" if args.policy == "scripted" else "rgb"

        print(f"Creating env: PickCube-v1 (obs={obs_mode}, control=pd_ee_delta_pose)")
        env = gym.make(
            "PickCube-v1", obs_mode=obs_mode, render_mode="cameras",
            control_mode="pd_ee_delta_pose", max_episode_steps=args.max_steps,
        )

        policy = None
        if args.policy == "scripted":
            policy = ScriptedPickCubePolicy()

        model = None
        if args.policy == "model":
            model = self._load_model(args)

        with self.stats_lock:
            self.stats["running"] = True
            self.stats["phase"] = "running"

        episode = 0
        while self._running:
            episode += 1
            obs, info = env.reset(seed=episode * 42)
            if policy:
                policy.reset()

            total_reward = 0.0
            success = False
            t0 = time.time()
            frame_times = []

            with self.stats_lock:
                self.stats["episode"] = episode
                self.stats["step"] = 0
                self.stats["total_reward"] = 0.0
                self.stats["success"] = False

            for step in range(args.max_steps):
                if not self._running:
                    break

                # Check commands
                cmd = self._process_commands()
                if cmd == "reset":
                    break

                # Pause
                with self.stats_lock:
                    paused = self.stats["paused"]
                if paused:
                    time.sleep(0.1)
                    continue

                ft0 = time.time()

                # Extract RGB
                rgb = self._extract_rgb(obs)

                # Encode frame
                self._encode_and_store_frame(rgb, step, episode, total_reward, success)

                # Get action
                if args.policy == "random":
                    action = env.action_space.sample()
                elif args.policy == "scripted":
                    action = policy.act(obs, info)
                elif args.policy == "model" and model is not None:
                    action = self._model_action(model, rgb, args)
                else:
                    action = env.action_space.sample()

                # Step
                obs, reward, term, trunc, info = env.step(action)
                total_reward += float(reward)
                done = bool(term) or bool(trunc)

                s = info.get("success", False)
                if isinstance(s, torch.Tensor):
                    s = s.item()
                if s:
                    success = True

                # Update stats
                frame_times.append(time.time() - ft0)
                with self.stats_lock:
                    self.stats["step"] = step + 1
                    self.stats["reward"] = float(reward)
                    self.stats["total_reward"] = total_reward
                    self.stats["success"] = success
                    if len(frame_times) > 10:
                        self.stats["fps"] = 1.0 / np.mean(frame_times[-10:])

                # FPS limiter
                target_dt = 1.0 / args.fps
                elapsed = time.time() - ft0
                if elapsed < target_dt:
                    time.sleep(target_dt - elapsed)

                if done:
                    break

            elapsed = time.time() - t0
            print(f"  Ep {episode}: {'SUCCESS' if success else 'FAIL'} | "
                  f"reward={total_reward:.2f} | steps={step+1} | time={elapsed:.1f}s")

        env.close()
        with self.stats_lock:
            self.stats["running"] = False
            self.stats["phase"] = "stopped"

    def _extract_rgb(self, obs):
        if isinstance(obs, dict) and "sensor_data" in obs:
            for cam in obs["sensor_data"]:
                rgb = obs["sensor_data"][cam]["rgb"]
                if isinstance(rgb, torch.Tensor):
                    rgb = rgb.squeeze(0).cpu().numpy()
                return rgb
        return np.zeros((128, 128, 3), dtype=np.uint8)

    def _encode_and_store_frame(self, rgb, step, episode, reward, success):
        """Encode RGB frame to JPEG with OSD overlay."""
        # Resize for streaming
        h, w = rgb.shape[:2]
        target_w, target_h = 640, 480
        if h != target_h or w != target_w:
            rgb = cv2.resize(rgb, (target_w, target_h), interpolation=cv2.INTER_LINEAR)

        # Convert RGB to BGR for OpenCV
        frame = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

        # OSD overlay
        cv2.putText(frame, f"Ep:{episode} Step:{step}", (10, 25),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        cv2.putText(frame, f"R:{reward:.2f}", (10, 50),
                     cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        if success:
            cv2.putText(frame, "SUCCESS!", (target_w - 150, 30),
                         cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

        # Encode JPEG
        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 75])

        with self.frame_lock:
            self.frame = jpeg.tobytes()

    def _load_model(self, args):
        """Load OpenVLA model for inference."""
        if not args.checkpoint:
            print("WARNING: --policy model requires --checkpoint, falling back to random")
            return None
        try:
            from transformers import AutoModelForVision2Seq, AutoProcessor
            from peft import LoraConfig, get_peft_model

            print(f"Loading model for streaming inference...")
            processor = AutoProcessor.from_pretrained("openvla/openvla-7b", trust_remote_code=True)
            model = AutoModelForVision2Seq.from_pretrained(
                "openvla/openvla-7b", torch_dtype=torch.bfloat16,
                trust_remote_code=True, low_cpu_mem_usage=True, device_map="auto",
            )
            lora_config = LoraConfig(
                r=32, lora_alpha=32,
                target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                                "gate_proj", "up_proj", "down_proj"],
                lora_dropout=0.0, bias="none", task_type="CAUSAL_LM",
            )
            model = get_peft_model(model, lora_config)
            ckpt = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
            model.load_state_dict(ckpt["model_state_dict"], strict=False)
            model.eval()
            self._model = model
            self._processor = processor
            print("Model loaded for streaming")
            return model
        except Exception as e:
            print(f"Failed to load model: {e}")
            return None

    def _model_action(self, model, rgb, args):
        """Get action from model."""
        try:
            pil = Image.fromarray(rgb).resize((224, 224), Image.BILINEAR)
            prompt = "In: What action should the robot take to pick up the cube and place it on the target?\nOut:"
            device = next(model.parameters()).device
            inputs = self._processor(prompt, pil).to(device, dtype=torch.bfloat16)
            with torch.no_grad():
                output = model.generate(**inputs, max_new_tokens=7, do_sample=False)
            input_len = inputs["input_ids"].shape[1]
            tokens = output[0, input_len:input_len + 7].cpu().numpy()

            bins = np.linspace(-1, 1, 256)
            bin_centers = (bins[:-1] + bins[1:]) / 2.0
            action = []
            for t in tokens[:7]:
                idx = 32000 - 1 - int(t)
                idx = max(0, min(len(bin_centers) - 1, idx))
                action.append(bin_centers[idx])
            return np.array(action, dtype=np.float32)
        except Exception as e:
            print(f"Model inference error: {e}")
            return np.zeros(7, dtype=np.float32)


# ── WebSocket Streaming ───────────────────────────────────────────────

async def ws_handler(websocket, engine):
    """Handle WebSocket connection — stream frames + receive commands."""
    print(f"Client connected: {websocket.remote_address}")
    try:
        # Start send/recv tasks
        send_task = asyncio.create_task(ws_send_frames(websocket, engine))
        recv_task = asyncio.create_task(ws_recv_commands(websocket, engine))
        done, pending = await asyncio.wait(
            [send_task, recv_task], return_when=asyncio.FIRST_COMPLETED
        )
        for task in pending:
            task.cancel()
    except websockets.exceptions.ConnectionClosed:
        pass
    print(f"Client disconnected: {websocket.remote_address}")


async def ws_send_frames(websocket, engine):
    """Send JPEG frames and stats via WebSocket."""
    while True:
        with engine.frame_lock:
            frame = engine.frame
        with engine.stats_lock:
            stats = dict(engine.stats)

        if frame:
            # Send stats as JSON text message
            await websocket.send(json.dumps({"type": "stats", "data": stats}))
            # Send frame as binary message
            await websocket.send(frame)

        await asyncio.sleep(1.0 / engine.args.fps)


async def ws_recv_commands(websocket, engine):
    """Receive commands from browser."""
    async for message in websocket:
        try:
            cmd = json.loads(message)
            action = cmd.get("action", "")
            if action in ("reset", "pause", "stop"):
                engine.send_command(action)
                print(f"Command received: {action}")
        except json.JSONDecodeError:
            pass


# ── HTTP Server (serves the viewer page) ──────────────────────────────

VIEWER_HTML = """<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>TT-VLA Simulation Streamer</title>
<style>
  * { margin: 0; padding: 0; box-sizing: border-box; }
  body { background: #0a0a1a; color: #eee; font-family: -apple-system, sans-serif; }
  .container { display: flex; height: 100vh; }
  .main { flex: 1; display: flex; flex-direction: column; align-items: center; justify-content: center; }
  .sidebar { width: 300px; background: #12122a; padding: 20px; overflow-y: auto; border-left: 1px solid #1a1a3e; }

  canvas { border: 2px solid #333; border-radius: 8px; background: #000; max-width: 100%; }

  h1 { font-size: 18px; color: #e94560; margin-bottom: 15px; text-align: center; }
  h2 { font-size: 14px; color: #4ecca3; margin: 15px 0 8px 0; text-transform: uppercase; letter-spacing: 1px; }

  .stat { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #1a1a2e; }
  .stat-label { color: #888; font-size: 13px; }
  .stat-value { font-weight: bold; font-size: 14px; }
  .stat-value.good { color: #4ecca3; }
  .stat-value.bad { color: #e94560; }
  .stat-value.warn { color: #f0a500; }

  .btn-group { display: flex; gap: 8px; margin: 12px 0; flex-wrap: wrap; }
  .btn { padding: 8px 16px; border: none; border-radius: 6px; cursor: pointer; font-size: 13px;
         font-weight: bold; transition: all 0.2s; }
  .btn:hover { transform: translateY(-1px); }
  .btn-green { background: #4ecca3; color: #000; }
  .btn-red { background: #e94560; color: #fff; }
  .btn-yellow { background: #f0a500; color: #000; }
  .btn-blue { background: #3a86ff; color: #fff; }

  .latency { font-size: 11px; color: #555; text-align: center; margin-top: 8px; }

  .log-box { background: #0a0a15; border-radius: 6px; padding: 10px; font-family: monospace;
             font-size: 11px; max-height: 200px; overflow-y: auto; color: #4ecca3; margin-top: 10px;
             white-space: pre-wrap; }

  .connection { display: flex; align-items: center; gap: 6px; margin-bottom: 10px; }
  .dot { width: 8px; height: 8px; border-radius: 50%; }
  .dot.on { background: #4ecca3; box-shadow: 0 0 6px #4ecca3; }
  .dot.off { background: #e94560; }
</style>
</head><body>
<div class="container">
  <div class="main">
    <canvas id="viewer" width="640" height="480"></canvas>
    <div class="latency" id="latency">Connecting...</div>
  </div>
  <div class="sidebar">
    <h1>TT-VLA Sim Stream</h1>

    <div class="connection">
      <div class="dot" id="connDot"></div>
      <span id="connText" style="font-size:12px; color:#888;">Disconnected</span>
    </div>

    <h2>Controls</h2>
    <div class="btn-group">
      <button class="btn btn-green" onclick="sendCmd('reset')">Reset</button>
      <button class="btn btn-yellow" onclick="sendCmd('pause')">Pause</button>
      <button class="btn btn-red" onclick="sendCmd('stop')">Stop</button>
    </div>

    <h2>Simulation</h2>
    <div class="stat"><span class="stat-label">Episode</span><span class="stat-value" id="s-episode">-</span></div>
    <div class="stat"><span class="stat-label">Step</span><span class="stat-value" id="s-step">-</span></div>
    <div class="stat"><span class="stat-label">Reward</span><span class="stat-value good" id="s-reward">-</span></div>
    <div class="stat"><span class="stat-label">Total Reward</span><span class="stat-value" id="s-total">-</span></div>
    <div class="stat"><span class="stat-label">Success</span><span class="stat-value" id="s-success">-</span></div>
    <div class="stat"><span class="stat-label">Policy</span><span class="stat-value" id="s-policy">-</span></div>

    <h2>Performance</h2>
    <div class="stat"><span class="stat-label">Sim FPS</span><span class="stat-value" id="s-fps">-</span></div>
    <div class="stat"><span class="stat-label">Stream Latency</span><span class="stat-value" id="s-latency">-</span></div>
    <div class="stat"><span class="stat-label">Frame Size</span><span class="stat-value" id="s-fsize">-</span></div>
    <div class="stat"><span class="stat-label">Bandwidth</span><span class="stat-value" id="s-bw">-</span></div>

    <h2>Log</h2>
    <div class="log-box" id="logbox">Waiting for connection...</div>
  </div>
</div>

<script>
const canvas = document.getElementById('viewer');
const ctx = canvas.getContext('2d');
let ws = null;
let frameCount = 0;
let byteCount = 0;
let lastBwCheck = Date.now();

function connect() {
  const host = location.hostname || 'localhost';
  const wsUrl = 'ws://' + host + ':WS_PORT';
  log('Connecting to ' + wsUrl + '...');

  ws = new WebSocket(wsUrl);
  ws.binaryType = 'arraybuffer';

  ws.onopen = () => {
    document.getElementById('connDot').className = 'dot on';
    document.getElementById('connText').textContent = 'Connected';
    log('Connected!');
  };

  ws.onclose = () => {
    document.getElementById('connDot').className = 'dot off';
    document.getElementById('connText').textContent = 'Disconnected';
    log('Disconnected. Reconnecting in 3s...');
    setTimeout(connect, 3000);
  };

  ws.onerror = (e) => { log('WebSocket error'); };

  ws.onmessage = (event) => {
    if (typeof event.data === 'string') {
      // JSON stats
      try {
        const msg = JSON.parse(event.data);
        if (msg.type === 'stats') updateStats(msg.data);
      } catch(e) {}
    } else {
      // Binary JPEG frame
      const t0 = performance.now();
      const blob = new Blob([event.data], {type: 'image/jpeg'});
      const url = URL.createObjectURL(blob);
      const img = new window.Image();
      img.onload = () => {
        ctx.drawImage(img, 0, 0, canvas.width, canvas.height);
        URL.revokeObjectURL(url);
        const lat = (performance.now() - t0).toFixed(1);
        document.getElementById('s-latency').textContent = lat + 'ms';
      };
      img.src = url;

      frameCount++;
      byteCount += event.data.byteLength;
      document.getElementById('s-fsize').textContent = (event.data.byteLength / 1024).toFixed(1) + 'KB';

      // Bandwidth calc every 2s
      const now = Date.now();
      if (now - lastBwCheck > 2000) {
        const bw = (byteCount * 8 / (now - lastBwCheck) * 1000 / 1e6).toFixed(2);
        document.getElementById('s-bw').textContent = bw + ' Mbps';
        byteCount = 0;
        lastBwCheck = now;
      }
    }
  };
}

function updateStats(d) {
  document.getElementById('s-episode').textContent = d.episode || '-';
  document.getElementById('s-step').textContent = d.step || '-';
  document.getElementById('s-reward').textContent = d.reward != null ? d.reward.toFixed(4) : '-';
  document.getElementById('s-total').textContent = d.total_reward != null ? d.total_reward.toFixed(2) : '-';

  const suc = document.getElementById('s-success');
  suc.textContent = d.success ? 'YES' : 'NO';
  suc.className = 'stat-value ' + (d.success ? 'good' : 'bad');

  document.getElementById('s-policy').textContent = d.policy || '-';
  document.getElementById('s-fps').textContent = d.fps ? d.fps.toFixed(1) : '-';
}

function sendCmd(action) {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({action: action}));
    log('Sent: ' + action);
  }
}

function log(msg) {
  const box = document.getElementById('logbox');
  const ts = new Date().toLocaleTimeString();
  box.textContent += ts + ' ' + msg + '\\n';
  box.scrollTop = box.scrollHeight;
}

document.getElementById('latency').textContent = 'Initializing...';
connect();
</script>
</body></html>"""


class ViewerHTTPHandler(http.server.BaseHTTPRequestHandler):
    ws_port = 8901

    def log_message(self, *args):
        pass

    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html")
        self.end_headers()
        html = VIEWER_HTML.replace("WS_PORT", str(self.ws_port))
        self.wfile.write(html.encode())


# ── Main ──────────────────────────────────────────────────────────────

async def start_ws_server(engine, port):
    async with websockets.serve(
        lambda ws: ws_handler(ws, engine),
        "0.0.0.0", port, max_size=2**20,
    ):
        print(f"WebSocket server: ws://0.0.0.0:{port}")
        await asyncio.Future()  # run forever


def main():
    parser = argparse.ArgumentParser(description="ManiSkill3 Simulation Streamer")
    parser.add_argument("--policy", choices=["random", "scripted", "model"], default="scripted")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--http-port", type=int, default=8900)
    parser.add_argument("--ws-port", type=int, default=8901)
    parser.add_argument("--fps", type=int, default=15)
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--gpu", type=int, default=None, help="GPU ID to use")
    args = parser.parse_args()

    if args.gpu is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)

    # Create simulation engine
    engine = SimulationEngine(args)

    # Start simulation in background thread
    sim_thread = threading.Thread(target=engine.run, daemon=True)
    sim_thread.start()

    # Start HTTP server for viewer page
    ViewerHTTPHandler.ws_port = args.ws_port
    http_server = http.server.HTTPServer(("0.0.0.0", args.http_port), ViewerHTTPHandler)
    http_thread = threading.Thread(target=http_server.serve_forever, daemon=True)
    http_thread.start()
    print(f"Viewer page:    http://0.0.0.0:{args.http_port}")

    # Start WebSocket server (main async loop)
    asyncio.run(start_ws_server(engine, args.ws_port))


if __name__ == "__main__":
    main()
