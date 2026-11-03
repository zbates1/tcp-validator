#!/usr/bin/env python3
"""
MJPEG stream server for Raspberry Pi 5 with dual cameras.

Copy to the Pi and run:
    python3 pi_stream_server.py

Streams (served on port 8080):
    http://<pi-ip>:8080/          -- browser preview + record button
    http://<pi-ip>:8080/cam0.mjpg -- XY (bottom) camera
    http://<pi-ip>:8080/cam1.mjpg -- Side camera

Recordings are downloaded directly to your browser (no files kept on the Pi).
Requires: python3-picamera2  python3-opencv  psutil
    sudo apt install -y python3-picamera2 python3-opencv
    pip3 install psutil
"""

import json
import os
import tempfile
import time
import threading
import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import cv2
from picamera2 import Picamera2

# ─── Config ────────────────────────────────────────────────────────────────────
PORT           = 8080
RESOLUTION     = (1280, 720)
FRAMERATE      = 15
JPEG_QUAL      = 75
STATS_INTERVAL = 2.0                              # seconds between stat lines
ZOOM_MAX       = 8.0                              # maximum scroll zoom level
# ───────────────────────────────────────────────────────────────────────────────

_locks     = [threading.Lock(), threading.Lock()]
_frames    = [None, None]
_available = []                                   # cam indices actually present

# ── Recording directory (changeable from browser) ──────────────────────────────
_rec_dir     = tempfile.gettempdir()              # default: /tmp
_rec_dir_lock = threading.Lock()

# ── Recording state ────────────────────────────────────────────────────────────
_rec_lock        = threading.Lock()
_recorders:  dict[int, cv2.VideoWriter] = {}
_rec_tmpfiles: dict[int, str] = {}          # cam_idx -> temp file path
_rec_zoom:    dict[int, float] = {}         # zoom level snapshotted at record-start
_recording       = False
_rec_start_time: float | None = None
_rec_ts          = ""                       # timestamp string for download filename

# ── Per-camera zoom levels (set via scroll wheel in browser) ──────────────────
_zoom      = [1.0, 1.0]
_zoom_lock = threading.Lock()

# ── Per-camera frame counters (for FPS) ───────────────────────────────────────
_frame_counts = [0, 0]


# ─── Stats thread ──────────────────────────────────────────────────────────────

def _cpu_temp() -> float:
    try:
        return int(Path("/sys/class/thermal/thermal_zone0/temp").read_text()) / 1000.0
    except Exception:
        return 0.0


def _supply_voltage() -> str:
    """Read Pi 5 supply voltage and throttle flags from vcgencmd."""
    import subprocess
    try:
        v = subprocess.check_output(
            ["vcgencmd", "measure_volts", "core"], text=True
        ).strip()          # e.g. "volt=0.8688V"
        volts = float(v.split("=")[1].rstrip("V"))

        flags_raw = subprocess.check_output(
            ["vcgencmd", "get_throttled"], text=True
        ).strip()          # e.g. "throttled=0x50000"
        flags = int(flags_raw.split("=")[1], 16)

        # Bit 0 = under-voltage detected, bit 1 = freq cap, bit 2 = throttled
        warn = ""
        if flags & 0x1:
            warn += " ⚠UNDERVOLT"
        if flags & 0x4:
            warn += " ⚠THROTTLED"
        if flags & 0x2:
            warn += " ⚠FREQCAP"

        return f"{volts:.3f}V{warn}"
    except Exception:
        return "n/a"


def _stats_loop():
    try:
        import psutil
    except ImportError:
        print("(psutil not installed — no live stats. Run: pip3 install psutil)")
        return

    prev_net    = psutil.net_io_counters()
    prev_time   = time.time()
    prev_counts = list(_frame_counts)

    while True:
        time.sleep(STATS_INTERVAL)
        now = time.time()
        dt  = max(now - prev_time, 0.001)

        cpu   = psutil.cpu_percent(interval=None)
        mem   = psutil.virtual_memory()
        temp  = _cpu_temp()
        net   = psutil.net_io_counters()
        volts = _supply_voltage()

        up_kb = (net.bytes_sent - prev_net.bytes_sent) / dt / 1024
        dn_kb = (net.bytes_recv - prev_net.bytes_recv) / dt / 1024

        fps_parts = []
        for idx in _available:
            fps = (_frame_counts[idx] - prev_counts[idx]) / dt
            fps_parts.append(f"cam{idx} {fps:.1f}fps")

        rec_tag = ""
        with _rec_lock:
            if _recording and _rec_start_time:
                rec_tag = f"  ● REC {int(now - _rec_start_time)}s"

        print(
            f"CPU {cpu:4.1f}%  {temp:.0f}°C  {volts}  "
            f"RAM {mem.used >> 20}/{mem.total >> 20}MB  "
            f"↑{up_kb:.0f}kB/s ↓{dn_kb:.0f}kB/s  "
            + "  ".join(fps_parts)
            + rec_tag,
            flush=True,
        )

        prev_net    = net
        prev_time   = now
        prev_counts = list(_frame_counts)


# ─── Capture threads ───────────────────────────────────────────────────────────

def _capture_loop(cam_idx: int):
    cam = Picamera2(cam_idx)
    cam.configure(cam.create_video_configuration(
        main={"size": RESOLUTION, "format": "RGB888"}
    ))
    cam.start()
    delay = 1.0 / FRAMERATE

    while True:
        arr = cam.capture_array()
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        # Apply zoom before recording so saved video matches what you see
        with _zoom_lock:
            zoom = _zoom[cam_idx]
        if zoom > 1.0:
            h, w = bgr.shape[:2]
            ch, cw = h // 2, w // 2
            crop_h, crop_w = int(h / zoom), int(w / zoom)
            y1, y2 = ch - crop_h // 2, ch + crop_h // 2
            x1, x2 = cw - crop_w // 2, cw + crop_w // 2
            disp = cv2.resize(bgr[y1:y2, x1:x2], (w, h), interpolation=cv2.INTER_LINEAR)
        else:
            disp = bgr.copy()

        with _rec_lock:
            if _recording and cam_idx in _recorders:
                _recorders[cam_idx].write(disp)

        # No crosshair drawn here: the browser page overlays one, and servo_to_center.py needs clean frames
        ok, enc = cv2.imencode('.jpg', disp, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUAL])
        if ok:
            with _locks[cam_idx]:
                _frames[cam_idx] = enc.tobytes()

        _frame_counts[cam_idx] += 1
        time.sleep(delay)


# ─── Recording control ─────────────────────────────────────────────────────────

def _set_rec_dir(path: str) -> dict:
    global _rec_dir
    with _rec_dir_lock:
        if _recording:
            return {"status": "error", "message": "Cannot change directory while recording"}
        p = Path(path).expanduser()
        if not p.exists():
            try:
                p.mkdir(parents=True)
            except Exception as e:
                return {"status": "error", "message": str(e)}
        if not os.access(p, os.W_OK):
            return {"status": "error", "message": f"{p} is not writable"}
        _rec_dir = str(p)
        print(f"Recording directory set to: {_rec_dir}", flush=True)
        return {"status": "ok", "path": _rec_dir}


def _start_recording() -> dict:
    global _recording, _rec_start_time, _rec_ts
    with _rec_lock:
        if _recording:
            return {"status": "already_recording"}
        _rec_ts  = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        fourcc   = cv2.VideoWriter_fourcc(*'mp4v')
        with _rec_dir_lock:
            save_dir = _rec_dir
        with _zoom_lock:
            for idx in _available:
                _rec_zoom[idx] = _zoom[idx]
        for idx in _available:
            fd, path = tempfile.mkstemp(suffix=f"_cam{idx}.mp4", dir=save_dir)
            os.close(fd)
            _rec_tmpfiles[idx] = path
            _recorders[idx]    = cv2.VideoWriter(path, fourcc, FRAMERATE, RESOLUTION)
        _recording      = True
        _rec_start_time = time.time()
        zoom_info = {idx: _rec_zoom[idx] for idx in _available}
        print(f"\n● Recording started → {save_dir}  zoom={zoom_info}\n", flush=True)
        return {"status": "started", "dir": save_dir, "zoom": zoom_info}


def _stop_recording() -> dict:
    global _recording, _rec_start_time
    with _rec_lock:
        if not _recording:
            return {"status": "not_recording"}
        _recording = False
        elapsed    = time.time() - (_rec_start_time or 0)
        for w in _recorders.values():
            w.release()
        _recorders.clear()
        _rec_start_time = None
        # Return which cam indices have files ready to download
        ready = list(_rec_tmpfiles.keys())
        print(f"\n■ Recording stopped ({elapsed:.1f}s) — ready for download\n", flush=True)
        return {"status": "stopped", "duration": round(elapsed, 1), "cams": ready}


# ─── HTTP handler ──────────────────────────────────────────────────────────────

class _Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        pass  # suppress per-request noise

    def do_GET(self):
        routes = {
            '/cam0.mjpg':         lambda: self._mjpeg(0),
            '/cam1.mjpg':         lambda: self._mjpeg(1),
            '/record/start':      lambda: self._json(_start_recording()),
            '/record/stop':       lambda: self._json(_stop_recording()),
            '/record/status':     self._rec_status,
            '/record/download/0': lambda: self._download(0),
            '/record/download/1': lambda: self._download(1),
            '/record/diskfree':   self._diskfree,
            '/zoom/status':       self._zoom_status,
            '/':                  self._index,
        }
        # /zoom?cam=0&level=2.5
        if self.path.startswith('/zoom?'):
            from urllib.parse import urlparse, parse_qs
            qs = parse_qs(urlparse(self.path).query)
            cam   = int(qs.get('cam', [0])[0])
            level = float(qs.get('level', [1.0])[0])
            level = max(1.0, min(ZOOM_MAX, level))
            with _zoom_lock:
                _zoom[cam] = level
            self._json({'cam': cam, 'zoom': round(level, 2)})
            return
        # /record/setdir?path=...
        if self.path.startswith('/record/setdir'):
            from urllib.parse import urlparse, parse_qs, unquote
            qs = parse_qs(urlparse(self.path).query)
            path = unquote(qs.get('path', ['/tmp'])[0])
            self._json(_set_rec_dir(path))
            return
        handler = routes.get(self.path)
        if handler:
            handler()
        else:
            self.send_error(404)

    def _json(self, data: dict):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(body)

    def _rec_status(self):
        with _rec_lock:
            self._json({
                "recording": _recording,
                "duration":  round(time.time() - _rec_start_time, 1) if _rec_start_time else 0,
            })

    def _zoom_status(self):
        with _zoom_lock:
            self._json({str(i): _zoom[i] for i in _available})

    def _diskfree(self):
        st = os.statvfs(_rec_dir)
        free_mb = (st.f_bavail * st.f_frsize) // (1024 * 1024)
        self._json({"free_mb": free_mb, "path": _rec_dir})

    def _download(self, idx: int):
        with _rec_lock:
            path = _rec_tmpfiles.get(idx)
        if not path or not os.path.exists(path):
            self.send_error(404, f"No recording for cam{idx}")
            return
        z = _rec_zoom.get(idx, 1.0)
        zoom_tag = f"_z{z:.2f}x" if z != 1.0 else ""
        fname = f"capture_{_rec_ts}_cam{idx}{zoom_tag}.mp4"
        size  = os.path.getsize(path)
        self.send_response(200)
        self.send_header('Content-Type',        'video/mp4')
        self.send_header('Content-Length',       str(size))
        self.send_header('Content-Disposition', f'attachment; filename="{fname}"')
        self.end_headers()
        with open(path, 'rb') as f:
            while chunk := f.read(65536):
                self.wfile.write(chunk)
        # Clean up temp file after sending
        try:
            os.unlink(path)
            with _rec_lock:
                _rec_tmpfiles.pop(idx, None)
            print(f"  Delivered & removed {path}", flush=True)
        except OSError:
            pass

    def _mjpeg(self, idx: int):
        if idx not in _available:
            self.send_error(404, f"Camera {idx} not connected")
            return
        self.send_response(200)
        self.send_header('Content-Type',  'multipart/x-mixed-replace; boundary=frame')
        self.send_header('Cache-Control', 'no-cache')
        self.end_headers()
        interval = 1.0 / FRAMERATE
        try:
            while True:
                with _locks[idx]:
                    frame = _frames[idx]
                if frame is None:
                    time.sleep(0.05)
                    continue
                self.wfile.write(
                    b'--frame\r\nContent-Type: image/jpeg\r\n\r\n'
                    + frame +
                    b'\r\n'
                )
                time.sleep(interval)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _index(self):
        cams_html = "".join(
            f'<div class="cam" id="cam-wrap-{i}">'
            f'<p>cam{i} — <a href="/cam{i}.mjpg">direct link</a>'
            f' &nbsp; <span id="zoom-label-{i}" style="color:#8cf">1.0x</span></p>'
            f'<div class="view"><img src="/cam{i}.mjpg" id="cam-img-{i}"><span class="ring"></span></div></div>'
            for i in _available
        )
        html = f"""<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>Pi Camera Streams</title>
<style>
  * {{ box-sizing: border-box; }}
  body {{ font-family: sans-serif; background: #111; color: #eee;
         text-align: center; margin: 0; padding: 16px; }}
  h2   {{ margin: 0 0 12px; }}
  .cams {{ display: flex; justify-content: center; gap: 8px; flex-wrap: wrap; }}
  .cam  {{ max-width: 660px; width: 100%; }}
  .cam img {{ width: 100%; border: 2px solid #333; display: block; }}
  .view {{ position: relative; }}
  .view::before, .view::after, .ring {{ position: absolute; pointer-events: none; }}
  .view::before {{ content: ""; left: 0; right: 0; top: 50%; border-top: 1px solid #0f0; }}
  .view::after  {{ content: ""; top: 0; bottom: 0; left: 50%; border-left: 1px solid #0f0; }}
  .ring {{ left: 50%; top: 50%; width: 12px; height: 12px; margin: -6px 0 0 -6px;
          border: 1px solid #0f0; border-radius: 50%; }}
  .cam p   {{ margin: 4px 0 8px; font-size: 12px; color: #666; }}
  #controls {{ margin: 12px 0; display: flex; align-items: center;
               justify-content: center; gap: 16px; flex-wrap: wrap; }}
  #rec-btn  {{ padding: 9px 26px; font-size: 15px; border: none; border-radius: 6px;
               background: #b00; color: #fff; cursor: pointer; transition: background .2s; }}
  #rec-btn:disabled {{ opacity: .5; cursor: default; }}
  #rec-btn.on {{ background: #444; }}
  #rec-label  {{ font-size: 14px; color: #aaa; min-width: 160px; text-align: left; }}
  #dir-row    {{ display: flex; gap: 6px; align-items: center; flex-wrap: wrap;
                 justify-content: center; margin-bottom: 8px; }}
  #dir-input  {{ font-size: 13px; padding: 5px 8px; border-radius: 4px;
                 border: 1px solid #555; background: #222; color: #eee; width: 260px; }}
  #dir-select {{ font-size: 13px; padding: 5px 6px; border-radius: 4px;
                 border: 1px solid #555; background: #222; color: #eee; }}
  #dir-btn    {{ font-size: 13px; padding: 5px 12px; border: none; border-radius: 4px;
                 background: #336; color: #eee; cursor: pointer; }}
  #dir-status {{ font-size: 12px; color: #888; }}
</style></head>
<body>
<h2>TCP Validator — Live Camera Streams</h2>
<div id="dir-row">
  <select id="dir-select" onchange="dirPreset()">
    <option value="">— presets —</option>
    <option value="/tmp">Temp (/tmp)</option>
    <option value="/home/live-feinberg">Home (~)</option>
    <option value="/mnt/usb">USB drive (/mnt/usb)</option>
  </select>
  <input id="dir-input" type="text" value="/tmp" placeholder="/path/to/save">
  <button id="dir-btn" onclick="setDir()">Set dir</button>
  <span id="dir-status">—</span>
</div>
<div id="controls">
  <button id="rec-btn" onclick="toggleRec()">⏺ Start Recording</button>
  <span id="rec-label">Not recording</span>
</div>
<div class="cams">{cams_html}</div>
<script>
let isRec = false;

// ── Scroll-to-zoom ────────────────────────────────────────────────────────────
const zoomLevels = {{}};
{''.join('zoomLevels[' + str(i) + '] = 1.0;\n' for i in _available)}[{', '.join(str(i) for i in _available)}].forEach(cam => {{
  const img = document.getElementById('cam-img-' + cam);
  img.addEventListener('wheel', e => {{
    e.preventDefault();
    const step = e.deltaY < 0 ? 0.25 : -0.25;
    zoomLevels[cam] = Math.round(Math.max(1.0, Math.min({ZOOM_MAX}, zoomLevels[cam] + step)) * 100) / 100;
    document.getElementById('zoom-label-' + cam).textContent = zoomLevels[cam].toFixed(2) + 'x';
    fetch('/zoom?cam=' + cam + '&level=' + zoomLevels[cam]);
  }}, {{ passive: false }});
  img.addEventListener('dblclick', () => {{
    zoomLevels[cam] = 1.0;
    document.getElementById('zoom-label-' + cam).textContent = '1.00x';
    fetch('/zoom?cam=' + cam + '&level=1.0');
  }});
}});

// Keep labels in sync when something else (servo_to_center.py) sets the zoom
async function syncZoom() {{
  const z = await (await fetch('/zoom/status')).json();
  for (const cam in z) {{
    zoomLevels[cam] = z[cam];
    document.getElementById('zoom-label-' + cam).textContent = z[cam].toFixed(2) + 'x';
  }}
}}
window.addEventListener('load', syncZoom);
setInterval(syncZoom, 2000);

function dirPreset() {{
  const v = document.getElementById('dir-select').value;
  if (v) document.getElementById('dir-input').value = v;
}}

async function setDir() {{
  const path = document.getElementById('dir-input').value.trim();
  if (!path) return;
  const st = document.getElementById('dir-status');
  st.textContent = 'Setting...';
  const d = await (await fetch('/record/setdir?path=' + encodeURIComponent(path))).json();
  if (d.status === 'ok') {{
    const df = await (await fetch('/record/diskfree')).json();
    st.textContent = `✓ ${{d.path}} (${{df.free_mb}} MB free)`;
    st.style.color = '#8f8';
  }} else {{
    st.textContent = `✗ ${{d.message}}`;
    st.style.color = '#f88';
  }}
}}

// Load initial disk free on page load
window.addEventListener('load', async () => {{
  const df = await (await fetch('/record/diskfree')).json();
  document.getElementById('dir-status').textContent = `${{df.path}} (${{df.free_mb}} MB free)`;
}});

async function toggleRec() {{
  const btn = document.getElementById('rec-btn');
  const lbl = document.getElementById('rec-label');
  btn.disabled = true;
  if (!isRec) {{
    const d = await (await fetch('/record/start')).json();
    isRec = d.status === 'started';
    btn.textContent = '⏹ Stop Recording';
    btn.className   = 'on';
    lbl.textContent = 'Recording...';
  }} else {{
    const d = await (await fetch('/record/stop')).json();
    isRec = false;
    btn.textContent = '⏺ Start Recording';
    btn.className   = '';
    lbl.textContent = `Downloading (${{d.duration}}s)...`;
    // Trigger a download for each recorded camera
    for (const idx of (d.cams || [])) {{
      const a = document.createElement('a');
      a.href = `/record/download/${{idx}}`;
      a.download = '';
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      await new Promise(r => setTimeout(r, 500)); // stagger multi-cam downloads
    }}
    lbl.textContent = 'Download complete';
  }}
  btn.disabled = false;
}}
setInterval(async () => {{
  if (!isRec) return;
  const d = await (await fetch('/record/status')).json();
  if (d.recording) document.getElementById('rec-label').textContent = `● REC ${{d.duration}}s`;
}}, 1000);
</script>
</body></html>""".encode()
        self.send_response(200)
        self.send_header('Content-Type', 'text/html; charset=utf-8')
        self.end_headers()
        self.wfile.write(html)


# ─── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    detected = Picamera2.global_camera_info()
    for info in detected:
        _available.append(info['Num'])

    if not _available:
        print("ERROR: No cameras detected. Check connections and try again.")
        raise SystemExit(1)

    print(f"\nDetected {len(_available)} camera(s): {_available}")

    for idx in _available:
        t = threading.Thread(target=_capture_loop, args=(idx,), daemon=True)
        t.start()

    threading.Thread(target=_stats_loop, daemon=True).start()

    print(f"\nStream server on port {PORT}")
    print(f"  Browser : http://raspberrypi.local:{PORT}/")
    for idx in _available:
        label = ["XY camera", "Side camera"][idx] if idx < 2 else f"Camera {idx}"
        print(f"  {label:<12}: http://raspberrypi.local:{PORT}/cam{idx}.mjpg")
    print("Press Ctrl+C to stop.\n")

    server = ThreadingHTTPServer(('0.0.0.0', PORT), _Handler)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        with _rec_lock:
            if _recording:
                for w in _recorders.values():
                    w.release()
            for p in _rec_tmpfiles.values():
                try:
                    os.unlink(p)
                except OSError:
                    pass
        print("\nStopped.")
