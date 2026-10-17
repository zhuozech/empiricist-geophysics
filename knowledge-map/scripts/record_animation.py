"""Record the 3-D knowledge map as an animation: figures/knowledge-map.gif (README) and an MP4 with a poster frame.

The recording is the real page, html/knowledge-map.html, run in headless Chrome one frame at a time: the page's clock
(performance.now and requestAnimationFrame) is replaced by a stepped one, so every frame is reproducible. The story:
the map grows from the bottom and spins; one idea is opened and its prerequisites light up; one of the ideas it unlocks
is opened from the card; the card is closed and the map spins on. The last frames fade back into the first, so it loops.

needs: Google Chrome, ffmpeg, websocket-client, Pillow, numpy (none of them is needed to read the course)
usage: python knowledge-map/scripts/record_animation.py [--video-dir build/_scratch_media]
"""
import argparse
import base64
import io
import json
import pathlib
import shutil
import socket
import subprocess
import tempfile
import time
import urllib.request

import numpy as np
import websocket
from PIL import Image

KM = pathlib.Path(__file__).resolve().parent.parent
ROOT = KM.parent
PAGE = ROOT / "html" / "knowledge-map.html"
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
W, H, FPS = 1280, 720, 24
SPEED = 1.5                       # page milliseconds per real millisecond: a slightly livelier spin than the page's
FIRST, SECOND = "eg_hierarchical", "Every choice on one inversion"
STORY = [(3.3, "open"), (6.3, "follow"), (9.6, "close")]     # seconds into the video
LENGTH, FADE = 12.6, 0.6                                      # total length and the fade back to the first frame (s)

STEPPED_CLOCK = """
(() => {
  let now = 0, queue = [];
  performance.now = () => now;
  window.requestAnimationFrame = cb => { queue.push(cb); return queue.length; };
  window.__step = t => { now = t; const run = queue; queue = []; run.forEach(cb => cb(t)); return run.length; };
  const css = document.createElement("style");
  css.textContent = "*{transition:none !important}";
  document.addEventListener("DOMContentLoaded", () => document.head.appendChild(css));
})();
"""
ACTIONS = {
    "open": f"location.hash = '#{FIRST}'",
    "follow": f"[...document.querySelectorAll('#rnext .row')].find(b => b.textContent.includes({json.dumps(SECOND)})).click()",
    "close": "document.getElementById('x').click()",
}


class Chrome:
    """Headless Chrome driven over the DevTools protocol."""

    def __init__(self):
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            self.port = s.getsockname()[1]
        self.profile = tempfile.mkdtemp(prefix="km_chrome_")
        self.proc = subprocess.Popen([CHROME, "--headless=new", f"--remote-debugging-port={self.port}",
                                      f"--user-data-dir={self.profile}", "--no-first-run", "--no-default-browser-check",
                                      "--hide-scrollbars", "--mute-audio", f"--window-size={W},{H}", "about:blank"],
                                     stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        self.ws, self.n = None, 0
        try:
            for _ in range(100):
                try:
                    targets = json.load(urllib.request.urlopen(f"http://127.0.0.1:{self.port}/json/list", timeout=1))
                    url = next(t["webSocketDebuggerUrl"] for t in targets if t["type"] == "page")
                    break
                except Exception:
                    time.sleep(0.1)
            else:
                raise RuntimeError("Chrome did not start")
            self.ws = websocket.create_connection(url, timeout=60, suppress_origin=True)
        except BaseException:
            self.close()                                   # never leave a headless Chrome behind
            raise

    def call(self, method, **params):
        self.n += 1
        self.ws.send(json.dumps(dict(id=self.n, method=method, params=params)))
        while True:
            msg = json.loads(self.ws.recv())
            if msg.get("id") == self.n:
                if "error" in msg:
                    raise RuntimeError(f"{method}: {msg['error']}")
                return msg.get("result", {})

    def js(self, expression):
        r = self.call("Runtime.evaluate", expression=expression, returnByValue=True, awaitPromise=True)
        if "exceptionDetails" in r:
            raise RuntimeError(f"{expression}: {r['exceptionDetails']}")
        return r.get("result", {}).get("value")

    def close(self):
        try:
            if self.ws is not None:
                self.ws.close()
        finally:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                self.proc.kill()
            shutil.rmtree(self.profile, ignore_errors=True)


def record():
    c = Chrome()
    frames = []
    try:
        c.call("Page.enable")                                  # needed before the clock script is injected
        c.call("Emulation.setDeviceMetricsOverride", width=W, height=H, deviceScaleFactor=2, mobile=False)
        c.call("Page.addScriptToEvaluateOnNewDocument", source=STEPPED_CLOCK)
        c.call("Page.navigate", url=PAGE.as_uri())
        for _ in range(100):
            if c.js("document.readyState === 'complete' && typeof window.__step === 'function'"):
                break
            time.sleep(0.1)
        c.js("document.fonts.ready.then(() => true)")
        todo = sorted((round(t * FPS), a) for t, a in STORY)
        for k in range(round(LENGTH * FPS)):
            while todo and todo[0][0] == k:
                c.js(ACTIONS[todo.pop(0)[1]])
                c.js("new Promise(r => setTimeout(r, 30))")            # let the hashchange event run
            c.js(f"__step({(k + 1) * 1000 / FPS * SPEED})")
            png = base64.b64decode(c.call("Page.captureScreenshot", format="png", fromSurface=True)["data"])
            frames.append(np.asarray(Image.open(io.BytesIO(png)).convert("RGB").resize((W, H), Image.LANCZOS)))
            if k % 24 == 0:
                print(f"  frame {k}/{round(LENGTH * FPS)}", flush=True)
    finally:
        c.close()
    n = round(FADE * FPS)                                                    # fade the end into the first frame
    for j in range(n):
        a = (j + 1) / (n + 1)
        frames[-n + j] = ((1 - a) * frames[-n + j] + a * frames[0]).astype(np.uint8)
    return frames


def write_mp4(frames, path):
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-c:v", "libx264", "-preset", "slow", "-crf", "23",
                           "-pix_fmt", "yuv420p", "-movflags", "+faststart", "-an", str(path)], stdin=subprocess.PIPE)
    for f in frames:
        ff.stdin.write(f.tobytes())
    ff.stdin.close()
    if ff.wait():
        raise RuntimeError("ffmpeg failed")


def write_gif(frames, path, width=800, fps=12):
    """GIF through ffmpeg: one 256-colour palette built from all frames, so the section colours stay true."""
    vf = (f"fps={fps},scale={width}:-1:flags=lanczos,split[a][b];[a]palettegen=max_colors=256:stats_mode=full[p];"
          "[b][p]paletteuse=dither=none")
    ff = subprocess.Popen(["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
                           "-r", str(FPS), "-i", "-", "-vf", vf, "-loop", "0", str(path)], stdin=subprocess.PIPE)
    for f in frames:
        ff.stdin.write(f.tobytes())
    ff.stdin.close()
    if ff.wait():
        raise RuntimeError("ffmpeg failed")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gif", default=str(ROOT / "figures" / "knowledge-map.gif"))
    ap.add_argument("--video-dir", default=str(ROOT / "build" / "_scratch_media"))
    args = ap.parse_args()
    frames = record()
    out = pathlib.Path(args.video_dir)
    out.mkdir(parents=True, exist_ok=True)
    write_mp4(frames, out / "knowledge-map.mp4")
    Image.fromarray(frames[round(8.4 * FPS)]).save(out / "knowledge-map-poster.jpg", quality=88)
    write_gif(frames, args.gif)
    for p in (out / "knowledge-map.mp4", out / "knowledge-map-poster.jpg", pathlib.Path(args.gif)):
        print(f"wrote {p.relative_to(ROOT) if p.is_relative_to(ROOT) else p} ({p.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()
