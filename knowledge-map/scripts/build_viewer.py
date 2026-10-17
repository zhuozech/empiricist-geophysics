"""Write html/knowledge-map.html (rotating 3-D map, data embedded) and figures/knowledge-map.png (300 dpi snapshot).

The picture follows the 3-D curriculum view of the Marble skill taxonomy (withmarble.com/curriculum): every dot is a
topic, colour is its section, height is reading order (notebook 2.01 at the bottom, 2.33 at the top), the funnel widens
upwards, and every thread is a prerequisite. Positions are computed here, once, so the page only rotates and draws.

usage: python knowledge-map/scripts/build_viewer.py   (after build.py)
"""
import colorsys
import json
import pathlib

import numpy as np

KM = pathlib.Path(__file__).resolve().parent.parent
ROOT = KM.parent
DATA = KM / "data"

topics = json.loads((DATA / "topics.json").read_text())["topics"]
deps = json.loads((DATA / "dependencies.json").read_text())["dependencies"]
clusters = json.loads((DATA / "clusters.json").read_text())["clusters"]
counts = json.loads((DATA / "manifest.json").read_text())["counts"]

# section colours, bottom (A) to top (G); neighbouring sections differ in hue or lightness
SECTION_COLOURS = ["#3D63D8", "#9DB8DC", "#EFE3C8", "#E76D95", "#E8B08A", "#DD3B31", "#E8B8C4"]
H = 1400.0                       # height of the funnel in layout units
SEED = 20261017


# ---------------------------------------------------------------- layout
def funnel_layout(T, D):
    """x, y, z for every topic: y from reading order, radius growing with y, angles from a force relaxation."""
    n = len(T)
    idx = {t["id"]: i for i, t in enumerate(T)}
    rng = np.random.default_rng(SEED)
    reading = sorted(range(n), key=lambda i: (float(T[i]["introducedIn"]), T[i]["section"], i))
    rank = np.empty(n)
    rank[reading] = np.arange(n)
    y = (rank + 0.5 + rng.uniform(-0.35, 0.35, n)) / n * H              # evenly spaced in reading order
    radius = (107.0 + 0.25 * y) * np.clip(rng.normal(1.0, 0.17, n), 0.68, 1.32)     # Marble's funnel
    E = np.array([(idx[d["topicId"]], idx[d["prerequisiteId"]], d["strength"] == "hard") for d in D])
    deg = np.bincount(E[:, :2].ravel(), minlength=n).astype(float)
    theta = rank * np.pi * (3 - np.sqrt(5))                               # golden-angle start: evenly spread
    P = np.stack([radius * np.cos(theta), y, radius * np.sin(theta)], 1)
    dy = np.abs(y[E[:, 0]] - y[E[:, 1]])                                  # links at about the same height pull harder,
    near = 1 + 3 * np.exp(-dy / (2 * H / 33))                             # so they do not cut across the funnel
    w = 0.03 / np.sqrt(deg[E[:, 0]] * deg[E[:, 1]]) * np.where(E[:, 2], 1.0, 0.5) * near
    step = 1.0
    for _ in range(700):                               # dots repel; linked topics pull together; height fixed
        d = P[:, None, :] - P[None, :, :]
        d2 = (d ** 2).sum(-1) + 25.0
        np.fill_diagonal(d2, np.inf)
        F = 9000.0 * (d / d2[..., None] ** 1.5).sum(1)
        delta = P[E[:, 1]] - P[E[:, 0]]
        np.add.at(F, E[:, 0], w[:, None] * delta)
        np.add.at(F, E[:, 1], -w[:, None] * delta)
        F[:, 1] = 0.0
        mv = step * F
        mv *= np.minimum(1.0, 12.0 / (np.linalg.norm(mv, axis=1, keepdims=True) + 1e-9))
        P += mv
        r = np.hypot(P[:, 0], P[:, 2])
        P[:, 0] *= radius / r
        P[:, 2] *= radius / r
        step *= 0.996
    return P, E


def shade(hex_colour, j):
    """Per-notebook shade of a section colour: small, deterministic shifts in hue and lightness."""
    r, g, b = (int(hex_colour[k:k + 2], 16) / 255 for k in (1, 3, 5))
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    h = (h + ((j * 5) % 17 - 8) / 360) % 1
    l = min(0.94, max(0.35, l + ((j * 3) % 7 - 3) * 0.015))
    r, g, b = colorsys.hls_to_rgb(h, l, s)
    return "#%02X%02X%02X" % tuple(round(v * 255) for v in (r, g, b))


P, E = funnel_layout(topics, deps)
domains = [c["domain"] for c in clusters]
sec_of = {d: k for k, d in enumerate(domains)}
nb_rank = {}
for d in domains:
    for j, nb in enumerate(sorted({t["introducedIn"] for t in topics if t["domain"] == d}, key=float)):
        nb_rank[nb] = j
colour = [shade(SECTION_COLOURS[sec_of[t["domain"]]], nb_rank[t["introducedIn"]]) for t in topics]
TYPE_WORD = {"CONCEPTUAL": "idea", "PROCEDURAL": "method", "REPRESENTATIONAL": "picture", "LANGUAGE": "vocabulary",
             "META": "judgement"}

nodes = []
for i, t in enumerate(topics):
    url = t["url"][len("html/"):] if t["url"] and t["url"].startswith("html/") else t["url"]
    nodes.append(dict(id=t["id"], x=round(float(P[i, 0]), 1), y=round(float(P[i, 1]), 1), z=round(float(P[i, 2]), 1),
                      s=sec_of[t["domain"]], nb=t["introducedIn"], sec=t["section"], c=t["centrality"],
                      col=colour[i], t=t["name"], d=t["description"], q=t["assessmentPrompt"], u=url,
                      k=TYPE_WORD[t["type"]]))
sections = [dict(key=d.split(" · ")[0], name=d.split(" · ", 1)[1], col=SECTION_COLOURS[k],
                 n=sum(t["domain"] == d for t in topics), first=c["firstNotebook"], last=c["lastNotebook"])
            for k, (d, c) in enumerate(zip(domains, clusters))]
payload = json.dumps(dict(H=H, sections=sections, nodes=nodes, edges=[[int(a), int(b), int(h)] for a, b, h in E]),
                     ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")

PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1, viewport-fit=cover">
<title>Knowledge map · Empiricist Geophysics</title>
<meta name="description" content="Topic 2 of Empiricist Geophysics as a rotating 3-D map: __NT__ ideas wired together by __NE__ prerequisite links.">
<style>
:root { --bg:#0A0C14; --panel:rgba(14,17,28,.82); --line:rgba(130,145,190,.18); --ink:#EDEFF7; --ink2:#CBD2E3;
  --mut:#98A0B8; --faint:#6B738B; --accent:#FF6B4A; }
* { box-sizing:border-box; }
html, body { margin:0; height:100%; background:var(--bg); }
#stage { position:relative; width:100%; height:100vh; height:100dvh; min-height:520px; overflow:hidden; color:var(--ink);
  background:radial-gradient(120% 90% at 55% 112%, rgba(52,70,140,.26), transparent 58%), var(--bg);
  font:500 14px/1.5 ui-sans-serif, -apple-system, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  -webkit-font-smoothing:antialiased; cursor:grab; user-select:none; -webkit-user-select:none; touch-action:none; }
#stage.drag { cursor:grabbing; }
canvas { position:absolute; inset:0; display:block; }
.num { font-family:ui-monospace, "SF Mono", Menlo, Consolas, monospace; font-variant-numeric:tabular-nums; }
a { color:inherit; }
.panel { background:var(--panel); border:1px solid var(--line); backdrop-filter:blur(12px); -webkit-backdrop-filter:blur(12px); }

#home { position:absolute; top:22px; left:28px; z-index:6; font-size:12.5px; font-weight:600; color:var(--mut);
  text-decoration:none; letter-spacing:.02em; }
#home:hover { color:var(--ink); }
#head { position:absolute; left:28px; top:64px; max-width:min(560px, 40vw); z-index:5; pointer-events:none;
  transition:opacity .25s; }
#head.away { opacity:0; }
#head.away #links { pointer-events:none; }
h1 { font:500 clamp(36px, 4.2vw, 62px)/1.04 "EB Garamond", "Iowan Old Style", "Palatino Linotype", Palatino, Georgia, serif;
  margin:0 0 16px; color:#fff; letter-spacing:.002em; text-wrap:balance; }
h1 .dot { color:var(--accent); }
#head .kicker { font-size:13.5px; font-weight:700; color:#fff; margin:0 0 6px; }
#head p { margin:0; font-size:13.5px; line-height:1.55; color:var(--ink2); max-width:46ch; }
#head p b { color:#fff; font-weight:700; }
#head .ctx { margin-top:10px; font-size:12.5px; color:var(--mut); max-width:46ch; }
#links { display:flex; flex-wrap:wrap; align-items:center; gap:10px 12px; margin-top:18px; pointer-events:auto; }
.btn { display:inline-flex; align-items:center; gap:7px; padding:8px 15px; border-radius:999px; text-decoration:none;
  border:1px solid rgba(130,145,190,.35); background:rgba(20,24,38,.6); color:var(--ink); font-size:13px; font-weight:700;
  transition:background .15s, border-color .15s; }
.btn:hover { background:rgba(130,145,190,.18); border-color:rgba(130,145,190,.6); }
#links .lic { font-size:11px; letter-spacing:.14em; text-transform:uppercase; color:var(--faint); font-weight:600; }

#legend { position:absolute; left:28px; bottom:24px; z-index:6; border-radius:18px; padding:12px; width:360px;
  max-height:min(52vh, 460px); overflow-y:auto; overscroll-behavior:contain; }
#find { width:100%; padding:8px 12px; border-radius:10px; border:1px solid var(--line); background:rgba(8,10,18,.6);
  color:var(--ink); font:inherit; font-size:13px; outline:none; user-select:text; -webkit-user-select:text; }
#find:focus { border-color:rgba(160,175,220,.55); }
#find::placeholder { color:var(--faint); }
#results { margin:6px 0 2px; display:none; flex-direction:column; gap:2px; }
#results.on { display:flex; }
.lg-title { display:flex; align-items:center; justify-content:space-between; font-size:10px; letter-spacing:.16em;
  text-transform:uppercase; color:var(--faint); margin:12px 4px 6px; font-weight:600; }
#fold { display:none; border:0; background:none; color:var(--mut); font:inherit; font-size:10px; letter-spacing:.14em;
  text-transform:uppercase; cursor:pointer; padding:0; }
.chip { display:flex; align-items:center; gap:9px; padding:4px 7px; border-radius:8px; cursor:pointer;
  transition:background .15s, opacity .15s; }
.chip:hover { background:rgba(130,145,190,.10); }
.chip.off { opacity:.32; }
.sw { width:10px; height:10px; border-radius:50%; flex:0 0 auto; }
.chip .nm { font-size:12.5px; font-weight:600; flex:1; min-width:0; white-space:nowrap; overflow:hidden;
  text-overflow:ellipsis; }
.chip .ct { font-size:11.5px; color:var(--mut); min-width:2ch; text-align:right; }

#tip { position:absolute; z-index:9; pointer-events:none; opacity:0; transform:translateY(4px); max-width:300px;
  transition:opacity .12s, transform .12s; background:rgba(10,13,22,.95); border:1px solid var(--line); border-radius:14px;
  padding:11px 13px; box-shadow:0 18px 50px rgba(0,0,0,.55); }
#tip.on { opacity:1; transform:none; }
.meta { display:flex; align-items:center; gap:8px; font-size:10.5px; letter-spacing:.1em; text-transform:uppercase;
  color:var(--mut); font-weight:600; margin-bottom:6px; }
.meta .sw { width:8px; height:8px; }
#tip .ttl { font-size:15px; font-weight:700; line-height:1.22; margin-bottom:6px; }
#tip .q { font-size:12.5px; line-height:1.5; color:var(--ink2); }

#card { position:absolute; right:24px; top:24px; z-index:8; width:340px; max-height:calc(100% - 48px); overflow-y:auto;
  overscroll-behavior:contain; border-radius:18px; padding:14px 18px 18px; opacity:0; transform:translateY(-6px);
  transition:opacity .18s, transform .18s; pointer-events:none; user-select:text; -webkit-user-select:text; cursor:auto; }
#card.on { opacity:1; transform:none; pointer-events:auto; }
.bar { display:flex; align-items:center; gap:8px; height:26px; margin-bottom:8px; }
.bar .sp { flex:1; }
.ghost { border:1px solid var(--line); background:transparent; color:var(--mut); cursor:pointer; height:26px; padding:0 10px;
  border-radius:8px; font:inherit; font-size:11.5px; font-weight:600; display:inline-flex; align-items:center; gap:5px; }
.ghost:hover { color:var(--ink); background:rgba(130,145,190,.12); }
#back { display:none; } #back.on { display:inline-flex; }
#x { width:26px; padding:0; justify-content:center; font-size:16px; line-height:1; }
#card .ttl { font-size:20px; font-weight:800; line-height:1.15; letter-spacing:-.015em; margin:2px 0 8px; }
#card .desc { font-size:13px; line-height:1.55; color:var(--ink2); margin:0 0 10px; }
#card .ask { font-size:12.5px; line-height:1.5; color:var(--mut); margin:0 0 12px; padding-left:10px;
  border-left:2px solid var(--line); }
#card .ask b { color:var(--ink2); font-weight:600; }
#read { display:inline-flex; align-items:center; gap:6px; font-size:12.5px; font-weight:700; text-decoration:none;
  padding:6px 12px; border-radius:999px; border:1px solid rgba(130,145,190,.35); background:rgba(20,24,38,.6); }
#read:hover { background:rgba(130,145,190,.18); }
#card .kind { font-size:11px; color:var(--faint); margin-left:8px; letter-spacing:.06em; }
.big { display:flex; align-items:baseline; gap:8px; margin:16px 0 3px; }
.big .n { font-size:34px; font-weight:800; letter-spacing:-.03em; line-height:1; }
.big .u { font-size:12.5px; color:var(--mut); font-weight:600; }
#card .sub { font-size:12px; color:#AEB6CC; line-height:1.5; }
.grp { margin-top:15px; }
.grp-t { display:flex; gap:7px; font-size:10px; letter-spacing:.14em; text-transform:uppercase; color:var(--faint);
  font-weight:600; margin-bottom:6px; }
.grp-t .k { color:var(--mut); }
.rows { display:flex; flex-direction:column; gap:2px; }
.row { display:flex; align-items:center; gap:9px; width:100%; text-align:left; cursor:pointer; background:transparent;
  border:1px solid transparent; border-radius:9px; padding:6px 8px; color:var(--ink); font:inherit; }
.row:hover, .row.cur { background:rgba(130,145,190,.11); border-color:var(--line); }
.row .rt { flex:1; font-size:12.5px; line-height:1.28; font-weight:600; }
.row .ra { font-size:11px; color:var(--faint); flex:0 0 auto; }
.none { font-size:12px; color:var(--faint); font-style:italic; padding:2px; }

#hint { position:absolute; right:24px; bottom:24px; z-index:5; font-size:11.5px; color:var(--faint); text-align:right;
  line-height:1.7; pointer-events:none; transition:opacity .2s; }
#hint b { color:var(--mut); font-weight:600; }
#card.on ~ #hint { opacity:0; }
button:focus-visible, .chip:focus-visible, a:focus-visible { outline:2px solid var(--accent); outline-offset:2px; }

@media (max-height:780px) and (min-width:721px) {
  h1 { font-size:clamp(32px, 3.6vw, 48px); margin-bottom:12px; }
  #head .ctx { display:none; }
  #legend { max-height:40vh; }
}
@media (max-width:720px) {
  #home { top:14px; left:16px; }
  #head { top:40px; left:16px; right:16px; transform:none; max-width:none; }
  h1 { font-size:32px; margin-bottom:8px; }
  #head .kicker, #head .ctx, #links .lic { display:none; }
  #head p { font-size:12.5px; }
  #links { margin-top:10px; }
  .btn { padding:6px 12px; font-size:12px; }
  #legend { left:12px; right:12px; bottom:12px; width:auto; padding:10px; max-height:46vh; }
  #legend.folded #chips { display:none; }
  #fold { display:inline; }
  #card { left:12px; right:12px; top:auto; bottom:12px; width:auto; max-height:52%; }
  #hint { display:none; }
}
@media (prefers-reduced-motion:reduce) { * { transition:none !important; } }
</style></head>
<body>
<div id="stage">
  <canvas id="cv" role="img" aria-label="Rotating 3-D map of the __NT__ ideas of Topic 2 and the prerequisite links between them. Use the search box to open any idea."></canvas>
  <a id="home" href="index.html">&larr; Empiricist Geophysics</a>
  <div id="head">
    <h1>__TITLE__<span class="dot">.</span></h1>
    <p class="kicker">__KICKER__</p>
    <p>__LEAD__</p>
    <p class="ctx">__CTX__</p>
    <div id="links">
      <a class="btn" href="index.html">Course index</a>
      <a class="btn" href="__REPO__/tree/main/knowledge-map">Data on GitHub</a>
      <span class="lic">Data &middot; CC BY 4.0</span>
    </div>
  </div>

  <div id="legend" class="panel folded">
    <input id="find" type="search" placeholder="Find an idea, e.g. Cholesky" aria-label="Find an idea" autocomplete="off" spellcheck="false">
    <div id="results" role="listbox" aria-label="Matching ideas"></div>
    <div class="lg-title"><span>Sections &middot; click to show or hide</span><button id="fold" aria-expanded="false">show</button></div>
    <div id="chips"></div>
  </div>

  <div id="tip" role="tooltip"><div class="meta"><span class="sw"></span><span class="where"></span></div>
    <div class="ttl"></div><div class="q"></div></div>

  <div id="card" class="panel" role="region" aria-label="Selected idea">
    <div class="bar"><button id="back" class="ghost" aria-label="Back to the previous idea">&larr; Back</button>
      <span class="sp"></span><button id="x" class="ghost" aria-label="Close">&times;</button></div>
    <div class="meta"><span class="sw"></span><span class="where"></span></div>
    <div class="ttl"></div>
    <p class="desc"></p>
    <p class="ask"><b>Check yourself:</b> <span></span></p>
    <a id="read" href="#">Read it in the notebook &rarr;</a><span class="kind"></span>
    <div class="big"><span class="n num">0</span><span class="u"></span></div>
    <div class="sub"></div>
    <div class="grp"><div class="grp-t">Builds directly on <span class="k num" id="kpre"></span></div><div class="rows" id="rpre"></div></div>
    <div class="grp"><div class="grp-t">Unlocks next <span class="k num" id="knext"></span></div><div class="rows" id="rnext"></div></div>
  </div>

  <div id="hint"><b>Drag</b> to spin &middot; <b>Scroll</b> to zoom &middot; <b>Click</b> a dot to trace what it builds on</div>
</div>
<script id="km-data" type="application/json">__DATA__</script>
<script>
(() => {
"use strict";
const G = JSON.parse(document.getElementById("km-data").textContent);
const N = G.nodes, E = G.edges, SEC = G.sections, H = G.H, n = N.length;
const reduce = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
const $ = s => document.querySelector(s);
const stage = $("#stage"), cv = $("#cv"), ctx = cv.getContext("2d");
const tip = $("#tip"), card = $("#card"), head = $("#head"), legend = $("#legend"), find = $("#find"), results = $("#results");
let foldedLegend = 86;                      // height of the folded legend on narrow screens, measured in resize()
if (window.matchMedia("(pointer: coarse)").matches) {      // touch screens: tap, not click
  head.querySelectorAll("b").forEach(b => { if (b.textContent === "Click any dot") b.textContent = "Tap any dot"; });
}
const MID = " · ";

// ---- graph: e = [topic, prerequisite, hard]
const pre = N.map(() => []), next = N.map(() => []), upEdges = N.map(() => []);
E.forEach((e, k) => { pre[e[0]].push(e[1]); next[e[1]].push(e[0]); upEdges[e[0]].push(k); });
const byId = new Map(N.map((d, i) => [d.id, i]));
const RGB = N.map(d => { const v = parseInt(d.col.slice(1), 16); return (v >> 16) + "," + ((v >> 8) & 255) + "," + (v & 255); });
const WX = Float32Array.from(N, d => d.x), WY = Float32Array.from(N, d => d.y - H / 2), WZ = Float32Array.from(N, d => d.z);
const RISE = Float32Array.from(N, d => d.y / H);

// ---- view and camera
let VW = 0, VH = 0, DPR = 1;
const narrow = () => VW <= 720;
const FOV = 1400, SPIN = 0.00018;
const cam = {yaw: 0.6, pitch: -0.32, zoom: 1, px: 0, py: 0, ox: 0.57, oy: 0.52};
let goal = null, spin = reduce ? 0 : SPIN;
let band = [0, 1];                       // narrow screens: free height between the header and the folded legend
const home = () => narrow() ? {ox: 0.5, oy: (band[0] + band[1]) / 2 / VH} : {ox: 0.57, oy: 0.52};
const scale0 = () => narrow() ? Math.min(VW / 1250, (band[1] - band[0]) / 1500) : Math.min(VW / 1500, VH / 1720);
const PX = new Float32Array(n), PY = new Float32Array(n), PF = new Float32Array(n);
function project() {
  const cy = Math.cos(cam.yaw), sy = Math.sin(cam.yaw), cp = Math.cos(cam.pitch), sp = Math.sin(cam.pitch);
  const s = scale0() * cam.zoom, X0 = VW * cam.ox, Y0 = VH * cam.oy;
  for (let i = 0; i < n; i++) {
    const x = WX[i] * cy + WZ[i] * sy, z = -WX[i] * sy + WZ[i] * cy;
    const y = WY[i] * cp - z * sp, d = WY[i] * sp + z * cp;
    const f = FOV / (FOV + d * 0.8);                  // fixed perspective: zoom only scales
    PX[i] = X0 + (x - cam.px) * s * f; PY[i] = Y0 - (y - cam.py) * s * f; PF[i] = f;
  }
}

// ---- state
const on = new Set(SEC.map((_, k) => k));
let hover = -1, sel = -1, lin = null, rowHot = -1, grow = reduce ? 1 : 0;
const trail = [];                         // ideas visited before the current one (for Back)
const shown = i => on.has(N[i].s) && RISE[i] <= grow;
const radius = i => (4 + Math.sqrt(N[i].c) * 9) * PF[i] * Math.min(1.3, Math.max(0.9, cam.zoom));
function resize() {
  DPR = Math.min(window.devicePixelRatio || 1, 2);
  VW = stage.clientWidth; VH = stage.clientHeight;
  cv.width = Math.round(VW * DPR); cv.height = Math.round(VH * DPR);
  cv.style.width = VW + "px"; cv.style.height = VH + "px";
  if (narrow() && legend.classList.contains("folded")) foldedLegend = legend.offsetHeight;
  band = [head.offsetTop + head.offsetHeight + 10, VH - 12 - foldedLegend - 10];
  if (band[1] - band[0] < 220) band = [Math.max(0, band[1] - 220), band[1]];
  if (sel < 0) Object.assign(cam, home());
}

function lineage(i) {                     // everything before i, all the way back
  const nodes = new Set([i]), edges = new Set(), todo = [i];
  while (todo.length) {
    const u = todo.pop();
    for (const k of upEdges[u]) { edges.add(k); const p = E[k][1]; if (!nodes.has(p)) { nodes.add(p); todo.push(p); } }
  }
  return {nodes, edges};
}
const wrapAngle = a => Math.atan2(Math.sin(a), Math.cos(a));
function viewBox() {                      // screen area left free by the card and the legend
  if (!card.classList.contains("on")) return {x: 0, y: 0, w: VW, h: VH};
  if (narrow()) return {x: 0, y: 0, w: VW, h: Math.max(160, VH - card.offsetHeight - 24)};
  const left = legend.offsetLeft + legend.offsetWidth + 16, right = card.offsetLeft - 16;
  return right - left > 360 ? {x: left, y: 16, w: right - left, h: VH - 32} : {x: 0, y: 0, w: Math.max(240, right), h: VH};
}
function aim(i) {                         // turn idea i to the front and fit its lineage into the free area
  const phi = Math.atan2(WZ[i], WX[i]);
  const yaw = cam.yaw + wrapAngle(phi + Math.PI / 2 - cam.yaw), pitch = -0.2;
  const cy = Math.cos(yaw), sy = Math.sin(yaw), cp = Math.cos(pitch), sp = Math.sin(pitch);
  let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
  lin.nodes.forEach(j => {
    const x = WX[j] * cy + WZ[j] * sy, z = -WX[j] * sy + WZ[j] * cy, y = WY[j] * cp - z * sp;
    x0 = Math.min(x0, x); x1 = Math.max(x1, x); y0 = Math.min(y0, y); y1 = Math.max(y1, y);
  });
  const b = viewBox(), s0 = scale0();
  const fit = Math.min(b.w * 0.8 / ((x1 - x0) * s0 + 1), b.h * 0.78 / ((y1 - y0) * s0 + 1));
  goal = {yaw, pitch, zoom: Math.max(1, Math.min(1.8, fit)), px: (x0 + x1) / 2, py: (y0 + y1) / 2,
          ox: (b.x + b.w / 2) / VW, oy: (b.y + b.h / 2) / VH};
}

// ---- selection
function select(i, push) {
  if (push && sel >= 0 && sel !== i) trail.push(sel);
  sel = i; lin = lineage(i); spin = 0; hideTip(); fillCard(i); head.classList.add("away");
  aim(i);
  if (location.hash !== "#" + N[i].id) window.history.replaceState(null, "", "#" + N[i].id);
}
function clearSel() {
  sel = -1; lin = null; trail.length = 0; rowHot = -1;
  card.classList.remove("on"); head.classList.remove("away");
  goal = Object.assign({pitch: -0.32, zoom: 1, px: 0, py: 0}, home());
  spin = reduce ? 0 : SPIN;
  if (location.hash) window.history.replaceState(null, "", location.pathname + location.search);
}
function back() { if (trail.length) select(trail.pop(), false); }

// ---- tooltip and card
const esc = s => String(s).replace(/[&<>"]/g, c => ({"&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;"}[c]));
function short(s) {                       // whole description if short, otherwise its first sentence
  if (s.length <= 200) return s;
  const re = /[.!?](?=\s+[A-Z(])/g;
  let m;
  while ((m = re.exec(s))) { const h = s.slice(0, m.index + 1); if (!/\b(e\.g|i\.e|vs|cf)\.$/i.test(h)) return h; }
  return s.slice(0, s.lastIndexOf(" ", 200)) + " …";
}
function showTip(i, e) {
  const d = N[i];
  tip.querySelector(".sw").style.background = d.col;
  tip.querySelector(".where").textContent = SEC[d.s].key + MID + SEC[d.s].name + MID + d.nb;
  tip.querySelector(".ttl").textContent = d.t;
  tip.querySelector(".q").textContent = short(d.d);
  tip.classList.add("on"); placeTip(e);
}
function placeTip(e) {
  const r = stage.getBoundingClientRect();
  let x = e.clientX - r.left + 16, y = e.clientY - r.top + 16;
  if (x + tip.offsetWidth > VW - 8) x = e.clientX - r.left - tip.offsetWidth - 16;
  if (y + tip.offsetHeight > VH - 8) y = e.clientY - r.top - tip.offsetHeight - 16;
  tip.style.left = Math.max(8, x) + "px"; tip.style.top = Math.max(8, y) + "px";
}
function hideTip() { tip.classList.remove("on"); }
const byReading = (a, b) => (parseFloat(N[a].nb) - parseFloat(N[b].nb)) || (N[a].sec - N[b].sec) || (a - b);
function rows(box, list) {
  box.innerHTML = "";
  if (!list.length) { box.innerHTML = '<div class="none">nothing in Topic 2</div>'; return; }
  list.slice().sort(byReading).forEach(j => {
    const d = N[j], b = document.createElement("button");
    b.className = "row";
    b.innerHTML = '<span class="sw" style="background:' + d.col + '"></span><span class="rt">' + esc(d.t) +
      '</span><span class="ra num">' + d.nb + "</span>";
    b.addEventListener("click", () => select(j, true));
    b.addEventListener("pointerenter", () => { rowHot = j; });
    b.addEventListener("pointerleave", () => { if (rowHot === j) rowHot = -1; });
    box.appendChild(b);
  });
}
function fillCard(i) {
  const d = N[i], before = lin.nodes.size - 1;
  card.querySelector(".meta .sw").style.background = d.col;
  card.querySelector(".meta .where").textContent = SEC[d.s].key + MID + SEC[d.s].name;
  card.querySelector(".ttl").textContent = d.t;
  card.querySelector(".desc").textContent = d.d;
  card.querySelector(".ask span").textContent = d.q;
  const read = $("#read");
  read.href = d.u || "#"; read.textContent = "Read it in " + d.nb + " §" + d.sec + " →";
  card.querySelector(".kind").textContent = d.k;
  card.querySelector(".big .n").textContent = before;
  card.querySelector(".big .u").textContent = before === 1 ? "prerequisite in total" : "prerequisites in total";
  card.querySelector(".sub").textContent = before > 0
    ? "Every idea it builds on, directly or through others, traced all the way back."
    : "A starting point: nothing in Topic 2 comes before it.";
  $("#kpre").textContent = pre[i].length || ""; $("#knext").textContent = next[i].length || "";
  rows($("#rpre"), pre[i]); rows($("#rnext"), next[i]);
  $("#back").classList.toggle("on", trail.length > 0);
  rowHot = -1; card.scrollTop = 0; card.classList.add("on");
}
$("#x").addEventListener("click", clearSel);
$("#back").addEventListener("click", back);
["pointerdown", "pointerup", "pointermove", "click", "wheel"].forEach(ev => {
  [card, legend, $("#links"), $("#home")].forEach(el => el.addEventListener(ev, e => e.stopPropagation()));
});

// ---- legend: one chip per section
const chips = $("#chips");
SEC.forEach((s, k) => {
  const el = document.createElement("div");
  el.className = "chip"; el.tabIndex = 0; el.setAttribute("role", "button"); el.setAttribute("aria-pressed", "true");
  el.title = "notebooks " + s.first + "–" + s.last + ", " + s.n + " ideas";
  el.innerHTML = '<span class="sw" style="background:' + s.col + '"></span><span class="nm">' + esc(s.key + MID + s.name) +
    '</span><span class="ct num">' + s.n + "</span>";
  const flip = () => { const off = on.has(k); off ? on.delete(k) : on.add(k); el.classList.toggle("off", off);
    el.setAttribute("aria-pressed", String(!off)); };
  el.addEventListener("click", flip);
  el.addEventListener("keydown", e => { if (e.key === "Enter" || e.key === " ") { e.preventDefault(); flip(); } });
  chips.appendChild(el);
});
$("#fold").addEventListener("click", () => { const f = legend.classList.toggle("folded");
  $("#fold").textContent = f ? "show" : "hide"; $("#fold").setAttribute("aria-expanded", String(!f)); });

// ---- search by name
let hits = [], hit = 0;
function showHits() {
  results.innerHTML = "";
  hits.forEach((j, k) => {
    const d = N[j], b = document.createElement("button");
    b.className = "row" + (k === hit ? " cur" : ""); b.setAttribute("role", "option");
    b.innerHTML = '<span class="sw" style="background:' + d.col + '"></span><span class="rt">' + esc(d.t) +
      '</span><span class="ra num">' + d.nb + "</span>";
    b.addEventListener("click", () => choose(j));
    results.appendChild(b);
  });
  results.classList.toggle("on", hits.length > 0);
}
function choose(j) { find.value = ""; hits = []; showHits(); find.blur(); grow = 1; select(j, true); }
find.addEventListener("input", () => {
  const q = find.value.trim().toLowerCase();
  hits = q.length < 2 ? [] : N.map((d, j) => [d.t.toLowerCase().indexOf(q), j]).filter(a => a[0] >= 0)
    .sort((a, b) => (a[0] > 0) - (b[0] > 0) || byReading(a[1], b[1])).slice(0, 8).map(a => a[1]);
  hit = 0; showHits();
});
find.addEventListener("keydown", e => {
  if (e.key === "ArrowDown" && hits.length) { hit = (hit + 1) % hits.length; showHits(); e.preventDefault(); }
  else if (e.key === "ArrowUp" && hits.length) { hit = (hit + hits.length - 1) % hits.length; showHits(); e.preventDefault(); }
  else if (e.key === "Enter" && hits.length) { choose(hits[hit]); }
  else if (e.key === "Escape") { find.value = ""; hits = []; showHits(); }
});

// ---- pointer: drag to spin, click to select, wheel or pinch to zoom
function nearest(mx, my) {
  let best = -1, bd = Infinity;
  for (let i = 0; i < n; i++) {
    if (!shown(i)) continue;
    const dx = PX[i] - mx, dy = PY[i] - my, d = dx * dx + dy * dy, r = Math.max(11, radius(i) + 6);
    if (d < r * r && d < bd) { bd = d; best = i; }
  }
  return best;
}
const pts = new Map();
let dragging = false, moved = false, lx = 0, ly = 0, pinch = 0;
stage.addEventListener("pointerdown", e => {
  pts.set(e.pointerId, [e.clientX, e.clientY]);
  if (pts.size === 1) { dragging = true; moved = false; lx = e.clientX; ly = e.clientY; stage.classList.add("drag");
    stage.setPointerCapture(e.pointerId); }
});
stage.addEventListener("pointermove", e => {
  if (pts.has(e.pointerId)) pts.set(e.pointerId, [e.clientX, e.clientY]);
  if (pts.size === 2) {
    const [a, b] = [...pts.values()], dd = Math.hypot(a[0] - b[0], a[1] - b[1]);
    if (pinch) { cam.zoom = Math.max(0.5, Math.min(4, cam.zoom * dd / pinch)); if (goal) delete goal.zoom; }
    pinch = dd; moved = true; return;
  }
  if (dragging) {
    const dx = e.clientX - lx, dy = e.clientY - ly;
    if (Math.abs(dx) + Math.abs(dy) > 3) { moved = true; if (goal) { delete goal.yaw; delete goal.pitch; } }
    cam.yaw += dx * 0.0055; cam.pitch = Math.max(-1.1, Math.min(0.15, cam.pitch - dy * 0.003));
    lx = e.clientX; ly = e.clientY;
    return;
  }
  const r = cv.getBoundingClientRect(), i = nearest(e.clientX - r.left, e.clientY - r.top);
  if (i !== hover) { hover = i; i >= 0 ? showTip(i, e) : hideTip(); } else if (i >= 0) placeTip(e);
  stage.style.cursor = i >= 0 ? "pointer" : "";
});
stage.addEventListener("pointerup", e => {
  pts.delete(e.pointerId); if (pts.size < 2) pinch = 0;
  if (!dragging) return;
  dragging = false; stage.classList.remove("drag");
  if (moved) return;
  const r = cv.getBoundingClientRect(), i = nearest(e.clientX - r.left, e.clientY - r.top);
  if (i >= 0) select(i, true); else if (sel >= 0) clearSel();
});
stage.addEventListener("pointercancel", e => { pts.delete(e.pointerId); pinch = 0; dragging = false; stage.classList.remove("drag"); });
stage.addEventListener("pointerleave", () => { hover = -1; hideTip(); });
stage.addEventListener("wheel", e => {
  e.preventDefault();
  cam.zoom = Math.max(0.5, Math.min(4, cam.zoom * Math.exp(-e.deltaY * 0.0016)));
  if (goal) delete goal.zoom;
}, {passive: false});
document.addEventListener("keydown", e => { if (e.key === "Escape" && document.activeElement !== find && sel >= 0) clearSel(); });
window.addEventListener("hashchange", () => {
  const j = byId.get(decodeURIComponent(location.hash.slice(1)));
  if (j !== undefined && j !== sel) { grow = 1; select(j, true); }
});

// ---- drawing
const order = N.map((_, i) => i);
function draw() {
  ctx.setTransform(DPR, 0, 0, DPR, 0, 0);
  ctx.clearRect(0, 0, VW, VH);
  project();
  ctx.lineCap = "round";
  ctx.lineWidth = 1;
  for (let k = 0; k < E.length; k++) {                 // threads behind the dots
    const a = E[k][0], b = E[k][1];
    if (!shown(a) || !shown(b) || (lin && lin.edges.has(k))) continue;
    const alpha = lin ? 0.035 : (E[k][2] ? 0.2 : 0.1) * (PF[a] + PF[b]) / 2;
    ctx.strokeStyle = "rgba(150,165,205," + alpha.toFixed(3) + ")";
    ctx.beginPath(); ctx.moveTo(PX[a], PY[a]); ctx.lineTo(PX[b], PY[b]); ctx.stroke();
  }
  if (lin) {                                            // the selected lineage, each thread in its prerequisite's colour
    ctx.lineWidth = 1.6;
    for (const k of lin.edges) {
      const a = E[k][0], b = E[k][1];
      if (!shown(a) || !shown(b)) continue;
      ctx.strokeStyle = "rgba(" + RGB[b] + ",.78)";
      ctx.beginPath(); ctx.moveTo(PX[a], PY[a]); ctx.lineTo(PX[b], PY[b]); ctx.stroke();
    }
  }
  order.sort((a, b) => PF[a] - PF[b]);                  // far to near
  for (const i of order) {
    if (!on.has(N[i].s)) continue;
    const fade = reduce ? 1 : Math.min(1, (grow - RISE[i]) / 0.03);
    if (fade <= 0) continue;
    const inLin = !lin || lin.nodes.has(i), focus = i === sel || i === hover || i === rowHot;
    const dim = inLin ? 1 : 0.1, r = radius(i) * (focus ? 1.6 : 1);
    const alpha = fade * dim * (0.55 + 0.45 * Math.min(1, PF[i] * PF[i]));
    if (focus || (lin && inLin)) { ctx.shadowColor = "rgb(" + RGB[i] + ")"; ctx.shadowBlur = focus ? 18 : 9; }
    ctx.fillStyle = "rgba(" + RGB[i] + "," + alpha.toFixed(3) + ")";
    ctx.beginPath(); ctx.arc(PX[i], PY[i], r, 0, 6.2832); ctx.fill();
    ctx.shadowBlur = 0;
    ctx.strokeStyle = "rgba(8,10,18," + (0.5 * dim * fade).toFixed(3) + ")";
    ctx.lineWidth = 1; ctx.stroke();
    if (focus) {
      ctx.strokeStyle = "rgba(255,255,255,.95)"; ctx.lineWidth = 1.6;
      ctx.beginPath(); ctx.arc(PX[i], PY[i], r + 2.5, 0, 6.2832); ctx.stroke();
    }
  }
}

// ---- animation: bottom-up growth, slow spin, camera easing
const T0 = performance.now();
let last = T0;
function tick(now) {
  const dt = Math.min(64, now - last); last = now;
  if (grow < 1) grow = Math.min(1, (now - T0) / 2800);
  if (!dragging && hover < 0 && sel < 0 && spin) cam.yaw += spin * dt;
  if (goal) {
    const k = reduce ? 1 : 1 - Math.exp(-dt / 130);
    let done = true;
    for (const key in goal) {
      const d = goal[key] - cam[key];
      cam[key] += d * k;
      if (Math.abs(d) > (key === "px" || key === "py" ? 0.5 : 0.002)) done = false;
    }
    if (done) goal = null;
  }
  draw();
  requestAnimationFrame(tick);
}
window.addEventListener("resize", () => { resize(); if (sel >= 0) aim(sel); });
resize();
const start = byId.get(decodeURIComponent(location.hash.slice(1)));
if (start !== undefined) { grow = 1; select(start, false); Object.assign(cam, goal); goal = null; }
requestAnimationFrame(tick);
})();
</script>
</body></html>
"""

TEXT = dict(
    TITLE="Inverse problems and Bayesian inversion",
    KICKER="A knowledge map of Topic 2 of Empiricist Geophysics.",
    LEAD=(f"<b>{counts['topics']}</b> ideas and <b>{counts['dependencies']}</b> prerequisite links, from the forward "
          "model to a full Bayesian inversion. Height is reading order, notebook 2.01 at the bottom and 2.33 at the top; "
          "colour is the section. <b>Click any dot</b> to see everything that comes before it."),
    CTX="Every idea links to the notebook section that introduces it. The format follows the open Marble skill taxonomy.",
)
page = PAGE
for key, val in TEXT.items():
    page = page.replace(f"__{key}__", val)
page = (page.replace("__NT__", str(counts["topics"])).replace("__NE__", str(counts["dependencies"]))
        .replace("__REPO__", "https://github.com/zhuozech/empiricist-geophysics").replace("__DATA__", payload))
(ROOT / "html" / "knowledge-map.html").write_text(page)
print(f"wrote html/knowledge-map.html ({len(page) / 1e3:.0f} kB)")

# ---------------------------------------------------------------- static snapshot (300 dpi), same layout and camera
import matplotlib  # noqa: E402
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.collections import LineCollection  # noqa: E402

BG = "#0A0C14"
yaw, pitch, fov = 0.6, -0.32, 1400.0
X, Y, Z = P[:, 0], P[:, 1] - H / 2, P[:, 2]
xr = X * np.cos(yaw) + Z * np.sin(yaw)
zr = -X * np.sin(yaw) + Z * np.cos(yaw)
yr = Y * np.cos(pitch) - zr * np.sin(pitch)
pf = fov / (fov + (Y * np.sin(pitch) + zr * np.cos(pitch)) * 0.8)  # same perspective as the page
sx, sy = xr * pf, -yr * pf
legend_h = 175                                                   # room below the funnel for the legend
x0, x1 = sx.min() - 110, sx.max() + 110
y0, y1 = sy.min() - 60, sy.max() + 60 + legend_h
DPI, W_IN = 300, 6.0
H_IN = W_IN * (y1 - y0) / (x1 - x0)
px_per_unit = W_IN * DPI / (x1 - x0)
fig = plt.figure(figsize=(W_IN, H_IN), dpi=DPI, facecolor=BG)
ax = fig.add_axes([0, 0, 1, 1], facecolor=BG)
ax.set_xlim(x0, x1)
ax.set_ylim(y1, y0)
ax.axis("off")
segs = [[(sx[a], sy[a]), (sx[b], sy[b])] for a, b, h in E]
cols = [(150 / 255, 165 / 255, 205 / 255, (0.22 if h else 0.11) * (pf[a] + pf[b]) / 2) for a, b, h in E]
ax.add_collection(LineCollection(segs, colors=cols, linewidths=0.35, zorder=1))
cent = np.array([t["centrality"] for t in topics])
r_px = (4 + np.sqrt(cent) * 9) * pf * 1.25 * px_per_unit           # dot radius in output pixels
order = np.argsort(pf)
alpha = 0.55 + 0.45 * np.minimum(1, pf ** 2)
ax.scatter(sx[order], sy[order], s=(2 * r_px[order] * 72 / DPI) ** 2,
           c=[matplotlib.colors.to_rgba(colour[i], alpha[i]) for i in order],
           edgecolors=(8 / 255, 10 / 255, 18 / 255, 0.5), linewidths=0.3, zorder=2)
for k, sec in enumerate(sections):
    yk = y1 - 40 - (len(sections) - 1 - k) * 19
    ax.scatter([x0 + 40], [yk], s=12, color=sec["col"], zorder=3)
    ax.text(x0 + 56, yk, f"{sec['key']} · {sec['name']}   {sec['first']}–{sec['last']}", color="#CBD2E3",
            fontsize=5.4, va="center")
ax.text(x1 - 40, y1 - 40, f"{counts['topics']} ideas · {counts['dependencies']} prerequisite links\n"
        "height = reading order (2.01 at the bottom)", color="#98A0B8", fontsize=5.4, ha="right", va="center",
        linespacing=1.6)
out = ROOT / "figures" / "knowledge-map.png"
fig.savefig(out, dpi=DPI, facecolor=BG)
plt.close(fig)
print(f"wrote figures/{out.name} ({W_IN * DPI:.0f} x {H_IN * DPI:.0f} px, {DPI} dpi)")
