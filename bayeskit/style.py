"""Shared look for every figure in Empiricist Geophysics.

One colour per sampler family, used in every notebook, so the colour itself becomes part of the picture:
    gradient samplers (HMC, NUTS)  blue
    random-walk Metropolis         orange
    SMC / tempering                aqua
    ensemble Kalman (EnKF/EKI/EnKI) violet
    exact answer                   ink (black/grey)
The four hues were checked as a set with the dataviz palette validator (all pairs, light surface).
"""
from __future__ import annotations

import base64
import io
import pathlib

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# ---------------------------------------------------------------- colours
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

BLUE, ORANGE, AQUA, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7"
GOOD, BAD = "#0ca30c", "#d03b3b"          # reserved for accept/reject, pass/fail (always with a symbol or label)

COLORS = {
    "hmc": BLUE, "nuts": BLUE, "grad": BLUE,
    "rwm": ORANGE, "metropolis": ORANGE,
    "smc": AQUA, "tempering": AQUA,
    "enki": VIOLET, "eki": VIOLET, "enkf": VIOLET,
    "exact": INK, "truth": INK,
}
CHAINS = [BLUE, ORANGE, AQUA, VIOLET]      # chain 1..4 in the diagnostics notebook

# single-hue grey ramp for "true posterior" regions (light = low probability)
GREYS = LinearSegmentedColormap.from_list("bx_greys", [SURFACE, "#ecebe6", "#d9d8d1", "#c3c2b7", "#a3a29b", "#7c7b75"])

# ---------------------------------------------------------------- paths
ROOT = pathlib.Path(__file__).resolve().parent.parent
FIG = ROOT / "figures"


def use():
    """Apply the house style to matplotlib."""
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "figure.dpi": 110, "savefig.dpi": 300, "savefig.bbox": "tight",   # saved figures at 300 dpi
        "font.family": ["Arial", "DejaVu Sans"],     # Arial has a real bold face; DejaVu fills missing glyphs (✓ ✗ ∝)
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 9.5, "axes.titlesize": 10.5, "axes.titleweight": "bold", "axes.labelsize": 9.5,
        "axes.edgecolor": AXIS, "axes.linewidth": 0.8, "axes.labelcolor": INK, "axes.titlecolor": INK,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.color": GRID, "grid.linewidth": 0.6, "grid.linestyle": "-",
        "xtick.color": INK2, "ytick.color": INK2, "xtick.labelsize": 8.5, "ytick.labelsize": 8.5,
        "xtick.major.size": 3, "ytick.major.size": 3, "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "lines.linewidth": 1.5, "lines.solid_capstyle": "round", "lines.solid_joinstyle": "round",
        "legend.frameon": False, "legend.fontsize": 8.5,
        "text.color": INK, "mathtext.fontset": "dejavusans",
        "animation.embed_limit": 60,
    })


def savefig(fig, name: str, show_path: bool = True):
    """Save a figure to figures/<name>.png and return the path."""
    FIG.mkdir(exist_ok=True)
    path = FIG / f"{name}.png"
    fig.savefig(path)
    if show_path:
        print(f"saved figures/{path.name}")
    return path


def gif_html(path, width: int | None = None) -> str:
    """Return an <img> tag with the GIF embedded (shows in Jupyter, VS Code and the HTML export)."""
    data = base64.b64encode(pathlib.Path(path).read_bytes()).decode("ascii")
    w = f' width="{width}"' if width else ""
    return f'<img src="data:image/gif;base64,{data}"{w} style="max-width:100%"/>'


def show_gif(path, width: int | None = None):
    from IPython.display import HTML, display
    display(HTML(gif_html(path, width)))


def frames_to_gif(frames, path, fps: int = 12, hold_last: int = 12):
    """Write a list of RGB arrays to an optimised GIF (palette-quantised, loops forever)."""
    from PIL import Image
    imgs = [Image.fromarray(f) for f in frames] + [Image.fromarray(frames[-1])] * hold_last
    pal = [im.convert("P", palette=Image.ADAPTIVE, colors=128) for im in imgs]
    pal[0].save(path, save_all=True, append_images=pal[1:], duration=int(1000 / fps), loop=0, optimize=True, disposal=2)
    return path


def fig_to_rgb(fig) -> np.ndarray:
    """Rasterise a figure to an RGB array (used to build GIF frames)."""
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())
    return buf[..., :3].copy()


def badge(ax, text, ok: bool | None = None, loc="upper left", fs=8.5):
    """Small label in a corner of an axis; ok=True/False adds a check or cross (never colour alone)."""
    if ok is None:
        sym = ""
    elif "$" in text:                     # mathtext has no ✓/✗ glyphs; use its own symbols
        sym = r"$\checkmark$ " if ok else r"$\times$ "
    else:
        sym = "✓ " if ok else "✗ "
    col = INK if ok is None else (GOOD if ok else BAD)
    xy = {"upper left": (0.03, 0.97, "left", "top"), "upper right": (0.97, 0.97, "right", "top"),
          "lower left": (0.03, 0.03, "left", "bottom"), "lower right": (0.97, 0.03, "right", "bottom")}[loc]
    t = ax.text(xy[0], xy[1], sym + text, transform=ax.transAxes, ha=xy[2], va=xy[3], fontsize=fs, color=INK,
                bbox=dict(boxstyle="round,pad=0.25", fc=SURFACE, ec=col if ok is not None else AXIS, lw=0.8, alpha=0.92))
    return t
