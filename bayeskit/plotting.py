"""Plot helpers shared by the notebooks (exact-posterior regions, samples, traces)."""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

from . import style as S
from .problems import Grid, grid as make_grid


def density(ax, g: Grid, probs=(0.5, 0.9, 0.99), fill=True, lines=True, lw=0.7, color=None, zorder=0):
    """Draw the exact posterior as nested highest-density regions (50 %, 90 %, 99 % of the probability)."""
    lev = [g.levels.get(p) or _level(g, p) for p in sorted(probs, reverse=True)]   # low density -> high
    lev = sorted(set(lev))
    if fill:
        shades = ["#ecebe6", "#d9d8d1", "#c3c2b7", "#a3a29b"][: len(lev)]
        ax.contourf(g.X, g.Y, g.P, levels=lev + [g.P.max() * 1.01], colors=shades, zorder=zorder)
    if lines:
        ax.contour(g.X, g.Y, g.P, levels=lev, colors=color or S.MUTED, linewidths=lw, zorder=zorder + 0.1)
    return ax


def _level(g, p):
    from .problems import hdr_levels
    return hdr_levels(g.P, (p,))[p]


def frame(ax, problem, equal=False, labels=True):
    ext = problem.extent
    ax.set_xlim(ext[0], ext[1])
    ax.set_ylim(ext[2], ext[3])
    if labels:
        ax.set_xlabel(problem.labels[0])
        ax.set_ylabel(problem.labels[1])
    if equal:
        ax.set_aspect("equal")
    return ax


def samples(ax, X, color, n_max=4000, s=3.0, alpha=0.35, rng=None, zorder=3, label=None):
    X = np.asarray(X)
    if len(X) > n_max:
        rng = rng or np.random.default_rng(0)
        X = X[rng.choice(len(X), n_max, replace=False)]
    ax.scatter(X[:, 0], X[:, 1], s=s, c=color, alpha=alpha, lw=0, zorder=zorder, label=label, rasterized=True)
    return ax


def trace(ax, x, color, lw=0.7, alpha=1.0, label=None):
    ax.plot(np.arange(len(x)), x, color=color, lw=lw, alpha=alpha, label=label)
    ax.set_xlim(0, len(x))
    return ax


def posterior_panel(ax, problem, g=None, X=None, color=None, title=None, equal=False, **kw):
    g = g or make_grid(problem)
    density(ax, g)
    if X is not None:
        samples(ax, X, color, **kw)
    frame(ax, problem, equal=equal)
    if title:
        ax.set_title(title)
    return g


def ellipse(ax, mean, cov, nsig=2.0, color=S.INK, lw=1.2, ls="-", label=None, zorder=4):
    """Draw the nsig-sigma ellipse of a 2-D Gaussian."""
    ev, U = np.linalg.eigh(cov)
    t = np.linspace(0, 2 * np.pi, 200)
    circ = np.stack([np.cos(t), np.sin(t)])
    pts = (U * (nsig * np.sqrt(np.maximum(ev, 0)))) @ circ + np.asarray(mean)[:, None]
    ax.plot(pts[0], pts[1], color=color, lw=lw, ls=ls, label=label, zorder=zorder)
    return ax


def legend_handles(items):
    """items: list of (label, color, kind) with kind in {'dot', 'line', 'region'}."""
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    hs = []
    for lab, col, kind in items:
        if kind == "dot":
            hs.append(Line2D([], [], marker="o", ls="", color=col, markersize=6, label=lab))
        elif kind == "region":
            hs.append(Patch(facecolor=col, edgecolor=S.MUTED, lw=0.6, label=lab))
        else:
            hs.append(Line2D([], [], color=col, lw=1.6, label=lab))
    return hs


def truth_legend():
    return legend_handles([("exact posterior: 50 / 90 / 99 % regions", "#c3c2b7", "region")])
