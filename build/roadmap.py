"""Draw figures/roadmap.png: the four topics, the reading order of Topic 2 (sections A-G) and Topic 3."""
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

import bayeskit as bk
from bayeskit import style as S

T2 = [  # (section title, [(id, short label, optional?)])
    ("A · Probability", [("2.01", "deterministic physics,\nuncertain answers", 0), ("2.02", "Bayes in pictures", 0),
                         ("2.03", "probability\ndistributions", 0), ("2.04", "conditional probability,\nBayes' rule", 0),
                         ("2.05", "LLN and CLT", 0), ("2.06", "matrix and\nnotation primer", 0)]),
    ("B · Linear inversion, priors", [("2.07", "least squares", 0), ("2.08", "Bayesian linear\ninversion, MAP", 0),
                                      ("2.09", "regularisation:\nsmoothing, L1, L2", 0), ("2.10", "loss functions,\noptimisation", 0),
                                      ("2.11", "choosing a prior", 0), ("2.12", "Gaussian-process\npriors", 0)]),
    ("C · Errors, weighting", [("2.13", "model errors\nC_d and C_p", 0), ("2.14", "noise models,\nbuilding C_d", 0)]),
    ("D · Sampling", [("2.15", "map of methods", 0), ("2.16", "Metropolis", 0), ("2.17", "JAX, autodiff\n(GPU: in preparation)", 0),
                      ("2.18", "HMC and NUTS", 0), ("2.19", "convergence,\nR-hat, MC error", 0), ("2.20", "tempering, SMC", 0),
                      ("2.21", "variational\ninference", 1), ("2.22", "adjoints", 1)]),
    ("E · Time and ensembles", [("2.23", "Kalman filter", 0), ("2.24", "EnKF, EKI, EnKI", 0),
                                ("2.25", "nonlinearity, EKF", 0), ("2.26", "localized EnKI", 1)]),
    ("F · Hyperparameters, weights", [("2.27", "entropy", 0), ("2.28", "KL divergence", 0), ("2.29", "ABIC, AIC, BIC", 0),
                                      ("2.30", "hierarchical Bayes,\ndata weights", 0), ("2.31", "to smooth or not", 0)]),
    ("G · Checking", [("2.32", "using and checking\na posterior", 0), ("2.33", "capstone:\none inversion", 0)]),
]
T3 = [("3.01", "surrogates 101"), ("3.02", "neural networks,\nResNets"), ("3.03", "GP surrogates,\nCES"),
      ("3.04", "physics-informed\nnetworks"), ("3.05", "neural operators\n(FNO)"), ("3.06", "reduced-order models,\nreal time"),
      ("3.07", "inference with\nsurrogates")]

S.use()
FW, FH = 20.0, 11.6
fig, ax = plt.subplots(figsize=(FW, FH))
ax.set_xlim(0, FW); ax.set_ylim(0, FH); ax.axis("off")
W, H, GAP = 2.3, 0.78, 0.18
pos = {}


def box(key, x, y, label, ec=S.AXIS, ls="-", lw=1.3, fs=8.6, show_id=True):
    ax.add_patch(FancyBboxPatch((x, y), W, H, boxstyle="round,pad=0.02,rounding_size=0.08", fc=S.SURFACE, ec=ec,
                                lw=lw, ls=ls, zorder=2))
    txt = (f"$\\bf{{{key}}}$  " if show_id else "") + label
    ax.text(x + W / 2, y + H / 2, txt, ha="center", va="center", fontsize=fs, color=S.INK, linespacing=1.2, zorder=3)
    pos[key] = (x, y)


def arrow(a, b, color=S.MUTED, lw=1.0, ls="-", rad=0.0):
    (xa, ya), (xb, yb) = pos[a], pos[b]
    if abs(ya - yb) > 1.5 * H or abs(xa - xb) < 1e-6:        # vertical link (same column, or down to another band)
        p0, p1 = ((xa + W / 2, ya), (xb + W / 2, yb + H)) if ya > yb else ((xa + W / 2, ya + H), (xb + W / 2, yb))
    else:                                                      # side by side: right of a to left of b
        p0, p1 = (xa + W, ya + H / 2), (xb, yb + H / 2)
    ax.annotate("", xy=p1, xytext=p0, arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, ls=ls, shrinkA=1, shrinkB=1,
                                                      connectionstyle=f"arc3,rad={rad}"), zorder=1)


# Topic 2: seven sections, read top to bottom, left to right
X0, DX, TOP = 2.95, 2.45, 10.25
ax.text(X0, 10.95, "Topic 2 · Inverse problems and Bayesian inversion", fontsize=13, fontweight="bold", color=S.INK)
titles = []
for j, (title, items) in enumerate(T2):
    x = X0 + j * DX
    titles.append(ax.text(x + W / 2, TOP + 0.25, title, ha="center", va="center", fontsize=9.6, fontweight="bold",
                          color=S.INK2))
    keys = []
    for i, (key, lab, opt) in enumerate(items):
        box(key, x, TOP - (i + 1) * (H + GAP) + GAP, lab, ec=(S.MUTED if opt else S.AXIS), ls=("--" if opt else "-"))
        keys.append(key)
    for a_, b_ in zip(keys[:-1], keys[1:]):
        arrow(a_, b_)
for j in range(len(T2) - 1):                                  # section order: arrows between the column titles
    xa, xb = X0 + j * DX + W / 2, X0 + (j + 1) * DX + W / 2
    ax.annotate("", xy=(xb - 1.08, TOP + 0.25), xytext=(xa + 1.08, TOP + 0.25),
                arrowprops=dict(arrowstyle="-|>", color=S.MUTED, lw=1.0), zorder=1)
# Topic 3
Y3 = 0.85
ax.text(X0 + 2 * DX, Y3 + H + 0.32, "Topic 3 · Forward surrogate models (under development; CNN, RNN, GNN planned)",
        fontsize=13, fontweight="bold", color=S.INK)
for j, (key, lab) in enumerate(T3):
    box(key, X0 + j * DX, Y3, lab, ec=S.INK2)
for a, b in zip([k for k, _ in T3][:-1], [k for k, _ in T3][1:]):
    arrow(a, b)
arrow("2.12", "3.01", color=S.INK2, lw=1.2, ls=(0, (4, 3)))
ym = (pos["2.12"][1] + Y3 + H) / 2                               # label to the left of the midpoint of the dashed link
xm = (pos["2.12"][0] + pos["3.01"][0]) / 2 + W / 2
ax.text(xm - 0.25, ym, "Topic 3 can start after\n2.10 (losses) and\n2.13 (model errors)", fontsize=8.4, color=S.INK2,
        va="center", ha="right")

# Topics 1 and 4 (planned)
box("T1", 0.25, pos["2.01"][1], "Topic 1  (planned)\nforward problems and\nnumerical methods", ec=S.MUTED, ls="--", fs=8.6,
    show_id=False)
arrow("T1", "2.01", color=S.MUTED, ls=(0, (4, 3)))
box("T4", pos["2.14"][0], pos["2.14"][1] - 2 * (H + GAP), "Topic 4  (planned)\ndata and data processing",
    ec=S.MUTED, ls="--", fs=8.6, show_id=False)
arrow("T4", "2.14", color=S.MUTED, ls=(0, (4, 3)))
pos["T4"] = (pos["T4"][0] + 1e-3, pos["T4"][1])               # (not used again)

ax.text(0.25, 0.22, "solid boxes: core path, read in order (arrows)    dashed boxes: optional on a first pass    "
        "dashed arrows: planned topics, and where Topic 3 branches off", fontsize=9, color=S.INK2)
fig.tight_layout()
bk.savefig(fig, "roadmap")
