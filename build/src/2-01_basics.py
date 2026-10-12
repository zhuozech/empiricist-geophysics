# %% [markdown]
# # 2.01 · Basics: deterministic physics, uncertain answers
#
# The physics in a geophysical inversion is deterministic. Give the model a fault slip (how far the two sides of a
# fault moved past each other, usually specified on many small fault *patches*) and it returns one exact
# displacement at every GPS site; give it an epicentre and it returns one exact distance to every station. The Earth
# also has one true slip distribution and one true epicentre. So why does the rest of this tutorial talk about
# probability, about posteriors that have *shapes*, and about *crowds* of samples? Two kinds of error turn that one
# Earth into many possible answers: noisy measurements (section 2) and imperfect physics (section 7).
#
# This notebook answers that with one small example: locating an earthquake from its distance to seismic stations.
# The unknowns are two numbers, the epicentre's east and north coordinates. That is what "two parameters" means in
# every notebook, and it is why every picture is a map: **each point on the map is one possible epicentre**.
#
# **Before this:** nothing. · **You will be able to:** explain why deterministic physics still needs probability;
# read a posterior as a crowd of possible Earths; say what a 90 % credible region does and does not promise; tell
# measurement noise from model error. · **Optional sections:** 9 (what "not applicable" means in 2.15).
#
# **Words this notebook defines:** deterministic, random, uncertain, forward model, measurement noise, model error,
# prior, likelihood, posterior, candidate, crowd (samples, ensemble), shape, true value, exact posterior, credible
# region, highest-density region, coverage, "not applicable".

# %% [markdown]
# **Problems in this notebook**
#
# | problem | unknowns | data | forward model | linear? |
# |---|---|---|---|---|
# | epicentre from station distances (sections 1-6) | 2 (east, north) | 1, 2, 3 or 6 distances | $d_i = \lvert\mathbf m - \mathbf s_i\rvert$ (distance to station $i$) | nonlinear |
# | epicentre with wave-speed (path) errors (section 7) | 2 | 3 distances | $d_i = (1+\varepsilon_i)\lvert\mathbf m - \mathbf s_i\rvert + e_i$, path error $\varepsilon_i$ unknown | nonlinear |
# | one direct measurement per unknown (section 8) | 1 to 700 | one per unknown | $d_i = m_i$ | linear |

# %%
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.ndimage import gaussian_filter
import jax.numpy as jnp

import bayeskit as bk
from bayeskit import problems as P, plotting as BP, style as S

STAR = np.array([0.3, 0.8])                 # the true epicentre (east, north), in units of 10 km
SIG = 0.06                                  # measurement noise of one distance: 0.06 units = 0.6 km
M0, PRIOR_SD = np.array([0.0, 0.5]), 1.0    # regional prior: earthquakes occur around here, give or take 10 km
ST = np.array([[-1.0, 0.0], [1.0, 0.0], [0.2, 1.6],          # station positions; stations 1-3 are the ones used
               [1.6, 1.4], [-1.3, 1.7], [0.6, -1.2]])        # in notebooks 2.15 and 2.02; 4-6 only in section 6
EXT = (-2.8, 2.6, -2.3, 3.1)                                 # map window


def exact_distance(m, stations):
    """The forward model: the distance from epicentre m to each station. Deterministic."""
    return np.linalg.norm(stations - np.asarray(m)[None, :], axis=1)


# %% [markdown]
# ## 1. The forward model is deterministic
#
# Put the earthquake somewhere (the star) and compute its distance to three stations (the triangles). This is the
# **forward model**: cause (where the earthquake is) goes in, effect (the distances) comes out. Running it twice
# gives the same numbers to the last digit, because there is nothing random in it.

# %%
d_once = exact_distance(STAR, ST[:3])
d_again = exact_distance(STAR, ST[:3])
print("station   position (east, north)   exact distance (x 10 km)")
for i in range(3):
    print(f"   {i + 1}       ({ST[i, 0]:+.1f}, {ST[i, 1]:+.1f})              {d_once[i]:.4f}")
print("same numbers when run again:", np.array_equal(d_once, d_again))

fig, ax = plt.subplots(figsize=(5.4, 4.6))
for i in range(3):
    ax.plot([STAR[0], ST[i, 0]], [STAR[1], ST[i, 1]], color=S.MUTED, lw=1.0, zorder=1)
    mid = 0.5 * (STAR + ST[i])
    ax.text(mid[0], mid[1], f" {d_once[i]:.3f}", fontsize=8.5, color=S.INK2, ha="left", va="bottom")
    ax.plot(*ST[i], "^", color=S.INK, ms=9, zorder=3)
    above = ST[i, 1] > STAR[1]
    ax.text(ST[i, 0], ST[i, 1] + (0.14 if above else -0.18), f"station {i + 1}", ha="center",
            va="bottom" if above else "top", fontsize=8.5)
ax.plot(*STAR, "*", color=S.INK, ms=14, mec=S.SURFACE, zorder=4)
ax.text(STAR[0] + 0.1, STAR[1] + 0.1, "epicentre", fontsize=8.5)
ax.set_xlim(-1.8, 2.0); ax.set_ylim(-0.7, 2.2); ax.set_aspect("equal")
ax.set_xlabel("east (x 10 km)"); ax.set_ylabel("north (x 10 km)")
ax.set_title("Forward model: epicentre in, exact distances out")
fig.tight_layout()
bk.savefig(fig, "2-01_forward")
plt.show()

# %% [markdown]
# The same is true for a slip inversion: slip on every fault patch goes in, the displacement at every GPS site comes
# out, through the Green's functions ($\mathbf d = \mathbf G\mathbf m$). A **Green's function** is the displacement at one
# site caused by one unit of slip on one patch; stacking them for every site and patch gives the matrix $\mathbf G$, and
# multiplying by the slip on every patch ($\mathbf m$) adds up their effects. One input, one output, every time.
#
# **No probability is needed to go forward.** The question this tutorial is about runs the other way.

# %% [markdown]
# ## 2. The first kind of error: measurement noise
#
# A real station does not report the exact distance. Picking the arrival time and clock errors add a small error that
# changes from one measurement to the next: the **measurement noise**. Its typical size is written σ (sigma), the
# **standard deviation** of the noise: for the bell-shaped (Gaussian) noise used here, about two thirds of the errors
# are smaller than σ and about 95 % are smaller than 2σ (2.03 defines all of this precisely). Here is the same distance
# measured 20 times with noise σ = 0.06 (0.6 km), and the same experiment repeated 10,000 times.

# %%
rng = np.random.default_rng(2)
d_true1 = d_once[0]
twenty = d_true1 + SIG * rng.standard_normal(20)
many = d_true1 + SIG * rng.standard_normal(10_000)
print(f"exact distance to station 1: {d_true1:.4f}")
print(f"20 measurements: from {twenty.min():.4f} to {twenty.max():.4f}")
print(f"share of the 10,000 measurements within ±2σ of the exact distance: {np.mean(np.abs(many - d_true1) < 2 * SIG):.3f}")

fig, axs = plt.subplots(1, 2, figsize=(11, 3.3), gridspec_kw=dict(width_ratios=[1.3, 1]))
ax = axs[0]
ax.axhspan(d_true1 - SIG, d_true1 + SIG, color="#ecebe6", lw=0, label="exact ± σ")
ax.axhline(d_true1, color=S.INK, lw=1.4, label="exact distance")
ax.plot(np.arange(1, 21), twenty, "o", color=S.INK2, ms=5, label="measurement")
ax.set_xlabel("measurement number"); ax.set_ylabel("distance (x 10 km)")
ax.set_xticks([1, 5, 10, 15, 20])
ax.set_title("The same distance, measured 20 times")
ax.legend(fontsize=8, loc="upper right", ncol=3)
ax = axs[1]
ax.hist(many, bins=60, color="#c3c2b7", edgecolor=S.SURFACE, lw=0.3)
ax.axvline(d_true1, color=S.INK, lw=1.4)
for k in (-2, 2):
    ax.axvline(d_true1 + k * SIG, color=S.MUTED, lw=0.8)
ax.set_xlabel("measured distance (x 10 km)"); ax.set_ylabel("how often")
ax.set_title("10,000 repeats: most within ±2σ")
fig.tight_layout()
bk.savefig(fig, "2-01_noise")
plt.show()

# %% [markdown]
# The Earth did not change between measurements, so the spread comes from the measurement noise alone: the part of
# the error that changes from one repeat to the next. It is the only *random* ingredient. There is a second kind of
# error that does not change between repeats: our forward model is not the real Earth (a wave speed, an elastic
# structure or a fault geometry that is slightly wrong). Sections 3 to 6 pretend the forward model is perfect;
# section 7 adds this **model error**. Either way the epicentre itself is fixed; what is *uncertain* is our knowledge
# of it.

# %% [markdown]
# ## 3. Going backwards: one station
#
# Now turn the question around. Station 1 reports a distance (one noisy measurement, value printed below). Where
# could the earthquake be?
#
# * **(a) Even a perfect, noise-free measurement does not give one answer.** Every point on a circle around the
#   station has exactly that distance. The forward model is deterministic, yet many causes produce the same effect,
#   so going backward there is a whole set of answers.
# * **(b) With noise**, the circle becomes a ring about ±2σ wide: points slightly inside or outside the circle would
#   also have produced this measurement with ordinary noise.
# * **(c) With what we knew beforehand** (the *prior*: earthquakes in this region occur around (0, 0.5), give or take
#   one map unit), the part of the ring near that region is more plausible than the far side. The result is an arc,
#   a banana.

# %%
D = exact_distance(STAR, ST) + np.r_[[0.02, -0.03, 0.01], SIG * np.random.default_rng(4).standard_normal(3)]
print(f"measured distance at station 1: {D[0]:.4f}  (exact {d_once[0]:.4f})")


def make_problem(idx, prior_sd=PRIOR_SD, ext=EXT):
    """Epicentre from the measured distances at the stations in idx, with a Gaussian prior around M0."""
    idx = list(idx)
    st = jnp.asarray(ST[idx])
    fwd = lambda m: jnp.sqrt(jnp.sum((m[None, :] - st) ** 2, axis=1) + 1e-12)
    return P.InverseProblem(fwd, D[idx], SIG, m0=M0, P=np.eye(2) * prior_sd ** 2, extent=ext,
                            labels=("east (x 10 km)", "north (x 10 km)"), truth=STAR)


def base_map(ax, idx, ext=EXT, star=True, labels=True):
    for i in idx:
        ax.plot(*ST[i], "^", color=S.INK, ms=8, zorder=6)
    if star:
        ax.plot(*STAR, "*", color=S.INK, ms=12, mec=S.SURFACE, mew=1.0, zorder=7)
    ax.set_xlim(ext[0], ext[1]); ax.set_ylim(ext[2], ext[3]); ax.set_aspect("equal")
    if labels:
        ax.set_xlabel("east (x 10 km)"); ax.set_ylabel("north (x 10 km)")


banana = make_problem([0])
ring_only = P.Target(lambda m: banana.loglik_fn(m), 2, extent=EXT)       # the measurement alone, no prior
th = np.linspace(0, 2 * np.pi, 400)
circ = lambda r: (ST[0, 0] + r * np.cos(th), ST[0, 1] + r * np.sin(th))

fig, axs = plt.subplots(1, 3, figsize=(14, 4.4))
ax = axs[0]
ax.plot(*circ(D[0]), color=S.INK, lw=1.4)
for a in np.deg2rad([20, 70, 130, 200, 260, 320]):
    p = ST[0] + D[0] * np.array([np.cos(a), np.sin(a)])
    ax.plot([ST[0, 0], p[0]], [ST[0, 1], p[1]], color=S.MUTED, lw=0.8)
    ax.plot(*p, "o", color=S.INK2, ms=5)
base_map(ax, [0])
ax.set_title("(a) noise-free: every point on the circle\nhas exactly the measured distance", fontsize=10)
ax = axs[1]
BP.density(ax, P.grid(ring_only, n=450))
for k in (-2, 2):
    ax.plot(*circ(D[0] + k * SIG), color=S.MUTED, lw=0.6)
base_map(ax, [0])
ax.set_title("(b) with noise: a ring about ±2σ wide", fontsize=10)
ax = axs[2]
BP.density(ax, P.grid(banana, n=450))
ax.add_patch(plt.Circle(M0, 2 * PRIOR_SD, fill=False, ls=(0, (1, 2)), color=S.INK2, lw=0.9))
ax.text(M0[0] + 1.45, M0[1] + 1.55, "prior:\n2 sd", fontsize=8, color=S.INK2)
base_map(ax, [0])
ax.set_title("(c) ring × prior: an arc (banana)", fontsize=10)
fig.tight_layout()
bk.savefig(fig, "2-01_backwards_one_station")
plt.show()

# %% [markdown]
# Grey shading means "plausible": the darkest band holds half of the plausibility, the next 90 %, the lightest 99 %
# (section 5 makes this precise). The star is the true epicentre; it lies on the ring, as it must.
#
# **This is the heart of the matter.** The answer to an inverse problem is not one point but a *region* of the
# parameter map, and inside that region some places are more plausible than others. Probability is simply the
# bookkeeping for "many possible answers, some more plausible than others". It does not mean the Earth is random.

# %% [markdown]
# ## 4. The posterior is a crowd of possible Earths
#
# Here is the most useful way to picture a posterior. It needs no formula beyond one weighted coin.
#
# 1. Draw many **candidate** epicentres from the prior (where we believed earthquakes occur).
# 2. For each candidate, run the deterministic forward model: what distance *would* station 1 have measured?
# 3. Compare with what it *did* measure, in units of the noise, and keep the candidate with probability
#
# $$p_\text{keep} = \exp\!\Big[-\tfrac12\Big(\frac{d_\text{measured} - d_\text{predicted}}{\sigma}\Big)^2\Big].$$
#
# In words: a candidate whose prediction matches the measurement perfectly is always kept; one that misses by one
# noise unit is kept a little more than half the time; one that misses by three units almost never (table below).
# The candidates that survive are the **posterior**. Nothing more is meant by the word. (This recipe is called
# *rejection sampling*. Its survivors follow Bayes' rule exactly: prior, because the candidates came from it; times
# likelihood, because each was kept in proportion to how well it fits.)

# %%
print("miss (in noise units)   chance of being kept")
for miss in [0, 0.5, 1, 2, 3, 4]:
    print(f"        {miss:3.1f}                {np.exp(-0.5 * miss ** 2):.4f}")


def crowd(prob, n, rng):
    """Rejection sampling: candidates from the prior, kept with probability exp(-misfit^2 / 2)."""
    C = prob.sample_prior(rng, n)                       # candidate epicentres, drawn from the prior
    pred = prob.forward_batch(C)                        # deterministic forward run for each candidate
    miss = (prob.y - pred) / prob.sigma                 # how far each prediction misses, in noise units
    keep_p = np.exp(-0.5 * np.sum(miss ** 2, axis=1))
    keep = rng.uniform(size=n) < keep_p                 # one weighted coin per candidate
    return C, pred, keep


rng_c = np.random.default_rng(3)
C1, pred1, keep1 = crowd(banana, 20_000, rng_c)
pred1 = pred1[:, 0]
print(f"\none station: tried {len(C1)}, kept {keep1.sum()} ({100 * keep1.mean():.1f} %)")

# %%
# the movie: four example candidates one at a time (three rejected, one kept), then everyone else in growing batches
rej_idx, kept_idx = np.flatnonzero(~keep1), np.flatnonzero(keep1)
first = [rej_idx[0], rej_idx[1], kept_idx[0], rej_idx[2]]
order = np.r_[first, np.setdiff1d(np.arange(len(C1)), first)]
Co, po, ko = C1[order], pred1[order], keep1[order]
counts = [1, 2, 3, 4, 7, 12, 20, 32, 50, 80, 125, 200, 320, 500, 800, 1250, 2000, 3200, 5000, 8000, 12500, 20000]
bins = np.linspace(0, 4.6, 93)


def movie_frame(n, highlight_single, final_name=None):
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 4.3), dpi=88, gridspec_kw=dict(width_ratios=[1.05, 1]))
    ax = axs[0]
    ax.plot(*circ(D[0]), color=S.INK, lw=1.0, zorder=1)
    old = slice(0, n)
    rej = ~ko[old]
    size = 28 if n <= 20 else (12 if n <= 300 else 4)            # few candidates: big dots, so each one is visible
    ax.scatter(Co[old][rej, 0], Co[old][rej, 1], s=size, color="#c3c2b7", lw=0, zorder=2)
    ax.scatter(Co[old][~rej, 0], Co[old][~rej, 1], s=size * 1.4, color=S.INK, lw=0, alpha=0.75, zorder=3)
    if highlight_single:
        c, ok = Co[n - 1], ko[n - 1]
        ax.plot([ST[0, 0], c[0]], [ST[0, 1], c[1]], color=S.GOOD if ok else S.BAD, lw=1.4, zorder=4)
        ax.plot(*c, "o" if ok else "X", color=S.GOOD if ok else S.BAD, ms=9, mec=S.SURFACE, zorder=5)
    base_map(ax, [0], star=False)
    ax.set_title("candidate epicentres (black: kept, grey: rejected)", fontsize=9.5)
    S.badge(ax, f"tried {n:,} · kept {ko[:n].sum():,}", loc="lower left", fs=8.5)
    ax = axs[1]
    ax.axvspan(D[0] - 2 * SIG, D[0] + 2 * SIG, color="#ecebe6", lw=0)
    ax.hist(po[:n], bins=bins, color="#d9d8d1", label="all candidates tried")
    ax.hist(po[:n][ko[:n]], bins=bins, color=S.INK, label="kept")
    ax.axvline(D[0], color=S.INK, lw=1.2, label="measured distance (band: ±2σ)")
    ax.set_xlabel("distance the candidate predicts at station 1 (x 10 km)")
    ax.set_ylabel("number of candidates")
    ax.set_title("each candidate's prediction vs the measurement", fontsize=9.5)
    top = ax.get_ylim()[1]
    ax.set_ylim(0, max(2.6, top * 1.35))                          # headroom for the legend and the badge
    ax.yaxis.set_major_locator(plt.MaxNLocator(integer=True))
    ax.legend(fontsize=7.5, loc="upper right")
    if highlight_single:
        c_p, ok = po[n - 1], ko[n - 1]
        miss = abs(D[0] - c_p) / SIG
        ax.plot(c_p, 0, "o" if ok else "X", color=S.GOOD if ok else S.BAD, ms=10, mec=S.SURFACE, clip_on=False, zorder=5)
        S.badge(ax, f"candidate {n} predicts {c_p:.2f}\nmisses by {miss:.1f} σ\n"
                    f"keep chance {np.exp(-0.5 * miss ** 2):.2f}: {'kept' if ok else 'rejected'}",
                ok=bool(ok), loc="upper left", fs=8)
    fig.tight_layout()
    if final_name:                                                # a still at full resolution instead of a GIF frame
        bk.savefig(fig, final_name)
        plt.show()
        return None
    rgb = S.fig_to_rgb(fig)
    plt.close(fig)
    return rgb


frames = []
for n in counts:
    single = n <= 4
    frames += [movie_frame(n, single)] * (9 if single else 2)
gif = S.frames_to_gif(frames, S.FIG / "2-01_crowd_forming.gif", fps=6, hold_last=10)
print("GIF size %.1f MB" % (gif.stat().st_size / 1e6))
S.show_gif(gif, width=760)
print("the last frame at full resolution:")
movie_frame(counts[-1], False, final_name="2-01_crowd_forming_final")

# %%
fig, axs = plt.subplots(1, 2, figsize=(11, 4.5))
ax = axs[0]
ax.scatter(*C1[~keep1].T, s=3, color="#d9d8d1", lw=0, label="rejected candidates", rasterized=True)
ax.scatter(*C1[keep1].T, s=4, color=S.INK, lw=0, alpha=0.6, label="kept candidates: the posterior", rasterized=True)
base_map(ax, [0])
ax.legend(fontsize=8, loc="upper right", markerscale=3)
ax.set_title(f"All {len(C1):,} candidates", fontsize=10)
ax = axs[1]
ax.scatter(*C1[keep1].T, s=4, color=S.INK, lw=0, alpha=0.6, rasterized=True)
base_map(ax, [0])
ax.set_title(f"The {keep1.sum():,} survivors: a crowd of possible epicentres", fontsize=10)
fig.tight_layout()
bk.savefig(fig, "2-01_crowd")
plt.show()

# %% [markdown]
# The survivors trace the same arc as the grey banana of section 3, without any formula for the arc: it emerges
# from "run the physics, keep what fits".
#
# **Each survivor is a complete, sharp, deterministic Earth.** The next picture uses all three stations, picks 8
# survivors and checks each one: its predicted distances (dots) against the measured ones (lines, with the ±2σ band).
# Every survivor's predictions sit close to every measurement: almost all inside the band (count printed below), and
# a point just outside is ordinary noise, since about one measurement in twenty lands outside ±2σ (section 2). One
# rejected candidate (open circle) is shown for contrast: it misses at least one station by several noise units.

# %%
three = make_problem([0, 1, 2])
C3, pred3, keep3 = crowd(three, 1_000_000, np.random.default_rng(5))
print(f"three stations: tried {len(C3):,}, kept {keep3.sum():,} ({100 * keep3.mean():.2f} %)")
rng8 = np.random.default_rng(8)
pick = rng8.choice(np.flatnonzero(keep3), 8, replace=False)
miss_all = np.abs((three.y - pred3) / SIG)
ZOOM = (STAR[0] - 0.32, STAR[0] + 0.32, STAR[1] - 0.3, STAR[1] + 0.3)
in_zoom = (np.abs(C3[:, 0] - STAR[0]) < 0.25) & (np.abs(C3[:, 1] - STAR[1]) < 0.22)
near_rej = np.flatnonzero((~keep3) & in_zoom & (miss_all.max(1) > 3) & (miss_all.max(1) < 6))
rej1 = near_rej[0]

fig, axs = plt.subplots(1, 4, figsize=(15, 3.9), gridspec_kw=dict(width_ratios=[1.15, 1, 1, 1]))
ax = axs[0]
ax.scatter(*C3[keep3].T, s=3, color="#a3a29b", lw=0, alpha=0.6, rasterized=True)
for k, j in enumerate(pick):
    ax.plot(*C3[j], "o", color=S.INK, ms=5)
    ax.text(C3[j, 0] + 0.008, C3[j, 1] + 0.008, str(k + 1), fontsize=8)
ax.plot(*C3[rej1], "o", mfc="none", mec=S.INK, ms=6)
ax.text(C3[rej1, 0] + 0.01, C3[rej1, 1], " rejected", fontsize=7.5, color=S.INK2)
base_map(ax, [], ext=ZOOM)
ax.set_title("8 survivors (numbered) and their crowd", fontsize=10)
for i in range(3):
    ax = axs[i + 1]
    ax.axhspan(D[i] - 2 * SIG, D[i] + 2 * SIG, color="#ecebe6", lw=0)
    ax.axhline(D[i], color=S.INK, lw=1.2)
    ax.plot(np.arange(1, 9), pred3[pick, i], "o", color=S.INK, ms=6)
    ax.plot(10, pred3[rej1, i], "o", mfc="none", mec=S.INK, ms=7)
    ax.set_xticks(list(range(1, 9)) + [10]); ax.set_xticklabels([str(k) for k in range(1, 9)] + ["rej."], fontsize=8)
    ax.set_ylim(D[i] - 7 * SIG, D[i] + 7 * SIG)
    ax.set_xlabel("candidate")
    ax.set_title(f"station {i + 1}: measured (line) vs predicted", fontsize=9.5)
axs[1].set_ylabel("distance (x 10 km)")
fig.tight_layout()
bk.savefig(fig, "2-01_eight_earths")
plt.show()
inside_band = np.abs(pred3[pick] - D[:3]) < 2 * SIG
print(f"{inside_band.sum()} of the {inside_band.size} predictions of the 8 survivors fall inside the ±2σ band;"
      f" largest miss {miss_all[pick].max():.1f} noise units")
print("rejected example misses the three stations by", np.round(miss_all[rej1], 1), "noise units")

# %% [markdown]
# **The image to keep: the posterior is not a blurry Earth. It is the list of sharp Earths you cannot rule out.**
# Its spread is your uncertainty. With more or better data, fewer candidates survive and the crowd shrinks; with
# worse data it grows. In a slip inversion each survivor is a complete slip map that fits all the GNSS (satellite
# positioning such as GPS) and tsunami data within their noise, and the "±2σ" uncertainty plotted at each patch is the
# spread of the crowd there.

# %% [markdown]
# ## 5. What "the shape of the posterior" means
#
# With two parameters every survivor is a dot on the map, so the crowd has an outline: that outline is its
# **shape**. The grey regions used in every notebook are nothing but a smooth way of saying "where the crowd is
# dense". From left to right: the crowd; how many survivors fall in each small square; the same, smoothed, drawn as
# the regions holding 50 %, 90 % and 99 % of the crowd; and the **exact posterior** computed by brute force on a
# fine grid (section 6), with the crowd on top.

# %%
Cb, _, kb = crowd(banana, 200_000, np.random.default_rng(6))
Xb = Cb[kb]
g_ban = P.grid(banana, n=450)
H, xe, ye = np.histogram2d(Xb[:, 0], Xb[:, 1], bins=[np.linspace(EXT[0], EXT[1], 61), np.linspace(EXT[2], EXT[3], 61)])
Hs = gaussian_filter(H.T, 1.0)
xc, yc = 0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1])
lev = sorted(set(P.hdr_levels(Hs).values()))

fig, axs = plt.subplots(1, 4, figsize=(15.5, 4.1))
axs[0].scatter(*Xb.T, s=2, color=S.INK, lw=0, alpha=0.35, rasterized=True)
axs[0].set_title(f"(a) the crowd: {len(Xb):,} survivors", fontsize=10)
axs[1].imshow(H.T, origin="lower", extent=(EXT[0], EXT[1], EXT[2], EXT[3]), cmap=S.GREYS, aspect="equal")
axs[1].set_title("(b) survivors per small square", fontsize=10)
axs[2].contourf(xc, yc, Hs, levels=lev + [Hs.max() * 1.01], colors=["#ecebe6", "#d9d8d1", "#c3c2b7"])
axs[2].contour(xc, yc, Hs, levels=lev, colors=S.MUTED, linewidths=0.7)
axs[2].set_title("(c) smoothed: regions holding\n50 %, 90 %, 99 % of the crowd", fontsize=10)
BP.density(axs[3], g_ban)
axs[3].scatter(*Xb[:3000].T, s=2, color=S.INK, lw=0, alpha=0.5, rasterized=True)
axs[3].set_title("(d) exact posterior (grey)\nwith the crowd on top", fontsize=10)
for ax in axs:
    base_map(ax, [0], labels=False)
fig.tight_layout()
bk.savefig(fig, "2-01_crowd_to_shape")
plt.show()
for q in (0.5, 0.9, 0.99):
    print(f"share of the crowd inside the exact posterior's {int(100 * q)} % region: {g_ban.inside(Xb, q):.3f}")

# %% [markdown]
# The shares printed above are what "the 50 % region" means: the darkest patch contains half of the crowd, the 90 %
# region nine tenths. Two independent routes (keeping candidates by coin flips, and evaluating every grid point)
# give the same shape, because both compute the same thing.
#
# The shape carries physical information. Same epicentre, same measurements; only the number of stations and the
# prior change:

# %%
shapes = [("1 station, broad prior:\na ring (direction unknown)", make_problem([0], prior_sd=3.0), [0], 400_000),
          ("1 station, regional prior:\na banana", banana, [0], 200_000),
          ("2 stations on a line:\ntwo islands (north or south?)", make_problem([0, 1]), [0, 1], 600_000),
          ("3 stations:\none small blob", three, [0, 1, 2], 1_000_000)]
fig, axs = plt.subplots(1, 4, figsize=(15.5, 4.2))
for k, (ax, (title, prob, idx, n)) in enumerate(zip(axs, shapes)):
    Cs, _, ks = crowd(prob, n, np.random.default_rng(20 + k))
    BP.density(ax, P.grid(prob, n=450))
    ax.scatter(*Cs[ks][:2500].T, s=2, color=S.INK, lw=0, alpha=0.45, rasterized=True)
    base_map(ax, idx, labels=(k == 0))
    ax.set_title(title, fontsize=10)
    print(f"{title.splitlines()[0]:28s}: tried {n:,}, kept {ks.sum():,} ({100 * ks.mean():.2f} %)")
fig.tight_layout()
bk.savefig(fig, "2-01_four_shapes")
plt.show()

# %% [markdown]
# How to read a shape:
# * **Long direction**: a combination of parameters the data cannot fix (along the ring you can move the epicentre
#   without changing its distance to station 1). In slip inversions this is the classic trade-off between
#   neighbouring patches.
# * **Narrow direction**: what the data do fix (the distance itself).
# * **Two islands**: two different explanations that fit equally well. Two stations on an east-west line measure
#   the same distances for an epicentre north of the line and its mirror image south of it. The northern island holds
#   more of the crowd only because the prior leans north.
# * **One small blob**: the data pin the answer down, and the shape is close to an ellipse (a *Gaussian*, the
#   two-dimensional version of the bell curve; 2.03).
#
# A sampler's job, in every later notebook, is to produce a crowd with the right shape. These two-parameter problems
# are where we can check it against the exact posterior.

# %% [markdown]
# ## 6. Two different things called "truth"
#
# Two words matter from here on, and in casual speech both get called "the truth":
#
# | name | symbol | what it is | do we know it? |
# |---|---|---|---|
# | **true value** | ★ | the epicentre we chose, used to compute the synthetic measurements before adding noise | here yes, because we made the data; in real life never |
# | **exact posterior** | grey 50/90/99 % regions | the correct crowd for the data we actually have, computed by brute force: evaluate prior × likelihood at every point of a fine grid | yes, but only because there are 2 parameters (section 8) |
#
# The exact posterior is the best possible answer *given the data*, and it is the answer key that samplers are
# graded against in the other notebooks ("inside the exact posterior's 90 % region" and so on). Even a perfect method
# can only return that region. **The true value is usually inside it, not necessarily at its centre, and it can be
# outside**: the noise nudged the measurements, the crowd is centred on what the nudged measurements say, and a 90 %
# region leaves 10 % of the plausibility outside by construction.

# %%
X3 = C3[keep3]
off = (X3.mean(0) - STAR) / X3.std(0)
print(f"3 stations: crowd centre {np.round(X3.mean(0), 3)}, true value {STAR}; offset (in crowd sd): {np.round(off, 2)}")
g3 = P.grid(three, n=500)
print(f"is the true value inside the exact posterior's 50 % region? {bool(g3.inside(STAR[None], 0.5) > 0)};"
      f" 90 % region? {bool(g3.inside(STAR[None], 0.9) > 0)}")

# %% [markdown]
# **What a 90 % region is.** A 90 % **credible region** is a region of the map that holds 90 % of the posterior
# probability, given the data we have and the model (prior, forward model, noise) we assumed. It is a statement about
# the crowd for *these* data. Many regions hold 90 %; the one drawn in these notebooks is the **highest-density
# region** (HDR), the smallest of them, which keeps the most plausible points.
#
# **What it implies about the true value** is a separate question, called **coverage**: repeat the synthetic
# experiment many times (same stations, new noise each time, a new exact posterior each time) and count how often the
# true value lands inside that experiment's 90 % region. We do it two ways: (a) always the same true epicentre;
# (b) a new true epicentre each time, drawn from the prior (as if nature picked earthquakes from the region's usual
# pattern).

# %%
def logpost_np(X, Y, d, idx, m0=M0, sd=PRIOR_SD):
    """log prior + log likelihood on a grid of points (numpy version of the same problem)."""
    lp = -0.5 * ((X - m0[0]) ** 2 + (Y - m0[1]) ** 2) / sd ** 2
    for j, i in enumerate(idx):
        lp = lp - 0.5 * ((d[j] - np.sqrt((X - ST[i, 0]) ** 2 + (Y - ST[i, 1]) ** 2)) / SIG) ** 2
    return lp


def one_experiment(star, rng, idx=(0, 1, 2)):
    """New noise, new exact posterior; is the true value inside its 50 % and 90 % regions?"""
    d = exact_distance(star, ST[list(idx)]) + SIG * rng.standard_normal(len(idx))
    xs, ys = np.linspace(-4, 4, 140), np.linspace(-3.5, 4.5, 140)      # coarse search for where the crowd is
    Xc, Yc = np.meshgrid(xs, ys)
    lpc = logpost_np(Xc, Yc, d, idx)
    sel = lpc > lpc.max() - 25
    pad = 2 * (xs[1] - xs[0])
    xf = np.linspace(Xc[sel].min() - pad, Xc[sel].max() + pad, 170)    # fine grid around it
    yf = np.linspace(Yc[sel].min() - pad, Yc[sel].max() + pad, 170)
    XF, YF = np.meshgrid(xf, yf)
    lpf = logpost_np(XF, YF, d, idx)
    mx = lpf.max()
    dA = (xf[1] - xf[0]) * (yf[1] - yf[0])
    Z = np.exp(lpf - mx).sum() * dA
    Pn = np.exp(lpf - mx) / Z
    levels = P.hdr_levels(Pn, (0.5, 0.9))
    at_star = np.exp(logpost_np(np.array(star[0]), np.array(star[1]), d, idx) - mx) / Z
    edge = (Pn[0].sum() + Pn[-1].sum() + Pn[:, 0].sum() + Pn[:, -1].sum()) * dA
    return {q: bool(at_star >= levels[q]) for q in levels}, (XF, YF, Pn, levels[0.9]), edge


N_REP = 400
cover = {}
kept_regions = []
for mode, seed in [("same true epicentre", 11), ("true epicentre drawn from the prior", 12)]:
    rr = np.random.default_rng(seed)
    hits = {0.5: 0, 0.9: 0}
    worst_edge = 0.0
    for k in range(N_REP):
        star = STAR if mode.startswith("same") else M0 + PRIOR_SD * rr.standard_normal(2)
        h, region, edge = one_experiment(star, rr)
        hits[0.5] += h[0.5]; hits[0.9] += h[0.9]
        worst_edge = max(worst_edge, edge)
        if mode.startswith("same") and k < 12:
            kept_regions.append((region, h[0.9]))
    cover[mode] = {q: hits[q] / N_REP for q in hits}
    print(f"{mode:38s}: inside the 50 % region {cover[mode][0.5]:.3f}, inside the 90 % region {cover[mode][0.9]:.3f}"
          f"   ({N_REP} experiments; largest probability left outside a grid window {worst_edge:.0e})")
se = {q: np.sqrt(q * (1 - q) / N_REP) for q in (0.5, 0.9)}
print(f"expected scatter from having only {N_REP} experiments (2 standard errors): 50 %: ±{2 * se[0.5]:.3f}, 90 %: ±{2 * se[0.9]:.3f}")
print(f"of the 12 regions drawn below, {sum(h for _, h in kept_regions)} contain the true value")

# %%
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw=dict(width_ratios=[1.1, 1]))
ax = axs[0]
for (XF, YF, Pn, l90), inside in kept_regions:
    ax.contour(XF, YF, Pn, levels=[l90], colors=[S.INK2 if inside else S.BAD], linewidths=1.0)
ax.plot(*STAR, "*", color=S.INK, ms=14, mec=S.SURFACE, zorder=5)
ax.set_xlim(STAR[0] - 0.25, STAR[0] + 0.25); ax.set_ylim(STAR[1] - 0.22, STAR[1] + 0.22); ax.set_aspect("equal")
ax.set_xlabel("east (x 10 km)"); ax.set_ylabel("north (x 10 km)")
ax.set_title("90 % regions from 12 repeated experiments\n(red outline: misses the true value ★)", fontsize=10)
ax = axs[1]
xpos = np.arange(2)
for j, (mode, mk) in enumerate([("same true epicentre", "o"), ("true epicentre drawn from the prior", "s")]):
    vals = [cover[mode][0.5], cover[mode][0.9]]
    ax.errorbar(xpos + (j - 0.5) * 0.18, vals, yerr=[2 * se[0.5], 2 * se[0.9]], fmt=mk, color=S.INK, ms=7,
                capsize=3, lw=1, label=mode)
for q, x in zip((0.5, 0.9), xpos):
    ax.plot([x - 0.35, x + 0.35], [q, q], color=S.MUTED, lw=1.2)
ax.set_xticks(xpos); ax.set_xticklabels(["50 % region", "90 % region"])
ax.set_ylim(0.3, 1.0)
ax.set_ylabel("share of experiments where ★ is inside")
ax.set_title("How often the region catches the true value\n(grey lines: the promise; bars: ±2 standard errors)", fontsize=10)
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
bk.savefig(fig, "2-01_coverage")
plt.show()

# %% [markdown]
# **In this example the 90 % regions caught the true value about 9 times out of 10, and the 50 % regions about half
# the time** (printed above, both ways). The regions move around from one experiment to the next because the noise
# changes; the true value stays put. The two ways are not equally guaranteed:
# * **(b) true value drawn from the prior:** if the prior, forward model and noise are all correct, the 90 % region
#   covers the true value in exactly 90 % of experiments *on average over the prior and the noise*. That is a
#   theorem.
# * **(a) one fixed true value:** no such guarantee. It came out near 0.90 here only because three stations pin the
#   epicentre down much more tightly than the prior does, so the prior hardly moves the crowd. The next cell shows a
#   case where it fails badly.
#
# Both statements also assume the forward model is right; section 7 shows what happens when it is not.

# %% [markdown]
# **A case where one fixed true value is not covered 90 % of the time.** (Notation: N(μ, s²) is a Gaussian, or
# normal, distribution with mean μ and standard deviation s; "~" reads "is distributed as".) Take one unknown θ, a
# prior θ ~ N(0, 1), and one measurement y = θ + noise with noise ~ N(0, 1). The posterior is then exactly N(y/2, 1/2), so its central 90 %
# interval is y/2 ± 1.645·√(1/2). The prior and the measurement count equally, so the prior pulls every answer halfway
# back toward 0. The cell computes, exactly, how often that interval covers a fixed θ, for every θ.

# %%
from scipy.stats import norm
z90 = norm.ppf(0.95)
half = z90 * np.sqrt(0.5)                                 # half-width of the central 90 % posterior interval


def cover_fixed(theta):
    """P(theta inside [y/2 - half, y/2 + half]) when y ~ N(theta, 1): exact, from the normal cdf."""
    return norm.cdf(2 * (theta + half) - theta) - norm.cdf(2 * (theta - half) - theta)


th = np.linspace(-5, 5, 1001)
cov_th = cover_fixed(th)
prior_avg = np.sum(cov_th * norm.pdf(th)) * (th[1] - th[0])
print(f"coverage of the 90 % posterior interval at a fixed true value: theta = 0: {cover_fixed(0.0):.3f}, "
      f"theta = 1: {cover_fixed(1.0):.3f}, theta = 2: {cover_fixed(2.0):.3f}, theta = 3: {cover_fixed(3.0):.3f}")
print(f"averaged over true values drawn from the prior N(0, 1): {prior_avg:.3f}")

fig, ax = plt.subplots(figsize=(7.6, 3.6))
ax.fill_between(th, 0, norm.pdf(th) / norm.pdf(0) * 0.35, color=S.GRID, lw=0, label="prior density (scaled)")
ax.plot(th, cov_th, color=S.INK, lw=1.8, label="coverage at this fixed true value")
ax.axhline(0.9, color=S.MUTED, lw=1, ls="--", label="0.90")
ax.plot(3.0, cover_fixed(3.0), "o", color=S.BAD, ms=6)
ax.annotate(f"θ = 3: {cover_fixed(3.0):.2f}", (3.0, cover_fixed(3.0)), xytext=(3.3, 0.45), fontsize=8.5,
            arrowprops=dict(arrowstyle="->", color=S.INK2, lw=0.8))
ax.set_xlabel("fixed true value θ"); ax.set_ylabel("share of repeats covered")
ax.set_ylim(0, 1.05)
ax.set_title("Coverage at a fixed true value: high near the prior, low far from it", fontsize=10)
ax.legend(fontsize=8, loc="upper right", bbox_to_anchor=(1.0, 0.83))   # below the 0.90 line
fig.tight_layout()
bk.savefig(fig, "2-01_fixed_truth_coverage")
plt.show()

# %% [markdown]
# **How to read it.** Near the prior mean the interval covers the true value *more* than 90 % of the time; far from it
# much less (printed: at θ = 3 about one time in four), because the prior pulls every answer toward 0. Weighted by how
# often the prior says each θ occurs, the coverage averages to 0.90 (printed). So a credible region is honest *on
# average over the world the prior describes*, and the prior matters most for true values the prior considers unusual.
# In the epicentre example above, three stations outweigh the prior and both kinds of coverage were close to 0.90.
#
# **The image to keep: a 90 % credible region holds 90 % of the posterior probability. It is not a promise to catch
# one particular true value 9 times in 10.**
#
# **More data, smaller crowd.** Adding stations removes candidates that cannot explain the new measurements. The
# crowd shrinks, and it keeps containing the true value:

# %%
more = [("1 station", [0], 200_000), ("3 stations", [0, 1, 2], 1_000_000), ("6 stations", list(range(6)), 4_000_000)]
fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.4))
for k, (ax, (title, idx, n)) in enumerate(zip(axs, more)):
    prob = make_problem(idx)
    kept = []
    rngm = np.random.default_rng(30 + k)
    for _ in range(n // 1_000_000 + (n % 1_000_000 > 0)):
        Cm, _, km = crowd(prob, min(n, 1_000_000), rngm)
        kept.append(Cm[km])
    Xm = np.vstack(kept)
    ax.scatter(*Xm[:3000].T, s=2, color=S.INK, lw=0, alpha=0.5, rasterized=True)
    base_map(ax, idx, labels=(k == 0))
    ax.set_title(f"{title}", fontsize=10)
    S.badge(ax, f"kept {len(Xm):,} of {n:,} tried", loc="lower left", fs=8)
    print(f"{title}: kept {len(Xm):,} of {n:,} ({100 * len(Xm) / n:.3f} %); crowd sd east {Xm[:, 0].std():.3f},"
          f" north {Xm[:, 1].std():.3f}")
fig.tight_layout()
bk.savefig(fig, "2-01_more_data")
plt.show()

# %% [markdown]
# ## 7. The second kind of error: the forward model is not the Earth
#
# **In plain words.** A station does not measure a distance directly. It measures how long the seismic wave took to
# arrive and converts that time into a distance with a wave speed taken from a velocity model. Each wave crosses its
# own rock on the way to its station. If that rock is a few percent slower than the model assumes, the converted
# distance comes out a few percent too long. This error is not noise: measure again and it is still there (the rock
# did not change), and it is larger for distant stations. It is an error of our physics, not of the instrument, and
# it is called **model error** (or prediction error).
#
# First, what repeating a measurement does to each kind of error. In this demonstration the converted distance to
# station 1 comes out 4 % too long (a value we choose), and the station measures it 10,000 times:

# %%
EPS1 = 0.04                                                  # path error at station 1, chosen for the demonstration
rng_me = np.random.default_rng(21)
converted1 = (1 + EPS1) * d_true1                            # what a perfect instrument reports through the wrong speed
reps = converted1 + SIG * rng_me.standard_normal(10_000)
run_mean = np.cumsum(reps) / np.arange(1, reps.size + 1)
print(f"true distance {d_true1:.4f}; converted with the wrong speed {converted1:.4f}; average of 10,000 repeats {reps.mean():.4f}")
print(f"the average misses the true distance by {(reps.mean() - d_true1) / SIG:.2f} noise units;"
      f" the uncertainty of an average of 10,000 from noise alone is {SIG / np.sqrt(reps.size):.4f}")

fig, ax = plt.subplots(figsize=(9.5, 3.4))
nn = np.arange(1, reps.size + 1)
ax.fill_between(nn, d_true1 - 2 * SIG / np.sqrt(nn), d_true1 + 2 * SIG / np.sqrt(nn), color="#ecebe6", lw=0,
                label="where averages of pure noise would stay (±2σ/√N around the true distance)")
ax.axhline(d_true1, color=S.INK, lw=1.4, label="true distance")
ax.axhline(converted1, color=S.BAD, lw=1.2, label="distance through the wrong wave speed")
ax.plot(nn, run_mean, color=S.INK2, lw=1.2, label="running average of the repeated measurements")
ax.set_xscale("log"); ax.set_xlim(1, reps.size)
ax.set_ylim(d_true1 - 0.08, d_true1 + 0.13)
ax.set_xlabel("number of repeated measurements N (log scale)"); ax.set_ylabel("distance (x 10 km)")
ax.set_title("Averaging removes measurement noise, not model error", fontsize=10.5)
ax.legend(fontsize=7.8, loc="upper right")
fig.tight_layout()
bk.savefig(fig, "2-01_averaging_vs_model_error")
plt.show()

# %% [markdown]
# The running average settles, as averages of noise always do (notebook 2.05), but it settles on the wrong distance:
# the grey band, where averages of pure noise around the true distance would stay, is left behind. More data cannot
# fix an error that every datum shares.
#
# **Precisely.** Write the data as $d_i = (1+\varepsilon_i)\,G_i(\mathbf m) + e_i$: the measurement noise
# $e_i \sim N(0, \sigma^2)$ changes with every repeat; the path error $\varepsilon_i \sim N(0, \sigma_\varepsilon^2)$ is
# fixed for a given station and Earth, and we do not know its value. Not knowing it and averaging over it widens each
# station's band from $\sigma$ to
#
# $$\sigma_i^2 \;=\; \sigma^2 + \big(\sigma_\varepsilon\,G_i(\mathbf m)\big)^2 .$$
#
# With many data, the error sizes are collected in a **covariance matrix**: one row and one column per datum, the
# squared error size (the **variance**, σ²) of each datum on the diagonal, and off the diagonal how strongly the errors
# of two data vary together (2.03 builds this from scratch). The **misfit** is the difference between measured and
# predicted data, and its covariance becomes $\mathbf C_\chi = \mathbf C_d + \mathbf C_p$: $\mathbf C_d$ for the
# instrument, $\mathbf C_p$ for the prediction error of the physics (Tarantola 2005; Duputel et al. 2014). Here
# $\mathbf C_p$ is diagonal with entries $\sigma_\varepsilon^2 G_i^2$, larger for distant stations. In words: a
# candidate is kept if its predicted distances match the measurements within the instrument noise *plus* the spread
# that our uncertainty about the rock would cause.
#
# Now a test with a better instrument (σ four times smaller than in sections 2 to 6) and path errors of about 4 %
# (σ_ε = 0.04), drawn at random for the three stations. The same data are inverted twice: with the measurement noise
# only, and with the model error included.

# %%
SIG_PRECISE, SIG_EPS = 0.015, 0.04


def logpost_me(X, Y, d, sig, sig_eps, idx=(0, 1, 2)):
    """log prior + log likelihood; each station's band is widened by the path-error uncertainty:
    variance_i = sig^2 + (sig_eps * predicted distance_i)^2 (the path errors averaged out exactly)."""
    lp = -0.5 * ((X - M0[0]) ** 2 + (Y - M0[1]) ** 2) / PRIOR_SD ** 2
    for j, i in enumerate(idx):
        g = np.sqrt((X - ST[i, 0]) ** 2 + (Y - ST[i, 1]) ** 2)
        var = sig ** 2 + (sig_eps * g) ** 2
        lp = lp - 0.5 * (d[j] - g) ** 2 / var - 0.5 * np.log(var)
    return lp


def exact_post_me(d, sig, sig_eps):
    """Exact posterior on a grid that zooms onto it; is the true value inside its 90 % region? how wide is it?"""
    xs, ys = np.linspace(-2.0, 2.6, 160), np.linspace(-1.5, 3.0, 160)
    Xc, Yc = np.meshgrid(xs, ys)
    lpc = logpost_me(Xc, Yc, d, sig, sig_eps)
    sel = lpc > lpc.max() - 30
    pad = 2 * (xs[1] - xs[0])
    xf = np.linspace(Xc[sel].min() - pad, Xc[sel].max() + pad, 180)
    yf = np.linspace(Yc[sel].min() - pad, Yc[sel].max() + pad, 180)
    XF, YF = np.meshgrid(xf, yf)
    lpf = logpost_me(XF, YF, d, sig, sig_eps)
    w = np.exp(lpf - lpf.max())
    Pn = w / w.sum()
    lev = P.hdr_levels(Pn, (0.5, 0.9, 0.99))
    at_star = np.exp(logpost_me(np.array(STAR[0]), np.array(STAR[1]), d, sig, sig_eps) - lpf.max()) / w.sum()
    mx, my = (Pn * XF).sum(), (Pn * YF).sum()
    sd = np.sqrt(0.5 * ((Pn * (XF - mx) ** 2).sum() + (Pn * (YF - my) ** 2).sum()))
    dx, dy = xf[1] - xf[0], yf[1] - yf[0]
    dens = Pn / (dx * dy)
    return dict(grid=P.Grid(XF, YF, dens, dx, dy, P.hdr_levels(dens)), inside90=bool(at_star >= lev[0.9]), sd=sd,
                mean=np.array([mx, my]))


rng_e = np.random.default_rng(23)
eps_paths = SIG_EPS * rng_e.standard_normal(3)
d_me = (1 + eps_paths) * exact_distance(STAR, ST[:3]) + SIG_PRECISE * rng_e.standard_normal(3)
one = {"noise only": exact_post_me(d_me, SIG_PRECISE, 0.0), "noise + model error": exact_post_me(d_me, SIG_PRECISE, SIG_EPS)}
print("path errors drawn for stations 1-3 (percent):", np.round(100 * eps_paths, 1))
for k, v in one.items():
    print(f"{k:20s}: posterior centre {np.round(v['mean'], 3)}, width (sd) {v['sd']:.4f}, true value inside its 90 % region: {v['inside90']}")

# repeat the experiment: new noise and new path errors each time, for instruments of different quality
N_ME, SIGS = 300, [0.06, 0.03, 0.015, 0.005]
cov_me = {"noise only": [], "noise + model error": []}
sd_me = {"noise only": [], "noise + model error": []}
for sig in SIGS:
    rr = np.random.default_rng(31)
    hits = {k: 0 for k in cov_me}
    widths = {k: [] for k in cov_me}
    for _ in range(N_ME):
        eps = SIG_EPS * rr.standard_normal(3)
        d = (1 + eps) * exact_distance(STAR, ST[:3]) + sig * rr.standard_normal(3)
        for name, se in (("noise only", 0.0), ("noise + model error", SIG_EPS)):
            r = exact_post_me(d, sig, se)
            hits[name] += r["inside90"]
            widths[name].append(r["sd"])
    for name in cov_me:
        cov_me[name].append(hits[name] / N_ME)
        sd_me[name].append(np.median(widths[name]))
print(f"\nshare of {N_ME} experiments in which the 90 % region contains the true value:")
print(f"{'instrument noise σ':>20s} " + " ".join(f"{s:>8.3f}" for s in SIGS))
for name in cov_me:
    print(f"{name:>20s} " + " ".join(f"{c:8.2f}" for c in cov_me[name]))
print("median width of the posterior (sd):")
for name in sd_me:
    print(f"{name:>20s} " + " ".join(f"{w:8.4f}" for w in sd_me[name]))
print(f"(expected scatter of a share near 0.9 from {N_ME} experiments: ±{2 * np.sqrt(0.09 / N_ME):.3f})")

# %%
ZM = (STAR[0] - 0.3, STAR[0] + 0.3, STAR[1] - 0.27, STAR[1] + 0.27)
fig, axs = plt.subplots(1, 4, figsize=(16, 3.9), gridspec_kw=dict(width_ratios=[1, 1, 1.15, 1.15]))
for ax, (name, r) in zip(axs[:2], one.items()):
    BP.density(ax, r["grid"])
    ax.plot(*STAR, "*", color=S.INK, ms=13, mec=S.SURFACE, zorder=5)
    ax.set_xlim(ZM[0], ZM[1]); ax.set_ylim(ZM[2], ZM[3]); ax.set_aspect("equal")
    ax.set_xlabel("east (x 10 km)")
    S.badge(ax, "90 % region " + ("contains ★" if r["inside90"] else "misses ★"), ok=r["inside90"], loc="lower left", fs=8)
axs[0].set_ylabel("north (x 10 km)")
axs[0].set_title("(a) likelihood with measurement\nnoise only", fontsize=10)
axs[1].set_title("(b) measurement noise + model error\n(C_d + C_p)", fontsize=10)
ax = axs[2]
cols = {"noise only": S.BAD, "noise + model error": S.COLORS["nuts"]}
for name in cov_me:
    ax.plot(SIGS, cov_me[name], "o-", color=cols[name], ms=6, label=name)
ax.axhline(0.9, color=S.MUTED, lw=1.0)
ax.text(SIGS[-1], 0.925, "the promise: 0.9", fontsize=8, color=S.INK2, ha="right")
ax.set_xscale("log"); ax.invert_xaxis()
ax.set_ylim(0, 1.02)
ax.set_xlabel("instrument noise σ (log scale; better →)"); ax.set_ylabel("share of experiments with ★ inside")
ax.set_title(f"(c) how often the 90 % region\ncatches the true value ({N_ME} experiments each)", fontsize=10)
ax.legend(fontsize=8, loc="lower left")
ax = axs[3]
for name in sd_me:
    ax.plot(SIGS, sd_me[name], "o-", color=cols[name], ms=6, label=name)
ax.set_xscale("log"); ax.set_yscale("log"); ax.invert_xaxis()
ax.set_xlabel("instrument noise σ (log scale; better →)"); ax.set_ylabel("width of the posterior (sd, x 10 km)")
ax.set_title("(d) the crowd's width: model error\nsets a floor", fontsize=10)
ax.legend(fontsize=8, loc="lower left")
fig.tight_layout()
bk.savefig(fig, "2-01_model_error")
plt.show()

# %% [markdown]
# **How to read it.**
# * (a) With only the measurement noise in the likelihood, the crowd is small and sits where the three slightly wrong
#   distances agree. It misses the true value.
# * (b) With each station's band widened by the path-error uncertainty, the crowd is larger and contains it (both
#   verdicts printed above).
# * (c) Over many repeated experiments, with new noise and new path errors each time, the region built from the noise
#   alone catches the true value less and less often as the instrument improves (shares printed above). The region
#   that includes the model error keeps its promise at every instrument quality.
# * (d) The reason: if model error is ignored, the crowd shrinks with the instrument noise without limit; if it is
#   included, the crowd cannot shrink below a floor set by how well we know the physics.
#
# **The image to keep: a better instrument does not fix imperfect physics, and ignoring the physics error makes you
# more confidently wrong.**
#
# **Common misreadings.**
# * *"Collect more data and the error averages out."* Measurement noise does; model error that the data share does not.
# * *"Inflate all the noise until the fit looks right."* Scaling every σ by one factor (a per-data-set $\sigma_k$) is a
#   crude version of $\mathbf C_p$. It cannot know that distant stations, or large signals, deserve wider bands, nor that
#   prediction errors are correlated between stations; notebook 2.13 measures what that costs.
#
# **Where model error comes from in geodesy:** the elastic structure (layering, rigidity), the fault geometry (dip,
# depth, curvature), approximations inside the Green's functions (a *half-space*, one uniform elastic block, instead
# of a layered Earth; shallow-water tsunami waves instead of dispersive ones, whose speed depends on wavelength), and
# processes the model leaves out (*afterslip*, the slow slip that follows an earthquake, recorded inside a window
# meant for the earthquake alone).
# Notebook 2.13 builds $\mathbf C_p$ for a slip inversion and compares it with noise scaling; 2.31 follows Jiang &
# Simons (2016), who built $\mathbf C_p$ for tsunami waveforms, and Zhao & Jiang (2026), who used it for layered
# rigidity.
#
# **In practice.** Tsunami Green's functions computed with the shallow-water equations leave out wave dispersion, so
# far from the source the computed waves arrive slightly early. That is this section's error exactly: it comes from the
# physics, not from the sensors; it repeats for every event; and a better pressure gauge would not remove it. Where
# the physics can be corrected (here, by a phase correction for dispersion), correct it; where it cannot, carry it
# into the error bars with $\mathbf C_p$.

# %% [markdown]
# ## 8. Why we need samplers at all
#
# For two parameters everything above is easy. The exact posterior needed one forward run per point of a
# 450 × 450 grid (count printed below), and rejection sampling kept a reasonable share of its candidates. Neither
# idea survives more unknowns.
#
# **Brute force on a grid** needs (points per axis) to the power (number of unknowns):

# %%
print(f"the grids in this notebook: 450 x 450 = {450 * 450:,} forward runs for 2 unknowns\n")
print("unknowns   grid points with 100 values per unknown")
for dim in (2, 10, 700):
    print(f"{dim:8d}   10^{2 * dim}")
age_s = 13.8e9 * 365.25 * 86400
budget = age_s * 1e6
print(f"\nfor scale: one forward run per microsecond since the Big Bang (13.8 billion years) is 10^{np.log10(budget):.1f} runs")

# %% [markdown]
# **Rejection sampling** dies too, because the share of candidates that survive shrinks with every measurement:
# you saw it fall from one station to three to six (printed in sections 4-6). To see how fast, take the simplest
# possible problem: $d$ unknowns, each measured once directly with noise σ = 0.1, prior standard deviation 1. For
# this problem the expected survival rate has an exact formula,
#
# $$\text{survival rate} \;=\; \prod_{i=1}^{d} \frac{\sigma}{\sqrt{1+\sigma^2}}\; \exp\!\Big[-\frac{d_i^2}{2(1+\sigma^2)}\Big],$$
#
# a product of one factor per unknown; each factor is at most $\sigma/\sqrt{1+\sigma^2}$, about one tenth here. We
# check it against brute-force counting for small $d$, then use it for large $d$.

# %%
sig_l = 0.1
rl = np.random.default_rng(5)
d_lin = rl.standard_normal(700) + sig_l * rl.standard_normal(700)          # one synthetic dataset, 700 unknowns
log10_rate = np.cumsum(np.log10(sig_l / np.sqrt(1 + sig_l ** 2)) - d_lin ** 2 / (2 * (1 + sig_l ** 2)) / np.log(10))
mc = {}
for dim in (1, 2, 3, 4):
    kept, tried = 0.0, 0
    for _ in range(4):
        m = rl.standard_normal((1_000_000, dim))
        kept += np.exp(-0.5 * np.sum(((d_lin[:dim] - m) / sig_l) ** 2, axis=1)).sum()
        tried += 1_000_000
    mc[dim] = kept / tried
    print(f"d = {dim}: counted survival rate {mc[dim]:.3e}   formula {10 ** log10_rate[dim - 1]:.3e}")
print(f"formula: d = 10 -> 10^{log10_rate[9]:.1f};  d = 700 -> 10^{log10_rate[699]:.0f}")

fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.9))
ax = axs[0]
dd = np.arange(1, 9)
ax.plot(dd, log10_rate[:8], "-", color=S.INK, lw=1.5, label="exact formula")
ax.plot(list(mc), np.log10(list(mc.values())), "o", color=S.INK2, ms=7, mfc="none", mew=1.5, label="counted (4 million candidates)")
ax.set_xlabel("number of unknowns d"); ax.set_ylabel("log10(share of candidates that survive)")
ax.set_title("Survival rate falls by a factor per unknown", fontsize=10)
ax.legend(fontsize=8)
ax = axs[1]
dd = np.arange(1, 701)
ax.plot(dd, log10_rate, color=S.INK, lw=1.5)
ax.axhline(-np.log10(budget), color=S.MUTED, lw=1.0)
ax.text(700, -np.log10(budget) - 18, "below this line: not one survivor expected in the\nage of the universe at a million candidates per second",
        ha="right", va="top", fontsize=8, color=S.INK2)
ax.plot(700, log10_rate[699], "o", color=S.INK, ms=6)
ax.annotate("700 unknowns (a slip model with 700 patches):\nsurvival rate " + rf"$10^{{{log10_rate[699]:.0f}}}$", (700, log10_rate[699]),
            xytext=(160, log10_rate[699] + 40), fontsize=8, arrowprops=dict(arrowstyle="->", color=S.INK2, lw=0.8))
ax.set_xlabel("number of unknowns d"); ax.set_ylabel("log10(share of candidates that survive)")
ax.set_title("At slip-inversion size: hopeless", fontsize=10)
fig.tight_layout()
bk.savefig(fig, "2-01_why_samplers")
plt.show()

# %% [markdown]
# So with hundreds of unknowns nobody can colour every point of the map, and nobody can wait for random candidates
# to survive. **Samplers are smarter ways to build the same crowd**: Metropolis walks from one survivor to a nearby
# candidate (2.16); NUTS uses the slope of the plausibility to glide along the crowd (2.18); SMC turns the data up
# gradually so a crowd can follow (2.20); EnKI moves a crowd with a Kalman rule (2.24). The two-parameter problems in this
# tutorial are the test bench: there, and only there, the exact posterior exists to check whether a sampler's crowd
# has the right shape.

# %% [markdown]
# ## 9. What "not applicable: no forward model" means
#
# In notebook 2.15 · Map some cells say *not applicable* (n/a). Methods differ in what they need to be told about a
# problem:

# %%
fig, ax = plt.subplots(figsize=(13, 4.1))
ax.set_xlim(0, 13); ax.set_ylim(0.7, 5.0); ax.axis("off")


def box(x, y, w, h, text, edge=S.AXIS, bold=False, fs=9):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.1", fc=S.SURFACE, ec=edge, lw=1.3))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs, color=S.INK,
            fontweight="bold" if bold else "normal", linespacing=1.3)


def arrow(x0, y0, x1, y1, color=S.MUTED):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2))


ax.text(0.1, 4.75, "An inverse problem (data + forward model): the epicentre, every slip inversion", fontsize=10, fontweight="bold")
box(0.1, 3.35, 1.7, 0.9, "candidate\nparameters m")
box(2.3, 3.35, 1.9, 0.9, "forward model\nG(m)")
box(4.7, 3.35, 1.8, 0.9, "predicted\ndata")
box(7.0, 3.35, 2.1, 0.9, "compare with the\nmeasured data d\n(noise σ)")
box(9.6, 3.35, 1.6, 0.9, "plausibility\nof m", bold=True)
for x0, x1 in [(1.8, 2.3), (4.2, 4.7), (6.5, 7.0), (9.1, 9.6)]:
    arrow(x0, 3.8, x1, 3.8)
ax.text(0.1, 2.2, "A shape written directly as a formula (no data): the funnel test problem", fontsize=10, fontweight="bold")
box(0.1, 0.85, 1.7, 0.9, "candidate\nparameters m")
box(2.3, 0.85, 6.8, 0.9, "formula p(m)     (no forward model, no predicted data, nothing to compare)")
box(9.6, 0.85, 1.6, 0.9, "plausibility\nof m", bold=True)
arrow(1.8, 1.3, 2.3, 1.3); arrow(9.1, 1.3, 9.6, 1.3)
ax.text(11.45, 4.35, "Metropolis, SMC:\nneed only this number", fontsize=8.5, color=S.INK, va="top")
ax.text(11.45, 3.65, "NUTS: this number\nand its slope", fontsize=8.5, color=S.INK, va="top")
ax.text(11.45, 1.75, "Metropolis, SMC, NUTS:\nstill work", fontsize=8.5, color=S.INK, va="top")
ax.text(4.0, 2.95, "EnKI needs predicted data for a crowd of candidates; the EKF needs predicted data and the\n"
        "slope of G (its Jacobian). Below, there are no predicted data: EnKI and EKF are not applicable.",
        fontsize=8.5, color=S.INK2, ha="left", va="top")
arrow(5.6, 3.35, 5.6, 3.0, color=S.COLORS["enki"])
fig.tight_layout()
bk.savefig(fig, "2-01_what_methods_need")
plt.show()

# %% [markdown]
# | method | what it must be able to compute for a candidate | works without data and a forward model? |
# |---|---|---|
# | Metropolis | the plausibility (log posterior) | yes |
# | SMC | the plausibility, split into prior and likelihood (or a reference and the rest) | yes |
# | NUTS / HMC | the plausibility and its slope (gradient) | yes |
# | EnKI / EKI / EnKF | the predicted data, for every member of a crowd | **no** |
# | EKF | the predicted data and the slope of the forward model (Jacobian) | **no** |
#
# The names: **Metropolis** (2.16) and **SMC**, sequential Monte Carlo (2.20), are samplers that need only the
# plausibility. **HMC**, Hamiltonian Monte Carlo, and its self-tuning version **NUTS** (2.18) also use its slope, the
# **gradient** (the rate of change in every parameter direction). The **log posterior** is the logarithm of prior ×
# likelihood, used because products of tiny numbers become sums. **EnKF**, **EKI** and **EnKI** are ensemble Kalman
# methods (2.24) and the **EKF** is the extended Kalman filter (2.25); the **Jacobian** is the table of slopes of every
# predicted datum with respect to every parameter.
#
# The **funnel** test problem is a plausibility shape written down directly as a formula, with no data. It copies
# the geometry of a **hierarchical** inversion, one that also estimates settings of the prior itself (here a scale
# parameter such as a smoothing weight α, which sets how much the slip may vary, together with the slip it scales;
# 2.09 and 2.30), so that samplers can be tested on that geometry alone. EnKI and the EKF work by comparing predicted data
# with measured data, so on the funnel they have nothing to compare: "not applicable". In a real slip inversion with
# α sampled there are data and a forward model, and EnKI could be run (it would treat α as one more unknown).

# %% [markdown]
# ## 10. The words in plain language
#
# | word | meaning |
# |---|---|
# | **deterministic** | same input, same output, every time (the forward model) |
# | **random** | changes unpredictably from one repeat to the next (the measurement noise) |
# | **uncertain** | not known exactly *to us*, although it has one true value (the epicentre) |
# | **parameter** | a number we want to know (east and north of the epicentre; slip on each fault patch) |
# | **data** | what was measured (the distances; GNSS displacements) |
# | **forward model** | the physics that turns parameters into predicted data (the distance formula; Green's functions) |
# | **measurement noise** | the part of the error that changes from one repeat to the next (arrival picking, clocks); averaging many repeats removes it |
# | **model error** (prediction error) | the part of the error that comes from imperfect physics (a wave speed, elastic structure or fault geometry slightly wrong); the same in every repeat, so averaging does not remove it; its covariance is $\mathbf C_p$ |
# | **prior** | what we believed about the parameters before measuring; candidates are drawn from it |
# | **likelihood** | how well a candidate's predictions match the measurements, given the noise (the chance of being kept) |
# | **posterior** | the candidates that survive: prior × likelihood; the crowd |
# | **candidate, sample, member** | one complete set of parameter values: one possible Earth |
# | **crowd, samples, ensemble** | many kept candidates; their spread is the uncertainty |
# | **shape** | the outline of the crowd in parameter space (blob, ridge, banana, islands) |
# | **true value** (★) | the parameters that generated synthetic data; unknown in real life |
# | **exact posterior** (grey) | the crowd computed exactly, by brute force on a grid; the answer key for grading samplers |
# | **50 % / 90 % credible region** | a region holding that share of the posterior probability, given these data and this model; it can miss the true value |
# | **highest-density region (HDR)** | the smallest credible region: the one drawn in grey in these notebooks |
# | **coverage** | how often the region contains the true value over repeated experiments; 0.90 on average when the true value is drawn from a correct prior, not guaranteed for one fixed true value (section 6) |
# | **not applicable (n/a)** | the method needs something the problem does not provide (EnKI and EKF need data and a forward model) |

# %% [markdown]
# **In practice.** In a fault-slip inversion the parameters are the slip on hundreds of fault patches (plus a few
# **hyperparameters**, settings of the prior or the noise model that are estimated too), the forward model is a matrix of Green's functions, and the posterior is a crowd of slip maps that
# all fit the geodetic data within their noise. A gradient-based sampler such as NUTS builds that crowd (2.18), because
# neither a grid nor rejection sampling can (section 8). The slip uncertainty plotted on each patch is the spread of the
# crowd there, and the shape questions of section 5 (ridges, bananas, islands) are asked of that high-dimensional crowd
# by the diagnostics of 2.19.
#
# Next: **2.02 · Bayes in pictures**.
