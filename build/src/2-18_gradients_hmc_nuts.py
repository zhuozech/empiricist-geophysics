# %% [markdown]
# # 2.18 · Gradients, HMC and NUTS: sliding instead of stumbling
#
# **Before this:** 2.16 (Metropolis), 2.17 (gradients and autodiff); 2.02 for the shapes.
# **You will be able to:** explain how HMC turns the gradient into long, well-aimed moves; tune or read its two knobs
# (step size, trajectory length) and what NUTS automates; diagnose divergences, saturated tree depth and a collapsed
# step size; choose centered or non-centered coordinates and a diagonal or dense mass matrix.
# **Optional sections:** 7 (head-to-head costs), 9 (mass matrix).
#
# Metropolis (notebook 2.16) proposes blind steps and asks "is it better here?". It never knows which way is
# uphill, so on a long, curved ridge it wanders like a random walk. Gradient-based samplers ask a better
# question at every point: *which way is uphill, and how steep is it?* That one extra piece of information is
# what this often-quoted sentence means:
#
# > "NUTS follows the gradient of the log-posterior, so it moves along curved, highly correlated ridges instead
# > of a random walk."
#
# This notebook draws that sentence, piece by piece, with real runs.
#
# **Words this notebook defines:** gradient of the log posterior, potential energy, momentum, leapfrog step,
# trajectory, energy error, divergence, U-turn, tree depth, step-size adaptation (dual averaging), mass matrix,
# centered / non-centered.

# %% [markdown]
# **Problems in this notebook** (size and forward model; notebook 2.25 explains the degree of nonlinearity)
#
# | problem | unknowns | data | forward model | linear? |
# |---|---|---|---|---|
# | trade-off | 2 | 2 | $\mathbf d = \mathbf G\mathbf m$ | linear |
# | banana, plus sharper versions (σ down to 0.0125) | 2 | 1 | distance to one station | nonlinear |
# | ill-conditioned Gaussian (7) | 100 | none | target density only | no data |
# | Neal's funnel (8) | 10 | none | hierarchical prior only | no data |
# | badly scaled, correlated Gaussian (9) | 2 | none | target density only | no data |

# %%
import sys, pathlib, inspect, time
sys.path.insert(0, str(pathlib.Path.cwd().parent))
import numpy as np
import matplotlib.pyplot as plt
import jax, jax.numpy as jnp
import bayeskit as bk
from bayeskit import problems as P, samplers as SM, diagnostics as D, plotting as BP, style as S

NUTS_C, HMC_C, RWM_C = S.COLORS["nuts"], S.COLORS["hmc"], S.COLORS["rwm"]
BLUES = ["#86b6ef", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0d366b"]   # ordinal ramp (doubling 0, 1, 2, ...)

lin = P.correlated()                        # the trade-off ridge from notebook 2.02
ban = P.location(kind="banana")             # the banana (one station + regional prior)
g_lin, g_ban = P.grid(lin, n=350), P.grid(ban, n=450)


def ess_capped(x):
    """ESS over chains, capped at S*log10(S) as Stan does. (bayeskit's D.ess has no cap, and an antithetic
    chain, where successive draws are negatively correlated, can make the uncapped estimate absurdly large.)"""
    x = np.atleast_2d(x)
    return min(D.ess(x), x.size * np.log10(x.size))

# %% [markdown]
# ## 1. The gradient as arrows
#
# The posterior density is a landscape; $\log p(\mathbf m\mid\mathbf d)$ is its height. The **gradient**
# $\nabla_{\mathbf m}\log p$ is a vector at every point: it points in the direction in which the log density
# rises fastest, and its length is how steep the rise is. Drawn as arrows over the three shapes from notebook 2.02:

# %%
fun2 = P.funnel(dim=2, centered=True, scale_v=1.5)
fun2.extent = (-5, 5, -9, 9)
fig, axs = plt.subplots(1, 3, figsize=(15, 4.4))
for ax, pr, title in zip(axs, [lin, ban, fun2], ["Trade-off ridge", "Banana", "Funnel"]):
    g = P.grid(pr, n=300)
    BP.density(ax, g)
    e, nx = pr.extent, 19
    XX, YY = np.meshgrid(np.linspace(e[0], e[1], nx), np.linspace(e[2], e[3], nx))
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    G = np.array(jax.vmap(jax.grad(pr._logp_fn))(jnp.asarray(pts)))    # the gradient at every grid point
    mag = np.linalg.norm(G, axis=1)
    length = (mag ** 0.3) / (mag ** 0.3).max()                          # compressed: steep walls would swamp the plot
    span = 0.95 * min(e[1] - e[0], e[3] - e[2]) / nx
    u = G / mag[:, None] * (length * span)[:, None]
    ax.quiver(pts[:, 0], pts[:, 1], u[:, 0], u[:, 1], angles="xy", scale_units="xy", scale=1,
              color=S.INK, width=0.0042, headwidth=3.6, zorder=4)
    BP.frame(ax, pr, equal=pr is ban)
    ax.set_title(title)
    print(f"{title:16s} steepness |grad log p| on this window: from {mag.min():.3g} to {mag.max():.3g}")
fig.tight_layout()
bk.savefig(fig, "2-18_gradient_arrows")
plt.show()

# %% [markdown]
# **How to read it.** Each arrow says "uphill is this way". Arrows cross the contours at right angles, because
# along a contour the height does not change. Far from the ridge the arrows are long and point straight at it;
# on the ridge they are short and point *along* it, towards the peak. The printed ranges show how different the
# steepness is: the funnel's narrow neck is orders of magnitude steeper than its wide mouth, where the arrows
# shrink to dots. Keep that in mind for section 8.
#
# Computing these arrows needed no formula from us: `jax.grad` differentiated the log density automatically
# (notebook 2.17 explains how).

# %% [markdown]
# ## 2. An optimiser uses the gradient to find ONE point
#
# The simplest use of the gradient is to walk uphill until it is flat: $\mathbf m \leftarrow \mathbf m + \eta\,\nabla\log p$.
# That is gradient ascent, the basic optimiser. It finds the peak (the MAP) and stops there.

# %%
starts = np.array([[-2.2, 0.8], [2.0, -2.0], [0.3, -1.6]])
eta, n_steps = 0.004, 4000
paths = []
for x in starts:
    x, path = x.copy(), [x.copy()]
    for _ in range(n_steps):
        x = x + eta * ban.grad(x)
        path.append(x.copy())
    paths.append(np.array(path))
print("end points of the three ascents:", np.round([p[-1] for p in paths], 4).tolist())

ban.reset_counts()
nu_ban = SM.nuts(ban, np.array([0.0, 0.0]), 2000, n_warmup=500, rng=np.random.default_rng(3))
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2))
BP.density(axs[0], g_ban)
for p in paths:
    axs[0].plot(p[:, 0], p[:, 1], color=S.INK, lw=1.3)
    axs[0].plot(*p[0], "o", color=S.INK, ms=5)
axs[0].plot(*paths[0][-1], "*", color=S.BAD, ms=14, mec=S.SURFACE, label="where every ascent stops (MAP)")
axs[0].legend(loc="lower left", fontsize=8)
BP.frame(axs[0], ban, equal=True)
axs[0].set_title("Gradient ascent: up, then along the ridge, then stop")
BP.density(axs[1], g_ban)
BP.samples(axs[1], nu_ban["chain"], NUTS_C, s=4, alpha=0.45)
BP.frame(axs[1], ban, equal=True)
axs[1].set_title("NUTS: uses the same gradient to cover the whole ridge")
fig.tight_layout()
bk.savefig(fig, "2-18_ascent_vs_sampling")
plt.show()

# %% [markdown]
# Left: three starts, three paths, one answer. The paths first climb straight up the steep wall, then crawl
# along the ridge to the peak, then stop: an optimiser answers *"where is the highest point?"*. Right: a
# sampler answers *"where is the probability?"*. It uses the same gradient, but never stops; it keeps moving so
# that the time it spends in each region is proportional to the probability there. The next sections show
# how adding *momentum* turns "walk uphill" into "explore".

# %% [markdown]
# ## 3. Hamiltonian Monte Carlo: a puck sliding on the landscape
#
# Turn the landscape upside down: define the **potential energy** $U(\mathbf m) = -\log p(\mathbf m\mid\mathbf d)$.
# High probability is now a valley floor. Hamiltonian Monte Carlo (HMC; the total energy $H$ below is called the
# **Hamiltonian**, which gives the method its name) does this, at every iteration:
#
# 1. **Kick.** Give a frictionless puck a random **momentum** $\mathbf p \sim N(0, \mathbf M)$ (a random push in a
#    random direction; $\mathbf M$ is the *mass matrix*, the identity for now).
# 2. **Slide.** Let it move under Newton's laws for a while. The force is $-\nabla U = \nabla\log p$: the
#    gradient. In the valley the puck speeds up, on the walls it slows down and turns. It follows the valley
#    because the walls keep pushing it back towards the floor.
# 3. **Check.** Total energy $H = U + \tfrac12\mathbf p^\top\mathbf M^{-1}\mathbf p$ should be conserved. The
#    computer integrates the motion in small steps (the **leapfrog** scheme below), which conserves $H$ only
#    approximately; the end point is accepted with probability $\min(1, e^{-\Delta H})$. Small energy error,
#    almost always accepted.
#
# The leapfrog step, as it runs in `bayeskit` (half kick, drift, kick, ..., half kick):

# %%
print(inspect.getsource(SM.leapfrog))

# %% [markdown]
# Here is the picture behind the sentence. To make the ridge the kind that defeats a random walk, the banana
# below is four times sharper than in notebook 2.02: the same station, with the distance measured four times more
# precisely. All three samplers start at the same point and spend about the same number of evaluations
# (Metropolis: log densities; HMC and NUTS: gradients). Section 7 measures how the comparison depends on this
# sharpness.

# %%
def banana(sigma):
    """The banana of notebook 2.02 with a different distance uncertainty sigma (it is 0.1 there)."""
    st = jnp.asarray([[0.0, -2.5]])
    fwd = lambda m: jnp.sqrt(jnp.sum((m[None, :] - st) ** 2, axis=1) + 1e-12)
    return P.InverseProblem(fwd, [2.5], sigma, m0=[0.0, 0.0], P=np.eye(2), name=f"banana, sigma {sigma}",
                            extent=(-2.6, 2.6, -2.9, 1.1), labels=("east (x 10 km)", "north (x 10 km)"))


def on_arc(points):
    """Move points radially onto the arc (radius 2.5 around the station): sensible starting points."""
    c = np.array([0.0, -2.5])
    v = np.atleast_2d(np.asarray(points, float)) - c
    return c + 2.5 * v / np.linalg.norm(v, axis=1, keepdims=True)


sharp = banana(0.025)
g_sharp = P.grid(sharp, n=900)
x0 = on_arc([-1.55, -0.54])[0]
eps_h, L_h, n_traj = 0.02, 100, 6
hmc_demo = SM.hmc(sharp, x0, n_traj, eps=eps_h, L=L_h, rng=np.random.default_rng(7), record=True)
budget = n_traj * L_h

# Metropolis gets the step size that worked best in a pilot scan (4 chains x 4000 steps each)
scan = {}
for st in [0.01, 0.025, 0.05, 0.08, 0.12]:
    r = SM.rwm(sharp, np.tile(x0, (4, 1)), 4000, step=st, rng=np.random.default_rng(5))
    scan[st] = min(ess_capped(r["samples"][:, 800:, i]) for i in range(2))
rw_step = max(scan, key=scan.get)
print("Metropolis pilot (16,000 evaluations per step size), min ESS:", {k: round(v, 1) for k, v in scan.items()},
      "-> step", rw_step)
rwm_demo = SM.rwm(sharp, x0, budget, step=rw_step, rng=np.random.default_rng(8))

nu_w = SM.nuts(sharp, x0, 10, n_warmup=500, rng=np.random.default_rng(9))         # warm-up only, to get the step size
nuts_demo = SM.nuts(sharp, x0, 300, n_warmup=0, eps=nu_w["step_size"], rng=np.random.default_rng(10), record=range(300))
cum = np.cumsum(nuts_demo["n_leapfrog"])
k_nuts = int(np.searchsorted(cum, budget)) + 1                                   # NUTS iterations that fit in the budget

fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.4))
ch = rwm_demo["chain"]
BP.density(axs[0], g_sharp)
axs[0].plot(np.r_[x0[0], ch[:, 0]], np.r_[x0[1], ch[:, 1]], color=RWM_C, lw=0.8)
axs[0].plot(ch[-1, 0], ch[-1, 1], "o", color=RWM_C, ms=6)
axs[0].set_title(f"Metropolis: {budget} log-density evaluations")
BP.density(axs[1], g_sharp)
for i, pth in enumerate(hmc_demo["paths"]):
    axs[1].plot(pth[:, 0], pth[:, 1], color=HMC_C, lw=0.7, alpha=0.9)
    axs[1].plot(*pth[-1], "o", color=HMC_C, ms=6, mec=S.SURFACE, zorder=5)
axs[1].set_title(f"HMC: {n_traj} trajectories × {L_h} leapfrog steps")
BP.density(axs[2], g_sharp)
for k in range(k_nuts):
    tr = nuts_demo["trees"][k]
    axs[2].plot(tr["path"][:, 0], tr["path"][:, 1], color=NUTS_C, lw=0.6, alpha=0.8)
xs_n = np.vstack([x0, nuts_demo["chain"][:k_nuts]])
axs[2].plot(xs_n[:, 0], xs_n[:, 1], "o", color=NUTS_C, ms=5, mec=S.SURFACE, zorder=5)
axs[2].set_title(f"NUTS: {k_nuts} iterations, {int(cum[k_nuts - 1])} gradient evaluations")
for ax in axs:
    ax.plot(*x0, "o", color=S.INK, ms=7, zorder=6)
    BP.frame(ax, sharp, equal=True)
    ax.set_ylim(-2.3, 0.6)
fig.tight_layout()
bk.savefig(fig, "2-18_random_walk_vs_gradient")
plt.show()
print(f"HMC energy errors (should be small): {np.round(hmc_demo['energy_error'], 3).tolist()}, "
      f"accepted {int(hmc_demo['accepted'].sum())} of {n_traj}")
print(f"Metropolis acceptance {rwm_demo['accept_rate']:.2f}. East-west range covered: "
      f"Metropolis {np.ptp(ch[:, 0]):.2f}, HMC {np.ptp(np.vstack(hmc_demo['paths'])[:, 0]):.2f}, "
      f"NUTS {np.ptp(xs_n[:, 0]):.2f}  (x 10 km; the arc spans about 4.8)")

# %% [markdown]
# **The image to keep.** Same start (black dot), same cost. The Metropolis chain (orange) shuffles inside a
# small patch: each blind step must be small enough not to fall off the thin arc, so its progress *along* the arc
# is a random walk (the pilot scan printed above shows that no step size does much better). The HMC and NUTS
# trajectories (blue) sweep along the arc. They zig-zag *across* it, because the walls push the puck back and
# forth, but the momentum carries them *along* it and around the bend. Nobody told them the arc is curved; the
# gradient told them at every step. That is the whole content of the sentence.
#
# The same HMC run, animated (each frame adds five leapfrog steps; dots are the accepted end points):

# %%
fig, ax = plt.subplots(figsize=(6.6, 3.9), dpi=95)
BP.density(ax, g_sharp)
BP.frame(ax, sharp, equal=True)
ax.set_ylim(-2.3, 0.6)
ax.plot(*x0, "o", color=S.INK, ms=7, zorder=6)
ttl = ax.set_title("")
frames = []
for i, pth in enumerate(hmc_demo["paths"]):
    line, = ax.plot([], [], color=HMC_C, lw=1.2, zorder=5)
    head, = ax.plot([], [], "o", color=HMC_C, ms=7, mec=S.SURFACE, zorder=7)
    for l in range(1, len(pth) + 1, 5):
        line.set_data(pth[:l, 0], pth[:l, 1])
        head.set_data([pth[l - 1, 0]], [pth[l - 1, 1]])
        ttl.set_text(f"HMC trajectory {i + 1} of {n_traj}: leapfrog step {min(l, len(pth) - 1)} of {L_h}")
        frames.append(S.fig_to_rgb(fig))
    line.set_data(pth[:, 0], pth[:, 1])
    line.set_alpha(0.3); line.set_linewidth(0.8)
    head.set_data([pth[-1, 0]], [pth[-1, 1]])
    head.set_markersize(5)
    frames.extend([S.fig_to_rgb(fig)] * 4)
final_png = bk.savefig(fig, "2-18_hmc_banana_final")          # the last frame as a 300-dpi still
plt.close(fig)
gif = S.frames_to_gif(frames, S.FIG / "2-18_hmc_banana.gif", fps=14)
print(f"saved figures/2-18_hmc_banana.gif ({gif.stat().st_size / 1e6:.1f} MB, {len(frames)} frames)")
S.show_gif(gif)
from IPython.display import Image as _Image, display as _display
print("The last frame at full resolution (300 dpi):")
_display(_Image(filename=str(final_png), width=600))

# %% [markdown]
# ## 4. The two HMC knobs: step size and trajectory length
#
# **Step size $\varepsilon$.** Leapfrog is a numerical integrator. With a small step it conserves energy almost
# exactly; with a larger step the energy error oscillates but stays bounded; beyond a stability limit (for a
# Gaussian, twice the smallest posterior standard deviation) it grows without bound. That blow-up is a
# **divergence**. On the trade-off ridge the narrow direction has standard deviation printed below, so the limit
# is about twice that:

# %%
mean_lin, C_lin = lin.exact
sd_eig, V_eig = np.linalg.eigh(C_lin)
sd_eig = np.sqrt(sd_eig)
print(f"posterior sd: narrow direction {sd_eig[0]:.4f}, long direction {sd_eig[1]:.4f}; "
      f"leapfrog stability limit 2 x narrow sd = {2 * sd_eig[0]:.4f}")

rng = np.random.default_rng(2)
xs0, ps0 = mean_lin + np.array([0.3, -0.3]), rng.standard_normal(2)
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
BP.density(axs[0], g_lin)
for eps, shade in zip([0.02, 0.10, 0.135], [S.INK2, HMC_C, S.BAD]):
    x, p = xs0.copy(), ps0.copy()
    lp, gr = lin.logp_and_grad(x)
    H0 = -lp + 0.5 * p @ p
    traj, dH = [x.copy()], [0.0]
    for _ in range(40):
        p = p + 0.5 * eps * gr; x = x + eps * p; lp, gr = lin.logp_and_grad(x); p = p + 0.5 * eps * gr
        traj.append(x.copy()); dH.append(-lp + 0.5 * p @ p - H0)
    traj, dH = np.array(traj), np.array(dH)
    keep = np.all(np.abs(traj - mean_lin) < 4, axis=1)
    lab = f"ε = {eps}: max |ΔH| = {np.abs(dH).max():.2g}"
    axs[0].plot(traj[keep, 0], traj[keep, 1], "-o", color=shade, ms=2.5, lw=1.0, label=lab, zorder=6 if eps == 0.02 else 4)
    axs[1].plot(np.abs(dH) + 1e-12, color=shade, lw=1.6, label=lab)
    print(lab)
BP.frame(axs[0], lin)
axs[0].set_xlim(-2.5, 3.0); axs[0].set_ylim(-2.5, 3.0)
axs[0].legend(loc="lower left", fontsize=8)
axs[0].set_title("40 leapfrog steps from the same start")
axs[1].set_yscale("log")
axs[1].axhline(1000, color=S.MUTED, lw=0.8)
axs[1].text(21, 2500, "NUTS calls an error above 1000 a divergence", fontsize=8, color=S.INK2)
axs[1].set_xlabel("leapfrog step"); axs[1].set_ylabel("|energy error ΔH|")
axs[1].set_title("Energy error along the trajectory")
axs[1].legend(loc="upper left", fontsize=8)
fig.tight_layout()
bk.savefig(fig, "2-18_step_size_energy")
plt.show()

# %% [markdown]
# Grey (small $\varepsilon$): the energy error stays tiny, every proposal would be accepted, but each step is
# short, so a long trajectory costs many gradients. Blue (moderate $\varepsilon$): the error oscillates but stays
# bounded; this is the regime samplers aim for. Red (just past the limit printed above): the trajectory is flung
# off the ridge and the energy error explodes: a **divergence**. A divergence is not a numerical nuisance to
# ignore; it says the sampler could not follow the curvature somewhere, and the region it could not enter is
# then under-sampled (section 8 shows the consequence).
#
# **Trajectory length $L\varepsilon$.** On a Gaussian the puck orbits like a pendulum. Too short, and each
# iteration moves a little, like a random walk with smaller steps. About half an orbit along the long direction,
# and each iteration lands far away. A full orbit, and the puck returns to where it started: all those gradients
# bought nothing. The curve below measures progress as the mean squared jump along the long direction per
# gradient evaluation. Each run jitters $\varepsilon$ by ±15 % from one iteration to the next, as practical HMC
# codes do (with an exactly fixed half orbit the chain would only mirror itself, $z \to -z$, and never change $|z|$).

# %%
def hmc_jittered(target, x0, n, eps, L, rng, jitter=0.15, record=False):
    """HMC with the step size drawn uniformly in eps*(1 +- jitter) at every iteration (built on SM.leapfrog)."""
    x = np.array(x0, float)
    lp, gr = target.logp_and_grad(x)
    chain, paths = [], []
    for _ in range(n):
        e = eps * (1 + jitter * (2 * rng.uniform() - 1))
        p0 = rng.standard_normal(x.size)
        H0 = -lp + 0.5 * p0 @ p0
        x1, p1, lp1, g1, path = SM.leapfrog(target, x, p0, gr, e, L, np.eye(x.size), record)
        if lp1 is not None and np.isfinite(lp1) and np.log(rng.uniform()) < H0 - (-lp1 + 0.5 * p1 @ p1):
            x, lp, gr = x1, lp1, g1
        chain.append(x.copy()); paths.append(path)
    return np.array(chain), paths


eps_L = 0.05
half = np.pi * sd_eig[1] / eps_L                       # leapfrog steps in half an orbit along the long direction
Ls = [1, 2, 4, 8, 12, 16, 20, 24, 28, 32, 36, 40, 44, 48, 56, 64, 72, 80, 88, 96, 104]
long_dir = V_eig[:, 1]
rng0 = np.random.default_rng(0)
jump = []
for L in Ls:
    start = mean_lin + np.linalg.cholesky(C_lin) @ rng0.standard_normal(2)     # a random start from the posterior
    chain, _ = hmc_jittered(lin, start, 400, eps_L, L, np.random.default_rng(L))
    jump.append(np.mean(np.diff(chain @ long_dir) ** 2) / L)
jump = np.array(jump)
print(f"half orbit = {half:.0f} leapfrog steps, full orbit = {2 * half:.0f}; "
      f"largest progress per gradient in the scan at L = {Ls[int(np.argmax(jump))]}; "
      f"progress at L = 88 is {jump[Ls.index(88)] / jump.max():.2f} of the best")

fig = plt.figure(figsize=(15, 4.0))
ax0 = fig.add_subplot(1, 4, 1)
ax0.plot(Ls, jump, "-o", color=HMC_C, ms=4)
for v_, t_ in [(half, "half orbit"), (2 * half, "full orbit")]:
    ax0.axvline(v_, color=S.MUTED, lw=0.8)
    ax0.text(v_ + 1.5, jump.max() * 1.02, t_, fontsize=8, color=S.INK2)
ax0.set_ylim(0, jump.max() * 1.12)
ax0.set_xlabel("leapfrog steps L"); ax0.set_ylabel("squared jump per gradient")
ax0.set_title("Progress per gradient")
start_demo = mean_lin + long_dir * 0.9
for j, (L, t_) in enumerate([(4, "L = 4: tiny moves"), (int(round(half)), f"L = {int(round(half))}: about half an orbit"),
                              (int(round(2 * half)), f"L = {int(round(2 * half))}: about a full orbit")]):
    ax = fig.add_subplot(1, 4, j + 2)
    BP.density(ax, g_lin)
    chain, pths = hmc_jittered(lin, start_demo, 12, eps_L, L, np.random.default_rng(40 + j), record=True)
    for pth in pths:
        ax.plot(pth[:, 0], pth[:, 1], color=HMC_C, lw=0.6, alpha=0.6)
    xs_ = np.vstack([start_demo, chain])
    ax.plot(xs_[:, 0], xs_[:, 1], "-o", color=S.INK, lw=0.9, ms=3.5, zorder=5)
    ax.plot(*xs_[0], "o", color=S.INK, ms=7, zorder=6)
    BP.frame(ax, lin, labels=False)
    ax.set_title(t_ + "\n(12 iterations)", fontsize=9.5)
fig.tight_layout()
bk.savefig(fig, "2-18_trajectory_length")
plt.show()

# %% [markdown]
# Left: progress per gradient is high for trajectories of roughly half an orbit and collapses near a full orbit
# (numbers printed above). Right: the black line joins 12 consecutive samples; blue lines are the trajectories.
# With $L = 4$ the samples creep; near half an orbit they jump from one part of the ridge to another; near a full
# orbit the long trajectories (blue) mostly carry the puck back towards where it began, so the net moves are
# short again. The best $L$ depends on the posterior, differs
# between directions, and is unknown in 695 dimensions. Choosing it automatically is exactly what NUTS adds.

# %% [markdown]
# ## 5. NUTS: HMC that stops when the trajectory starts to turn back
#
# The No-U-Turn Sampler (Hoffman & Gelman 2014) removes the choice of $L$. Each iteration it:
#
# 1. starts a trajectory at the current point with a random momentum;
# 2. **doubles** it, again and again: 1 leapfrog step, then 2 more, then 4, 8, 16, ..., each time extending
#    forwards *or backwards in time* (a coin flip), so the trajectory grows at both ends;
# 3. stops when the two ends start moving back towards each other: a **U-turn**. In symbols, with ends
#    $\mathbf m^-$, $\mathbf m^+$ and their momenta $\mathbf p^-$, $\mathbf p^+$, it continues only while
#    $(\mathbf m^+ - \mathbf m^-)\cdot\mathbf p^- \ge 0$ and $(\mathbf m^+ - \mathbf m^-)\cdot\mathbf p^+ \ge 0$;
# 4. picks the next sample from the points visited, favouring those with small energy error.
#
# The number of doublings is the **tree depth**; depth $j$ means up to $2^j - 1$ leapfrog steps. NumPyro caps it
# at 10 (1023 steps), a cap that badly scaled problems, such as the centered funnel of section 8, run into.
#
# The U-turn rule as a picture, on one long trajectory around a Gaussian (arrows: momentum at each end; grey
# arrow: from one end to the other):

# %%
gauss = P.linear(np.eye(2), y=[0.0, 0.0], sigma=[1.0, 1.0], m0=[0.0, 0.0], P=np.eye(2) * 1e6,
                 name="Gaussian orbit", extent=(-2.6, 2.6, -2.6, 2.6))
g_g = P.grid(gauss, n=300)
xA, pA, eps_g = np.array([1.2, -0.5]), np.array([0.2, 1.4]), 0.1
x, p = xA.copy(), pA.copy()
lp, gr = gauss.logp_and_grad(x)
orbit, moms = [x.copy()], [p.copy()]
for _ in range(60):
    p = p + 0.5 * eps_g * gr; x = x + eps_g * p; lp, gr = gauss.logp_and_grad(x); p = p + 0.5 * eps_g * gr
    orbit.append(x.copy()); moms.append(p.copy())
orbit, moms = np.array(orbit), np.array(moms)
ok = [(orbit[T] - xA) @ pA >= 0 and (orbit[T] - xA) @ moms[T] >= 0 for T in range(1, 61)]
T_fail = 1 + int(np.argmin(ok)) if not all(ok) else 60
print(f"one full orbit = {2 * np.pi / eps_g:.0f} leapfrog steps; the U-turn criterion first fails after {T_fail} steps")
fig, axs = plt.subplots(1, 3, figsize=(13, 4.3))
for ax, T in zip(axs, [T_fail // 3, T_fail - 4, T_fail]):
    BP.density(ax, g_g)
    ax.plot(orbit[:T + 1, 0], orbit[:T + 1, 1], color=NUTS_C, lw=1.4)
    d_ = orbit[T] - xA
    ax.annotate("", xy=orbit[T], xytext=xA, arrowprops=dict(arrowstyle="->", color=S.MUTED, lw=1.2))
    for xx, pp in [(xA, pA), (orbit[T], moms[T])]:
        ax.annotate("", xy=xx + 0.45 * pp, xytext=xx, arrowprops=dict(arrowstyle="-|>", color=S.INK, lw=1.4))
        ax.plot(*xx, "o", color=S.INK, ms=5)
    ax.text(xA[0] + 0.12, xA[1] - 0.25, "m-", fontsize=9, color=S.INK)
    ax.text(orbit[T][0] + 0.12, orbit[T][1] + 0.12, "m+", fontsize=9, color=S.INK)
    keep_going = bool(d_ @ pA >= 0 and d_ @ moms[T] >= 0)
    S.badge(ax, f"{T} steps:  d·p- = {d_ @ pA:+.2f},  d·p+ = {d_ @ moms[T]:+.2f}", loc="lower left", fs=8)
    ax.set_title("ends still moving apart: keep going" if keep_going else "leading end turning back: U-turn, stop",
                 fontsize=9.5)
    BP.frame(ax, gauss, equal=True, labels=False)
fig.tight_layout()
bk.savefig(fig, "2-18_uturn_rule")
plt.show()

# %% [markdown]
# Here $d = m^+ - m^-$ is the grey arrow, and black arrows are the momenta at the two ends. While both dot
# products are positive, the ends are still moving apart. Once the leading end's momentum points back towards the
# other end (a negative dot product), continuing would only retrace the orbit, so NUTS stops. On this round
# Gaussian that happens after about half an orbit (compare the two numbers printed above): NUTS lands near the
# efficient length of section 4 without being told.
#
# Now a real NUTS iteration on the banana, recorded leapfrog by leapfrog. Colours run from light to dark blue for
# successive doublings; the black dot is the starting point and the star is the sample NUTS picked.

# %%
nu_rec = SM.nuts(ban, np.array([0.0, 0.0]), 400, n_warmup=500, rng=np.random.default_rng(11), record=range(500, 900))


def end_momenta(tr, target):
    """Exact momenta at both ends of a recorded trajectory (identity mass), rebuilt from positions and gradients."""
    path, e = tr["path"], tr["eps"]
    p_plus = (path[-1] - path[-2]) / e + 0.5 * e * target.grad(path[-1])
    p_minus = (path[1] - path[0]) / e - 0.5 * e * target.grad(path[0])
    return path[0], p_minus, path[-1], p_plus


# show a long trajectory (at least 15 leapfrog steps) whose U-turn is the clearest (most negative dot product)
def uturn_strength(tr):
    a, pa, b, pb = end_momenta(tr, ban)
    return min((b - a) @ pa, (b - a) @ pb)


long_trees = {k: tr for k, tr in nu_rec["trees"].items() if len(tr["states"]) >= 15}
k_show = min(long_trees, key=lambda k: uturn_strength(long_trees[k]))
tree = long_trees[k_show]
print(f"{len(long_trees)} of {len(nu_rec['trees'])} recorded iterations used 15 or more leapfrog steps")
xm, pm, xp, pp = end_momenta(tree, ban)
print(f"iteration {k_show}: {len(tree['doublings'])} doublings, {len(tree['states'])} leapfrog steps, "
      f"directions {[('forward' if v == 1 else 'backward') for v, *_ in tree['doublings']]}")
print(f"at the end: (m+ - m-)·p- = {(xp - xm) @ pm:+.3f}, (m+ - m-)·p+ = {(xp - xm) @ pp:+.3f}  (a negative value is the U-turn)")

nd = len(tree["doublings"])
fig, axs = plt.subplots(1, nd, figsize=(3.4 * nd, 2.6))
for j, ax in enumerate(axs):
    BP.density(ax, g_ban)
    shown = tree["stage"] <= j
    order = np.argsort(np.r_[0, tree["times"][shown]])
    pts = np.vstack([tree["x0"], tree["states"][shown]])[order]
    ax.plot(pts[:, 0], pts[:, 1], color=S.INK2, lw=0.7, zorder=4)
    for s_ in range(j + 1):
        sel = tree["stage"] == s_
        ax.scatter(tree["states"][sel, 0], tree["states"][sel, 1], s=16, color=BLUES[min(s_, 5)], zorder=5,
                   edgecolors=S.SURFACE, linewidths=0.4)
    ax.plot(*tree["x0"], "o", color=S.INK, ms=6, zorder=6)
    v, jj, a, b = tree["doublings"][j]
    ax.set_title(f"doubling {j + 1}: +{b - a} step{'s' if b - a != 1 else ''} {'forward' if v == 1 else 'backward'}", fontsize=9)
    if j == nd - 1:
        ax.plot(*tree["chosen"], "*", color=S.INK, ms=13, mec=S.SURFACE, zorder=7)
        for xx, pp_ in [(xm, pm), (xp, pp)]:
            ax.annotate("", xy=xx + 0.25 * pp_, xytext=xx, arrowprops=dict(arrowstyle="-|>", color=S.INK, lw=1.3), zorder=8)
        S.badge(ax, "U-turn: stop", loc="lower left", fs=8)
    BP.frame(ax, ban, equal=True, labels=False)
    ax.set_xlim(-2.7, 2.7); ax.set_ylim(-2.3, 0.6)
fig.tight_layout()
bk.savefig(fig, "2-18_nuts_doubling")
plt.show()

# %% [markdown]
# Read the panels left to right. Each doubling adds as many steps as the trajectory already has, in a random
# direction of time. The trajectory runs along the arc in both directions from the start. In the last panel the
# red arrows are the momenta at the two ends: at least one of them now points back towards the other end (a
# negative dot product, printed above), so NUTS stops and draws the star from the points it visited. The same iteration,
# animated:

# %%
fig, ax = plt.subplots(figsize=(6.6, 4.0), dpi=95)
BP.density(ax, g_ban)
BP.frame(ax, ban, equal=True)
ax.set_xlim(-2.7, 2.7); ax.set_ylim(-2.3, 0.6)
ax.plot(*tree["x0"], "o", color=S.INK, ms=7, zorder=6)
ttl = ax.set_title("")
frames = [S.fig_to_rgb(fig)] * 6
creation = np.arange(len(tree["states"]))
for (v, j, a, b) in tree["doublings"]:
    for k in range(a, b):
        ax.scatter(*tree["states"][k], s=22, color=BLUES[min(j, 5)], zorder=5, edgecolors=S.SURFACE, linewidths=0.4)
        ttl.set_text(f"doubling {j + 1} ({'forward' if v == 1 else 'backward'} in time): step {k - a + 1} of {b - a}")
        frames.append(S.fig_to_rgb(fig))
    frames.extend([S.fig_to_rgb(fig)] * 5)
for xx, pp_ in [(xm, pm), (xp, pp)]:
    ax.annotate("", xy=xx + 0.25 * pp_, xytext=xx, arrowprops=dict(arrowstyle="-|>", color=S.BAD, lw=1.4), zorder=8)
ttl.set_text("U-turn: the ends turn towards each other. Stop.")
frames.extend([S.fig_to_rgb(fig)] * 10)
ax.plot(*tree["chosen"], "*", color=S.INK, ms=14, mec=S.SURFACE, zorder=9)
ttl.set_text("Next sample: drawn from the visited points (star)")
frames.extend([S.fig_to_rgb(fig)] * 6)
final_png = bk.savefig(fig, "2-18_nuts_tree_final")           # the last frame as a 300-dpi still
plt.close(fig)
gif = S.frames_to_gif(frames, S.FIG / "2-18_nuts_tree.gif", fps=8)
print(f"saved figures/2-18_nuts_tree.gif ({gif.stat().st_size / 1e6:.1f} MB, {len(frames)} frames)")
S.show_gif(gif)
from IPython.display import Image as _Image, display as _display
print("The last frame at full resolution (300 dpi):")
_display(_Image(filename=str(final_png), width=600))

# %% [markdown]
# ## 6. Choosing the step size automatically: dual averaging
#
# NUTS still needs $\varepsilon$. During **warm-up** it tunes $\varepsilon$ so that the average acceptance
# statistic hits a target (0.8 by default in NumPyro and Stan), using the dual-averaging scheme of Hoffman &
# Gelman (2014). After warm-up $\varepsilon$ is frozen; warm-up draws are discarded because the chain was
# changing its own rules while they were made.

# %%
it = np.arange(len(nu_rec["eps"]))
nw = 500
win = 25
acc_run = np.convolve(nu_rec["accept_stat"], np.ones(win) / win, mode="valid")
fig, axs = plt.subplots(1, 3, figsize=(15, 3.4))
axs[0].plot(it, nu_rec["eps"], color=NUTS_C, lw=0.8)
axs[0].set_yscale("log"); axs[0].set_ylabel("step size ε"); axs[0].set_title("Step size")
axs[1].plot(np.arange(win - 1, len(it)), acc_run, color=NUTS_C, lw=1.0)
axs[1].axhline(0.8, color=S.INK, lw=0.9)
axs[1].text(700, 0.788, "target 0.8", fontsize=8, color=S.INK2)
axs[1].set_ylabel(f"acceptance statistic ({win}-iteration mean)"); axs[1].set_title("Acceptance statistic")
axs[2].plot(it, nu_rec["n_leapfrog"], color=NUTS_C, lw=0.6)
axs[2].set_ylabel("leapfrog steps per iteration"); axs[2].set_title("Trajectory length chosen by NUTS")
for ax in axs:
    ax.axvspan(0, nw, color="#ecebe6", lw=0, zorder=0)
    ax.set_xlabel("iteration")
    ax.set_xlim(0, len(it))
axs[0].text(nw / 2, nu_rec["eps"].max() * 0.8, "warm-up", ha="center", fontsize=8.5, color=S.INK2)
fig.tight_layout()
bk.savefig(fig, "2-18_step_size_adaptation")
plt.show()
print(f"final step size {nu_rec['step_size']:.4f}; mean acceptance statistic: last 100 warm-up iterations "
      f"{nu_rec['accept_stat'][nw - 100:nw].mean():.2f}, after warm-up {nu_rec['accept_stat'][nw:].mean():.2f}; "
      f"tree depth after warm-up: {np.bincount(nu_rec['depth'][nw:]).tolist()} (count of depth 0, 1, 2, ...)")

# %% [markdown]
# Shaded = warm-up. The step size jumps around while it is being tuned, then is frozen. The acceptance
# statistic settles near its target during warm-up; after warm-up it runs a little higher, because the frozen
# step size is a smoothed average of the warm-up values (the printed numbers). The trajectory length on the
# right varies from iteration to iteration: NUTS chooses it afresh each time, depending on where it is and which
# way it was kicked.

# %% [markdown]
# ## 7. Head to head: Metropolis vs HMC vs NUTS at equal cost
#
# Each sampler runs 4 chains and gets the same number of evaluations as NUTS used, warm-up included
# (Metropolis: log densities; HMC and NUTS: gradients). Metropolis gets its best step size from a pilot scan;
# HMC gets NUTS's adapted step size for free plus a hand-chosen $L = 20$ (jittered). So the comparison is
# generous to both. Efficiency is the **effective sample size** (ESS, notebook 2.16) of the worst parameter per 1000
# evaluations. Every comparison runs with two seeds, so the run-to-run spread is visible. First, the two problems
# from notebook 2.02:

# %%
def compare(pr, starts, rw_steps, seed):
    out = {}
    ch, ng, costs = [], 0, []
    for c in range(4):
        nu = SM.nuts(pr, starts[c], 1000, n_warmup=500, rng=np.random.default_rng(seed * 100 + c))
        ch.append(nu["chain"]); ng += int(nu["n_leapfrog"].sum())
        costs.append(np.r_[0, np.cumsum(nu["n_leapfrog"][500:])])          # gradients spent, per chain
    out["NUTS"] = dict(S=np.array(ch), budget=ng, cost=costs, eps=nu["step_size"])
    scan_ = {}
    for st in rw_steps:
        r = SM.rwm(pr, starts, 4000, step=st, rng=np.random.default_rng(seed))
        scan_[st] = min(ess_capped(r["samples"][:, 800:, i]) for i in range(pr.dim))
    st = max(scan_, key=scan_.get)
    n_rw = ng // 4
    r = SM.rwm(pr, starts, n_rw, step=st, rng=np.random.default_rng(seed + 1))
    out["Metropolis"] = dict(S=r["samples"][:, n_rw // 5:], budget=4 * n_rw, step=st, acc=r["accept_rate"])
    L = 20
    n_h = ng // (4 * L)
    hs = [SM.hmc(pr, starts[c], n_h, eps=out["NUTS"]["eps"], L=L, rng=np.random.default_rng(seed * 10 + c),
                 jitter_L=True)["chain"] for c in range(4)]
    out["HMC"] = dict(S=np.array(hs)[:, n_h // 5:], budget=4 * n_h * L)
    for k, v in out.items():
        e = [ess_capped(v["S"][:, :, i]) for i in range(pr.dim)]
        v["ess"], v["per1000"] = min(e), 1000 * min(e) / v["budget"]
        v["rhat"] = max(D.rhat(v["S"][:, :, i]) for i in range(pr.dim))
    return out


t0 = time.time()
starts_ban = np.array([[0, 0], [1, -0.4], [-1, -0.4], [0.5, -0.1]], float)
starts_lin = np.array([[0.2, 0.7], [0.6, 0.3], [-0.2, 1.1], [0.0, 0.9]], float)
results = {}
for name, pr, st0, steps in [("banana", ban, starts_ban, [0.05, 0.1, 0.2, 0.4, 0.6]),
                             ("trade-off", lin, starts_lin, [0.03, 0.06, 0.1, 0.15, 0.25])]:
    results[name] = [compare(pr, st0, steps, seed) for seed in (1, 2)]
print(f"(ran in {time.time() - t0:.0f} s)")
print(f"{'problem':10s} {'sampler':11s} {'evaluations':>12s} {'min ESS':>15s} {'ESS per 1000 eval.':>20s} {'max R-hat':>16s}")
for name, (a, b) in results.items():
    for k in ["Metropolis", "HMC", "NUTS"]:
        print(f"{name:10s} {k:11s} {a[k]['budget']:12d} {a[k]['ess']:7.0f} /{b[k]['ess']:6.0f} "
              f"{a[k]['per1000']:10.1f} /{b[k]['per1000']:6.1f} {a[k]['rhat']:9.3f} /{b[k]['rhat']:6.3f}")
print("(two numbers = seed 1 / seed 2; Metropolis step sizes chosen by the pilot scan:",
      {name: (a["Metropolis"]["step"], b["Metropolis"]["step"]) for name, (a, b) in results.items()}, ")")

# %%
fig, axs = plt.subplots(2, 3, figsize=(14.5, 7.0))
for row, (name, pr, g) in enumerate([("banana", ban, g_ban), ("trade-off", lin, g_lin)]):
    res = results[name][0]
    for col, k in enumerate(["Metropolis", "HMC", "NUTS"]):
        ax = axs[row, col]
        BP.density(ax, g)
        X = res[k]["S"].reshape(-1, 2)
        BP.samples(ax, X, RWM_C if k == "Metropolis" else NUTS_C, n_max=1500, s=4, alpha=0.5)
        BP.frame(ax, pr, equal=pr is ban, labels=False)
        S.badge(ax, f"ESS per 1000 evaluations: {res[k]['per1000']:.1f}", loc="lower left", fs=8)
        ax.set_title(f"{k} ({name})")
fig.tight_layout()
bk.savefig(fig, "2-18_head_to_head_samples")
plt.show()

# %% [markdown]
# **How to read it.** All three samplers end up in the right place (the figure). In the table, compare ESS per
# 1000 evaluations within each problem and against the seed-to-seed spread. On these two problems Metropolis and
# NUTS are about equally efficient, and the hand-tuned HMC is the most efficient (several times more so on the
# trade-off ridge). That is honest and expected. In two dimensions, with ridges about ten times longer than they are wide, a random walk is not yet
# hopeless, and NUTS pays for its automatic stopping rule (its warm-up is counted against it, while HMC got its
# step size for free). NUTS's value is that nobody has to find a good $L$, which in hundreds of dimensions nobody
# can.
#
# The sentence you asked about is about **long, thin, curved ridges**, so turn exactly that knob. Keep the
# banana and make it sharper (the distance measured more precisely): the ridge gets thinner while its length
# stays the same. NUTS against Metropolis, at equal cost, 4 chains each, two seeds per width:

# %%
sigmas = [0.1, 0.05, 0.025, 0.0125]
starts_arc = on_arc([[0, 0], [1, -0.2], [-1, -0.2], [0.5, -0.05],              # seed 1
                     [0.3, 0], [-0.7, -0.3], [1.2, -0.5], [-0.2, -0.05]])      # seed 2
rows, keep = [], {}
t0 = time.time()
for sig_ in sigmas:
    pr = banana(sig_)
    for seed in (1, 2):
        st2 = starts_arc[4 * (seed - 1): 4 * seed]
        ch, ng, costs = [], 0, []
        for c in range(4):
            nu = SM.nuts(pr, st2[c], 1000, n_warmup=500, rng=np.random.default_rng(1000 * seed + c))
            ch.append(nu["chain"]); ng += int(nu["n_leapfrog"].sum())
            costs.append(np.r_[0, np.cumsum(nu["n_leapfrog"][500:])])
        ch = np.array(ch)
        e_n = min(ess_capped(ch[:, :, i]) for i in range(2))
        best = None
        for f_ in (0.5, 1, 2, 4, 6):                                   # Metropolis pilot scan, step = f * width
            r = SM.rwm(pr, st2, 3000, step=f_ * sig_, rng=np.random.default_rng(seed))
            e = min(ess_capped(r["samples"][:, 600:, i]) for i in range(2))
            if best is None or e > best[1]:
                best = (f_ * sig_, e)
        n_rw = ng // 4
        r = SM.rwm(pr, st2, n_rw, step=best[0], rng=np.random.default_rng(seed + 10))
        s_ = r["samples"][:, n_rw // 5:]
        e_r = min(ess_capped(s_[:, :, i]) for i in range(2))
        rows.append(dict(sigma=sig_, seed=seed, budget=ng, nuts=1000 * e_n / ng, rwm=1000 * e_r / ng,
                         step=best[0], acc=r["accept_rate"], leap=nu["n_leapfrog"][500:].mean()))
        if seed == 1:
            keep[sig_] = dict(nuts=ch, nuts_cost=costs, rwm=s_)
print(f"(ran in {time.time() - t0:.0f} s)")
print(f"{'width':>7s} {'seed':>4s} {'evaluations':>11s} {'NUTS ESS/1000':>14s} {'Metropolis ESS/1000':>20s} "
      f"{'ratio':>6s} {'NUTS steps/iter':>16s} {'Metr. step, acc.':>17s}")
for r_ in rows:
    print(f"{r_['sigma']:7.4f} {r_['seed']:4d} {r_['budget']:11d} {r_['nuts']:14.2f} {r_['rwm']:20.3f} "
          f"{r_['nuts'] / r_['rwm']:6.1f} {r_['leap']:16.1f} {r_['step']:10.4f}, {r_['acc']:.2f}")

# %%
fig, axs = plt.subplots(1, 2, figsize=(12, 4.2))
sg = np.array([r_["sigma"] for r_ in rows])
for key, col, lab in [("nuts", NUTS_C, "NUTS (per 1000 gradients)"), ("rwm", RWM_C, "Metropolis (per 1000 log densities)")]:
    val = np.array([r_[key] for r_ in rows])
    axs[0].plot(sg, val, "o", color=col, ms=6, mec=S.SURFACE, label=lab)
    means = [np.exp(np.mean(np.log(val[sg == s]))) for s in sigmas]
    axs[0].plot(sigmas, means, "-", color=col, lw=1.5)
ref = np.array(sigmas)
m_n = np.exp(np.mean(np.log([r_["nuts"] for r_ in rows if r_["sigma"] == sigmas[0]])))
m_r = np.exp(np.mean(np.log([r_["rwm"] for r_ in rows if r_["sigma"] == sigmas[0]])))
axs[0].plot(ref, m_n * (ref / ref[0]), color=S.MUTED, lw=0.8)
axs[0].plot(ref, m_r * (ref / ref[0]) ** 2, color=S.MUTED, lw=0.8)
xm_ = 0.035                                                            # label the reference lines mid-way
axs[0].text(xm_, m_n * (xm_ / ref[0]) * 1.25, "∝ width", fontsize=8, color=S.INK2, ha="center")
axs[0].text(xm_, m_r * (xm_ / ref[0]) ** 2 * 0.62, "∝ width²", fontsize=8, color=S.INK2, ha="center")
axs[0].set_xscale("log"); axs[0].set_yscale("log")
axs[0].set_xlabel("ridge width: distance uncertainty σ (x 10 km)")
axs[0].set_ylabel("ESS per 1000 evaluations")
axs[0].set_title("Thinner ridge: random walk loses quadratically")
axs[0].legend(loc="lower right", fontsize=8)
ratio = [np.exp(np.mean(np.log([r_["nuts"] / r_["rwm"] for r_ in rows if r_["sigma"] == s]))) for s in sigmas]
axs[1].plot(sigmas, ratio, "-o", color=S.INK, ms=6)
for r_ in rows:
    axs[1].plot(r_["sigma"], r_["nuts"] / r_["rwm"], "o", color=S.INK2, ms=3)
axs[1].axhline(1, color=S.MUTED, lw=0.8)
axs[1].set_xscale("log"); axs[1].set_yscale("log")
axs[1].set_xlabel("ridge width σ (x 10 km)")
axs[1].set_ylabel("NUTS efficiency / Metropolis efficiency")
axs[1].set_title("The advantage of following the gradient")
for ax in axs:
    ax.set_xticks(sigmas)
    ax.set_xticklabels([str(s) for s in sigmas])
    ax.xaxis.set_minor_locator(plt.NullLocator())
    ax.invert_xaxis()
fig.tight_layout()
bk.savefig(fig, "2-18_sharpness_scaling")
plt.show()

# trace of the east coordinate on the sharpest banana against evaluations spent
sig_t = sigmas[-1]
budget_show = 20000
fig, axs = plt.subplots(2, 1, figsize=(12, 4.4), sharex=True, sharey=True)
xr = keep[sig_t]["rwm"][0, :budget_show, 0]
axs[0].plot(np.arange(len(xr)), xr, color=RWM_C, lw=0.6)
cost_n = keep[sig_t]["nuts_cost"][0][1:]
xn = keep[sig_t]["nuts"][0, :, 0]
sel = cost_n <= budget_show
axs[1].plot(cost_n[sel], xn[sel], "-o", color=NUTS_C, lw=0.6, ms=1.8)
axs[0].set_title(f"Metropolis on the sharpest banana (σ = {sig_t}): one log density per step", fontsize=9.5, loc="left")
axs[1].set_title("NUTS on the same banana: gradients per iteration chosen by NUTS", fontsize=9.5, loc="left")
for ax in axs:
    ax.set_ylabel("east (x 10 km)")
axs[1].set_xlabel("evaluations spent (chain 1, after warm-up / burn-in)")
axs[0].set_xlim(0, budget_show)
fig.tight_layout()
bk.savefig(fig, "2-18_head_to_head_traces")
plt.show()

# %% [markdown]
# **The sentence in numbers.** As the ridge gets thinner, NUTS's efficiency falls roughly in proportion to the
# width (upper grey line), while Metropolis's falls roughly as the square of the width (lower grey line); the
# table has the numbers. A random walk's distance grows like the square root of the number of steps, while a
# trajectory that follows the gradient travels in proportion to its number of steps (Neal 2011). So the advantage
# of NUTS grows as the ridge gets thinner (right panel): about even on the banana of 2.02, several times at
# the sharpest one, and still rising. The traces show the same thing over the first 20,000 evaluations: the Metropolis
# trace drifts slowly along the arc, while NUTS crosses it again and again.
#
# **100 dimensions.** A Gaussian whose standard deviations range from 0.05 to 1 (independent parameters; identity
# mass matrix for NUTS, isotropic proposals for Metropolis, so neither gets to rescale the axes).

# %%
d = 100
sig = np.geomspace(0.05, 1.0, d)
sj = jnp.asarray(sig)
g100 = P.Target(lambda m: -0.5 * jnp.sum((m / sj) ** 2), d, name="100-D Gaussian")
x0s = np.random.default_rng(0).standard_normal((2, d)) * sig
t0 = time.time()
ch100, ng100 = [], 0
for c in range(2):
    nu = SM.nuts(g100, x0s[c], 600, n_warmup=300, rng=np.random.default_rng(10 + c))
    ch100.append(nu["chain"]); ng100 += int(nu["n_leapfrog"].sum())
ch100 = np.array(ch100)
cost100 = np.r_[0, np.cumsum(nu["n_leapfrog"][300:])]
e_n = np.array([ess_capped(ch100[:, :, i]) for i in range(d)])
best = None
for st in [0.006, 0.012, 0.02, 0.03]:
    r = SM.rwm(g100, x0s, ng100 // 2, step=st, rng=np.random.default_rng(1))
    s_ = r["samples"][:, (ng100 // 2) // 5:]
    e_r = np.array([ess_capped(s_[:, :, i]) for i in range(d)])
    print(f"Metropolis step {st}: acceptance {r['accept_rate']:.2f}, ESS of the widest parameter {e_r[-1]:.1f}, "
          f"of the narrowest {e_r[0]:.0f}")
    if best is None or e_r.min() > best[1].min():
        best = (st, e_r, s_)
print(f"(ran in {time.time() - t0:.0f} s)")
print(f"budget: {ng100} evaluations for each sampler (2 chains)")
print(f"NUTS:       min ESS over the 100 parameters {e_n.min():7.1f}  ->  {1000 * e_n.min() / ng100:8.3f} per 1000 gradients")
print(f"Metropolis: min ESS over the 100 parameters {best[1].min():7.1f}  ->  {1000 * best[1].min() / ng100:8.4f} per 1000 log densities (best step {best[0]})")
print(f"ratio of efficiencies (NUTS / Metropolis): {e_n.min() / best[1].min():.0f}")

fig, axs = plt.subplots(2, 1, figsize=(12, 4.6), sharey=True)
xr = best[2][0, :, -1]
axs[0].plot(np.arange(len(xr)), xr, color=RWM_C, lw=0.6)
axs[0].set_xlim(0, len(xr))
axs[0].set_title("Metropolis: the widest parameter (sd 1) against log-density evaluations", fontsize=9.5, loc="left")
axs[1].plot(cost100[1:], ch100[1, :, -1], color=NUTS_C, lw=0.6)
axs[1].set_xlim(0, cost100[-1])
axs[1].set_title("NUTS: the same parameter against gradient evaluations", fontsize=9.5, loc="left")
for ax in axs:
    ax.axhline(2, color=S.MUTED, lw=0.6); ax.axhline(-2, color=S.MUTED, lw=0.6)
    ax.set_ylabel("value")
axs[1].set_xlabel("evaluations spent (one chain, after warm-up / burn-in)")
fig.tight_layout()
bk.savefig(fig, "2-18_dimension_100")
plt.show()

# %% [markdown]
# Grey lines mark ±2 standard deviations of that parameter. In 100 dimensions a blind proposal has to be tiny
# to be accepted (it must not step off the *narrowest* direction), so the widest parameter crawls: the
# Metropolis trace crosses its own distribution only about once in the whole run. NUTS crosses it many times for
# the same cost. The ratio
# printed above is the practical meaning of "orders of magnitude" (theory: Metropolis needs a number of steps
# per independent sample that grows like the dimension $d$, HMC like $d^{1/4}$; Beskos et al. 2013). For a
# 695-node slip model this is the difference between feasible and not.

# %% [markdown]
# ## 8. The funnel: when no single step size works
#
# Neal's funnel in 10 dimensions: $v \sim N(0, 3^2)$ and nine parameters $x_i \sim N(0, e^{v})$, the shape of a
# hierarchical inversion that samples a scale ($v$, like $\log\alpha^2$) together with the parameters it scales
# ($x$, like slip). Sampled two ways with NumPyro's NUTS, 4 chains each:
# * **centered**: sample $(v, \mathbf x)$ directly;
# * **non-centered**: sample $(v, \mathbf z)$ with $\mathbf z \sim N(0, I)$ and set $\mathbf x = e^{v/2}\mathbf z$
#   afterwards. Same model, different coordinates (notebook 2.02, section 6).

# %%
from numpyro.infer import MCMC, NUTS

funnel_runs = {}
for centered in (True, False):
    f = P.funnel(dim=10, centered=centered, scale_v=3.0)
    mc = MCMC(NUTS(potential_fn=lambda z, f=f: -f._logp_fn(z)), num_warmup=1000, num_samples=2000, num_chains=4,
              chain_method="sequential", progress_bar=False)
    init = jnp.asarray(np.random.default_rng(0).standard_normal((4, 10)) * 0.5)
    t0 = time.time()
    mc.run(jax.random.PRNGKey(2), init_params=init, extra_fields=("diverging", "num_steps"))
    z = np.asarray(mc.get_samples(group_by_chain=True))
    ex = mc.get_extra_fields(group_by_chain=True)
    v = z[:, :, 0]
    x1 = z[:, :, 1] if centered else np.exp(v / 2) * z[:, :, 1]      # back to the physical x for the picture
    funnel_runs[centered] = dict(v=v, x1=x1, div=np.asarray(ex["diverging"]), steps=np.asarray(ex["num_steps"]),
                                 eps=np.asarray(mc.last_state.adapt_state.step_size))
    r_ = funnel_runs[centered]
    print(f"{'centered    ' if centered else 'non-centered'}: divergent transitions {int(r_['div'].sum()):4d} of {r_['div'].size}, "
          f"step size per chain {np.round(r_['eps'], 3).tolist()}, leapfrog steps per draw: mean {r_['steps'].mean():.1f}, "
          f"max {r_['steps'].max()}; R-hat(v) {D.rhat(v):.3f}, ESS(v) {ess_capped(v):.0f}; "
          f"P(v < -3) = {np.mean(v < -3):.3f}  ({time.time() - t0:.0f} s)")
from scipy.stats import norm
print(f"exact: P(v < -3) = {norm.cdf(-3, scale=3):.3f}")

# %%
fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw=dict(width_ratios=[1, 1, 0.9]))
vv = np.linspace(-9, 9, 300)
for ax, centered, t_ in [(axs[0], True, "Centered: samples and divergences"), (axs[1], False, "Non-centered (mapped back to x)")]:
    r_ = funnel_runs[centered]
    ax.scatter(r_["v"].ravel(), r_["x1"].ravel(), s=3, color=NUTS_C, alpha=0.35, lw=0, rasterized=True)
    dv = r_["div"].astype(bool)
    if dv.any():
        ax.scatter(r_["v"][dv], r_["x1"][dv], s=28, color=S.BAD, marker="x", lw=1.3, label="divergent transition", zorder=5)
        ax.legend(loc="upper left", fontsize=8)
    ax.plot(vv, 2 * np.exp(vv / 2), color=S.MUTED, lw=0.8)
    ax.plot(vv, -2 * np.exp(vv / 2), color=S.MUTED, lw=0.8)
    ax.set_xlim(-9, 9); ax.set_ylim(-12, 12)
    ax.set_xlabel("$v$ (log variance)"); ax.set_ylabel("$x_1$")
    ax.set_title(t_)
axs[0].text(-8.6, -11, "grey lines: ±2 sd of $x$ given $v$", fontsize=8, color=S.INK2)
for centered, col, lab in [(True, S.INK2, "centered"), (False, NUTS_C, "non-centered")]:
    axs[2].hist(funnel_runs[centered]["v"].ravel(), bins=np.linspace(-9, 9, 46), density=True, histtype="step",
                color=col, lw=1.6, label=lab)
axs[2].plot(vv, norm.pdf(vv, scale=3), color=S.INK, lw=1.0, label="exact N(0, 3²)")
axs[2].axvspan(-9, -3, color="#ecebe6", lw=0, zorder=0)
axs[2].text(-8.6, 0.135, "the neck\n(v < -3)", fontsize=8, color=S.INK2)
axs[2].set_xlabel("$v$"); axs[2].set_ylabel("density"); axs[2].set_title("The scale parameter $v$")
axs[2].legend(loc="upper right", fontsize=8)
fig.tight_layout()
bk.savefig(fig, "2-18_funnel_numpyro")
plt.show()

# %% [markdown]
# **The image to keep.** In centered coordinates the neck (small $v$) is extremely narrow and steep (section 1),
# while the mouth is wide and flat. The adapted step size is a compromise that is too big for the neck, so
# trajectories that enter it diverge (red crosses, where there are any) and the chain rarely goes there: compare
# the printed $P(v<-3)$ with the exact value, and the grey histogram with the black curve on the right. The chain is
# **biased**, and it says so only through divergences, long trajectories (the maximum printed above is the
# depth-10 cap) and a poor R-hat/ESS for $v$. In non-centered coordinates every direction has the same scale, the
# step size is several times larger, trajectories are short, and the neck is sampled correctly.
#
# The same geometry appears in hierarchical inversions that sample a smoothing or prior scale $\alpha$ together with
# the slip $\mathbf m$ it scales: written centered, $(\alpha, \mathbf m)$ is a funnel; written non-centered,
# $\mathbf m = \alpha\mathbf u$ with $\mathbf u$ standard normal, it is round as long as the data are weak (next
# paragraph). When NUTS struggles, change the coordinates before changing the sampler.
#
# One caution: which coordinates are round depends on the data. Non-centering wins when the data say little about
# $\mathbf x$ compared with the prior (this funnel has no data at all). When the data pin $\mathbf x$ down tightly,
# the *centered* form is the round one and the non-centered form develops the funnel instead (Papaspiliopoulos,
# Roberts & Sköld 2007; Betancourt & Girolami 2015). Notebook 2.33 shows both cases on a fault-slip problem.

# %% [markdown]
# ## 9. The mass matrix: rescaling and rotating the landscape
#
# The momentum kick $\mathbf p\sim N(0,\mathbf M)$ and the motion $\dot{\mathbf m} = \mathbf M^{-1}\mathbf p$ use a
# **mass matrix** $\mathbf M$. Choosing $\mathbf M^{-1}$ equal to the posterior covariance is the same as running
# HMC on a **whitened** posterior, where the ridge has become a round bowl. (Whitening is the change of coordinates,
# built from the Cholesky factor of a covariance, that turns a correlated Gaussian into independent ones with unit
# variance; 2.06, section 5 draws it.) During warm-up NumPyro estimates
# $\mathbf M^{-1}$ from the draws: by default only its diagonal (one scale per parameter); with
# `dense_mass=True` the full matrix (scales *and* correlations). Test: a 2-D Gaussian with standard deviations
# 10 and 0.1 and correlation 0.95.

# %%
s1, s2, rho = 10.0, 0.1, 0.95
Cov = np.array([[s1 ** 2, rho * s1 * s2], [rho * s1 * s2, s2 ** 2]])
Pinv = jnp.asarray(np.linalg.inv(Cov))
mass_runs = {}
for dense in (False, True):
    mc = MCMC(NUTS(potential_fn=lambda m: 0.5 * m @ Pinv @ m, dense_mass=dense), num_warmup=1000, num_samples=2000,
              num_chains=4, chain_method="sequential", progress_bar=False)
    mc.run(jax.random.PRNGKey(3), init_params=jnp.zeros((4, 2)), extra_fields=("num_steps",))
    z = np.asarray(mc.get_samples(group_by_chain=True))
    ns = np.asarray(mc.get_extra_fields(group_by_chain=True)["num_steps"])
    imm = np.asarray(mc.last_state.adapt_state.inverse_mass_matrix)[0]          # chain 1's adapted M^-1
    e = min(ess_capped(z[:, :, i]) for i in range(2))
    mass_runs[dense] = dict(z=z, steps=ns, imm=imm, ess=e, per1000=1000 * e / ns.sum())
    print(f"{'dense   ' if dense else 'diagonal'} mass matrix: leapfrog steps per draw {ns.mean():5.1f}, "
          f"min ESS {e:6.0f}, ESS per 1000 gradients {1000 * e / ns.sum():6.1f}")

# %%
rng = np.random.default_rng(0)
Xs = mass_runs[False]["z"].reshape(-1, 2)[::4]
imm_d = mass_runs[False]["imm"]
imm_f = mass_runs[True]["imm"]
Wd = Xs / np.sqrt(imm_d)                                     # diagonal mass: rescale each axis
Wf = np.linalg.solve(np.linalg.cholesky(imm_f), Xs.T).T     # dense mass: rescale and rotate (whiten)
fig, axs = plt.subplots(1, 3, figsize=(13.5, 4.0))
panels = [(Xs, "The posterior as the sampler sees it\n(identity mass; axes not to scale)", False),
          (Wd, "Diagonal mass matrix:\neach axis rescaled, the tilt remains", True),
          (Wf, "Dense mass matrix:\nrescaled and rotated, a round bowl", True)]
for ax, (W, t_, eq) in zip(axs, panels):
    ax.scatter(W[:, 0], W[:, 1], s=3, color=NUTS_C, alpha=0.35, lw=0, rasterized=True)
    ax.set_title(t_, fontsize=9.5)
    if eq:
        ax.set_aspect("equal"); ax.set_xlim(-4.5, 4.5); ax.set_ylim(-4.5, 4.5)
axs[0].set_xlabel("$m_1$ (sd 10)"); axs[0].set_ylabel("$m_2$ (sd 0.1)")
for ax in axs[1:]:
    ax.set_xlabel("whitened $m_1$"); ax.set_ylabel("whitened $m_2$")
S.badge(axs[1], f"{mass_runs[False]['steps'].mean():.1f} steps/draw, ESS/1000 grad {mass_runs[False]['per1000']:.0f}", loc="lower right", fs=8)
S.badge(axs[2], f"{mass_runs[True]['steps'].mean():.1f} steps/draw, ESS/1000 grad {mass_runs[True]['per1000']:.0f}", loc="lower right", fs=8)
fig.tight_layout()
bk.savefig(fig, "2-18_mass_matrix")
plt.show()

# %% [markdown]
# A diagonal mass matrix fixes *scales* (sd 10 vs 0.1 become 1 vs 1) but not *correlation*: the whitened
# cloud is still a tilted needle, so NUTS still zig-zags across it and needs more steps per draw. A dense mass
# matrix also undoes the correlation, and the bowl is round. So a poor R-hat (2.19) from NUTS can come from a strongly
# correlated pair of parameters met by a diagonal mass matrix, not from separated modes; a dense block (or
# reparameterising the pair) addresses that, while tempering (2.20) targets a different problem.
# The catch: a dense mass matrix has $d(d+1)/2$ entries to estimate during warm-up and costs $O(d^2)$ per step,
# so for hundreds of slip patches one often uses dense blocks for a few strongly correlated parameters, not one
# dense matrix for everything.

# %% [markdown]
# ## 10. Summary
#
# | word | picture |
# |---|---|
# | gradient of $\log p$ | an arrow at every point: uphill direction and steepness (section 1) |
# | potential energy $U = -\log p$ | the landscape turned upside down: probable regions are valleys |
# | momentum | a random kick at the start of each trajectory; it carries the sampler along valleys |
# | leapfrog | the numerical integrator: half kick, drift, kick, ..., half kick |
# | trajectory | the path of the puck during one iteration (blue lines) |
# | energy error, divergence | how badly leapfrog conserved energy; a divergence is a blow-up where the curvature is too strong for the step size |
# | U-turn, tree depth | NUTS's stopping rule (ends moving back towards each other) and the number of doublings it took |
# | dual averaging | the warm-up rule that tunes $\varepsilon$ to a target acceptance (0.8) |
# | mass matrix | a change of scale (diagonal) or of scale and rotation (dense) that makes the posterior round |
# | centered / non-centered | the same hierarchical model in coordinates where scale and parameters are tangled (funnel) or independent (round) |
#
# **In practice.** A slip inversion with a linear forward model (precomputed Green's functions) is a case where NUTS
# shines: the gradient through it is cheap with autodiff, and hundreds of correlated slip patches are far beyond what
# a random walk can explore (section 7's 100-D test). The common pathologies map onto this notebook one to one:
# step-size collapse with trajectories at the depth cap is the funnel (section 8; non-center, with the caution at the
# end of section 8); a poor R-hat with a smooth, unimodal posterior can be a correlated ridge met by a diagonal mass
# matrix (section 9); saturated trees with a collapsed step size can also mean a non-differentiable log density
# (2.17, section 6). When a NUTS run goes wrong, read its divergences, tree depths and step size first: they point to
# which picture you are in.
#
# Next: **2.19 · Convergence, mixing and R-hat**.
