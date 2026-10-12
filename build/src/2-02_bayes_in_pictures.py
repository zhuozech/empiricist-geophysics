# %% [markdown]
# # 2.02 · Bayes in pictures: what a posterior looks like
#
# A Bayesian inversion answers one question: **which parameter values agree with the data and with what we
# believed before seeing the data?** The answer is not one model but a **probability density** over all models (a
# curve, or a surface over the parameter map, whose height says how plausible each value is), the *posterior*:
#
# $$p(\mathbf m \mid \mathbf d) \;\propto\; \underbrace{p(\mathbf m)}_{\text{prior}}\;\times\;\underbrace{p(\mathbf d \mid \mathbf m)}_{\text{likelihood}}$$
#
# Read $p(\mathbf m \mid \mathbf d)$ as "the probability of $\mathbf m$ given $\mathbf d$". The sign ∝ means "proportional
# to": the right-hand side is multiplied by one constant so that the total probability is 1 (*normalising*).
#
# With one or two unknowns we can evaluate this on a grid and simply look at it. That is what this notebook
# does. Every later notebook is about getting the same picture when there are hundreds of unknowns and a grid
# is impossible (a 700-patch slip model on a 10-point grid per axis would need $10^{700}$ evaluations).
#
# **Before this:** 2.01. · **You will be able to:** sketch prior, likelihood and posterior for a small problem; tell a
# Gaussian posterior from a trade-off ridge, a banana, separate islands and a funnel, and say what causes each; say when
# a posterior is exactly Gaussian. · **Optional sections:** 6 (funnel) and 7 (smooth fields) can be skimmed first time.
#
# **Words this notebook defines with pictures:** prior, likelihood, posterior, Gaussian posterior, flat
# (unconstrained) direction, trade-off, banana, multimodal, funnel, smooth.

# %% [markdown]
# **Problems in this notebook** (size and forward model; notebook 2.25 explains the degree of nonlinearity)
#
# | problem | unknowns | data | forward model | linear? |
# |---|---|---|---|---|
# | slip on one patch (section 1) | 1 | 1 or 10 | $d = m$ + noise | linear |
# | two patches, data as bands (2) | 2 | 1 or 2 | $d_1 = m_1 + m_2$, $d_2 = m_1 - 0.6\,m_2$ | linear |
# | Gaussian and trade-off (3, 8) | 2 | 2 | $\mathbf d = \mathbf G\mathbf m$ | linear |
# | epicentre from distances (3, 4) | 2 | 1 to 3 | $d_i = \lvert\mathbf m - \mathbf s_i\rvert$ | nonlinear |
# | locking depth vs slip rate (5) | 2 | 5 | $v = \frac{s}{\pi}\arctan(x/D)$ | nonlinear |
# | funnel (6) | 2 | none | hierarchical prior only | no data |
# | slip profile with 60 patches (7) | 60 | none | prior draws only | no data |

# %%
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))

import numpy as np
import matplotlib.pyplot as plt
import jax, jax.numpy as jnp
from scipy.optimize import minimize

import bayeskit as bk
from bayeskit import problems as P, plotting as BP, style as S

rng = np.random.default_rng(1)

# %% [markdown]
# ## 1. Bayes' rule is a multiplication of two curves
#
# One unknown: slip $m$ on a fault patch. One GPS station measures $d = m + \varepsilon$ with noise
# $\varepsilon \sim N(0, \sigma^2)$. The **likelihood** is how well each candidate $m$ predicts the datum,
# $\exp[-\tfrac12 (d - m)^2/\sigma^2]$. The **prior** is what we believed before. Multiply them point by point
# and normalise: that is the posterior. Four situations:

# %%
m = np.linspace(-2.5, 4.5, 1400)


def gauss(x, mu, s):
    return np.exp(-0.5 * ((x - mu) / s) ** 2) / (s * np.sqrt(2 * np.pi))


ten = 1.6 + 0.7 * rng.standard_normal(10)                     # ten noisy measurements of a true slip of 1.6
cases = [
    ("(a) Gaussian prior × one datum", (0.0, 1.0), [2.0], 0.7),
    ("(b) Flat prior × one datum", None, [2.0], 0.7),
    ("(c) Strong prior × noisy datum", (0.0, 0.3), [2.0], 1.5),
    ("(d) Gaussian prior × ten data", (0.0, 1.0), list(ten), 0.7),
]
fig, axs = plt.subplots(1, 4, figsize=(13, 3.0), sharey=True)
for ax, (title, prior, data, sig) in zip(axs, cases):
    lik = np.exp(-0.5 * sum(((d - m) / sig) ** 2 for d in data))
    lik /= np.trapezoid(lik, m)                               # scaled to unit area only so it fits the axes
    pri = gauss(m, *prior) if prior else np.full_like(m, 0.12)
    post = pri * lik
    post /= np.trapezoid(post, m)
    ax.fill_between(m, pri, color="#e1e0d9", lw=0, label="prior")
    ax.plot(m, lik, color=S.INK2, lw=1.0, label="likelihood")
    ax.plot(m, post, color=S.INK, lw=2.2, label="posterior")
    ax.plot(data, np.zeros(len(data)), "|", color=S.INK2, ms=10, mew=1.2)
    mu = np.trapezoid(m * post, m)
    sd = np.sqrt(np.trapezoid((m - mu) ** 2 * post, m))
    ax.set_title(title, fontsize=9.5)
    ax.set_xlabel("slip m (m)")
    S.badge(ax, f"posterior {mu:.2f} ± {sd:.2f} m", loc="upper left", fs=8)
axs[0].set_ylabel("density")
axs[0].legend(loc="center left", bbox_to_anchor=(0.0, 0.55), fontsize=8)
axs[1].text(-2.3, 0.14, "flat prior: any constant", fontsize=8, color=S.INK2)
fig.tight_layout()
bk.savefig(fig, "2-02_bayes_1d")
plt.show()

# %% [markdown]
# **How to read it.** The ticks on the axis are the data.
# * (a) The posterior sits between prior and data, and is narrower than either: two independent pieces of
#   information.
# * (b) With a **flat prior** the posterior is just the likelihood, renormalised. "Flat" means *every value is
#   equally plausible beforehand*, so the data alone decide.
# * (c) A confident prior and a noisy datum: the posterior barely moves from the prior. Nothing is wrong
#   here; the data simply carry little information compared with the prior.
# * (d) Ten data: the likelihood is $\sqrt{10}$ times narrower and dominates the prior.
#
# The posterior width printed in each badge is the *uncertainty*. It shrinks when information is added
# (prior or data) and never because the sampler was told to stop.

# %% [markdown]
# ## 2. Two unknowns: each datum is a band, the posterior is where the bands overlap
#
# Two neighbouring slip patches, $m_1$ and $m_2$. A station far away sees only their sum,
# $d_1 = m_1 + m_2 + \varepsilon$. Its likelihood is constant along every line $m_1 + m_2 = \text{const}$: a
# **band** in the plane. Along the band the datum has no opinion at all. That direction is a *flat direction*
# of the likelihood, and only the prior can limit it. A second station with a different view ($d_2 = m_1 - 0.6\,m_2$)
# contributes a band at a different angle.

# %%
x = np.linspace(-3, 3, 401)
X, Y = np.meshgrid(x, x)
prior = np.exp(-0.5 * (X ** 2 + Y ** 2) / 1.0 ** 2)
lik1 = np.exp(-0.5 * ((1.0 - (X + Y)) / 0.15) ** 2)
lik2 = np.exp(-0.5 * ((0.2 - (X - 0.6 * Y)) / 0.15) ** 2)
panels = [
    ("(a) prior", prior),
    ("(b) likelihood of datum 1\n(sees $m_1+m_2$)", lik1),
    ("(c) prior × datum 1", prior * lik1),
    ("(d) FLAT prior × datum 1", lik1),
    ("(e) likelihood of datum 2\n(sees $m_1-0.6\\,m_2$)", lik2),
    ("(f) prior × datum 1 × datum 2", prior * lik1 * lik2),
]
fig, axs = plt.subplots(1, 6, figsize=(15, 2.9))
for ax, (title, img) in zip(axs, panels):
    ax.imshow(img / img.max(), extent=(-3, 3, -3, 3), origin="lower", cmap=S.GREYS, vmin=0, vmax=1)
    ax.set_title(title, fontsize=9)
    ax.set_xticks([-2, 0, 2]); ax.set_yticks([-2, 0, 2])
    ax.set_xlabel("$m_1$")
axs[0].set_ylabel("$m_2$")
# exact Gaussian answers (closed form) drawn on top: they coincide with the grid pictures
mc, Cc = P.linear_gaussian_posterior(np.array([[1.0, 1.0]]), np.array([1.0]), 0.15, np.zeros(2), np.eye(2))
mf, Cf = P.linear_gaussian_posterior(np.array([[1.0, 1.0], [1.0, -0.6]]), np.array([1.0, 0.2]), 0.15, np.zeros(2), np.eye(2))
BP.ellipse(axs[2], mc, Cc, color=S.INK, lw=1.0)
BP.ellipse(axs[5], mf, Cf, color=S.INK, lw=1.0)
axs[3].annotate("goes on forever:\nimproper posterior", xy=(2.9, -1.95), xytext=(-2.85, -2.6), fontsize=7.5, color=S.INK2,
                arrowprops=dict(arrowstyle="->", color=S.INK2, lw=0.7))
fig.tight_layout()
bk.savefig(fig, "2-02_bands_2d")
plt.show()
print("posterior after datum 1:  sd along m1-m2 = %.2f, across = %.3f, correlation = %.3f"
      % (np.sqrt(np.linalg.eigvalsh(Cc)[1]), np.sqrt(np.linalg.eigvalsh(Cc)[0]), Cc[0, 1] / np.sqrt(Cc[0, 0] * Cc[1, 1])))
print("posterior after both data: correlation = %.3f" % (Cf[0, 1] / np.sqrt(Cf[0, 0] * Cf[1, 1])))

# %% [markdown]
# **The images to keep:**
# * A datum that sees a *combination* of parameters gives a band, and the posterior inherits its
#   orientation. That tilted, elongated ellipse in (c) is a **trade-off**, a strong *correlation* between the two
#   parameters (knowing one tells you a lot about the other; 2.03 defines it): you can increase $m_1$ if you decrease
#   $m_2$ by the same amount and the data cannot tell.
# * (d) **Flat prior + flat direction = improper posterior.** The band never ends, so there is no normalised
#   distribution to sample. A sampler run on it just drifts along the band forever (we watch this happen in
#   notebook 2.19).
# * The black ellipses are the closed-form Gaussian answer (2σ). They fall exactly on the grid picture,
#   because this problem is linear.
#
# **A word on "exact".** The grey grid pictures in this course are a *numerical reference*: prior × likelihood
# evaluated at every point of a fine grid. They are exact for the model we wrote down, up to the grid spacing and the
# window, and that is why they can grade samplers. They are not the physical truth: if the forward model, the noise
# model or the prior is wrong, the grid computes the wrong posterior perfectly (2.01, section 7).

# %% [markdown]
# ## 3. "The posterior is Gaussian" means: linear forward model + Gaussian prior + Gaussian noise
#
# A forward model is **linear** when doubling the parameters doubles the predictions, and adding two parameter sets
# adds their predictions; then it is a matrix product, $\mathbf G\mathbf m$. If $\mathbf d = \mathbf G\mathbf m + \boldsymbol\varepsilon$
# with a Gaussian prior $N(\mathbf m_0, \mathbf P)$ (mean $\mathbf m_0$, covariance matrix $\mathbf P$) and Gaussian noise
# $N(0, \mathbf R)$ (covariance $\mathbf R$), the log posterior is a quadratic bowl and the posterior is exactly
# Gaussian, with a **closed form**, a formula that gives the answer without any search or sampling. It is shown here for
# reference (the Kalman update; 2.06 explains the matrix notation, 2.08 derives it, 2.24 uses it again):
#
# $$\bar{\mathbf m} = \mathbf m_0 + \mathbf K(\mathbf d - \mathbf G\mathbf m_0),\qquad
# \mathbf C = \mathbf P - \mathbf K\mathbf G\mathbf P,\qquad \mathbf K = \mathbf P\mathbf G^\top(\mathbf G\mathbf P\mathbf G^\top + \mathbf R)^{-1}.$$
#
# A Gaussian is completely described by its mean and covariance: an ellipse (in 2-D) or ellipsoid. No sampler
# is needed. **A nonlinear forward model bends the bands**, and the posterior stops being Gaussian. Example:
# locating an earthquake from its distance to one station. The datum "distance = 2.5" is a *ring* around the
# station, and a regional prior keeps only an arc of it: a **banana**.

# %%
lin = P.correlated()
ban = P.location(kind="banana")
g_lin, g_ban = P.grid(lin, n=400), P.grid(ban, n=500)

# MAP (the peak) and the Gaussian (Laplace) approximation built from the curvature at the peak
nlp = lambda z: -ban._logp_fn(jnp.asarray(z))
res = minimize(lambda z: float(nlp(z)), x0=np.array([0.4, -0.2]), jac=lambda z: np.asarray(jax.grad(nlp)(jnp.asarray(z))))
H = np.asarray(jax.hessian(nlp)(jnp.asarray(res.x)))
C_lap = np.linalg.inv(H)

fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.0))
BP.density(axs[0], g_lin, lines=False)
for q in (0.5, 0.9, 0.99):                                   # closed-form ellipse holding probability q
    BP.ellipse(axs[0], *lin.exact, nsig=np.sqrt(-2 * np.log(1 - q)), color=S.INK, lw=0.9)
BP.frame(axs[0], lin)
axs[0].set_title("Linear forward model: exactly Gaussian")
S.badge(axs[0], "grey = exact posterior on a grid (50/90/99 %), black = closed form", loc="lower left", fs=8)

BP.density(axs[1], g_ban)
BP.ellipse(axs[1], res.x, C_lap, nsig=1, color=S.BAD, lw=1.2)
BP.ellipse(axs[1], res.x, C_lap, nsig=2, color=S.BAD, lw=1.2)
axs[1].plot(*res.x, "o", color=S.BAD, ms=6, label="peak (MAP)")
axs[1].plot(*g_ban.mean, "P", color=S.INK, ms=9, mec=S.SURFACE, mew=1.2, label="posterior mean")
axs[1].plot(0, -2.5, "^", color=S.INK, ms=9, label="station")
BP.frame(axs[1], ban, equal=True)
axs[1].set_title("Nonlinear forward model: a banana")
axs[1].legend(loc="lower left", fontsize=8)
S.badge(axs[1], "red = Gaussian fitted at the peak (Laplace)", loc="upper right", fs=8)
fig.tight_layout()
bk.savefig(fig, "2-02_gaussian_vs_banana")
plt.show()
inside = g_ban.P[np.unravel_index(np.argmin((g_ban.X - g_ban.mean[0]) ** 2 + (g_ban.Y - g_ban.mean[1]) ** 2), g_ban.P.shape)]
print("density at the posterior mean / density at the peak = %.4f" % (inside / g_ban.P.max()))
print("probability inside the Laplace 2-sigma ellipse vs 0.865 if the posterior were exactly Gaussian: %.3f"
      % g_ban.mass(lambda X, Y: np.einsum("...i,ij,...j->...", np.stack([X - res.x[0], Y - res.x[1]], -1), H, np.stack([X - res.x[0], Y - res.x[1]], -1)) < 4))

# %% [markdown]
# Two lessons from the banana that matter later:
# * **The mean is not a typical model.** The posterior mean (black cross) falls below the arc, where the
#   density is a small fraction of the peak density (ratio printed above). The more curved the banana, the
#   further off the ridge it falls. For non-Gaussian posteriors, report samples or credible regions, not only
#   a mean.
# * **A Gaussian approximation at the peak misses the shape.** The red ellipse is what every method that
#   assumes Gaussianity "sees" (the *Laplace approximation*, a Gaussian fitted to the curvature at the peak, 2.08;
#   linearised error bars; and, in a different way, ensemble Kalman methods, 2.24). It covers the wrong area, as the probability printed above shows.

# %% [markdown]
# ## 4. Same physics, different data: one problem, four shapes
#
# The posterior shape is a property of the *problem* (physics + data + prior), not of the sampler. Here is the
# same epicentre problem with one, two and three stations:

# %%
probs = [P.location(kind="ring"), P.location(kind="banana"), P.location(kind="bimodal"), P.location(kind="gaussian")]
stations = [np.array([[-0.6, 0.0]]), np.array([[0.0, -2.5]]), np.array([[-1.0, 0.0], [1.0, 0.0]]),
            np.array([[-1.0, 0.0], [1.0, 0.0], [0.2, 1.6]])]
titles = ["1 station, broad prior:\nring (a curved flat direction)", "1 station, regional prior:\nbanana",
          "2 stations:\ntwo modes (mirror ambiguity)", "3 stations:\none blob (nearly Gaussian)"]
fig, axs = plt.subplots(1, 4, figsize=(14, 3.7))
for ax, pr, st, t in zip(axs, probs, stations, titles):
    g = P.grid(pr, n=450)
    BP.density(ax, g)
    ax.plot(st[:, 0], st[:, 1], "^", color=S.INK, ms=8)
    if pr.truth is not None and pr.name.startswith(("Two", "One", "Ring")):
        ax.plot(*pr.truth, "*", color=S.INK, ms=10, mec=S.SURFACE)
    BP.frame(ax, pr, equal=True)
    ax.set_title(t, fontsize=9.5)
fig.tight_layout()
bk.savefig(fig, "2-02_location_zoo")
plt.show()
gb = P.grid(probs[2], n=500)
print("2-station problem: probability of the northern mode = %.2f, southern mode = %.2f"
      % (gb.mass(lambda X, Y: Y > 0), gb.mass(lambda X, Y: Y < 0)))

# %% [markdown]
# Triangles are stations, the star is the true epicentre. The two-station case is **multimodal**: two
# separate islands of probability with almost nothing in between. The northern island holds more probability
# (printed above) because the prior is centred slightly north. A sampler that visits only one island reports a
# confident, wrong answer. Notebooks 2.19 and 2.20 show symptoms to look for and how *tempering* (turning the data on
# gradually, so the sampler can spread over every island first) can help.

# %% [markdown]
# ## 5. A geodetic trade-off: locking depth vs slip rate
#
# Between earthquakes (the *interseismic* period) the two sides of a **strike-slip** fault, where the blocks slide
# horizontally past each other, keep moving at depth while the shallow part of the fault is stuck. GPS stations then
# see a smooth velocity step across the fault: $v(x) = \frac{s}{\pi}\arctan(x/D)$ (Savage & Burford 1973), with $x$
# the distance from the fault, $s$ the long-term **slip rate** and $D$ the **locking depth**, the depth above which
# the fault is stuck. With stations only beyond 20 km from the fault, a deeper locking depth can be compensated by a
# faster slip rate.

# %%
sd = P.screw_dislocation()
g_sd = P.grid(sd, n=400)


def grid_draws(g, n, rng):
    """Draw parameter values from a gridded posterior (pick cells with probability P, jitter inside the cell)."""
    w = g.P.ravel() / g.P.sum()
    idx = rng.choice(w.size, n, p=w)
    return np.column_stack([g.X.ravel()[idx] + (rng.uniform(size=n) - 0.5) * g.dx,
                            g.Y.ravel()[idx] + (rng.uniform(size=n) - 0.5) * g.dy])


draws = grid_draws(g_sd, 40, rng)
xx = np.linspace(-5, 100, 300)
fig, axs = plt.subplots(1, 2, figsize=(11, 3.9))
BP.density(axs[0], g_sd)
axs[0].plot(draws[:, 0], draws[:, 1], "o", color=S.INK, ms=3)
axs[0].plot(*sd.truth, "*", color=S.INK, ms=11, mec=S.SURFACE)
BP.frame(axs[0], sd)
axs[0].set_title("Posterior: a trade-off ridge")
for D_, s_ in draws:
    axs[1].plot(xx, s_ / np.pi * np.arctan(xx / D_), color=S.INK2, lw=0.5, alpha=0.6)
axs[1].errorbar(sd.stations_km, sd.y, yerr=sd.sigma, fmt="o", color=S.INK, ms=5, capsize=2, lw=1, zorder=5)
axs[1].set_xlabel("distance from fault (km)")
axs[1].set_ylabel("velocity (mm/yr)")
axs[1].set_title("40 posterior draws: each one fits the data")
fig.tight_layout()
bk.savefig(fig, "2-02_locking_depth_tradeoff")
plt.show()
Cs = g_sd.cov
print("posterior correlation between D and s: %.2f" % (Cs[0, 1] / np.sqrt(Cs[0, 0] * Cs[1, 1])))

# %% [markdown]
# Every grey curve on the right is a different $(D, s)$ pair from the posterior, and every one passes through
# the error bars. The data constrain a *combination*; the ridge on the left is the set of combinations that work.
# This is what "the parameters trade off" means, drawn.

# %% [markdown]
# ## 6. The funnel: one parameter sets the scale of the others
#
# In a hierarchical inversion some parameters set the scale of others: the smoothing weight $\alpha$ sets
# how large slip can be, a noise scale $\sigma_k$ sets how far predictions may stray from data. Neal's funnel is
# the simplest version: $v$ is the logarithm of a variance and $x \sim N(0, e^{v})$, so $v$ sets how widely $x$ spreads.

# %%
fun_c, fun_n = P.funnel(dim=2, centered=True, scale_v=1.5), P.funnel(dim=2, centered=False, scale_v=1.5)
fun_c.extent = fun_n.extent = (-5, 5, -9, 9)
fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
BP.posterior_panel(axs[0], fun_c, title="Centered: sample $(v, x)$: a funnel")
BP.posterior_panel(axs[1], fun_n, title="Non-centered: sample $(v, z)$ with $x = e^{v/2} z$: round")
axs[1].set_ylabel("$z$ (standardised)")
fig.tight_layout()
bk.savefig(fig, "2-02_funnel")
plt.show()

# %% [markdown]
# Left: when $v$ is small, $x$ is squeezed into a narrow neck; when $v$ is large, $x$ spreads into a wide mouth.
# A sampler needs tiny steps in the neck and big steps in the mouth, and no single step size does both. The same
# shape appears in a hierarchical slip inversion that samples the smoothing weight $\alpha$ together with the slip
# $\mathbf m$ it scales, where it can shrink a gradient sampler's step size to almost nothing. Right: **the same model** written in different coordinates,
# $x = e^{v/2} z$ with $z \sim N(0,1)$. Nothing about the posterior changed, only how we describe it, and the
# shape becomes easy. That is what *non-centered parameterisation* means. Notebook 2.18 runs NUTS on both.

# %% [markdown]
# ## 7. Gaussian, flat, smooth: the three words
#
# | word | what it means in a picture | where it comes from |
# |---|---|---|
# | **Gaussian** posterior | elliptical contours; mean and covariance tell the whole story | linear forward model with Gaussian prior and noise (section 3) |
# | **flat** | the density does not change along some direction | a *flat prior* (no preference), or a *flat direction* the data cannot see (section 2). Both together: an improper posterior |
# | **smooth** (a solution) | neighbouring parameters take similar values | a prior that correlates neighbours: Laplacian smoothing or a Gaussian process (notebook 2.12) |
# | **smooth** (a log-posterior) | the surface has no jumps or kinks, so its slope exists everywhere | differentiable forward model and prior; needed by gradient samplers (notebook 2.17) |
#
# A quick look at "smooth solution" for a fault with 60 depth patches: draws from two Gaussian priors that
# differ only in how they correlate neighbours.

# %%
z = np.linspace(0, 30, 60)                                     # depth (km)
C_iid = np.eye(z.size)
C_smooth = np.exp(-0.5 * (z[:, None] - z[None, :]) ** 2 / 4.0 ** 2) + 1e-8 * np.eye(z.size)
fig, axs = plt.subplots(1, 2, figsize=(10, 2.9), sharey=True)
for ax, C, t in zip(axs, [C_iid, C_smooth], ["independent patches (iid Gaussian prior): rough",
                                              "neighbours correlated over 4 km (GP prior): smooth"]):
    Lc = np.linalg.cholesky(C)
    for k in range(4):
        ax.plot(z, Lc @ rng.standard_normal(z.size), lw=1.0, color=[S.INK, S.INK2, S.MUTED, "#b5b4ad"][k])
    ax.set_title(t, fontsize=9.5)
    ax.set_xlabel("depth (km)")
axs[0].set_ylabel("slip (prior draw)")
fig.tight_layout()
bk.savefig(fig, "2-02_smooth_vs_rough")
plt.show()

# %% [markdown]
# Both priors are Gaussian with the same variance at every patch. They differ only in the off-diagonal
# covariance, and that alone decides rough vs smooth. Notebook 2.12 explains this fully (Gaussian processes),
# and notebook 2.33 inverts a fault with each choice.

# %% [markdown]
# ## 8. The zoo: six shapes we will test every sampler on

# %%
zoo = [P.gaussian(), P.correlated(), P.flat_ridge("weak"), P.location(kind="banana"), P.location(kind="bimodal"),
       fun_c]
desc = ["Gaussian", "Trade-off (ρ = %.2f)" % (lin.exact[1][0, 1] / np.sqrt(lin.exact[1][0, 0] * lin.exact[1][1, 1])),
        "Flat direction (weak prior)", "Banana", "Two modes", "Funnel"]
fig, axs = plt.subplots(1, 6, figsize=(16, 2.95))
for ax, pr, t in zip(axs, zoo, desc):
    BP.posterior_panel(ax, pr, title=t, equal=pr.name.startswith(("Banana", "Two")))
    ax.set_xlabel(""); ax.set_ylabel("")
fig.tight_layout()
bk.savefig(fig, "2-02_zoo")
plt.show()

# %% [markdown]
# Grey shading = the exact posterior on a grid (the numerical reference): darkest region holds 50 % of the
# probability, then 90 %, then 99 %.
#
# **What a sampler must do:** produce points whose scatter looks like these pictures. In 2-D we can check
# against the grid; in 700 dimensions we cannot, which is why diagnostics exist (notebook 2.19).
#
# **In practice.** A slip inversion with precomputed Green's functions is linear in slip. With fixed
# hyperparameters, a Gaussian prior and Gaussian noise, its posterior is exactly Gaussian, an ellipsoid in
# hundreds of dimensions, and the closed form above gives it without sampling. The posterior stops being
# Gaussian when the hyperparameters are sampled (a smoothing weight $\alpha$ or data-set noise scales $\sigma_k$: the
# funnel of section 6), when slip is constrained to be positive, or when the forward model is nonlinear (afterslip
# driven by stress, rupture timing). That is where samplers earn their keep.
#
# Next: **2.03 · Probability distributions**.
