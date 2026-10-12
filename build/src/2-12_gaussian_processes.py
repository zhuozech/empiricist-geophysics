# %% [markdown]
# # 2.12 · Gaussian processes: priors on whole functions
#
# A slip model is not two numbers; it is a whole function (slip at every point of the fault). To put a prior on
# a function we need to say how values at *different* points relate: if slip is 2 m here, how likely is 2 m one
# kilometre away? A **Gaussian process** (GP) answers this with one function, the kernel. It is the prior behind
# "smooth" solutions, behind the Matérn covariances common in slip and friction inversions, and (in disguise) behind
# Laplacian smoothing.
#
# This notebook uses a GP as a **prior on a field** (slip along a fault). Topic 3 (3.03) uses the same mathematics for
# a different job, as an **emulator**: a cheap approximation of a forward model learned from a set of model runs. The
# formulas are shared; what the GP describes is not.
#
# **Words this notebook defines:** Gaussian process, mean function, kernel (covariance function), length scale,
# amplitude, smoothness $\nu$ (Matérn), stationarity, conditioning, GP regression, posterior predictive,
# hyperparameters, marginal likelihood, Laplacian smoothing as a Gaussian prior (GMRF), null space.
#
# **Before this:** 2.04, 2.08, 2.09. · **You will be able to:** choose a kernel, length scale and smoothness for a
# field prior and predict what its draws look like; condition a GP on point data; explain why sampling the
# hyperparameters gives more honest bands than fixing them; see Laplacian smoothing as a GP-like prior with a null
# space. · **Optional sections:** 7 (two-dimensional fields and bounded slip).

# %% [markdown]
# **Problems in this notebook** (size and forward model; notebook 2.25 explains the degree of nonlinearity)
#
# | problem | unknowns | data | forward model | linear? |
# |---|---|---|---|---|
# | GP prior draws (1-3, 6, 7) | values on a grid | none | prior only | no data |
# | GP regression (4) | values on a grid | 14 | point observations | linear |
# | hierarchical GP (5) | same + L, amplitude, noise | 14 | point observations | nonlinear in the hyperparameters |

# %%
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.linalg import cho_factor, cho_solve, cholesky, solve_triangular
from scipy.special import kv, gamma as Gamma
import jax, jax.numpy as jnp
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
import bayeskit as bk
from bayeskit import problems as P, samplers as SM, diagnostics as D, plotting as BP, style as S

rng = np.random.default_rng(7)
GREYS4 = [S.INK, S.INK2, S.MUTED, "#b5b4ad"]                  # four shades + direct labels for kernel identity
DIVERGING = LinearSegmentedColormap.from_list("bx_div", ["#0d366b", "#3987e5", "#f0efec", "#e34948", "#7a1f1f"])
BLUES = LinearSegmentedColormap.from_list("bx_blues", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def matern(r, L, nu, amp=1.0):
    """Matérn covariance k(r) = amp^2 2^(1-nu)/Gamma(nu) (sqrt(2 nu) r/L)^nu K_nu(sqrt(2 nu) r/L).
    nu = 0.5, 1.5, 2.5 have closed forms; nu = inf is the squared exponential."""
    r = np.abs(np.asarray(r, float))
    if np.isinf(nu):
        return amp ** 2 * np.exp(-0.5 * (r / L) ** 2)
    if nu == 0.5:
        return amp ** 2 * np.exp(-r / L)
    if nu == 1.5:
        a = np.sqrt(3) * r / L
        return amp ** 2 * (1 + a) * np.exp(-a)
    if nu == 2.5:
        a = np.sqrt(5) * r / L
        return amp ** 2 * (1 + a + a ** 2 / 3) * np.exp(-a)
    a = np.sqrt(2 * nu) * r / L
    with np.errstate(invalid="ignore"):
        k = amp ** 2 * 2 ** (1 - nu) / Gamma(nu) * a ** nu * kv(nu, a)
    return np.where(r == 0, amp ** 2, k)


def sym_sqrt(K):
    """Symmetric square root K^(1/2). Using it (instead of Cholesky) with the SAME white noise z makes draws from
    different kernels share their large-scale shape, so only the roughness changes between panels."""
    ev, U = np.linalg.eigh(K)
    return (U * np.sqrt(np.clip(ev, 0, None))) @ U.T

# %% [markdown]
# ## 1. A Gaussian process is just a very large multivariate Gaussian
#
# Pick any set of points $x_1, \dots, x_n$ along the fault. A GP says the values $f(x_1), \dots, f(x_n)$ are jointly
# Gaussian with mean $0$ (the *mean function*; it could be any function) and covariance
# $\mathrm{Cov}[f(x_i), f(x_j)] = k(x_i, x_j)$, the **kernel**. Nothing more. Here $k$ depends only on the distance
# $|x_i - x_j|$ (a **stationary** kernel: the same rules everywhere along the fault). The kernel below is the squared
# exponential with length scale $L$ = 15 km and amplitude 1 m.

# %%
x = np.linspace(0, 100, 201)                                  # position along strike (km)
K = matern(x[:, None] - x[None, :], L=15.0, nu=np.inf) + 1e-9 * np.eye(x.size)
Lc = cholesky(K, lower=True)
F = (Lc @ rng.standard_normal((x.size, 3000))).T              # 3000 prior draws of the whole function
ia, ib, ic = np.searchsorted(x, [40.0, 48.0, 90.0])

fig, axs = plt.subplots(1, 4, figsize=(15, 3.4), gridspec_kw=dict(width_ratios=[1.05, 1.6, 1, 1]))
im = axs[0].imshow(K, extent=(0, 100, 100, 0), cmap=S.GREYS, vmin=0, vmax=1)
axs[0].set_title("(a) covariance matrix $k(x_i, x_j)$")
axs[0].set_xlabel("$x_j$ (km)"); axs[0].set_ylabel("$x_i$ (km)")
fig.colorbar(im, ax=axs[0], fraction=0.046, pad=0.03, label="covariance (m²)")
for k, c in enumerate(GREYS4[:3] + ["#b5b4ad", S.INK2]):
    axs[1].plot(x, F[k], color=c, lw=1.2)
for xi, lab in [(x[ia], "A"), (x[ib], "B"), (x[ic], "C")]:
    axs[1].axvline(xi, color=S.AXIS, lw=0.8, zorder=0)
    axs[1].text(xi, 2.9, lab, ha="center", fontsize=9, color=S.INK2)
axs[1].set_ylim(-3.2, 3.2)
axs[1].set_title("(b) five draws of the whole function")
axs[1].set_xlabel("along strike x (km)"); axs[1].set_ylabel("slip (m)")
for ax, j, lab in [(axs[2], ib, "B"), (axs[3], ic, "C")]:
    ax.plot(F[:, ia], F[:, j], "o", color=S.INK, ms=1.6, alpha=0.35, rasterized=True)
    ax.set_xlim(-3.5, 3.5); ax.set_ylim(-3.5, 3.5); ax.set_aspect("equal")
    ax.set_xlabel("slip at A (m)"); ax.set_ylabel(f"slip at {lab} (m)")
    ax.set_title(f"({'c' if lab == 'B' else 'd'}) A vs {lab}: {abs(x[j] - x[ia]):.0f} km apart")
fig.tight_layout()
bk.savefig(fig, "2-12_gp_is_a_big_gaussian")
plt.show()
for j, lab in [(ib, "B"), (ic, "C")]:
    print(f"correlation A-{lab}: kernel says {K[ia, j]:.3f}, the 3000 draws give {np.corrcoef(F[:, ia], F[:, j])[0, 1]:.3f}")

# %% [markdown]
# **The image to keep.** (a) is the whole prior: a covariance matrix with a bright band along the diagonal (each point
# is strongly tied to its neighbours) fading with distance. (b) Each curve is *one* random function from that prior.
# (c, d) Take many draws and look at just two points: nearby points move together (a tilted ellipse, the same
# trade-off picture as in notebook 2.02), distant points are independent (a round cloud). The correlations printed
# above come straight from the kernel, and the draws reproduce them.

# %% [markdown]
# ## 2. The kernel decides roughness: squared exponential and the Matérn family
#
# Same length scale (15 km), same amplitude, and **the same random numbers** in every panel. Only the kernel shape
# changes. The Matérn parameter $\nu$ counts derivatives: a Matérn-$\nu$ draw is $\lceil\nu\rceil - 1$ times
# differentiable, where $\lceil\nu\rceil$ is ν rounded up (Rasmussen & Williams 2006, §4.2). So $\nu = 1/2$ gives continuous but nowhere-differentiable
# draws (jagged), $\nu = 3/2$ once differentiable, $\nu = 5/2$ twice, and the squared exponential ($\nu \to \infty$)
# infinitely smooth.

# %%
kernels = [("squared exponential ($\\nu \\to \\infty$)", np.inf), ("Matérn $\\nu$ = 5/2", 2.5), ("Matérn $\\nu$ = 3/2", 1.5),
           ("Matérn $\\nu$ = 1/2 (exponential)", 0.5)]
Z = rng.standard_normal((x.size, 3))
fig, axs = plt.subplots(1, 5, figsize=(16, 3.0), sharey=False)
for ax, (name, nu) in zip(axs[:4], kernels):
    Kk = matern(x[:, None] - x[None, :], 15.0, nu)
    Fk = sym_sqrt(Kk) @ Z
    for k in range(3):
        ax.plot(x, Fk[:, k], color=GREYS4[k], lw=1.0)
    ax.set_title(name, fontsize=9.5)
    ax.set_xlabel("x (km)")
    ax.set_ylim(-3, 3)
axs[0].set_ylabel("prior draw (m)")
r = np.linspace(0, 60, 300)
for (name, nu), c in zip(kernels, GREYS4):
    axs[4].plot(r, matern(r, 15.0, nu), color=c, lw=1.6, label=name.split(" (")[0])
axs[4].legend(loc="upper right", fontsize=7.5)
axs[4].set_xlabel("distance r (km)"); axs[4].set_ylabel("correlation k(r)")
axs[4].set_title("kernel vs distance (L = 15 km)", fontsize=9.5)
fig.tight_layout()
bk.savefig(fig, "2-12_kernels_roughness")
plt.show()

# %% [markdown]
# Left to right the draws get rougher while their large-scale wandering stays recognisably the same (same random
# numbers). The right panel shows why: at short distance the squared exponential is flat (neighbours almost
# identical), the exponential drops linearly right from $r = 0$ (neighbours already differ). **Roughness lives in the
# kernel's behaviour near $r = 0$; the length scale controls how far correlation reaches.**

# %% [markdown]
# ## 3. Length scale and amplitude
#
# The two knobs every kernel has. Matérn $\nu = 5/2$, same random numbers throughout.

# %%
fig, axs = plt.subplots(2, 3, figsize=(13, 5.2), sharex=True)
Z = rng.standard_normal((x.size, 3))
for ax, Lk in zip(axs[0], [3.0, 10.0, 30.0]):
    Fk = sym_sqrt(matern(x[:, None] - x[None, :], Lk, 2.5)) @ Z
    for k in range(3):
        ax.plot(x, Fk[:, k], color=GREYS4[k], lw=1.0)
    ax.set_title(f"length scale L = {Lk:.0f} km, amplitude 1 m", fontsize=9.5)
    ax.set_ylim(-3.5, 3.5)
for ax, amp in zip(axs[1], [0.3, 1.0, 3.0]):
    Fk = sym_sqrt(matern(x[:, None] - x[None, :], 10.0, 2.5, amp)) @ Z
    for k in range(3):
        ax.plot(x, Fk[:, k], color=GREYS4[k], lw=1.0)
    ax.set_title(f"amplitude {amp:g} m, L = 10 km", fontsize=9.5)
    ax.set_ylim(-9, 9)
    ax.set_xlabel("x (km)")
axs[0, 0].set_ylabel("prior draw (m)"); axs[1, 0].set_ylabel("prior draw (m)")
fig.tight_layout()
bk.savefig(fig, "2-12_lengthscale_amplitude")
plt.show()

# %% [markdown]
# Top: the length scale is the typical distance between bumps. Bottom: the amplitude only stretches the vertical
# axis (note the wider y-limits). In an inversion the amplitude plays the role of the prior slip size (the
# smoothing scale $\alpha$, notebook 2.02 section 6), and the length scale the role of the smoothing distance.

# %% [markdown]
# ## 4. GP regression: conditioning the prior on data
#
# Observe the function at a few points with noise, $y_i = f(x_i) + \varepsilon_i$. Because prior and noise are
# Gaussian and the "forward model" (pick the value at $x_i$) is linear, the posterior is again a GP, in closed form
# (the same conditioning formula as 2.04, section 5, and 2.08). Here $\mathbf K$ is the kernel between the data points,
# $\mathbf K_*$ between the data points and the points where we want predictions, $\mathbf K_{**}$ between the
# prediction points, and $\sigma_n$ the noise sd:
#
# $$\bar f(\mathbf x_*) = \mathbf K_{*}^\top(\mathbf K + \sigma_n^2\mathbf I)^{-1}\mathbf y, \qquad
# \mathrm{Cov}[f(\mathbf x_*)] = \mathbf K_{**} - \mathbf K_{*}^\top(\mathbf K + \sigma_n^2\mathbf I)^{-1}\mathbf K_{*}.$$
#
# This is *conditioning*: keep only the prior functions that pass near the data. Below, 14 noisy observations of a
# known "true" profile, with a deliberate gap between 40 and 60 km and nothing beyond 0-100 km.

# %%
def truth(x):
    return 1.5 * np.exp(-((x - 28) / 11) ** 2) + 0.9 * np.exp(-((x - 72) / 7) ** 2) - 0.2


xd = np.sort(np.r_[rng.uniform(2, 38, 8), rng.uniform(62, 98, 6)])
sn_true = 0.15
yd = truth(xd) + sn_true * rng.standard_normal(xd.size)
xs = np.linspace(-15, 115, 261)


def gp_post(xd, yd, xs, L, nu=2.5, amp=1.0, sn=0.15):
    """GP posterior mean and covariance at xs, plus the log marginal likelihood of the data."""
    Kdd = matern(xd[:, None] - xd[None, :], L, nu, amp) + sn ** 2 * np.eye(xd.size)
    Ksd = matern(xs[:, None] - xd[None, :], L, nu, amp)
    Kss = matern(xs[:, None] - xs[None, :], L, nu, amp)
    c = cho_factor(Kdd, lower=True)
    a = cho_solve(c, yd)
    mean = Ksd @ a
    V = solve_triangular(c[0], Ksd.T, lower=True)
    cov = Kss - V.T @ V
    logml = -0.5 * yd @ a - np.sum(np.log(np.diag(c[0]))) - 0.5 * xd.size * np.log(2 * np.pi)
    return mean, cov, logml


def band(ax, xs, mean, cov, draws=0, color=S.INK, fill="#d9d8d1", rng=rng, lw=1.6):
    sd = np.sqrt(np.clip(np.diag(cov), 0, None))
    ax.fill_between(xs, mean - 2 * sd, mean + 2 * sd, color=fill, lw=0, zorder=0)
    if draws:
        Lp = np.linalg.cholesky(cov + 1e-8 * np.eye(xs.size))
        for k in range(draws):
            ax.plot(xs, mean + Lp @ rng.standard_normal(xs.size), color=S.MUTED, lw=0.6, zorder=1)
    ax.plot(xs, mean, color=color, lw=lw, zorder=2)
    return sd


def data_and_truth(ax):
    ax.plot(xs, truth(xs), color=S.INK2, lw=1.0, ls=(0, (1, 1.5)), zorder=3)
    ax.plot(xd, yd, "o", color=S.INK, ms=4.5, mec=S.SURFACE, mew=0.8, zorder=4)
    ax.axvspan(40, 60, color="#f0efec", zorder=-1, lw=0)


mean, cov, lml = gp_post(xd, yd, xs, L=10.0)
fig, ax = plt.subplots(figsize=(11, 3.6))
sd = band(ax, xs, mean, cov, draws=6)
data_and_truth(ax)
ax.axhline(0, color=S.AXIS, lw=0.7, zorder=-1)
ax.text(50, -2.25, "no data (40-60 km)", ha="center", fontsize=8, color=S.INK2)
from matplotlib.lines import Line2D
hs = BP.legend_handles([("data", S.INK, "dot"), ("posterior mean", S.INK, "line"), ("posterior draws", S.MUTED, "line"),
                        ("mean ± 2 sd", "#d9d8d1", "region")])
hs.insert(1, Line2D([], [], color=S.INK2, lw=1.0, ls=(0, (1, 1.5)), label="true profile"))
ax.legend(handles=hs, loc="upper right", ncol=5, fontsize=8)
ax.set_ylim(-2.4, 2.9); ax.set_xlabel("along strike x (km)"); ax.set_ylabel("slip (m)")
ax.set_title("GP posterior (Matérn 5/2, L = 10 km, amplitude 1 m, noise 0.15 m)")
fig.tight_layout()
bk.savefig(fig, "2-12_gp_regression")
plt.show()
for where, xx in [("at a datum (x = %.1f km)" % xd[3], xd[3]), ("middle of the gap (x = 50 km)", 50.0),
                  ("far outside the data (x = 115 km)", 115.0)]:
    i = np.argmin(np.abs(xs - xx))
    print(f"posterior sd {where}: {sd[i]:.3f} m   (prior sd 1.000 m, noise sd {sn_true} m)")

# %% [markdown]
# **How to read it.** Near data the band pinches to about the noise level; in the gap it opens up; beyond the data it
# returns to the prior (mean 0, sd 1 m, printed above). The posterior draws (thin grey) are *all* the functions
# still consistent with data and prior. The posterior mean is smoother than any single draw: it is an average, not a
# typical function, the 1-D version of the banana lesson in notebook 2.02.
#
# Now change one knob at a time: the length scale, then the noise level we *assume*.

# %%
fig, axs = plt.subplots(2, 3, figsize=(14, 5.4), sharex=True, sharey=True)
print(f"{'setting':>26s} {'RMS vs true profile, 0-100 km':>30s} {'log marginal likelihood':>25s}")
inside = (xs >= 0) & (xs <= 100)
for ax, Lk in zip(axs[0], [2.5, 10.0, 40.0]):
    m_, c_, lml = gp_post(xd, yd, xs, L=Lk)
    band(ax, xs, m_, c_)
    data_and_truth(ax)
    ax.set_title(f"L = {Lk:g} km", fontsize=9.5)
    print(f"{'L = %g km' % Lk:>26s} {np.sqrt(np.mean((m_ - truth(xs))[inside] ** 2)):30.3f} {lml:25.2f}")
for ax, snk in zip(axs[1], [0.01, 0.15, 0.6]):
    m_, c_, lml = gp_post(xd, yd, xs, L=10.0, sn=snk)
    band(ax, xs, m_, c_)
    data_and_truth(ax)
    ax.set_title(f"assumed noise σn = {snk:g} m  (true 0.15 m)", fontsize=9.5)
    ax.set_xlabel("x (km)")
    print(f"{'noise %g m' % snk:>26s} {np.sqrt(np.mean((m_ - truth(xs))[inside] ** 2)):30.3f} {lml:25.2f}")
axs[0, 0].set_ylabel("slip (m)"); axs[1, 0].set_ylabel("slip (m)")
axs[0, 0].set_ylim(-2.4, 2.9)
fig.tight_layout()
bk.savefig(fig, "2-12_gp_knobs")
plt.show()

# %% [markdown]
# * **Too short a length scale** (top left): between data the prior has no reason to connect the points, so the mean
#   falls back towards 0 and the band balloons. **Too long** (top right): the prior refuses to bend fast enough and the
#   narrow second bump is flattened.
# * **Assumed noise too small** (bottom left): the mean is forced through every noisy datum (overfitting).
#   **Too large**: the data are discounted and the band is wide everywhere.
# * The table above scores each choice two ways: error against the true profile (which we only know in a synthetic test)
#   and the *log marginal likelihood*, which does not need it. Compare their rankings; section 5 explains the second
#   number.

# %% [markdown]
# ## 5. Hyperparameters: let the data choose L, amplitude and noise
#
# $L$, the amplitude and the noise level are **hyperparameters**: parameters of the prior and noise model rather
# than of the function itself. The *marginal likelihood* is the probability of the data with the function integrated
# out. For a GP it is closed form:
#
# $$\log p(\mathbf y\mid L, a, \sigma_n) = \underbrace{-\tfrac12\mathbf y^\top\mathbf K_y^{-1}\mathbf y}_{\text{data fit}}
# \;\underbrace{-\tfrac12\log|\mathbf K_y|}_{\text{complexity penalty}} - \tfrac n2\log 2\pi,\qquad \mathbf K_y = \mathbf K + \sigma_n^2\mathbf I.$$
#
# A flexible prior (short $L$, big amplitude) fits any data, so it spreads its probability thinly over many possible
# datasets and pays through the determinant term. The surface below holds the noise at its true value.

# %%
Ls = np.exp(np.linspace(np.log(1.5), np.log(80), 70))
amps = np.exp(np.linspace(np.log(0.15), np.log(5), 70))
LML = np.array([[gp_post(xd, yd, xd[:1], L=Lk, amp=ak)[2] for Lk in Ls] for ak in amps])
ia_, il_ = np.unravel_index(np.argmax(LML), LML.shape)
L_ml, a_ml = Ls[il_], amps[ia_]
fit_t, cplx_t = [], []
for Lk in Ls:
    Ky = matern(xd[:, None] - xd[None, :], Lk, 2.5, a_ml) + sn_true ** 2 * np.eye(xd.size)
    fit_t.append(-0.5 * yd @ np.linalg.solve(Ky, yd))
    cplx_t.append(-0.5 * np.linalg.slogdet(Ky)[1])

fig, axs = plt.subplots(1, 2, figsize=(12, 3.9))
cs = axs[0].contourf(Ls, amps, LML, levels=np.linspace(LML.max() - 25, LML.max(), 11), cmap=S.GREYS, extend="min")
axs[0].contour(Ls, amps, LML, levels=[LML.max() - 2], colors=S.INK, linewidths=1.0, linestyles="solid")
axs[0].plot(L_ml, a_ml, "o", color=S.INK, ms=7, mec=S.SURFACE)
axs[0].set_xscale("log"); axs[0].set_yscale("log")
from matplotlib.ticker import FixedLocator, NullFormatter, ScalarFormatter
for axis, ticks in [(axs[0].xaxis, [2, 5, 10, 20, 50]), (axs[0].yaxis, [0.2, 0.5, 1, 2, 5])]:
    axis.set_major_locator(FixedLocator(ticks)); axis.set_major_formatter(ScalarFormatter()); axis.set_minor_formatter(NullFormatter())
axs[0].set_xlabel("length scale L (km)"); axs[0].set_ylabel("amplitude (m)")
axs[0].set_title("log marginal likelihood (dot = maximum)")
fig.colorbar(cs, ax=axs[0], label="log p(y | L, amplitude)")
axs[1].plot(Ls, fit_t, color=S.INK2, lw=1.4)
axs[1].plot(Ls, cplx_t, color=S.MUTED, lw=1.4)
axs[1].plot(Ls, np.array(fit_t) + np.array(cplx_t), color=S.INK, lw=2.2)
axs[1].text(Ls[3], fit_t[3] - 1, "data fit", fontsize=8.5, color=S.INK2, va="top")
axs[1].text(Ls[3], cplx_t[3] + 1, "complexity penalty", fontsize=8.5, color=S.MUTED, va="bottom")
axs[1].text(Ls[-25], (np.array(fit_t) + np.array(cplx_t))[-25] + 1.5, "sum", fontsize=8.5, color=S.INK)
axs[1].axvline(L_ml, color=S.AXIS, lw=0.8)
axs[1].set_xscale("log")
axs[1].xaxis.set_major_locator(FixedLocator([2, 5, 10, 20, 50])); axs[1].xaxis.set_major_formatter(ScalarFormatter())
axs[1].xaxis.set_minor_formatter(NullFormatter())
axs[1].set_xlabel("length scale L (km)"); axs[1].set_ylabel("log-likelihood terms")
axs[1].set_title("the two terms along L (amplitude at its maximum)")
fig.tight_layout()
bk.savefig(fig, "2-12_marginal_likelihood")
plt.show()
print(f"maximum of the marginal likelihood (ML-II): L = {L_ml:.1f} km, amplitude = {a_ml:.2f} m")

# %% [markdown]
# The black contour encloses hyperparameters within 2 log-units of the best: a whole ridge of $(L, a)$ pairs explains
# the data almost equally well. Picking the single best point ("ML-II", or tuning by eye) ignores that spread.
# The hierarchical alternative samples the hyperparameters together with everything else. With NumPyro NUTS
# (NumPyro is a Python library for Bayesian models; NUTS, its sampler, is explained in 2.18)
# (priors: $L \sim$ LogNormal(log 15 km, 0.75), amplitude $\sim$ HalfNormal(2 m), noise $\sim$ HalfNormal(0.5 m)):

# %%
xdj, ydj = jnp.asarray(xd), jnp.asarray(yd)


def gp_model(x, y):
    L = numpyro.sample("L", dist.LogNormal(jnp.log(15.0), 0.75))
    amp = numpyro.sample("amplitude", dist.HalfNormal(2.0))
    sn = numpyro.sample("noise", dist.HalfNormal(0.5))
    a = jnp.sqrt(5.0) * jnp.abs(x[:, None] - x[None, :]) / L
    Ky = amp ** 2 * (1 + a + a ** 2 / 3) * jnp.exp(-a) + (sn ** 2 + 1e-9) * jnp.eye(x.size)
    numpyro.sample("y", dist.MultivariateNormal(jnp.zeros(x.size), covariance_matrix=Ky), obs=y)


mcmc = MCMC(NUTS(gp_model), num_warmup=600, num_samples=1000, num_chains=4, chain_method="sequential",
            progress_bar=False)
mcmc.run(jax.random.PRNGKey(3), xdj, ydj, extra_fields=("diverging",))
post = {k: np.asarray(v) for k, v in mcmc.get_samples(group_by_chain=True).items()}
names = ["L", "amplitude", "noise"]
D.print_summary(np.stack([post[k] for k in names], -1), names, title="hyperparameter posterior (4 chains x 1000)")
print("divergent transitions:", int(np.sum(mcmc.get_extra_fields()["diverging"])))

# %%
flat = {k: v.reshape(-1) for k, v in post.items()}
sel = rng.choice(flat["L"].size, 300, replace=False)
pool = []
for i in sel:                                                 # draw functions under each hyperparameter sample
    m_, c_, _ = gp_post(xd, yd, xs, L=flat["L"][i], amp=flat["amplitude"][i], sn=flat["noise"][i])
    Lp = np.linalg.cholesky(c_ + 1e-8 * np.eye(xs.size))
    pool.append(m_[None] + (Lp @ rng.standard_normal((xs.size, 10))).T)
pool = np.concatenate(pool)
lo, hi = np.percentile(pool, [2.5, 97.5], axis=0)
m_ml, c_ml, _ = gp_post(xd, yd, xs, L=L_ml, amp=a_ml, sn=sn_true)
sd_ml = np.sqrt(np.clip(np.diag(c_ml), 0, None))

fig, axs = plt.subplots(1, 2, figsize=(13, 3.8), gridspec_kw=dict(width_ratios=[1, 2.2]))
lg = np.linspace(0.5, 80, 400)
axs[0].hist(flat["L"], bins=60, range=(0, 60), density=True, color="#d9d8d1", lw=0)
axs[0].plot(lg, np.exp(dist.LogNormal(jnp.log(15.0), 0.75).log_prob(jnp.asarray(lg))), color=S.MUTED, lw=1.3)
axs[0].axvline(L_ml, color=S.INK, lw=1.2)
axs[0].set_xlim(0, 60)
axs[0].set_xlabel("length scale L (km)"); axs[0].set_ylabel("density")
axs[0].set_title("L: prior (line) and posterior (bars)")
axs[0].text(L_ml + 1, axs[0].get_ylim()[1] * 0.9, "ML-II point", fontsize=8, color=S.INK)
axs[1].fill_between(xs, lo, hi, color="#d9d8d1", lw=0, label="95 % band, hyperparameters sampled")
axs[1].plot(xs, m_ml - 2 * sd_ml, color=S.INK, lw=1.0)
axs[1].plot(xs, m_ml + 2 * sd_ml, color=S.INK, lw=1.0, label="± 2 sd, hyperparameters fixed at ML-II")
data_and_truth(axs[1])
axs[1].legend(loc="upper right", fontsize=8)
axs[1].set_ylim(-2.4, 2.9); axs[1].set_xlabel("x (km)"); axs[1].set_ylabel("slip (m)")
axs[1].set_title("predictions: plug-in vs hierarchical")
fig.tight_layout()
bk.savefig(fig, "2-12_hierarchical_gp")
plt.show()
gap = (xs > 42) & (xs < 58)
print(f"mean 95 % band width in the gap: hierarchical {np.mean((hi - lo)[gap]):.2f} m, "
      f"plug-in {np.mean(4 * sd_ml[gap]):.2f} m")
print(f"mean 95 % band width near the data (0-38 km): hierarchical {np.mean((hi - lo)[(xs > 2) & (xs < 38)]):.2f} m, "
      f"plug-in {np.mean(4 * sd_ml[(xs > 2) & (xs < 38)]):.2f} m")
out = (xs > 103) | (xs < -3)
print(f"mean 95 % band width beyond the data (x < -3 or x > 103 km): hierarchical {np.mean((hi - lo)[out]):.2f} m, "
      f"plug-in {np.mean(4 * sd_ml[out]):.2f} m")

# %% [markdown]
# The posterior of $L$ is broad (the ridge of the previous figure), and the hierarchical band (grey) is the average
# over all those plausible priors. Compare the widths printed above: where the data are dense the two treatments
# agree; where the prior does the work (the gap, beyond the data) fixing $L$ at one value claims more certainty than
# the data support.
#
# This is the hierarchical idea in miniature: regularisation weights and prior scales are sampled, not tuned
# (2.30 develops it). Note also what makes it work. The marginal likelihood carries the $-\tfrac12\log|\mathbf K_y|$ term. Drop
# that normalising term and the hyperparameters run off to an extreme, which is the same reason a sampled smoothing
# weight needs its log-determinant term (section 6 shows it).

# %% [markdown]
# ## 6. Laplacian smoothing is a Gaussian prior too (and where it is flat)
#
# The classic regularised inversion minimises $\|\mathbf d - \mathbf G\mathbf m\|^2/\sigma^2 + \alpha\|\mathbf D\mathbf m\|^2$
# with $\mathbf D$ a second-difference (1-D Laplacian) operator. Read the penalty as $-\log$ of a prior and it is the
# Gaussian $p(\mathbf m) \propto \exp(-\tfrac{\alpha}{2}\mathbf m^\top\mathbf D^\top\mathbf D\,\mathbf m)$: a prior
# specified by its **precision** (inverse covariance) $\mathbf Q = \alpha\mathbf D^\top\mathbf D$ instead of its
# covariance. Such priors are called Gaussian Markov random fields (GMRF): $\mathbf Q$ is sparse because each value only
# talks to its neighbours, while the implied covariance $\mathbf Q^{-1}$ is dense.
#
# One catch. $\mathbf D$ returns zero for any constant or straight line, so $\mathbf D^\top\mathbf D$ has a **null
# space** (eigenvalue 0) that the penalty never sees. In those directions the prior is *flat*, exactly the flat
# direction of notebook 2.02. Draws make this visible: add a tiny ridge $\epsilon\mathbf I$ (a small amount added to the
# diagonal, the "jitter" of 2.03) so we can draw at all.

# %%
n = 101
xg = np.linspace(0, 100, n)
h = xg[1] - xg[0]
Dm = np.zeros((n - 2, n))
for i in range(n - 2):
    Dm[i, i:i + 3] = [1.0, -2.0, 1.0]
Dm /= h ** 2
DtD = Dm.T @ Dm
ev = np.linalg.eigvalsh(DtD)
print("four smallest eigenvalues of D^T D:", np.array2string(ev[:4], precision=3))
alpha_s, eps = 2.0e4, 1e-6
Q = alpha_s * DtD + eps * np.eye(n)
Lq = np.linalg.cholesky(Q)
Zs = rng.standard_normal((n, 4))
Fq = solve_triangular(Lq.T, Zs, lower=False)                 # draws from N(0, Q^-1)
coef = np.polynomial.polynomial.polyfit(xg, Fq, 1)
trend = np.polynomial.polynomial.polyval(xg, coef).T
Fd = Fq - trend
Kgp = matern(xg[:, None] - xg[None, :], 25.0, 1.5, amp=0.4)
Fg = np.linalg.cholesky(Kgp + 1e-9 * np.eye(n)) @ Zs

fig, axs = plt.subplots(1, 3, figsize=(14, 3.3))
for k in range(4):
    axs[0].plot(xg, Fq[:, k], color=GREYS4[k], lw=1.1)
    axs[1].plot(xg, Fd[:, k], color=GREYS4[k], lw=1.1)
    axs[2].plot(xg, Fg[:, k], color=GREYS4[k], lw=1.1)
axs[0].set_title("Laplacian prior + tiny ridge: raw draws", fontsize=9.5)
axs[1].set_title("same draws, straight-line part removed", fontsize=9.5)
axs[2].set_title("GP draws, Matérn 3/2, L = 25 km, amplitude 0.4", fontsize=9.5)
for ax in axs:
    ax.set_xlabel("x (km)")
axs[0].set_ylabel("prior draw")
fig.tight_layout()
bk.savefig(fig, "2-12_laplacian_vs_gp_draws")
plt.show()
print(f"sd of the straight-line part across draws (set by the ridge eps): {np.std(trend[:, [0, -1]]):.1f}")
print(f"sd of what remains after removing it:                      {np.std(Fd):.2f}")

# %% [markdown]
# Left: the draws are dominated by enormous straight lines. Those are the null-space directions, limited only by the
# arbitrary tiny ridge $\epsilon$ (sizes printed above). Remove the straight-line part (middle) and what is left is
# smooth and wiggly, like the GP draws on the right (whose length scale and amplitude were picked to look
# comparable; the Laplacian prior's wiggle size is set by $\alpha$ and the fault length instead). **Laplacian
# smoothing = a GP-like prior plus a flat direction for offsets and trends.** In a slip inversion the data, edge constraints (zero slip at the fault
# edges) or an extra small-norm term usually close the flat directions; if nothing does, the posterior is improper in
# them (notebook 2.02, panel d).
#
# Sparse precision vs dense covariance, side by side:

# %%
fig, axs = plt.subplots(1, 3, figsize=(12, 3.6))
axs[0].imshow(np.abs(Q) > 1e-12 * np.abs(Q).max(), cmap=S.GREYS, interpolation="nearest")
axs[0].set_title("precision Q (non-zero entries)", fontsize=9.5)
Qi = np.linalg.inv(alpha_s * DtD + 1e-2 * np.eye(n))
axs[1].imshow(Qi / np.abs(Qi).max(), cmap=DIVERGING, vmin=-1, vmax=1, interpolation="nearest")
axs[1].set_title("its inverse: covariance (dense)", fontsize=9.5)
axs[2].imshow(Kgp / Kgp.max(), cmap=DIVERGING, vmin=-1, vmax=1, interpolation="nearest")
axs[2].set_title("GP covariance (dense)", fontsize=9.5)
for ax in axs:
    ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
bk.savefig(fig, "2-12_precision_vs_covariance")
plt.show()
print(f"non-zero entries: Q {np.count_nonzero(np.abs(Q) > 1e-12 * np.abs(Q).max())} of {n * n};  "
      f"GP covariance {np.count_nonzero(np.abs(Kgp) > 1e-12)} of {n * n}")

# %% [markdown]
# For the middle panel the ridge was raised to $10^{-2}$ only to make the inverse finite; with the null space open the
# covariance would be infinite in the constant and trend directions. Even so, the trend directions dominate its
# pattern: the two ends of the profile are anti-correlated (blue corners), the signature of a free tilt.
#
# **Why a sampled smoothing weight needs its log-determinant.** Under this prior the density of a slip profile given
# $\alpha$ is
# $p(\mathbf m\mid\alpha) \propto \alpha^{r/2}\exp(-\tfrac{\alpha}{2}\|\mathbf D\mathbf m\|^2)$, with $r$ the rank of
# $\mathbf D^\top\mathbf D$. The factor $\alpha^{r/2}$ is the normalising constant. As a function of a *fixed* $\alpha$
# it is irrelevant, so it is easy to drop. When $\alpha$ is *sampled* it is essential:

# %%
m_fix = truth(xg)                                             # one smooth profile
r_rank = n - 2
q = np.sum((Dm @ m_fix) ** 2)
la = np.linspace(np.log(1e1), np.log(1e8), 400)
with_det = 0.5 * r_rank * la - 0.5 * np.exp(la) * q
without = -0.5 * np.exp(la) * q
fig, ax = plt.subplots(figsize=(7.5, 3.2))
ax.plot(np.exp(la), with_det - with_det.max(), color=S.INK, lw=2.0)
ax.plot(np.exp(la), without - without.max(), color=S.MUTED, lw=1.5)
ax.set_xscale("log"); ax.set_ylim(-60, 5)
ax.set_xlabel("smoothing weight α"); ax.set_ylabel("log p(m | α) + const")
ax.text(np.exp(la[40]), 1.0, "without (r/2) log α: best α is as small as possible", fontsize=8.5, color=S.MUTED)
ax.text(r_rank / q * 4, -12, "with it: a peak", fontsize=8.5, color=S.INK)
ax.axvline(r_rank / q, color=S.AXIS, lw=0.8)
ax.set_title("the normalising term decides where a sampled α goes")
fig.tight_layout()
bk.savefig(fig, "2-12_logdet_term")
plt.show()
print(f"peak with the term at alpha = r / |D m|^2 = {r_rank / q:.3g};  without it the curve rises monotonically "
      f"towards alpha -> 0")

# %% [markdown]
# Without the normalising term, any smoothing costs probability and nothing rewards it, so $\alpha$ slides to zero (no
# smoothing); flip the sign convention and it slides to infinity. With the term the density has a proper peak. The
# same holds in a slip inversion with a sampled smoothing weight: it needs its $(r/2)\log\alpha$ term (2.30).

# %% [markdown]
# ## 7. Two-dimensional fields on a fault plane
#
# The same kernels work in 2-D with $r$ = distance between two points on the fault. A 120 km × 60 km fault on a 2 km
# grid, the same random numbers in every panel:

# %%
gx, gz = np.meshgrid(np.arange(1, 120, 2.0), np.arange(1, 60, 2.0))
pts = np.column_stack([gx.ravel(), gz.ravel()])
R = np.sqrt(((pts[:, None, :] - pts[None, :, :]) ** 2).sum(-1))
Zf = rng.standard_normal(len(pts))
settings = [(10.0, 0.5), (10.0, 2.5), (40.0, 0.5), (40.0, 2.5)]
fields = [sym_sqrt(matern(R, Lk, nuk)) @ Zf for Lk, nuk in settings]
fig, axs = plt.subplots(2, 2, figsize=(12, 5.4), sharex=True, sharey=True)
for ax, (Lk, nuk), f in zip(axs.flat, settings, fields):
    im = ax.imshow(f.reshape(gx.shape), extent=(0, 120, 60, 0), cmap=DIVERGING, vmin=-2.5, vmax=2.5)
    ax.set_title(f"Matérn L = {Lk:g} km, $\\nu$ = {nuk:g}", fontsize=9.5)
for ax in axs[:, 0]:
    ax.set_ylabel("down dip (km)")
for ax in axs[1]:
    ax.set_xlabel("along strike (km)")
fig.colorbar(im, ax=axs, fraction=0.025, pad=0.02, label="prior draw (standard units)")
bk.savefig(fig, "2-12_fault_fields")
plt.show()

# %% [markdown]
# Rows change the length scale, columns the smoothness. Short $L$ and $\nu = 1/2$ (top left) gives patchy, grainy
# slip; long $L$ and $\nu = 5/2$ (bottom right) gives a few broad, smooth lobes.
#
# **Bounded fields.** Slip is often bounded (non-negative, and below some maximum). A common construction puts a
# Matérn prior on an unbounded field $\xi$ and maps it to bounded slip by $x = lb + (ub - lb)\,\mathrm{sigmoid}(\xi)$
# ($lb$, $ub$: lower and upper bounds; the sigmoid, 2.08, maps any number into the interval 0 to 1).
# Example values for this toy: $L$ = 100 km, $\nu$ = 1, standard deviation 1.4 on $\xi$, and bounds 0-3.5 m. What such
# prior draws look like on this toy fault:

# %%
S_sim = sym_sqrt(matern(R, 100.0, 1.0, amp=1.4))
xi4 = S_sim @ rng.standard_normal((len(pts), 4))              # four independent prior draws of xi
fig, axs = plt.subplots(1, 5, figsize=(16, 2.6), gridspec_kw=dict(width_ratios=[1, 1, 1, 1, 0.8]))
for k in range(4):
    im1 = axs[k].imshow((3.5 / (1 + np.exp(-xi4[:, k]))).reshape(gx.shape), extent=(0, 120, 60, 0), cmap=BLUES,
                        vmin=0, vmax=3.5)
    axs[k].set_title(f"prior draw {k + 1}", fontsize=9.5)
    axs[k].set_xlabel("along strike (km)")
axs[0].set_ylabel("down dip (km)")
fig.colorbar(im1, ax=axs[3], fraction=0.05, pad=0.03, label="dip slip (m)")
xi1 = rng.normal(0, 1.4, 200000)
axs[4].hist(3.5 / (1 + np.exp(-xi1)), bins=50, range=(0, 3.5), density=True, color="#d9d8d1", lw=0)
axs[4].set_xlabel("dip slip at one point (m)"); axs[4].set_ylabel("density")
axs[4].set_title("implied prior at one point", fontsize=9.5)
fig.tight_layout()
bk.savefig(fig, "2-12_bounded_field_draw")
plt.show()
h_, e_ = np.histogram(3.5 / (1 + np.exp(-xi1)), bins=10, range=(0, 3.5), density=True)
print("implied prior density of dip slip in ten 0.35-m bins (flat would be 0.286 everywhere):")
print(np.array2string(h_, precision=3))
print("within-draw spread (sd over the fault) of each slip map:", np.array2string((3.5 / (1 + np.exp(-xi4))).std(0), precision=2), "m")

# %% [markdown]
# With $L$ = 100 km on a 120 km fault, each draw varies smoothly across the whole plane (one broad level, tilt or
# bulge; within-draw spread printed above), while different draws sit at quite different levels. The implied prior at
# a single point (right) is flat over most of the range and thins only near the two bounds: with a standard deviation
# of 1.4 on $\xi$ the sigmoid spreads the prior almost evenly over 0-3.5 m (bin densities printed above). All the fine
# structure in a posterior slip map must come from the data.
#
# **In practice.**
# * The Matérn kernels used in many slip and friction inversions are exactly the covariances of this notebook: $L$ and
#   $\nu$ choose the length scale and roughness of every prior draw.
# * In ensemble Kalman inversion the kernel matters twice (2.24): the posterior is a combination of the initial draws,
#   so the kernel decides which slip patterns are reachable at all, not just which are preferred.
# * When $L$ and $\nu$ are fixed rather than sampled, section 5 shows what is lost: bands that are too narrow wherever
#   the prior does the work.
# * Laplacian smoothing (section 6) is a GP-like prior with a null space; a sampled smoothing weight needs its
#   log-determinant term.
#
# Next: **2.13 · Model errors: C_d and C_p**.
