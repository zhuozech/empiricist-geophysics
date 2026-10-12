# %% [markdown]
# # 2.33 · Capstone: one fault-slip inversion, every choice
#
# **Before this:** 2.08 (Bayesian linear inversion), 2.09 (smoothing priors), 2.12 (Gaussian-process priors),
# 2.18 (NUTS), 2.24 (EnKI); 2.30 helps with section 5  
# **You will be able to:** run one inversion end to end and change one choice at a time; read prior draws next to the
# posterior; choose between centered and non-centered hyperparameters by their divergences; check an ensemble method
# against an exact answer; read a resolution matrix  
# **Optional sections:** 7 (EnKI), 8 (resolution)
#
# This is the synthesis of Topic 2: the pieces of the earlier notebooks, used together on one problem.
#
# This notebook runs one small but realistic inversion from start to finish and changes one choice at a time:
# the prior (flat, independent, smoothing, Gaussian process), the hyperparameters (fixed or sampled), the
# parameterisation (centered or non-centered), positivity, and the solver (closed form, NUTS, EnKI). Because the
# forward model is linear, the exact answer is available for most cases, so every method can be checked.
#
# The problem: a vertical strike-slip fault in 2-D antiplane strain, 30 slip patches from the surface to 30 km
# depth, 24 surface stations measuring the fault-parallel displacement.
#
# **Words this notebook uses together, and where each is explained:** Green's function matrix and singular values
# (2.06, 2.07); χ² and resolution matrix (2.07); flat and improper priors (2.09, 2.11); ridge (iid) prior and Laplacian
# smoothing (2.09); Matérn / Gaussian-process prior and length scale (2.12); half-normal and half-Cauchy priors (2.11,
# 2.30); hierarchical hyperparameters (2.30); NUTS, leapfrog steps, divergences, centered and non-centered
# parameterisation (2.18); R-hat and effective sample size (2.19); positivity (2.30); EnKI, ensemble size and the
# subspace property (2.24); coverage (2.01).

# %% [markdown]
# **Problems in this notebook** (size and forward model; notebook 2.25 explains the degree of nonlinearity)
#
# | problem | unknowns | data | forward model | linear? |
# |---|---|---|---|---|
# | antiplane fault slip | 30 patches | 24 stations | screw-dislocation Green's functions, $\mathbf d = \mathbf G\mathbf s$ | linear |
# | hierarchical version (5) | 30 + 2 hyperparameters | 24 | linear in slip | nonlinear jointly |
# | positivity (6) | 30, slip ≥ 0 | 24 | linear, bounded prior | linear, bounded |

# %%
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))
import numpy as np
import matplotlib.pyplot as plt
import bayeskit as bk
from bayeskit import problems as P, samplers as SM, diagnostics as D, plotting as BP, style as S

import numpyro
numpyro.set_host_device_count(4)                 # 4 NUTS chains in parallel (must come before any JAX computation)
import jax, jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from scipy.integrate import quad
from scipy.special import kv, gamma
import pandas as pd

rng = np.random.default_rng(0)
C_EX, C_NUTS, C_ENKI = S.INK, S.COLORS["nuts"], S.COLORS["enki"]
TRUE_KW = dict(color="#b5b4ad", lw=4.5, solid_capstyle="round", zorder=1, label="true slip")

# %% [markdown]
# ## 1. The forward model and the Green's function matrix
#
# A patch between depths $z_1$ and $z_2$ slipping by $s$ moves a surface point at distance $x$ from the fault by
#
# $$u(x) = \frac{s}{\pi}\left[\arctan\frac{z_2}{x} - \arctan\frac{z_1}{x}\right]$$
#
# (screw dislocation in a half-space; Segall 2010). Stacking one column per patch gives the Green's function
# matrix $\mathbf G$, and the whole forward model is $\mathbf d = \mathbf G\mathbf s + \boldsymbol\varepsilon$:
# linear in slip. First, three numerical checks of the formula and its sign.

# %%
N = 30
z_edges = np.linspace(0.0, 30.0, N + 1)                      # patch edges (km)
zc = 0.5 * (z_edges[1:] + z_edges[:-1])                      # patch centres (km)
xs = np.array([1, 2, 3, 4, 6, 9, 13, 18, 25, 35, 50, 70.0])
xs = np.sort(np.r_[-xs, xs])                                 # 24 stations, both sides of the fault (km)


def patch_disp(x, z1, z2):
    """Surface displacement per unit slip from a patch between depths z1 and z2 (same units as slip)."""
    return (np.arctan(z2 / x) - np.arctan(z1 / x)) / np.pi


# check 1: the formula is the depth integral of a line dislocation's kernel x / (pi (x^2 + z^2))
x0, z1, z2 = 7.0, 4.0, 9.0
print("check 1  formula %.10f   numerical integral %.10f" % (patch_disp(x0, z1, z2), quad(lambda z: x0 / (np.pi * (x0 ** 2 + z ** 2)), z1, z2)[0]))
# check 2: 3000 thin patches with equal slip add up to one thick patch
fine = np.linspace(0, 30, 3001)
xt = np.array([-20.0, -3.0, 0.5, 10.0, 60.0])
thin = sum(patch_disp(xt, fine[i], fine[i + 1]) for i in range(3000))
print("check 2  max |sum of thin patches - one thick patch| = %.1e" % np.abs(thin - patch_disp(xt, 0, 30)).max())
# check 3: slip from depth D to infinity gives the interseismic arctan(x/D)/pi (Savage & Burford 1973)
print("check 3  max |deep-slip limit - atan(x/D)/pi| = %.1e" % np.abs(patch_disp(xt, 15.0, 1e9) - np.arctan(xt / 15.0) / np.pi).max())
print("         surface-breaking patch: u(0+) = %.4f s, u(0-) = %.4f s  (a jump of s across the fault)" % (patch_disp(1e-9, 0, 10), patch_disp(-1e-9, 0, 10)))

G = 1000.0 * np.array([[patch_disp(x, z_edges[j], z_edges[j + 1]) for j in range(N)] for x in xs])   # mm per m of slip
s_true = 2.0 * np.exp(-0.5 * ((zc - 12.0) / 4.0) ** 2)      # true slip: 2 m peak at 12 km depth, 4 km wide (m)
sigma = 3.0                                                  # data noise (mm)
d = G @ s_true + sigma * np.random.default_rng(42).standard_normal(xs.size)
M = xs.size
print(f"\n{M} data, {N} unknowns; chi^2 of the TRUE slip against the noisy data: {np.sum(((d - G @ s_true) / sigma) ** 2):.1f} (expected about {M})")
sv = np.linalg.svd(G, compute_uv=False)
print("rank of G (independent slip combinations the stations can see at all): %d" % np.linalg.matrix_rank(G))
print("number of singular values of G above the noise level (sigma = %.0f mm): %d of %d" % (sigma, np.sum(sv > sigma), sv.size))
for j in [0, 11, 29]:
    print(f"patch at {zc[j]:4.1f} km: largest surface displacement per metre of slip = {np.abs(G[:, j]).max():6.1f} mm")

# %%
fig, axs = plt.subplots(1, 4, figsize=(16, 3.7), gridspec_kw=dict(width_ratios=[1.25, 1, 1.25, 1]))
xx = np.linspace(-70, 70, 1401)
xx = xx[np.abs(xx) > 1e-6]
ax = axs[0]
ax.plot(xx, 1000 * sum(s_true[j] * patch_disp(xx, z_edges[j], z_edges[j + 1]) for j in range(N)), color=S.INK2, lw=1.2, label="true displacement")
ax.errorbar(xs, d, yerr=2 * sigma, fmt="o", ms=4, color=S.INK, capsize=0, lw=1, label="data ± 2σ")
ax.set_xlabel("distance from fault (km)"); ax.set_ylabel("displacement (mm)")
ax.set_title("(a) Data: 24 stations"); ax.legend(loc="upper left", fontsize=8)
ax = axs[1]
im = ax.imshow(G, aspect="auto", cmap="RdBu_r", vmin=-500, vmax=500, extent=(0, 30, M, 0))
ax.set_xlabel("patch depth (km)"); ax.set_ylabel("station (sorted west to east)")
ax.set_title("(b) Green's functions G")
fig.colorbar(im, ax=ax, label="mm per m of slip", shrink=0.85)
ax = axs[2]
for j, shade in zip([0, 5, 11, 20, 29], [S.INK, "#3d3c39", S.INK2, S.MUTED, "#b5b4ad"]):
    col = 1000 * patch_disp(xx, z_edges[j], z_edges[j + 1])
    ax.plot(xx, col / np.abs(col).max(), color=shade, lw=1.4, label=f"{zc[j]:.1f} km (peak {np.abs(col).max():.0f} mm/m)")
ax.set_xlabel("distance from fault (km)"); ax.set_ylabel("displacement / its peak")
ax.set_title("(c) Columns of G: deeper = broader, weaker"); ax.legend(fontsize=7.2, loc="lower right", title="patch depth", title_fontsize=7.5)
ax = axs[3]
ax.semilogy(np.arange(1, sv.size + 1), sv, "o-", color=S.INK, ms=4, lw=1)
ax.axhline(sigma, color=S.BAD, lw=1)
ax.text(sv.size, sigma * 1.6, "noise level σ", ha="right", fontsize=8, color=S.INK2)
ax.set_xlabel("index"); ax.set_ylabel("singular value (mm per m)")
ax.set_title("(d) What the data can see"); ax.set_ylim(1e-4, 2e3)
fig.tight_layout()
bk.savefig(fig, "2-33_setup")
plt.show()

# %% [markdown]
# **How to read it.** (c) is the key to everything below: a shallow patch makes a sharp, large signal near the
# fault; a deep patch makes a broad, weak one that looks almost the same as its neighbour's. (d) says the same in
# one curve. Each singular value measures how strongly one combination of patch slips shows up in the data;
# only the combinations above the red noise line are actually *measured* (the count is printed above). Every
# other combination must come from the prior. That is why the choice of prior matters so much at depth.
#
# Why half of the singular values are zero: the stations sit in mirror pairs ($\pm x$), and for a vertical
# strike-slip fault the displacement is exactly antisymmetric, $u(-x) = -u(x)$. The two stations of a pair see the
# same slip combination (their independent noise still averages down), so $\mathbf G$ has the rank printed above,
# not 24: at most that many combinations could be measured even with perfect data.

# %% [markdown]
# ## 2. A flat prior: "the data alone decide", and they cannot
#
# A flat prior is approximated by a Gaussian with a huge standard deviation (100 m). The posterior is computed in
# closed form. (With such a broad prior the usual covariance formula $\mathbf P - \mathbf K\mathbf G\mathbf P$ loses
# precision, so here it is computed in the equivalent *precision* form
# $\mathbf C = (\mathbf G^\top\mathbf R^{-1}\mathbf G + \mathbf P^{-1})^{-1}$.)

# %%
def posterior(Pm, d=d, sigma=sigma):
    """Exact linear-Gaussian posterior (zero prior mean) in precision form: returns mean, covariance, precision."""
    Q = G.T @ G / sigma ** 2 + np.linalg.inv(Pm)
    C = np.linalg.inv(Q)
    C = 0.5 * (C + C.T)
    return C @ (G.T @ d / sigma ** 2), C, Q


def draws_from(mean, Q, n, rng):
    """Posterior draws using the Cholesky factor of the precision matrix (stable even for huge prior variance)."""
    Lq = np.linalg.cholesky(Q)
    return mean + np.linalg.solve(Lq.T, rng.standard_normal((mean.size, n))).T


def stats(mean, sd, lo=None, hi=None):
    lo = mean - 2 * sd if lo is None else lo
    hi = mean + 2 * sd if hi is None else hi
    return dict(rms_err=np.sqrt(np.mean((mean - s_true) ** 2)), chi2=np.sum(((d - G @ mean) / sigma) ** 2),
                sd_shallow=sd[zc < 10].mean(), sd_deep=sd[zc > 20].mean(),
                coverage=np.mean((s_true >= lo) & (s_true <= hi)))


def profile(ax, mean, sd=None, dr=None, color=C_EX, title=None, lo=None, hi=None, label="posterior mean"):
    """Slip vs depth (depth increases downwards), true slip in light grey."""
    ax.plot(s_true, zc, **TRUE_KW)
    if sd is not None or lo is not None:
        lo_ = mean - 2 * sd if lo is None else lo
        hi_ = mean + 2 * sd if hi is None else hi
        ax.fill_betweenx(zc, lo_, hi_, color=color, alpha=0.13, lw=0, label="± 2σ" if lo is None else "95 % interval")
    if dr is not None:
        for k, r in enumerate(dr):
            ax.plot(r, zc, color=color, lw=0.6, alpha=0.55, label="posterior draws" if k == 0 else None)
    ax.plot(mean, zc, color=color, lw=2.0, label=label)
    ax.set_ylim(30, 0)
    ax.set_ylabel("depth (km)"); ax.set_xlabel("slip (m)")
    if title:
        ax.set_title(title)


results = {}
m_flat, C_flat, Q_flat = posterior(100.0 ** 2 * np.eye(N))
sd_flat = np.sqrt(np.diag(C_flat))
results["flat prior (sd 100 m), exact"] = stats(m_flat, sd_flat)
fig, axs = plt.subplots(1, 2, figsize=(10, 4.0))
profile(axs[0], m_flat, title="(a) Flat prior: posterior mean")
axs[0].legend(fontsize=8, loc="lower left")
axs[1].semilogx(sd_flat, zc, color=C_EX, lw=2, label="flat prior (sd 100 m)")
axs[1].set_ylim(30, 0); axs[1].set_xlabel("posterior sd (m, log scale)"); axs[1].set_ylabel("depth (km)")
axs[1].set_title("(b) Flat prior: posterior uncertainty")
axs[1].legend(fontsize=8)
fig.tight_layout()
bk.savefig(fig, "2-33_flat_prior")
plt.show()
print("flat prior: posterior mean ranges from %.1f to %.1f m (true slip is between 0 and 2 m)" % (m_flat.min(), m_flat.max()))
print("flat prior: posterior sd from %.2f m (best resolved patch) to %.1f m (worst)" % (sd_flat.min(), sd_flat.max()))
print("chi^2 of the posterior-mean prediction: %.1f for %d data (it fits the noise)" % (results["flat prior (sd 100 m), exact"]["chi2"], M))

# %% [markdown]
# **The image to keep.** With a flat prior the mean swings far beyond the true slip (the range is printed above),
# and the posterior standard deviation is of the order of the prior's 100 m almost everywhere below the shallowest
# patches. Nothing is wrong with the arithmetic: the 24 data measure only a handful of slip combinations (section
# 1), and a flat prior says nothing about the rest. The low $\chi^2$ is the other symptom. The mean uses its
# freedom to fit the noise.

# %% [markdown]
# ## 3-4. Three proper priors: independent, smoothing, Gaussian process
#
# * **iid Gaussian** (ridge, "damping"): every patch $N(0, 1\,\text{m}^2)$, independent of its neighbours.
# * **Laplacian smoothing**: the same damping plus a penalty on the second difference,
#   $p(\mathbf s) \propto \exp[-\tfrac12\|\mathbf s\|^2 - \tfrac12\|\mathbf D\mathbf s\|^2/a^2]$ with $a = 0.1$ m.
#   This is the classic smoothing regularisation written as a prior.
# * **Matérn Gaussian process** with $\nu = 1$ (the kernel family of notebook 2.12), amplitude 1 m and
#   length scale $L = 5$ km: neighbouring patches are correlated over about $L$.
#
# Top row: what each prior believes before seeing data (prior draws). Bottom row: posterior.

# %%
def matern(z, L, nu=1.0, amp=1.0):
    r = np.abs(z[:, None] - z[None, :])
    x = np.sqrt(2 * nu) * r / L
    with np.errstate(invalid="ignore", divide="ignore"):
        C = amp ** 2 * 2 ** (1 - nu) / gamma(nu) * x ** nu * kv(nu, x)
    C[r == 0] = amp ** 2
    return C + 1e-9 * np.eye(z.size)


Dlap = np.zeros((N - 2, N))
for i in range(N - 2):
    Dlap[i, i:i + 3] = [1.0, -2.0, 1.0]
a_smooth = 0.1
priors = {"iid Gaussian (sd 1 m)": np.eye(N),
          "Laplacian smoothing (a = 0.1 m)": np.linalg.inv(np.eye(N) + Dlap.T @ Dlap / a_smooth ** 2),
          "Matérn GP (L = 5 km, ν = 1)": matern(zc, 5.0)}
fig, axs = plt.subplots(2, 3, figsize=(13, 7.2), sharey=True)
for k, (name, Pm) in enumerate(priors.items()):
    Lp = np.linalg.cholesky(Pm)
    for j, shade in enumerate([S.INK, S.INK2, S.MUTED]):
        axs[0, k].plot(Lp @ rng.standard_normal(N), zc, color=shade, lw=1.1)
    axs[0, k].set_ylim(30, 0); axs[0, k].set_xlim(-3.2, 3.2)
    axs[0, k].set_title(f"Prior draws: {name}".replace("ν", "$\\nu$"), fontsize=9.5)
    axs[0, k].set_xlabel("slip (m)")
    m, C, Q = posterior(Pm)
    sd = np.sqrt(np.diag(C))
    results[name + ", exact"] = stats(m, sd)
    profile(axs[1, k], m, sd, draws_from(m, Q, 3, rng), title=f"Posterior: {name}".replace("ν", "$\\nu$"))
    axs[1, k].set_xlim(-1.5, 3.5)
    r = results[name + ", exact"]
    S.badge(axs[1, k], f"rms error {r['rms_err']:.2f} m\nχ² {r['chi2']:.1f} (N = {M})", loc="lower right", fs=8)
    out = np.where(np.abs(s_true - m) > 2 * sd)[0]
    print(f"{name:34s} true slip outside ±2σ at depths (km): {zc[out]}")
axs[0, 0].set_ylabel("depth (km)")
h_, l_ = axs[1, 0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.02))
fig.tight_layout(rect=(0, 0.04, 1, 1))
bk.savefig(fig, "2-33_priors")
plt.show()

# %% [markdown]
# **How to read it.** Read each column top to bottom: the prior's draws tell you what the posterior will do
# wherever the data are blind.
# * **iid**: rough prior draws, so rough posterior draws. Below the resolved depths the mean falls back to the
#   prior mean (zero) and the band widens to the prior's ± 2 m. This is the "deep slip shrinks to zero" artefact
#   of damping.
# * **Laplacian smoothing**: smooth draws, but notice that the prior mostly constrains *curvature*. Straight
#   trends are cheap, so at depth the posterior mean follows the trend it has at the bottom of the resolved zone
#   and can cross zero. That is a flat-ish direction of this prior (compare notebook 2.02, section 2).
# * **Matérn GP**: draws wiggle on a 5 km scale; the posterior is smooth where the data speak and relaxes to
#   zero with the prior's uncertainty below.
#
# All three fit the data equally well: their $\chi^2$ values (badges) are close to that of the true slip itself
# (printed in section 1). The data cannot tell them apart; only the prior decides what happens at depth. The
# Laplacian band is also the narrowest: its stiffness is mistaken for information, and the true slip falls outside
# it where the printout above says.

# %% [markdown]
# ### The length scale: too short, about right, too long

# %%
fig, axs = plt.subplots(1, 3, figsize=(13, 4.0), sharey=True)
for ax, L in zip(axs, [1.0, 5.0, 20.0]):
    m, C, Q = posterior(matern(zc, L))
    sd = np.sqrt(np.diag(C))
    name = f"Matérn GP (L = {L:g} km, ν = 1)"
    results.setdefault(name + ", exact", stats(m, sd))
    profile(ax, m, sd, draws_from(m, Q, 3, rng), title=f"L = {L:g} km")
    ax.set_xlim(-1.5, 3.5)
    r = stats(m, sd)
    S.badge(ax, f"rms error {r['rms_err']:.2f} m\nχ² {r['chi2']:.1f}\ntrue slip inside ±2σ: {r['coverage']:.0%} of patches", loc="lower right", fs=8)
h_, l_ = axs[0].get_legend_handles_labels()
fig.legend(h_, l_, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.03))
fig.tight_layout(rect=(0, 0.05, 1, 1))
bk.savefig(fig, "2-33_lengthscale")
plt.show()

# %% [markdown]
# * $L = 1$ km (the patch size): the GP prior is almost iid. Rough draws and wide bands; honest but vague.
# * $L = 20$ km: the prior insists on long, gentle shapes. The band is narrow, but the posterior spreads the slip
#   peak deeper than the true slip and leaves the true slip outside its band at several depths (badge). It is confident
#   *and* wrong, the dangerous combination.
# * $L = 5$ km, close to the width of the true slip: a narrower band than $L = 1$ km that still contains the true slip.
#
# The posterior means of $L = 1$ and $5$ km are about equally close to the true slip (badges); what the length scale
# mostly controls is how wide the uncertainty is and whether it can be trusted. In a real inversion nobody tells
# you the right $L$: either compare choices like this, or sample the hyperparameters (next section, and notebook 2.12).

# %% [markdown]
# ## 5. Hierarchical: let the data choose the prior scale and the noise scale
#
# Now the prior amplitude $\alpha$ (in metres) and a single noise multiplier are *sampled*:
# $\alpha \sim \text{HalfNormal}(3)$, `noise_scale` $\sim \text{HalfCauchy}(1)$, slip $\sim N(0, \alpha^2\mathbf K)$
# with the $L = 5$ km Matérn correlation $\mathbf K$. With one data set this is a plain noise multiplier,
# not a $\sigma_k$ (those balance several data sets). Two ways to write the same model for NUTS:
#
# * **centered**: sample $\mathbf s \sim N(0, \alpha^2\mathbf K)$ directly;
# * **non-centered**: sample $\mathbf u \sim N(0, \mathbf I)$ and set $\mathbf s = \alpha\,\mathbf L_K\mathbf u$
#   (the scale is then a single multiplier in front of a standard-normal vector).
#
# **An exact yardstick.** Given $\alpha$ and the noise scale, the problem is linear-Gaussian again, so
# $p(\mathbf d\mid\alpha, \text{noise}) = N(\mathbf 0,\ \alpha^2\mathbf G\mathbf K\mathbf G^\top + \text{noise}^2\sigma^2\mathbf I)$
# is known in closed form. A 400 × 400 grid over the two hyperparameters then gives their exact posterior, and
# mixing the closed-form slip posteriors with those grid weights gives the exact hierarchical slip posterior.
# NUTS can be graded against it.

# %%
K5 = matern(zc, 5.0)
LK = np.linalg.cholesky(K5)
Gj, LKj = jnp.asarray(G), jnp.asarray(LK)


def exact_hierarchical(d, sigma, na=400, nn=400, amax=6.0, nmax=4.0, stride=8):
    A = G @ K5 @ G.T
    lam, V = np.linalg.eigh(A)
    r = V.T @ d
    a = np.linspace(1e-3, amax, na)
    n = np.linspace(1e-3, nmax, nn)
    AA, NN = np.meshgrid(a, n, indexing="ij")
    Svar = AA[..., None] ** 2 * lam + (NN[..., None] * sigma) ** 2           # eigenvalues of the data covariance
    loglik = -0.5 * np.sum(np.log(Svar), -1) - 0.5 * np.sum(r ** 2 / Svar, -1)
    logprior = -0.5 * (AA / 3.0) ** 2 - np.log1p(NN ** 2)                    # HalfNormal(3) x HalfCauchy(1)
    W = np.exp(loglik + logprior - (loglik + logprior).max())
    W /= W.sum()
    pa = W.sum(1)
    out = dict(a=a, dens_a=pa / (a[1] - a[0]), alpha_mean=(pa * a).sum())
    out["alpha_sd"] = np.sqrt((pa * (a - out["alpha_mean"]) ** 2).sum())
    pn = W.sum(0)
    out["noise_mean"] = (pn * n).sum()
    # slip moments: mix the conditional Gaussian posteriors over a coarser hyperparameter grid
    Wc = W[::stride, ::stride] / W[::stride, ::stride].sum()
    m1, m2 = np.zeros(N), np.zeros((N, N))
    for i, ai in enumerate(a[::stride]):
        for j, nj in enumerate(n[::stride]):
            if Wc[i, j] < 1e-12:
                continue
            Pm = ai ** 2 * K5
            Kg = Pm @ G.T @ np.linalg.inv(G @ Pm @ G.T + (nj * sigma) ** 2 * np.eye(M))
            mm = Kg @ d
            m1 += Wc[i, j] * mm
            m2 += Wc[i, j] * (Pm - Kg @ G @ Pm + np.outer(mm, mm))
    out["slip_mean"], out["slip_sd"] = m1, np.sqrt(np.diag(m2 - np.outer(m1, m1)))
    return out


def run_nuts(d, sigma, centered, seed, warmup=2000, samples=2000):
    dj = jnp.asarray(d)

    def model():
        alpha = numpyro.sample("alpha", dist.HalfNormal(3.0))
        noise = numpyro.sample("noise_scale", dist.HalfCauchy(1.0))
        if centered:
            s = numpyro.sample("s", dist.MultivariateNormal(jnp.zeros(N), scale_tril=alpha * LKj))
        else:
            u = numpyro.sample("u", dist.Normal(0.0, 1.0).expand([N]).to_event(1))
            s = numpyro.deterministic("s", alpha * (LKj @ u))
        numpyro.sample("d", dist.Normal(Gj @ s, noise * sigma).to_event(1), obs=dj)

    mc = MCMC(NUTS(model), num_warmup=warmup, num_samples=samples, num_chains=4, chain_method="parallel",
              progress_bar=False)
    mc.run(jax.random.PRNGKey(seed), extra_fields=("diverging", "num_steps"))
    smp = mc.get_samples(group_by_chain=True)
    ex = mc.get_extra_fields(group_by_chain=True)
    a = np.asarray(smp["alpha"])
    return dict(alpha=a, noise=np.asarray(smp["noise_scale"]), s=np.asarray(smp["s"]).reshape(-1, N),
                div=int(np.sum(ex["diverging"])), div_mask=np.asarray(ex["diverging"]), steps=float(np.mean(ex["num_steps"])),
                rhat=D.rhat(a), ess=D.ess(a))


hier = {}
for sig_case, d_case in [(sigma, d), (300.0, G @ s_true + 300.0 * np.random.default_rng(42).standard_normal(M))]:
    hier[sig_case] = dict(exact=exact_hierarchical(d_case, sig_case))
    for centered in (True, False):
        for seed in (0, 1):
            hier[sig_case][(centered, seed)] = run_nuts(d_case, sig_case, centered, seed)

rows = []
for sig_case, h in hier.items():
    e = h["exact"]
    rows.append(dict(noise=f"{sig_case:g} mm", version="EXACT (grid)", seed="", divergences="", steps_per_draw="",
                     alpha=f"{e['alpha_mean']:.3f} ± {e['alpha_sd']:.3f}", rhat_alpha="", ess_alpha=""))
    for centered in (True, False):
        for seed in (0, 1):
            r = h[(centered, seed)]
            rows.append(dict(noise=f"{sig_case:g} mm", version="centered" if centered else "non-centered", seed=seed,
                             divergences=r["div"], steps_per_draw=f"{r['steps']:.0f}",
                             alpha=f"{r['alpha'].mean():.3f} ± {r['alpha'].std():.3f}",
                             rhat_alpha=f"{r['rhat']:.3f}", ess_alpha=f"{r['ess']:.0f}"))
print("NUTS: 4 chains x 2000 draws after 2000 warm-up; divergences counted over all 8000 draws")
print(pd.DataFrame(rows).to_string(index=False))

# %%
fig = plt.figure(figsize=(15, 4.6))
gsp = fig.add_gridspec(2, 3, height_ratios=[4, 0.75], width_ratios=[1, 1, 0.9], hspace=0.06, wspace=0.28)
for col, sig_case, t_ in [(0, sigma, "(a) Our data (σ = 3 mm): data are strong"),
                          (1, 300.0, "(b) Same stations, σ = 300 mm: data are weak")]:
    ax = fig.add_subplot(gsp[0, col])
    axr = fig.add_subplot(gsp[1, col], sharex=ax)
    h = hier[sig_case]
    bins = np.linspace(0, 6, 61)
    dv_c = sum(h[(True, s)]["div"] for s in (0, 1)); dv_n = sum(h[(False, s)]["div"] for s in (0, 1))
    ax.hist(np.concatenate([h[(True, s)]["alpha"].ravel() for s in (0, 1)]), bins=bins, density=True,
            color=C_NUTS, alpha=0.35, label=f"NUTS centered (2 seeds, {dv_c} divergences)")
    ax.hist(np.concatenate([h[(False, s)]["alpha"].ravel() for s in (0, 1)]), bins=bins, density=True,
            histtype="step", color=C_NUTS, lw=1.8, label=f"NUTS non-centered (2 seeds, {dv_n} divergences)")
    ax.plot(h["exact"]["a"], h["exact"]["dens_a"], color=C_EX, lw=1.6, label="exact")
    ax.set_yscale("log"); ax.set_ylim(2e-3, 4); ax.set_xlim(0, 6)
    ax.set_ylabel("posterior density (log scale)")
    ax.set_title(t_, fontsize=9.5)
    ax.legend(fontsize=7.5, loc="upper right")
    ax.tick_params(labelbottom=False)
    for yk, centered in [(1, True), (0, False)]:                   # where the divergent transitions happened
        a_div = np.concatenate([h[(centered, s)]["alpha"][h[(centered, s)]["div_mask"]] for s in (0, 1)])
        axr.plot(a_div, np.full(a_div.size, yk), "|", color=S.BAD, ms=7, alpha=0.5)
    axr.set_ylim(-0.7, 1.7); axr.set_yticks([0, 1]); axr.set_yticklabels(["non-centered", "centered"], fontsize=7.5)
    axr.set_xlabel("prior amplitude α (m);  red ticks = α at divergent draws")
ax3 = fig.add_subplot(gsp[:, 2])
e = hier[sigma]["exact"]
m_fix, C_fix, _ = posterior(K5)
profile(ax3, e["slip_mean"], e["slip_sd"], color=C_EX, title="(c) Slip: sampled vs fixed α", label="hierarchical (exact)")
ax3.plot(m_fix, zc, color=S.INK2, lw=1.2, label="fixed α = 1 m, noise × 1 (± 2σ thin)")
ax3.plot(m_fix - 2 * np.sqrt(np.diag(C_fix)), zc, color=S.INK2, lw=0.7)
ax3.plot(m_fix + 2 * np.sqrt(np.diag(C_fix)), zc, color=S.INK2, lw=0.7)
ax3.set_xlim(-1.5, 3.5)
ax3.legend(fontsize=7.2, loc="lower right")
bk.savefig(fig, "2-33_hierarchical")
plt.show()
for sig_case, h in hier.items():
    da = h["exact"]["a"][1] - h["exact"]["a"][0]
    p2 = np.sum(h["exact"]["dens_a"][h["exact"]["a"] > 2]) * da
    line = f"sigma {sig_case:g} mm: P(alpha > 2 m) exact {p2:.3f}"
    for centered in (True, False):
        al = np.concatenate([h[(centered, s)]["alpha"].ravel() for s in (0, 1)])
        ad = np.concatenate([h[(centered, s)]["alpha"][h[(centered, s)]["div_mask"]] for s in (0, 1)])
        line += f" | {'centered' if centered else 'non-centered'}: P(alpha > 2) {np.mean(al > 2):.3f}, median alpha all draws {np.median(al):.2f}, at divergences {np.median(ad) if ad.size else float('nan'):.2f}"
    print(line)
for centered in (True, False):
    for seed in (0, 1):
        r = hier[sigma][(centered, seed)]
        print("%-13s seed %d: max |slip mean - exact| = %.3f m, slip sd / exact sd from %.2f to %.2f"
              % ("centered" if centered else "non-centered", seed, np.abs(r["s"].mean(0) - e["slip_mean"]).max(),
                 (r["s"].std(0) / e["slip_sd"]).min(), (r["s"].std(0) / e["slip_sd"]).max()))
print("hierarchical vs fixed hyperparameters: max |mean difference| = %.3f m, sd ratio from %.2f to %.2f"
      % (np.abs(e["slip_mean"] - m_fix).max(), (e["slip_sd"] / np.sqrt(np.diag(C_fix))).min(),
         (e["slip_sd"] / np.sqrt(np.diag(C_fix))).max()))
results["hierarchical α + noise, exact (grid)"] = stats(e["slip_mean"], e["slip_sd"])
for centered in (True, False):
    r = hier[sigma][(centered, 0)]
    results[f"hierarchical, NUTS {'centered' if centered else 'non-centered'} (seed 0)"] = stats(r["s"].mean(0), r["s"].std(0))

# %% [markdown]
# **Read the table and panel (a) together.** With our informative data, compare each NUTS version with the exact
# row:
# * The **non-centered** runs report excellent R-hat and large ESS for $\alpha$, yet both seeds give an $\alpha$
#   posterior that is too narrow compared with the exact one, and the same in both seeds, so it is not noise.
#   Hundreds of **divergences** and trajectories near the 1023-step limit are the warning. On the log axis the
#   missing tail is visible, and the printed $P(\alpha > 2\,\text{m})$ quantifies it. The divergent draws
#   (red ticks) lean towards larger $\alpha$ than the posterior as a whole (medians printed above). There, the
#   slip combinations the data pin down force $\mathbf u$ to shrink exactly as $\alpha$ grows, a narrow curved
#   ridge in $(\alpha, \mathbf u)$ that the sampler cannot follow, so it under-visits that tail.
# * The **centered** runs have no divergences and match the exact answer.
#
# Panel (b) repeats the experiment with 100 times noisier data, so the slip is mostly prior-controlled. Now the
# centered model is the struggling one: its divergences lean towards *small* $\alpha$, the neck of the funnel
# (notebook 2.02, section 6), its R-hat for $\alpha$ is above 1.01, and it needs more leapfrog steps per draw for
# a fraction of the ESS (table). The non-centered model is cheaper, mixes well and matches the exact curve; its
# divergences lean towards large $\alpha$, where with weak data little probability lives. Each form has a
# blind end, and which one hurts depends on where the probability is.
#
# **The rule** (Papaspiliopoulos, Roberts & Sköld 2007; Betancourt & Girolami 2015): non-center what the *prior*
# controls, center what the *data* control. A slip model has both kinds at once (shallow combinations are
# data-controlled, deep ones prior-controlled), so neither form is right everywhere. In another problem, with more
# directions the data cannot see, the balance can tip the other way from panel (a): the centered form falls into the
# funnel of panel (b) and the non-centered form works. That is the same rule with a different balance of resolved and
# unresolved directions. Practical consequence: divergences
# are never cosmetic; when they appear, try the other parameterisation and compare against a reference.
#
# Panel (c): sampling $\alpha$ and the noise scale barely moves the slip mean here, and widens the uncertainty
# compared with fixing them (numbers printed above). That widening is the honest price of not
# knowing the hyperparameters.

# %% [markdown]
# ## 6. Positivity: slip cannot be negative
#
# Same prior as the Laplacian-smoothing case in section 3, with one change: slip $\geq 0$
# (HalfNormal(1) per patch times the same smoothing penalty). The posterior is no longer Gaussian, so there is no
# closed form; NUTS samples it (NumPyro samples $\log s$ internally, so the boundary at zero is never hit).

# %%
Dj = jnp.asarray(Dlap)


def model_positive():
    s = numpyro.sample("s", dist.HalfNormal(1.0).expand([N]).to_event(1))
    numpyro.factor("smoothing", -0.5 * jnp.sum((Dj @ s) ** 2) / a_smooth ** 2)
    numpyro.sample("d", dist.Normal(Gj @ s, sigma).to_event(1), obs=jnp.asarray(d))


mc = MCMC(NUTS(model_positive), num_warmup=2000, num_samples=2000, num_chains=4, chain_method="parallel", progress_bar=False)
mc.run(jax.random.PRNGKey(3), extra_fields=("diverging", "num_steps"))
s_pos = np.asarray(mc.get_samples(group_by_chain=True)["s"])
ex = mc.get_extra_fields(group_by_chain=True)
rh = [D.rhat(s_pos[:, :, i]) for i in range(N)]
es = [D.ess(s_pos[:, :, i]) for i in range(N)]
print(f"positivity NUTS: divergences {int(np.sum(ex['diverging']))}, mean leapfrog steps {np.mean(ex['num_steps']):.0f}, "
      f"max R-hat {max(rh):.3f}, min ESS {min(es):.0f}")
s_pos = s_pos.reshape(-1, N)
m_lap, C_lap, _ = posterior(priors["Laplacian smoothing (a = 0.1 m)"])
sd_lap = np.sqrt(np.diag(C_lap))
lo, hi = np.percentile(s_pos, [2.5, 97.5], axis=0)
results["Laplacian + positivity, NUTS"] = stats(s_pos.mean(0), s_pos.std(0), lo, hi)

fig, axs = plt.subplots(1, 4, figsize=(16, 3.9), gridspec_kw=dict(width_ratios=[1.2, 1, 1, 1]))
profile(axs[0], m_lap, sd_lap, color=C_EX, title="(a) Without vs with positivity", label="Gaussian (no positivity)")
axs[0].fill_betweenx(zc, lo, hi, color=C_NUTS, alpha=0.18, lw=0, label="slip ≥ 0: 95 % interval")
axs[0].plot(s_pos.mean(0), zc, color=C_NUTS, lw=2, label="slip ≥ 0 (NUTS), mean")
axs[0].axvline(0, color=S.AXIS, lw=0.8, zorder=0)
axs[0].set_xlim(-1.5, 3.0); axs[0].legend(fontsize=7.5, loc="lower right")
for ax, j in zip(axs[1:], [0, 11, 28]):
    ax.hist(s_pos[:, j], bins=60, density=True, color=C_NUTS, alpha=0.4, label="slip ≥ 0 (NUTS)")
    xg = np.linspace(min(m_lap[j] - 4 * sd_lap[j], -0.05), max(m_lap[j] + 4 * sd_lap[j], s_pos[:, j].max()), 400)
    ax.plot(xg, np.exp(-0.5 * ((xg - m_lap[j]) / sd_lap[j]) ** 2) / (sd_lap[j] * np.sqrt(2 * np.pi)), color=C_EX, lw=1.5,
            label="Gaussian (no positivity)")
    ax.axvline(s_true[j], color="#b5b4ad", lw=3, label="true slip")
    ax.axvline(0, color=S.AXIS, lw=0.8)
    ax.set_title(f"({'bcd'[[0, 11, 28].index(j)]}) patch at {zc[j]:.1f} km", fontsize=9.5)
    ax.set_xlabel("slip (m)")
axs[1].set_ylabel("posterior density"); axs[3].legend(fontsize=7.5)
fig.tight_layout()
bk.savefig(fig, "2-33_positivity")
plt.show()
miss = np.where((s_true < lo) | (s_true > hi))[0]
print("slip >= 0: true slip outside the 95 % interval at depths (km):", zc[miss])
for j in miss:
    print(f"   {zc[j]:4.1f} km: true {s_true[j]:.4f} m, 95 % interval [{lo[j]:.3f}, {hi[j]:.3f}] m  ->  true slip {'above' if s_true[j] > hi[j] else 'below'} it")
for j in [0, 11, 28]:
    print(f"patch {zc[j]:4.1f} km: true {s_true[j]:.3f} m | Gaussian mean {m_lap[j]:6.3f}, P(slip<0) = {np.mean(np.random.default_rng(1).normal(m_lap[j], sd_lap[j], 100000) < 0):.2f}"
          f" | positive: mean {s_pos[:, j].mean():.3f}, median {np.median(s_pos[:, j]):.3f}")

# %% [markdown]
# **How to read it.** Where the data are strong (the 11.5 km patch, panel c) positivity changes little: the
# Gaussian already lies far from zero. The small shift it does show is possible only because the smoothing
# penalty couples every patch to its neighbours, so a constraint at depth can be felt further up. Where the data are weak (the 28.5 km patch, panel d) the Gaussian
# posterior puts a large share of its probability on negative slip (printed above). The positive posterior piles
# up against zero with a long tail to the right: skewed, so its mean and median differ and "mean ± 2σ" no
# longer describes it. That is why panel (a) shows a 95 % interval for the positive case. A constraint that
# matters only where the data are blind is exactly the kind of choice that should be stated in a paper.
#
# One honest caveat about the summary table below: the positive model's coverage has two kinds of miss (printed
# above). At the slip peak the stiff smoothing prior flattens the peak, so the true slip lies *above* the interval;
# its Gaussian version missed at the peak too (section 3). At the bottom of the fault the true slip is essentially
# zero, and the 95 % interval of a strictly positive quantity starts slightly above zero, so the true slip sits just
# *below* it, on the boundary rather than far outside.

# %% [markdown]
# ## 7. EnKI on the same problem: ensemble size and the subspace property
#
# Now the Gaussian-process prior again (Matérn, $L = 5$ km, fixed amplitude), solved with the tempered EnKI of
# notebook 2.24 (`SM.enki`). Its initial ensemble is $J$ draws from the GP prior. Because the
# problem is linear-Gaussian, a large ensemble should reproduce the exact posterior.

# %%
prob = P.linear(G, d, sigma, np.zeros(N), K5)
m_ex, C_ex = prob.exact
sd_ex = np.sqrt(np.diag(C_ex))
enk = {}
for J in (20, 100, 1000):
    for seed in (1, 2):
        e_ = SM.enki(prob, J=J, rng=np.random.default_rng(seed))
        X0, XT = e_["history"][0]["X"], e_["samples"]
        A0 = (X0 - X0.mean(0)).T                               # initial anomalies, (N, J)

        def outside(v):
            c, *_ = np.linalg.lstsq(A0, v, rcond=None)
            return np.linalg.norm(v - A0 @ c) / np.linalg.norm(v)
        enk[(J, seed)] = dict(mean=XT.mean(0), sd=XT.std(0, ddof=1), iters=len(e_["history"]) - 1,
                              theta=e_["theta"], out_enki=outside(XT.mean(0) - X0.mean(0)),
                              out_exact=outside(m_ex - X0.mean(0)))
        results[f"GP prior, EnKI J={J} (seed {seed})"] = stats(XT.mean(0), XT.std(0, ddof=1))
print(f"{'J':>5s} {'seed':>4s} {'iters':>5s} {'rms(mean - exact)':>18s} {'median sd/exact sd':>19s} {'EnKI mean outside span':>23s} {'exact mean outside span':>24s}")
for (J, seed), r in enk.items():
    print(f"{J:5d} {seed:4d} {r['iters']:5d} {np.sqrt(np.mean((r['mean'] - m_ex) ** 2)):18.3f} {np.median(r['sd'] / sd_ex):19.2f}"
          f" {r['out_enki']:23.1e} {r['out_exact']:24.2f}")
print("theta history, J = 1000, seed 1:", np.round(enk[(1000, 1)]["theta"], 3))

# %%
fig, axs = plt.subplots(1, 4, figsize=(16, 4.0), sharey=True)
for ax, J in zip(axs[:3], (20, 100, 1000)):
    ax.fill_betweenx(zc, m_ex - 2 * sd_ex, m_ex + 2 * sd_ex, color=C_EX, alpha=0.12, lw=0, label="exact ± 2σ")
    ax.plot(s_true, zc, **TRUE_KW)
    ax.plot(m_ex, zc, color=C_EX, lw=1.6, label="exact mean")
    for seed, lw in ((1, 2.0), (2, 1.0)):
        ax.plot(enk[(J, seed)]["mean"], zc, color=C_ENKI, lw=lw, label=f"EnKI mean, seed {seed}")
    ax.set_ylim(30, 0); ax.set_xlim(-1.5, 3.5); ax.set_xlabel("slip (m)")
    ax.set_title(f"J = {J} members ({'fewer' if J <= N else 'more'} than {N} unknowns)", fontsize=9.5)
axs[0].set_ylabel("depth (km)"); axs[0].legend(fontsize=7.5, loc="lower right")
ax = axs[3]
ax.plot(sd_ex, zc, color=C_EX, lw=2, label="exact")
for J, lw in ((20, 0.8), (100, 1.3), (1000, 2.0)):
    ax.plot(enk[(J, 1)]["sd"], zc, color=C_ENKI, lw=lw, label=f"EnKI J = {J}")
ax.set_xlabel("posterior sd (m)"); ax.set_title("Uncertainty, seed 1", fontsize=9.5); ax.legend(fontsize=7.5)
fig.tight_layout()
bk.savefig(fig, "2-33_enki_ensemble_size")
plt.show()

# %% [markdown]
# **The images to keep.**
# * With **J = 20 < 30 unknowns**, the two seeds give visibly different answers (table: rms difference from the
#   exact mean). The last two columns show why. The EnKI mean change lies *exactly* in the span of the 20 initial
#   draws (the "outside span" fraction is at round-off level), while the exact posterior mean has a part outside
#   that span. EnKI cannot reach it, however many iterations it runs. This is the subspace property
#   (Tong & Morzfeld 2023); a new seed is a new subspace, hence a new answer.
# * With **J = 100 and 1000** the initial draws span all 30 dimensions, the subspace limit disappears, and what
#   remains is ordinary sampling noise, which shrinks with $J$.
# * The seed test is a cheap check: if two seeds differ by more than the quoted uncertainty, the ensemble is too small
#   for the problem. Large slip inversions often run fewer members than unknowns, which puts them on the J < unknowns
#   side of this figure; localisation (2.26) is the usual remedy.

# %% [markdown]
# ## 8. Resolution: how much did the data teach us at each depth?
#
# Two pictures. The ratio of posterior to prior standard deviation (1 = learned nothing, 0 = learned
# everything), and rows of the **resolution matrix** $\mathbf R = \mathbf K_{\text{gain}}\mathbf G$. The posterior
# mean is $\hat{\mathbf s} = \mathbf R\,\mathbf s_{\text{true}} + \text{(noise and prior terms)}$, so row $i$ shows
# which true patches are blended into the estimate at patch $i$.

# %%
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.0))
for (name, Pm), shade in zip(priors.items(), [S.MUTED, S.INK2, S.INK]):
    _, C, _ = posterior(Pm)
    axs[0].plot(np.sqrt(np.diag(C)) / np.sqrt(np.diag(Pm)), zc, color=shade, lw=1.8, label=name)
axs[0].set_ylim(30, 0); axs[0].set_xlim(0, 1.05)
axs[0].set_xlabel("posterior sd / prior sd"); axs[0].set_ylabel("depth (km)")
axs[0].set_title("(a) What the data taught us"); axs[0].legend(fontsize=7.5, loc="lower left")
Kgain = K5 @ G.T @ np.linalg.inv(G @ K5 @ G.T + sigma ** 2 * np.eye(M))
R = Kgain @ G
for i, shade in zip([2, 9, 17, 26], [S.INK, S.INK2, S.MUTED, "#b5b4ad"]):
    axs[1].plot(zc, R[i], color=shade, lw=1.8, label=f"estimate at {zc[i]:.1f} km (sum {R[i].sum():.2f})")
    axs[1].axvline(zc[i], color=shade, lw=0.6)
axs[1].set_xlabel("depth of the true patch (km)"); axs[1].set_ylabel("weight")
axs[1].set_title("(b) Rows of the resolution matrix (GP prior, L = 5 km)")
axs[1].legend(fontsize=7.5)
fig.tight_layout()
bk.savefig(fig, "2-33_resolution")
plt.show()
print("trace of the resolution matrix (effective number of resolved slip parameters): %.1f of %d" % (np.trace(R), N))

# %% [markdown]
# The shallow row is a narrow spike near its own depth (sharp resolution). Deeper rows are broad and centred away
# from their own depth: the estimate at 26.5 km is an average over many kilometres, weighted towards the bottom of
# the fault. Their sums stay close to one (legend), so a uniform slip would be recovered everywhere; it is the
# *detail* at depth that is blurred away. The trace of $\mathbf R$ counts how many slip parameters the data
# effectively determine (printed above). Compare it with the singular-value count in section 1: two views of the
# same fact.

# %% [markdown]
# ## 9. Summary: every choice on one table
#
# Columns: rms error of the posterior mean against the true slip (available only because this is a synthetic
# test), $\chi^2$ of the mean prediction ($N = 24$ data), mean posterior sd at 0-10 km and 20-30 km, and coverage
# (fraction of patches whose true slip lies inside the 95 % interval).

# %%
tab = pd.DataFrame(results).T
tab.columns = ["rms error (m)", "chi^2", "sd 0-10 km (m)", "sd 20-30 km (m)", "coverage"]
pd.set_option("display.width", 160)
print(tab.round(3).to_string())

# %% [markdown]
# Things this table shows at a glance (read the numbers above):
# * The flat prior is the only row whose error is far larger than the slip itself.
# * Prior choices that all fit the data ($\chi^2$ near $N$) still differ in deep uncertainty by factors: the
#   deep sd column is decided by the prior, not by the data.
# * The two NUTS rows and the exact hierarchical row agree on the slip, even though section 5 showed that the
#   non-centered run misrepresents $\alpha$: getting the main parameters right does not certify the
#   hyperparameters.
# * EnKI with fewer members than unknowns gives seed-dependent answers; with more members it approaches the exact
#   posterior as $J$ grows.
#
# **In practice.**
# * The prior decides everything below the resolved depth. Show the prior draws next to the posterior (section
#   3) so a reader can see which features are inherited from the prior.
# * When you sample $\alpha$ (or $\sigma_k$), divergences are the message. Check the other parameterisation, and
#   where a smaller version of the problem has an exact or reference answer (as here), grade NUTS against it. In
#   this test good R-hat and ESS coexisted with a biased hyperparameter.
# * Positivity changes the answer only where the data are blind, and there it makes the posterior skewed: quote
#   intervals, not mean ± 2σ.
# * For EnKI, members versus unknowns is the first number to check. Below the dimension, run a second seed before
#   trusting the error bars, and consider localisation.
#
# This closes Topic 2. **2.15 · Map** has the overview of all the samplers on all the test problems.
#
# Next: Topic 3, forward surrogate models (under development): **3.01 · Surrogates 101**.
