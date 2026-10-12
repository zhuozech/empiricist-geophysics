# %% [markdown]
# # 2.30 · Hierarchical Bayes: who sets the weights?
#
# **Before this:** 2.08 (Bayesian linear inversion), 2.09 (smoothing priors), 2.13 (C_d and C_p), 2.18 (NUTS),
# 2.29 (ABIC and the marginal likelihood)  
# **You will be able to:** write a slip inversion as a hierarchy with data and smoothing weights; say when ABIC's single
# best setting and the fully Bayesian answer agree and when the plug-in is overconfident; see why positivity breaks the
# ABIC formula; read an estimated data weight as a misfit level; test how much a hyperprior matters  
# **Optional sections:** 3 (positivity and orthant probabilities)
#
# Every slip inversion has two knobs that the physics does not set: how much to trust the data (the data weight) and
# how strongly to smooth the slip (the smoothing weight). Setting them by hand until the result "looks reasonable" is
# subjective. Two data-driven alternatives dominate the geodetic literature: ABIC (Akaike 1980; Yabuki & Matsu'ura
# 1992), which picks the single best setting, and the fully Bayesian (hierarchical) approach of Fukuda & Johnson
# (2008, 2010), which keeps every setting weighted by its probability. This notebook compares them on small problems
# where the exact answer is known, reproduces the two criticisms Fukuda & Johnson made of ABIC, and then shows what the
# estimated weights actually measure.
#
# **Words this notebook defines:** hyperparameter, hyperprior, hierarchical model, marginal likelihood, ABIC,
# empirical Bayes, hierarchical (fully) Bayes, plug-in posterior, law of total variance, positivity, orthant
# probability, data weight σ_k, mixed linear–nonlinear inversion, improper prior, objective (in the sense used in this
# literature).
#
# Quotes from the papers are verbatim, with journal page numbers.

# %% [markdown]
# **Problems in this notebook**
#
# | problem | unknowns | data | forward model | linear? |
# |---|---|---|---|---|
# | antiplane fault in depth (sections 1, 2, 6) | 24 patches + σ², α² | 5 to 120 stations on one side, σ = 2 mm (0.05 mm in one test) | screw-dislocation Green's functions, $\mathbf d = \mathbf G\mathbf s$ | linear in slip |
# | along-strike creep profile (sections 2, 6) | 120 patches + σ², α² | 6 to 120 stations | each station averages the nearby slip (Gaussian kernel) | linear in slip |
# | small antiplane fault with positive slip (section 3) | 8 patches ≥ 0 + σ², α² | 12 stations, σ = 3 mm | screw dislocation | linear, bounded |
# | two parallel strike-slip faults, two data sets (sections 4, 5) | $D_1, D_2, \dot s_1, \dot s_2, \sigma_1, \sigma_2$ (+ InSAR offset and ramp) | 25 GPS-like at 1.5 mm/yr + 300 InSAR-like at 4.5 mm/yr | $v(x) = \sum_k \frac{\dot s_k}{\pi}\arctan\frac{x - x_k}{D_k}$ | nonlinear in $D_k$, linear in $\dot s_k$ |
#
# The fault toys use stations on one side of the fault only. With stations in mirror pairs ($\pm x$) an antiplane fault
# gives $u(-x) = -u(x)$, so each pair measures the same slip combination twice (notebook 2.33).

# %%
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path.cwd().parent))
import numpyro
numpyro.set_host_device_count(4)                 # 4 NUTS chains in parallel (before any JAX computation)
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from numpyro.diagnostics import split_gelman_rubin
import jax, jax.numpy as jnp
jax.config.update("jax_enable_x64", True)
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.linalg import eigh
from scipy.stats import norm, multivariate_normal as mvn, truncnorm
from scipy.optimize import nnls, minimize_scalar
from scipy.ndimage import gaussian_filter
import bayeskit as bk
from bayeskit import problems as P, samplers as SM, diagnostics as D, plotting as BP, style as S

C_ABIC, C_FULL, C_CONV, C_TRUE = S.BLUE, S.VIOLET, S.ORANGE, S.INK      # ABIC / fully Bayesian / conventional / truth
LIGHT = {C_ABIC: "#b9d3f2", C_FULL: "#c4bde6", C_CONV: "#f7c3ad"}


def second_difference(M):
    """Second difference with zero-slip rows at both ends: full rank, as Fukuda & Johnson require for D^T D."""
    Dm = np.zeros((M, M))
    for i in range(M):
        Dm[i, i] = -2.0
        if i > 0:
            Dm[i, i - 1] = 1.0
        if i < M - 1:
            Dm[i, i + 1] = 1.0
    return Dm


def screw_G(xs, z_edges):
    """Surface displacement (mm) per metre of slip on each depth patch of a vertical strike-slip fault."""
    return 1000.0 * np.array([[(np.arctan(z_edges[j + 1] / x) - np.arctan(z_edges[j] / x)) / np.pi
                               for j in range(len(z_edges) - 1)] for x in xs])


# %% [markdown]
# ## 1. The hierarchy: knobs above the slip
#
# **In plain words.** A slip inversion has three layers. At the bottom are the data. Above them is the slip, which
# predicts the data through the Green's functions. Above the slip sit the knobs: σ² says how far the data may stray
# from the prediction, and α² says how rough the slip may be. In a hierarchical model the knobs are unknowns too. They
# get their own priors (hyperpriors), and the data are allowed to tell us about them.
#
# **Precisely** (Fukuda & Johnson 2008, eqs 7 and 9, in the notation of these notebooks: slip $\mathbf s$, difference
# operator $\mathbf D$, which they call $\mathbf L$):
#
# $$\mathbf d \mid \mathbf s, \sigma^2 \sim N(\mathbf G\mathbf s,\; \sigma^2\boldsymbol\Sigma_d), \qquad
# \mathbf D\mathbf s \mid \alpha^2 \sim N(\mathbf 0,\; \alpha^2\mathbf I), \qquad (\sigma^2, \alpha^2) \sim p(\sigma^2, \alpha^2).$$
#
# Here $\boldsymbol\Sigma_d$ is the shape of the data covariance (reminder: $\mathbf C_d$, 2.07 and 2.13) and σ² scales it. The
# smoothing prior is $p(\mathbf s\mid\alpha^2) = (2\pi\alpha^2)^{-N_m/2}\,|\mathbf D^\top\mathbf D|^{1/2}\exp(-\lVert\mathbf D\mathbf s\rVert^2/2\alpha^2)$,
# where the factor in front is its normalising constant (it makes the density integrate to one; 2.29) and
# $|\cdot|$ is the determinant (2.06): a large α² allows rough slip, a small α² forces smooth slip. The MAP (most
# probable slip, 2.08) for fixed knobs minimises
# $\lVert\mathbf d - \mathbf G\mathbf s\rVert^2/\sigma^2 + \lVert\mathbf D\mathbf s\rVert^2/\alpha^2$, so with σ known the
# smoothing weight λ of notebook 2.29 is $1/\alpha^2$.
#
# **"Objective"** in this literature means that no number was picked by hand after looking at the result. Fukuda &
# Johnson's opening sentence: "In inversion of geodetic data for spatial distribution of fault slip under smoothing and
# positivity constraints, subjective or incorrect techniques are often employed to select a smoothing parameter that
# determines the relative weight placed on fitting the data versus smoothing the slip distribution" (2008, p. 1128).
# It does not mean free of assumptions: the form of the smoothing operator, the noise model and the hyperpriors are
# still chosen by the analyst, and sections 5 and 6 show they matter.

# %%
M = 24
ze = np.linspace(0.0, 24.0, M + 1)
zc = 0.5 * (ze[1:] + ze[:-1])
Dop = second_difference(M)
DD = Dop.T @ Dop
a2_true = 1.0 / np.mean(np.diag(np.linalg.inv(DD)))       # chosen so the prior slip sd is about 1 m on average
sig_true = 2.0                                              # mm
print("alpha^2 used to generate the slip (prior slip sd about 1 m): %.4g ; data noise sigma = %.1f mm" % (a2_true, sig_true))

rng = np.random.default_rng(1)
xs_demo = np.geomspace(0.5, 60, 40)
G_demo = screw_G(xs_demo, ze)
fig = plt.figure(figsize=(14.5, 4.4))
ax0 = fig.add_axes([0.01, 0.06, 0.27, 0.88]); ax0.axis("off"); ax0.set_xlim(0, 1); ax0.set_ylim(0, 1)
boxes = [(0.5, 0.76, "hyperparameters\nσ² (data weight), α² (smoothing)"), (0.5, 0.45, "slip s\n(24 depth patches)"),
         (0.5, 0.14, "data d\n(surface displacements)")]
for (x, y, t) in boxes:
    ax0.add_patch(FancyBboxPatch((x - 0.36, y - 0.08), 0.72, 0.16, boxstyle="round,pad=0.01,rounding_size=0.03",
                                 fc=S.SURFACE, ec=S.AXIS, lw=1.3))
    ax0.text(x, y, t, ha="center", va="center", fontsize=9.5, color=S.INK)
for (y0, y1, lab) in [(0.965, 0.85, "hyperprior p(σ², α²)"), (0.67, 0.54, "prior p(s | α²)"), (0.36, 0.23, "likelihood p(d | s, σ²)")]:
    ax0.annotate("", xy=(0.5, y1), xytext=(0.5, y0), arrowprops=dict(arrowstyle="-|>", color=S.MUTED, lw=1.2))
    ax0.text(0.54, (y0 + y1) / 2, lab, fontsize=8.5, color=S.INK2, va="center")
ax0.set_title("The hierarchy", fontsize=10.5, fontweight="bold", loc="left")
ax1 = fig.add_axes([0.33, 0.13, 0.3, 0.75]); ax2 = fig.add_axes([0.69, 0.13, 0.3, 0.75])
s_one = np.sqrt(a2_true) * np.linalg.cholesky(np.linalg.inv(DD)) @ rng.standard_normal(M)   # one slip from the prior
d_demo = G_demo @ s_one + sig_true * rng.standard_normal(xs_demo.size)
for fac, col in [(0.01, S.CHAINS[0]), (1.0, S.CHAINS[1]), (100.0, S.CHAINS[2])]:
    s_map = np.linalg.solve(G_demo.T @ G_demo / sig_true ** 2 + DD / (fac * a2_true), G_demo.T @ d_demo / sig_true ** 2)
    ax1.plot(s_map, zc, color=col, lw=1.6, label=f"best fit, α² = {fac:g} × reference")
ax1.plot(s_one, zc, color=S.INK, lw=2.0, label="true slip")
ax1.invert_yaxis(); ax1.set_xlabel("slip (m)"); ax1.set_ylabel("depth (km)")
ax1.set_title("α² sets how strongly the slip is smoothed", fontsize=10.5); ax1.legend(fontsize=8)
for k, (sg, col) in enumerate([(8.0, S.CHAINS[2]), (2.0, S.CHAINS[1]), (0.5, S.CHAINS[0])]):
    ax2.plot(xs_demo, sg * rng.standard_normal(xs_demo.size), "o", ms=3.8, color=col, label=f"σ = {sg:g} mm")
ax2.axhline(0, color=S.INK, lw=1.2, label="prediction of the true slip")
ax2.set_xscale("log"); ax2.set_xlabel("distance from the fault (km)"); ax2.set_ylabel("data − prediction (mm)")
ax2.set_title("σ² sets the scatter of the data around the prediction", fontsize=10.5)
ax2.legend(fontsize=8, ncol=2, loc="upper center")
ax2.set_ylim(-28, 34)
bk.savefig(fig, "2-30_hierarchy")
plt.show()

# %% [markdown]
# **How to read it.** Left: the model as three layers, each conditioned on the one above. Middle: the best-fitting slip
# (MAP) for three values of α², all computed from the same 40 data points with σ = 2 mm. With a small α² every bit of
# curvature is expensive, so the slip comes out smooth and too small; with a large α² the slip is free to bend and
# follows the noise. Right: the data minus the prediction of the true slip, for three noise levels; the displacements
# themselves are much larger than these deviations. Estimating the hyperparameters means asking which values of these
# two knobs make the observed data most plausible.
#
# Note that α² alone does not change the *shape* of slip drawn from the prior: a draw is $\sqrt{\alpha^2}$ times a fixed
# random curve, so α² sets its size and with it the size of its curvature. The smoothing appears when the prior is
# balanced against the data, through the ratio α²/σ².
#
# **Common misreadings.**
# * "Estimated weights remove all subjectivity." They remove the hand-picked numbers; the forms (smoothing operator,
#   noise model, hyperprior) remain choices.
# * "α² is a regularisation trick, not a probability." In the hierarchical model it is the variance of $\mathbf D\mathbf s$,
#   a statement about how rough slip is believed to be, and it can be estimated like any other unknown.

# %% [markdown]
# ## 2. ABIC (empirical Bayes) or fully Bayesian (hierarchical Bayes)
#
# **In plain words.** Both approaches start from the same quantity: for every setting of the knobs, how probable were
# the observed data, averaging over all slip models allowed by that setting? (That is the marginal likelihood; reminder:
# 2.12 and 2.29.) ABIC takes the single best setting and then treats it as exactly right: this "estimate the knobs as
# single numbers, then plug them in as if known" strategy is called **empirical Bayes**, and the resulting slip
# posterior is the **plug-in** posterior. The fully Bayesian approach keeps every setting, weighted by how probable it made the data,
# and averages the slip posteriors over them. When one setting clearly dominates, the two answers agree. When many
# settings are about equally good, ABIC's answer is too confident.
#
# **Precisely.** The marginal likelihood (evidence) of the hyperparameters integrates the slip out:
# $p(\mathbf d\mid\sigma^2,\alpha^2) = \int p(\mathbf d\mid\mathbf s,\sigma^2)\,p(\mathbf s\mid\alpha^2)\,d\mathbf s$, closed form for
# this linear-Gaussian model. $\mathrm{ABIC} = -2\ln p(\mathbf d\mid\hat\sigma^2,\hat\alpha^2) + 2N_h$ at its minimum (Akaike 1980), and the
# ABIC solution is the *plug-in* posterior $p(\mathbf s\mid\mathbf d,\hat\sigma^2,\hat\alpha^2)$. The fully Bayesian solution is
#
# $$p(\mathbf s\mid\mathbf d) = \iint p(\mathbf s\mid\mathbf d,\sigma^2,\alpha^2)\;p(\sigma^2,\alpha^2\mid\mathbf d)\;d\sigma^2\,d\alpha^2,
# \qquad p(\sigma^2,\alpha^2\mid\mathbf d)\propto p(\mathbf d\mid\sigma^2,\alpha^2)\,p(\sigma^2,\alpha^2).$$
#
# Fukuda & Johnson (2010) state when the two agree, "If p(m, σ, α|d) has a dominant peak at its maximum" and the
# conditional slip posterior changes slowly near that peak (p. 1449), and when they do not: "if p(m, σ, α|d) is a broad
# distribution with possibly multiple peaks, p(s|d, m̃, σ̃, α̃) may be significantly different from p(s|d)" (p. 1450).
# In their notation m are nonlinear parameters such as fault geometry (none here) and the tilde marks the ABIC values.
# Their summary: "the ABIC-based method and our method can be regarded as an empirical Bayes method and a hierarchical
# Bayes method, respectively" (p. 1450).
#
# Computing tool: diagonalise $\mathbf G^\top\mathbf G$ and $\mathbf D^\top\mathbf D$ together (generalised eigenvectors
# $\mathbf V$ with $\mathbf V^\top\mathbf D^\top\mathbf D\mathbf V = \mathbf I$, $\mathbf V^\top\mathbf G^\top\mathbf G\mathbf V = \mathrm{diag}(\mu_i)$). In words:
# a set of slip patterns that the prior penalises equally and that the data see independently, pattern $i$ with
# strength $\mu_i$ (an extension of the eigen-decomposition of 2.06). In these patterns every matrix is diagonal, so the
# formulas reduce to one number per pattern. Then, with
# $r = \alpha^2/\sigma^2$ and $\mathbf c = \mathbf V^\top\mathbf G^\top\mathbf d$,
# $\ln p(\mathbf d\mid\sigma^2,\alpha^2) = -\tfrac12\big[N_d\ln 2\pi\sigma^2 + \sum_i\ln(1 + r\mu_i) + \mathbf d^\top\mathbf d/\sigma^2 - (\alpha^2/\sigma^4)\sum_i c_i^2/(1 + r\mu_i)\big]$,
# and the conditional posterior has mean $r\,\mathbf V[\mathbf c/(1 + r\boldsymbol\mu)]$ and covariance
# $\alpha^2\mathbf V\,\mathrm{diag}(1/(1 + r\mu_i))\mathbf V^\top$. Every quantity below is evaluated on a whole grid of
# $(\sigma^2, \alpha^2)$ at once. The hyperprior here is log-uniform (uniform in $\ln\sigma^2$ and $\ln\alpha^2$ over the grid
# box); section 6 changes it.

# %%
def simdiag(G, DDm):
    mu, V = eigh(G.T @ G, DDm)
    return np.clip(mu, 0.0, None), V              # round-off can leave tiny negative eigenvalues where G is blind


def make_grid(ls2, la2):
    return np.meshgrid(np.exp(ls2), np.exp(la2), indexing="ij")


def log_marglik(d, G, mu, V, S2, A2):
    c = V.T @ G.T @ d
    q = 1.0 + (A2 / S2)[..., None] * mu
    ll = -0.5 * (d.size * np.log(2 * np.pi * S2) + np.log(q).sum(-1) + d @ d / S2 - (A2 / S2 ** 2) * (c ** 2 / q).sum(-1))
    return ll, c


def cond_moments(mu, V, c, s2, a2):
    """Conditional posterior means and marginal variances for arrays of (s2, a2)."""
    s2, a2 = np.atleast_1d(s2), np.atleast_1d(a2)
    r = (a2 / s2)[:, None]
    q = 1.0 + r * mu
    means = (r * (c / q)) @ V.T
    var = a2[:, None] * (1.0 / q) @ (V.T ** 2)
    return means, var


def analyse(d, G, mu, V, S2, A2, log_prior=0.0, s_true=None, wfloor=1e-7):
    ll, c = log_marglik(d, G, mu, V, S2, A2)
    i, j = np.unravel_index(np.argmax(ll), ll.shape)                      # ABIC = maximum marginal likelihood
    mp, vp = cond_moments(mu, V, c, S2[i, j], A2[i, j])
    lw = ll + log_prior
    w = np.exp(lw - lw.max()); w /= w.sum()
    keep = w > wfloor * w.max()
    ws = w[keep] / w[keep].sum()
    means, var = cond_moments(mu, V, c, S2[keep], A2[keep])
    mmix = ws @ means
    vmix = ws @ (var + means ** 2) - mmix ** 2
    out = dict(ll=ll, w=w, ij=(i, j), mp=mp[0], vp=vp[0], ws=ws, means=means, var=var, mmix=mmix, vmix=vmix,
               n_resolved=int(np.sum(A2[i, j] * mu / S2[i, j] > 1.0)))
    if s_true is not None:
        Fp = norm.cdf((s_true - out["mp"]) / np.sqrt(out["vp"]))
        Fm = (ws[:, None] * norm.cdf((s_true - means) / np.sqrt(var))).sum(0)
        out["cover_plug"] = float(np.mean((Fp > 0.05) & (Fp < 0.95)))
        out["cover_full"] = float(np.mean((Fm > 0.05) & (Fm < 0.95)))
    return out


def mixture_quantiles(ws, means, var, qs=(0.05, 0.95), n=600):
    """Quantiles of a Gaussian mixture, patch by patch, by evaluating its CDF on a fine grid."""
    lo = (means - 6 * np.sqrt(var)).min(0); hi = (means + 6 * np.sqrt(var)).max(0)
    out = np.empty((len(qs), means.shape[1]))
    for j in range(means.shape[1]):
        g = np.linspace(lo[j], hi[j], n)
        F = (ws[:, None] * norm.cdf((g[None, :] - means[:, j:j + 1]) / np.sqrt(var[:, j:j + 1]))).sum(0)
        out[:, j] = np.interp(qs, F, g)
    return out


ls2 = np.linspace(np.log(1e-4), np.log(400.0), 121)
la2 = np.linspace(np.log(a2_true) - 7, np.log(a2_true) + 7, 111)
S2, A2 = make_grid(ls2, la2)
Lc_prior = np.linalg.cholesky(a2_true * np.linalg.inv(DD))
cases = {"8 stations": np.geomspace(1, 40, 8), "40 stations": np.geomspace(0.5, 60, 40)}
ops = {name: (screw_G(xs, ze),) + simdiag(screw_G(xs, ze), DD) for name, xs in cases.items()}


def realization(k):
    """True slip drawn from the smoothing prior, and 40 noise values (the 8-station set uses the first 8)."""
    return Lc_prior @ np.random.default_rng(100 + k).standard_normal(M), sig_true * np.random.default_rng(200 + k).standard_normal(40)


cov30 = np.zeros((30, 4))                        # 30 realizations; the one drawn below is the most typical of them
for k in range(30):
    st, e = realization(k)
    for c_, (G, mu, V) in enumerate(ops.values()):
        o = analyse(G @ st + e[:G.shape[0]], G, mu, V, S2, A2, s_true=st)
        cov30[k, 2 * c_:2 * c_ + 2] = o["cover_plug"], o["cover_full"]
k_show = int(np.argmin(np.abs(cov30 - cov30.mean(0)).sum(1)))
print("share of patches inside the 90 %% intervals, mean of 30 realizations: 8 stations plug-in %.2f, fully Bayesian %.2f;"
      " 40 stations plug-in %.2f, fully Bayesian %.2f" % tuple(cov30.mean(0)))
print("realization drawn below: the one closest to these means (number %d: %.2f, %.2f; %.2f, %.2f)" % ((k_show,) + tuple(cov30[k_show])))
s_true2, e2 = realization(k_show)
sgn = np.sign(s_true2.sum())         # the prior is symmetric: a draw and its negative are equally typical, and flipping
s_true2, e2 = sgn * s_true2, sgn * e2   # slip and noise together leaves every coverage number unchanged; draw positive slip
single = {}
for name, (G, mu, V) in ops.items():
    o = analyse(G @ s_true2 + e2[:G.shape[0]], G, mu, V, S2, A2, s_true=s_true2)
    single[name] = (G, mu, V, o); i, j = o["ij"]
    print(f"{name}: noise rms {np.sqrt(np.mean(e2[:G.shape[0]] ** 2)):.2f} mm; ABIC choice sigma = {np.sqrt(S2[i, j]):.2f} mm,"
          f" alpha^2 = {A2[i, j]:.3g}; resolved combinations {o['n_resolved']};"
          f" mean slip sd: plug-in {np.sqrt(o['vp']).mean():.3f} m, fully Bayesian {np.sqrt(o['vmix']).mean():.3f} m")

# %%
ext = []                                         # axis limits that contain both 99 % regions and the true values
for name in cases:
    w2 = single[name][3]["w"]
    ii, jj = np.nonzero(w2 >= P.hdr_levels(w2, (0.99,))[0.99])
    ext.append((0.5 * ls2[ii].min(), 0.5 * ls2[ii].max(), la2[jj].min() / np.log(10), la2[jj].max() / np.log(10)))
ext = np.array(ext)
xlim = (min(ext[:, 0].min(), np.log(sig_true)) - 0.25, max(ext[:, 1].max(), np.log(sig_true)) + 0.25)
ylim = (min(ext[:, 2].min(), np.log10(a2_true)) - 0.35, max(ext[:, 3].max(), np.log10(a2_true)) + 0.6)
fig, axs = plt.subplots(2, 2, figsize=(12.5, 8.6), gridspec_kw=dict(width_ratios=[1.1, 1]))
for row, name in enumerate(cases):
    G, mu, V, o = single[name]
    ax = axs[row, 0]
    w2 = o["w"]
    lev = sorted(set(P.hdr_levels(w2, (0.5, 0.9, 0.99)).values()))
    ax.contourf(0.5 * ls2, la2 / np.log(10), w2.T, levels=lev + [w2.max() * 1.01], colors=["#ecebe6", "#d9d8d1", "#c3c2b7"])
    ax.contour(0.5 * ls2, la2 / np.log(10), w2.T, levels=lev, colors=S.MUTED, linewidths=0.6)
    i, j = o["ij"]
    ax.plot(0.5 * ls2[i], la2[j] / np.log(10), "X", color=C_ABIC, ms=11, mec=S.SURFACE, label="ABIC minimum")
    ax.plot(np.log(sig_true), np.log10(a2_true), "*", color=C_TRUE, ms=13, mec=S.SURFACE, label="values used to make the data")
    ax.set_xlim(*xlim); ax.set_ylim(*ylim)
    xt = [v for v in [0.25, 0.5, 1, 2, 4, 8, 16, 32] if xlim[0] <= np.log(v) <= xlim[1]]
    ax.set_xticks(np.log(xt)); ax.set_xticklabels([f"{v:g}" for v in xt])
    ax.set_xlabel("σ (mm)"); ax.set_ylabel("log10 α²")
    ax.set_title(f"{name}: posterior of the knobs (50/90/99 %)", fontsize=10.5)
    ax.legend(fontsize=8, loc="upper right")
    ax = axs[row, 1]
    sdp = np.sqrt(o["vp"])
    ax.fill_betweenx(zc, o["mp"] - 1.645 * sdp, o["mp"] + 1.645 * sdp, color=LIGHT[C_ABIC], alpha=0.8, lw=0, label="ABIC plug-in, 90 %")
    ax.plot(o["mp"], zc, color=C_ABIC, lw=1.2, ls=(0, (3, 2)), label="ABIC plug-in mean")
    q = mixture_quantiles(o["ws"], o["means"], o["var"])
    ax.plot(q[0], zc, color=C_FULL, lw=1.2); ax.plot(q[1], zc, color=C_FULL, lw=1.2, label="fully Bayesian, 90 %")
    ax.plot(s_true2, zc, color=C_TRUE, lw=2.0, label="true slip")
    ax.invert_yaxis(); ax.set_xlabel("slip (m)"); ax.set_ylabel("depth (km)")
    S.badge(ax, f"inside the 90 % band: plug-in {100 * o['cover_plug']:.0f} %, fully Bayesian {100 * o['cover_full']:.0f} % of patches",
            loc="lower right", fs=7.8)
    ax.set_title(f"{name}: slip intervals", fontsize=10.5)
    ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
bk.savefig(fig, "2-30_abic_vs_full")
plt.show()

# %% [markdown]
# **How to read it.** The realization drawn is the most typical of 30 (printed above), not a lucky or an unlucky one.
# Left: the grey regions are the posterior of the two knobs (the marginal likelihood times the log-uniform hyperprior);
# the cross is ABIC's single choice; the star marks the values used to generate the data. With 8 stations the posterior
# is broad, so ABIC's point is only one of many settings that explain the data about as well; with 40 stations it is
# compact and surrounds the values used to make the data.
# Right: the blue band is the 90 % interval of the plug-in posterior, the violet lines the 90 % interval of the fully
# Bayesian posterior. The fully Bayesian band is wider below the top few kilometres, where the data see the slip poorly
# and the knobs matter most. Even the compact 40-station posterior still spans a range of α² that widens the slip
# intervals at depth, and the badges show the plug-in bands missing the true slip more often.
#
# One realization proves nothing about calibration. The next cell repeats the experiment many times, each time drawing a
# new true slip from the smoothing prior and new noise, and counts how often the 90 % intervals contain the true slip
# (the coverage; reminder: 2.01). The benchmark is 90 %: the posterior with the true knobs plugged in would reach it on
# average, because the true slips are drawn from that same prior.

# %%
def coverage_study(G, DDm, a2t, sigt, S2g, A2g, R, seed):
    rng_ = np.random.default_rng(seed)
    Mm = DDm.shape[0]
    mu_, V_ = simdiag(G, DDm)
    Lc_ = np.linalg.cholesky(a2t * np.linalg.inv(DDm) + 1e-12 * np.eye(Mm))
    cp, cf, ratio, nres, wid, wid_s, edge = [], [], [], [], [], [], []
    la_, ls_ = np.log(A2g[0]), np.log(S2g[:, 0])
    for _ in range(R):
        st = Lc_ @ rng_.standard_normal(Mm)
        dd_ = G @ st + sigt * rng_.standard_normal(G.shape[0])
        o_ = analyse(dd_, G, mu_, V_, S2g, A2g, s_true=st, wfloor=1e-6)
        cp.append(o_["cover_plug"]); cf.append(o_["cover_full"]); nres.append(o_["n_resolved"])
        ratio.append(np.mean(np.sqrt(o_["vmix"] / o_["vp"])))
        for w_, x_, out_ in [(o_["w"].sum(0), la_, wid), (o_["w"].sum(1), ls_, wid_s)]:   # spread of ln alpha^2 and ln sigma^2
            out_.append(np.sqrt((w_ * (x_ - (w_ * x_).sum()) ** 2).sum()))
        edge.append(o_["ij"][1] == 0)                 # ABIC optimum at the smallest alpha^2 of the grid: zero slip
    return dict(plug=np.mean(cp), full=np.mean(cf), ratio=np.mean(ratio), resolved=np.median(nres), sd_log_a2=np.median(wid),
                sd_log_s2=np.median(wid_s), edge=np.mean(edge))


t0 = time.time()
study = []
XS_AP = {5: np.geomspace(1, 30, 5), 8: np.geomspace(1, 40, 8), 40: np.geomspace(0.5, 60, 40), 120: np.geomspace(0.3, 60, 120)}
S2_, A2_ = make_grid(np.linspace(np.log(1e-4), np.log(400.0), 81), la2)
for nst, sg in [(5, 2.0), (8, 2.0), (40, 2.0), (120, 0.05)]:                        # antiplane fault in depth
    r_ = coverage_study(screw_G(XS_AP[nst], ze), DD, a2_true, sg, S2_, A2_, R=200, seed=nst)
    r_.update(toy="antiplane fault in depth", label=f"{nst} stations, σ {sg:g} mm"); study.append(r_)
Ms = 120
xa = np.arange(Ms) + 0.5
DDs = second_difference(Ms).T @ second_difference(Ms)
a2s = 1.0 / np.mean(np.diag(np.linalg.inv(DDs)))
la2s = np.linspace(np.log(a2s) - 7, np.log(a2s) + 7, 111)
for w_k, nst, sg in [(1.0, 120, 0.02), (2.0, 120, 0.05), (5.0, 120, 0.1), (10.0, 40, 0.2), (20.0, 12, 0.3), (30.0, 6, 0.5)]:
    xs_ = np.linspace(0.5, Ms - 0.5, nst)
    K = np.exp(-0.5 * ((xs_[:, None] - xa[None, :]) / w_k) ** 2); K /= K.sum(1, keepdims=True)
    S2_, A2_ = make_grid(np.linspace(np.log(1e-4), np.log(10.0), 81), la2s)
    r_ = coverage_study(K, DDs, a2s, sg, S2_, A2_, R=100, seed=int(w_k * 10))
    r_.update(toy="along-strike profile", label=f"kernel {w_k:g} km, {nst} st."); study.append(r_)
print(f"{'toy':26s} {'case':24s} {'resolved':>8s} {'sd ln a2':>8s} {'sd ln s2':>8s} {'plug-in':>8s} {'fully B.':>8s}"
      f" {'width ratio':>11s} {'ABIC at alpha^2 -> 0':>20s}")
for r_ in study:
    print(f"{r_['toy']:26s} {r_['label']:24s} {r_['resolved']:8.1f} {r_['sd_log_a2']:8.2f} {r_['sd_log_s2']:8.2f} {r_['plug']:8.3f}"
          f" {r_['full']:8.3f} {r_['ratio']:11.2f} {r_['edge']:20.2f}")
print("(coverage = share of patches whose true slip lies inside the 90 %% interval, averaged over repeats; last column = share"
      " of repeats whose ABIC optimum is the smallest alpha^2 on the grid; %.0f s)" % (time.time() - t0))

# %%
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.3))
for r_ in study:
    mk = "o" if r_["toy"].startswith("antiplane") else "s"
    x = r_["sd_log_a2"]
    axs[0].plot([x, x], [r_["plug"], r_["full"]], color=S.AXIS, lw=1.0, zorder=1)
    axs[0].plot(x, r_["plug"], mk, color=C_ABIC, ms=7, mec=S.SURFACE, zorder=2)
    axs[0].plot(x, r_["full"], mk, color=C_FULL, ms=7, mec=S.SURFACE, zorder=2)
    axs[1].plot(x, r_["ratio"], mk, color=S.INK2, ms=7, mec=S.SURFACE)
    axs[1].annotate(f"{r_['resolved']:.0f}", (x, r_["ratio"]), xytext=(5, 4), textcoords="offset points", fontsize=7.5, color=S.MUTED)
axs[0].axhline(0.9, color=S.MUTED, lw=0.8)
for ax in axs:
    ax.set_xlabel("spread of the posterior of α² (sd of ln α²)")
axs[0].set_ylabel("share of true slip inside the 90 % interval")
axs[1].set_ylabel("interval width: fully Bayesian / plug-in")
axs[0].set_title("Plug-in intervals fall short as the knob posterior broadens", fontsize=10.5)
axs[1].set_title("How much the knob uncertainty widens the intervals", fontsize=10.5)
axs[0].legend(handles=BP.legend_handles([("ABIC plug-in", C_ABIC, "dot"), ("fully Bayesian", C_FULL, "dot")]) +
              [plt.Line2D([], [], marker="o", ls="", color=S.INK2, label="antiplane fault in depth"),
               plt.Line2D([], [], marker="s", ls="", color=S.INK2, label="along-strike profile")], fontsize=8, loc="lower left")
axs[1].axhline(1.0, color=S.MUTED, lw=0.8)
S.badge(axs[1], "numbers: slip combinations resolved", loc="upper left", fs=7.8)
fig.tight_layout()
bk.savefig(fig, "2-30_coverage")
plt.show()

# %% [markdown]
# **How to read it.** Each linked pair of markers is one data set-up repeated many times (table above): the plug-in in
# blue, the fully Bayesian answer in violet. The horizontal axis is how broad the posterior of the smoothing knob is.
# Where it is narrow, the two kinds of interval coincide and both contain the truth about 90 % of the time. As it
# broadens, the plug-in intervals miss the truth more and more often, and the fully Bayesian intervals become wider than
# the plug-in ones by the factor in the right panel. This is Fukuda & Johnson's dominant-peak condition, measured.
#
# What makes the knob posterior broad? The small numbers in the right panel count the slip combinations that the data
# measure better than the prior does (generalised eigenvalues with $\alpha^2\mu_i/\sigma^2 > 1$): the fewer, the broader.
# Stations alone do not decide it: with the same 120 stations the antiplane fault resolves fewer combinations than the
# along-strike profile with its two sharpest kernels (table), because only the shallow part of the fault is seen. The
# two toys do not fall on one curve: the antiplane cases with 5 and 8 stations sit below the along-strike points of
# similar spread, and the table shows that their σ² posterior is also much broader (column sd ln s2). Our reading, an
# inference from this table, is that the uncertainty of σ² adds to that of α².
#
# Two warnings for the weakest set-ups. The fully Bayesian intervals fall short of 90 % there too, because with data
# this weak the answer depends on the hyperprior, and even on where its range stops (section 6). And ABIC's optimum is,
# in a share of the repeats (last column), the smallest α² on the grid: nearly zero slip with nearly zero uncertainty.
#
# **Worked example (why the fully Bayesian interval is wider).** For one patch the fully Bayesian variance obeys the law of
# total variance: $\mathrm{Var}[s_j\mid\mathbf d] = \mathrm{E}\big[\mathrm{Var}[s_j\mid\mathbf d,\sigma^2,\alpha^2]\big] + \mathrm{Var}\big[\mathrm{E}[s_j\mid\mathbf d,\sigma^2,\alpha^2]\big]$:
# the average spread for fixed knobs, plus how much the best estimate moves when the knobs move. The plug-in keeps only
# the first term, evaluated at one setting.

# %%
G8, mu8, V8, o8 = single["8 stations"]
j = int(np.argmin(np.abs(zc - 14.5)))
E_var = o8["ws"] @ o8["var"][:, j]
Var_E = o8["ws"] @ (o8["means"][:, j] - o8["mmix"][j]) ** 2
print(f"patch at {zc[j]:.1f} km depth, 8 stations:")
print(f"  average conditional variance  E[Var]  = {E_var:.5f} m^2")
print(f"  variance of conditional means Var[E]  = {Var_E:.5f} m^2")
print(f"  fully Bayesian variance (sum)         = {E_var + Var_E:.5f} m^2  (direct: {o8['vmix'][j]:.5f})")
print(f"  plug-in variance at the ABIC knobs    = {o8['vp'][j]:.5f} m^2")

# %% [markdown]
# In this example the first term alone already exceeds the plug-in variance: the knob settings the data allow include
# larger σ² and α² than the ABIC pair, and those give a larger conditional spread. The second term adds less.

# %% [markdown]
# **Common misreadings.**
# * "ABIC is wrong." Without positivity it is the peak of the same posterior of the knobs (with a flat hyperprior), and
#   Fukuda & Johnson (2010, p. 1449) give the condition under which the two answers agree. What the plug-in discards is
#   the uncertainty of the knobs.
# * "Many stations make the knobs well determined." What counts is the number of independently resolved slip
#   combinations and, for σ², the number of data left over after them.
#
# **In practice.** A slip inversion that samples the smoothing scale and the data scales (with NUTS or another
# sampler) is the fully Bayesian column of this comparison. A plug-in answer understates the slip uncertainty most where
# the data resolve least.

# %% [markdown]
# ## 3. Positivity breaks the ABIC formula
#
# **In plain words.** Many slip inversions forbid slip in the wrong direction (back-slip on a thrust, for example). The
# closed-form ABIC formula averages over *all* slip models, including the forbidden negative ones. With positivity, the average must run over the allowed models only,
# and the share of the posterior that lies on the allowed side changes with the knobs. Leave that share out and ABIC
# picks the wrong knobs.
#
# **Precisely.** Fukuda & Johnson (2008) state it in the abstract: "A popular objective method based on Akaike's Bayesian
# information criterion (ABIC) is incorrect if positivity constraints are employed" (p. 1128). In their appendix, the
# conventional procedure takes the non-negative least-squares (NNLS: least squares with every slip forced to be ≥ 0) solution $\mathbf s^\ast$ for each knob setting and
# evaluates the closed-form marginal likelihood (their eq. A7) with it, but "the analytical expression for the marginal
# likelihood, (A7), is derived from the prior PDF (9) that does not include positivity constraints" (p. 1146), and the
# correct integral "cannot be computed analytically" (p. 1145).
#
# It can be computed numerically for a few patches. The positivity prior is the Gaussian prior truncated to
# $\mathbf s\ge 0$ (cut off below zero; reminder: truncated distributions, 2.11) and renormalised by
# $P(\mathbf s\ge\mathbf 0\mid\alpha^2)$ so that it integrates to one again. Integrating over the allowed region only gives
#
# $$p_+(\mathbf d\mid\sigma^2,\alpha^2) = p(\mathbf d\mid\sigma^2,\alpha^2)\;\frac{P(\mathbf s\ge\mathbf 0\mid\mathbf d,\sigma^2,\alpha^2)}{P(\mathbf s\ge\mathbf 0\mid\alpha^2)},$$
#
# where both probabilities are *orthant probabilities* of Gaussians: of the unconstrained posterior in the numerator, of
# the prior in the denominator. An orthant is the region where every coordinate has a fixed sign, here all positive: in
# two dimensions the positive quarter-plane, in three the positive eighth of space. An orthant probability is the share
# of a Gaussian's probability that falls inside it (in two dimensions, the share of the ellipse's probability in the
# quarter-plane). In closed form only for very few dimensions; here it is computed numerically for 8 patches. For a zero-mean prior the denominator does not depend on α², because scaling a zero-mean
# Gaussian leaves the share of it in the positive orthant unchanged. So the whole correction comes from the posterior,
# whose position relative to zero moves with the knobs. The picture below shows both statements in two dimensions.

# %%
# orthant probabilities in two dimensions: the share of a Gaussian in the quarter-plane where both slips are >= 0
rng_o = np.random.default_rng(31)
C_o = np.array([[1.0, 0.6], [0.6, 1.0]])                 # prior correlation of two neighbouring patches (m^2 per unit alpha^2)
G_o = np.array([[1.0, 0.4], [0.2, 0.7]])                 # two stations, two patches
d_o, s_o = np.array([0.35, 0.05]), 0.15                  # data and noise sd (same units as slip, for simplicity)


def post_o(a2):
    C = np.linalg.inv(np.linalg.inv(a2 * C_o) + G_o.T @ G_o / s_o ** 2)
    return C @ G_o.T @ d_o / s_o ** 2, C


def orthant_mc(m, C, n=2_000_000):
    return float(np.mean(np.all(rng_o.multivariate_normal(m, C, n) >= 0, axis=1)))


def ellipse90(m, C):
    ev, U = np.linalg.eigh(C)
    t_ = np.linspace(0, 2 * np.pi, 200)
    return (m[:, None] + U @ (np.sqrt(ev * chi2_90)[:, None] * np.vstack([np.cos(t_), np.sin(t_)])))


chi2_90 = -2 * np.log(0.1)                                 # 90 % region of a 2-D Gaussian: Mahalanobis distance^2 < 4.61
rho_o = C_o[0, 1]
print(f"prior orthant probability, closed form 1/4 + arcsin(rho)/(2 pi) = {0.25 + np.arcsin(rho_o) / (2 * np.pi):.4f}")
fig, axs = plt.subplots(1, 2, figsize=(11, 4.7))
for ax, lim in zip(axs, [(-3.6, 3.6), (-0.55, 0.85)]):
    ax.fill_between([0, lim[1]], 0, lim[1], color="#ecebe6", lw=0, zorder=0)
    ax.axhline(0, color=S.AXIS, lw=0.8); ax.axvline(0, color=S.AXIS, lw=0.8)
    ax.set_xlim(*lim); ax.set_ylim(*lim); ax.set_aspect("equal")
    ax.set_xlabel("slip of patch 1"); ax.set_ylabel("slip of patch 2")
    ax.text(lim[1] * 0.96, lim[1] * 0.93, "positive orthant\n(both slips ≥ 0)", ha="right", va="top", fontsize=8, color=S.INK2)
for a2, col in [(0.25, S.BLUE), (2.0, S.ORANGE)]:
    po = orthant_mc(np.zeros(2), a2 * C_o)
    e = ellipse90(np.zeros(2), a2 * C_o)
    axs[0].plot(e[0], e[1], color=col, lw=1.8, label=f"prior, α² = {a2:g}: P(both ≥ 0) = {po:.3f}")
    m_, C_ = post_o(a2)
    pp = orthant_mc(m_, C_)
    e = ellipse90(m_, C_)
    axs[1].plot(e[0], e[1], color=col, lw=1.8, label=f"posterior, α² = {a2:g}: P(both ≥ 0 | d) = {pp:.3f}")
    axs[1].plot(*m_, "o", color=col, ms=5)
    print(f"alpha^2 = {a2:4.2f}: prior orthant probability {po:.4f} (Monte Carlo), posterior {pp:.4f}, posterior mean {np.round(m_, 3)}")
axs[0].set_title("(a) Prior: the share does not depend on α²")
axs[1].set_title("(b) Posterior: the share moves with the knobs")
axs[1].plot([], [], "o", color=S.MUTED, ms=5, label="dots: posterior means")
for ax in axs:                                               # below the axes: the ellipses fill the panels
    ax.legend(fontsize=7.8, loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=1,
              title="curves: 90 % regions of the Gaussian (before the bound)", title_fontsize=7.4)
fig.tight_layout()
bk.savefig(fig, "2-30_orthant")
plt.show()

# %% [markdown]
# **How to read it.** Each curve is the 90 % region of a two-patch Gaussian; the shaded quarter-plane is the positive
# orthant. (a) Two zero-mean priors of different width: the larger one is the smaller one magnified about the origin, so
# the share of probability inside the quarter-plane is the same (printed, with the two-dimensional closed form). This is
# why the denominator of the formula above does not depend on α². (b) The posteriors for the same data at the same two
# settings sit at different places relative to the axes, so their shares differ (printed). That share is the numerator:
# it is what the conventional formula leaves out, and it changes with the knobs.

# %% [markdown]
# Test problem: 8 patches from 0 to 16 km, slip $(1.2, 1.5, 1.0, 0.3, 0, 0, 0, 0)$ m from the top down, 12 stations on one
# side, σ = 3 mm. The noise realization is chosen with a root-mean-square close to σ, so the example is typical (printed).
# Hyperpriors uniform in σ² and α², as in Fukuda & Johnson: "we assume that p(σ², α²) is a uniform PDF" (2008, p. 1131).

# %%
Mp = 8
zp = np.linspace(0.0, 16.0, Mp + 1)
xs_p = np.array([0.5, 1, 2, 3, 4.5, 6, 8, 11, 15, 20, 30, 45.0])
Np = xs_p.size
Gp = screw_G(xs_p, zp)
s_true_p = np.array([1.2, 1.5, 1.0, 0.3, 0.0, 0.0, 0.0, 0.0])
e_p = 3.0 * np.random.default_rng(4).standard_normal(Np)
d_p = Gp @ s_true_p + e_p
print("noise realization rms: %.2f mm (sigma = 3 mm)" % np.sqrt(np.mean(e_p ** 2)))
Lp = second_difference(Mp)
LtLp = Lp.T @ Lp
P0p = np.linalg.inv(LtLp)
logdet_LtLp = np.linalg.slogdet(LtLp)[1]


def A_p(s2, a2):
    return Gp.T @ Gp / s2 + LtLp / a2


def f_p(s, s2, a2, d):
    r = d - Gp @ s
    q = Lp @ s
    return r @ r / s2 + q @ q / a2


def logml_A7(s, s2, a2, d):
    """Fukuda & Johnson (2008) eq. A7 with s plugged in: exact only when s is the unconstrained least-squares solution."""
    return (-0.5 * Np * np.log(2 * np.pi * s2) - 0.5 * Mp * np.log(a2) + 0.5 * logdet_LtLp
            - 0.5 * np.linalg.slogdet(A_p(s2, a2))[1] - 0.5 * f_p(s, s2, a2, d))


def s_free(s2, a2, d):
    return np.linalg.solve(A_p(s2, a2), Gp.T @ d / s2)


def s_nnls(s2, a2, d):
    return nnls(np.vstack([Gp / np.sqrt(s2), Lp / np.sqrt(a2)]), np.r_[d / np.sqrt(s2), np.zeros(Mp)])[0]


def log_orthant(m, C):
    return float(np.log(max(mvn(-m, C, allow_singular=True, seed=0).cdf(np.zeros(len(m))), 1e-300)))


def logml_exact(s2, a2, d, lp0):
    sf = s_free(s2, a2, d)
    C = np.linalg.inv(A_p(s2, a2)); C = 0.5 * (C + C.T)
    return logml_A7(sf, s2, a2, d) + log_orthant(sf, C) - lp0


for a2 in [0.01, 1.0, 100.0]:
    print("prior probability P(s >= 0 | alpha^2 = %6g) = %.5f" % (a2, np.exp(log_orthant(np.zeros(Mp), a2 * P0p))))
lp0 = log_orthant(np.zeros(Mp), P0p)

# %% [markdown]
# **Worked example (one knob setting).** At σ = 3 mm and α² = 0.5 the three pieces of the exact formula are printed below:
# the closed-form log marginal likelihood (no positivity), the log posterior orthant probability, and the log prior
# orthant probability; the exact value is the first plus the second minus the third. The conventional procedure's value
# (NNLS solution inside eq. A7) is printed for comparison.

# %%
s2w, a2w = 9.0, 0.5
sf = s_free(s2w, a2w, d_p); Cw = np.linalg.inv(A_p(s2w, a2w)); Cw = 0.5 * (Cw + Cw.T)
t1 = logml_A7(sf, s2w, a2w, d_p); t2 = log_orthant(sf, Cw)
print(f"ln p(d | sigma^2, alpha^2), closed form without positivity = {t1:.4f}")
print(f"ln P(s >= 0 | d, sigma^2, alpha^2)  (posterior orthant)     = {t2:.4f}")
print(f"ln P(s >= 0 | alpha^2)              (prior orthant)         = {lp0:.4f}")
print(f"exact ln p_+(d | sigma^2, alpha^2) = {t1:.4f} + {t2:.4f} - ({lp0:.4f}) = {t1 + t2 - lp0:.4f}")
print(f"conventional value (NNLS solution in eq. A7)              = {logml_A7(s_nnls(s2w, a2w, d_p), s2w, a2w, d_p):.4f}")
print("unconstrained solution at this setting:", np.round(sf, 3), "(negative entries are forbidden slip)")

# %%
t0 = time.time()
ls2p = np.linspace(np.log(0.5), np.log(400.0), 46)
la2p = np.linspace(np.log(1e-3), np.log(1e3), 70)
conv = np.zeros((ls2p.size, la2p.size)); exact = np.zeros_like(conv); free = np.zeros_like(conv)
for i_, a in enumerate(ls2p):
    for j_, b in enumerate(la2p):
        s2, a2 = np.exp(a), np.exp(b)
        sfree = s_free(s2, a2, d_p)
        free[i_, j_] = logml_A7(sfree, s2, a2, d_p)
        conv[i_, j_] = logml_A7(s_nnls(s2, a2, d_p), s2, a2, d_p)
        C_ = np.linalg.inv(A_p(s2, a2)); C_ = 0.5 * (C_ + C_.T)
        exact[i_, j_] = free[i_, j_] + log_orthant(sfree, C_) - lp0


def knob_posterior(F):
    W = F + ls2p[:, None] + la2p[None, :]           # uniform priors in sigma^2 and alpha^2 on a log grid: Jacobian
    W = np.exp(W - W.max()); W /= W.sum()
    return W


def grid_quantiles(p, x, qs=(0.05, 0.5, 0.95)):
    """Quantiles of a distribution given as probabilities p at grid points x (each point holds its whole cell)."""
    return np.interp(qs, np.cumsum(p) - 0.5 * p, x)


def gibbs_positive(d, n=20000, burn=2000, seed=1):
    """Exact Gibbs sampler of Fukuda & Johnson's model: truncated-normal slip, inverse-gamma sigma^2 and alpha^2."""
    r_ = np.random.default_rng(seed)
    s = np.maximum(np.linalg.lstsq(Gp, d, rcond=None)[0], 0.05); s2, a2 = 9.0, 1.0
    out = np.empty((n, Mp + 2))
    for it in range(n + burn):
        Ci = A_p(s2, a2); b = Gp.T @ d / s2
        for k in range(Mp):
            prec = Ci[k, k]; m_k = (b[k] - Ci[k] @ s + Ci[k, k] * s[k]) / prec; sd_k = 1.0 / np.sqrt(prec)
            s[k] = truncnorm.rvs(-m_k / sd_k, np.inf, loc=m_k, scale=sd_k, random_state=r_)
        res = d - Gp @ s
        s2 = (res @ res / 2) / r_.gamma(Np / 2 - 1)       # uniform prior on sigma^2 -> inverse gamma (N/2 - 1)
        q = Lp @ s
        a2 = (q @ q / 2) / r_.gamma(Mp / 2 - 1)           # uniform prior on alpha^2 -> inverse gamma (M/2 - 1)
        if it >= burn:
            out[it - burn] = np.r_[s, s2, a2]
    return out


smp = gibbs_positive(d_p)
print("grid and Gibbs done in %.0f s" % (time.time() - t0))
for name, F in [("closed form without positivity", free), ("conventional (NNLS in eq. A7)", conv), ("exact with positivity", exact)]:
    i_, j_ = np.unravel_index(np.argmax(F), F.shape)
    W = knob_posterior(F)
    q5, q50, q95 = np.exp(grid_quantiles(W.sum(0), la2p))
    print(f"{name:32s}: ABIC-type choice alpha^2 = {np.exp(la2p[j_]):.3f} (sigma {np.sqrt(np.exp(ls2p[i_])):.2f} mm);"
          f" posterior of alpha^2: median {q50:.3f}, 90 % [{q5:.3f}, {q95:.3f}]; sigma median {np.exp(0.5 * grid_quantiles(W.sum(1), ls2p)[1]):.2f} mm")
qg = np.percentile(smp[:, Mp + 1], [5, 50, 95])
print(f"{'exact Gibbs sampling':32s}: posterior of alpha^2: median {qg[1]:.3f}, 90 % [{qg[0]:.3f}, {qg[2]:.3f}];"
      f" sigma median {np.sqrt(np.median(smp[:, Mp])):.2f} mm")
ess_g = D.ess(np.log(smp[:, Mp + 1]))
print("Gibbs: effective sample size of ln alpha^2 %.0f of %d draws; Monte Carlo error of its mean %.3f (grid step in ln alpha^2: %.3f)"
      % (ess_g, len(smp), np.log(smp[:, Mp + 1]).std() / np.sqrt(ess_g), la2p[1] - la2p[0]))
Wx = knob_posterior(exact)
print("share of the exact knob posterior in the outermost grid rows (should be ~0): top of sigma^2 %.4f, bottom %.4f;"
      " top of alpha^2 %.4f, bottom %.4f" % (Wx.sum(1)[-1], Wx.sum(1)[0], Wx.sum(0)[-1], Wx.sum(0)[0]))

# %% [markdown]
# The last lines check the computation itself. A Gibbs sampler is an MCMC method (reminder: Markov chains, 2.16) that
# updates one unknown at a time by drawing it from its exact conditional distribution given all the others; with
# positivity each conditional is a one-dimensional truncated Gaussian, easy to draw. The Monte Carlo error of the Gibbs
# sampler (2.05) and the grid step bound how
# closely the two exact methods can agree, and the grid must contain the posterior. Now the same comparison over 40
# noise realizations, with σ held at its true value so that only the choice of α² differs.

# %%
t0 = time.time()
rng_p = np.random.default_rng(11)
ratio_conv, ratio_free = [], []
for rep in range(40):
    dd = Gp @ s_true_p + 3.0 * rng_p.standard_normal(Np)
    opt = {}
    for name, fn in [("conv", lambda b: -logml_A7(s_nnls(9.0, np.exp(b), dd), 9.0, np.exp(b), dd)),
                     ("free", lambda b: -logml_A7(s_free(9.0, np.exp(b), dd), 9.0, np.exp(b), dd)),
                     ("exact", lambda b: -logml_exact(9.0, np.exp(b), dd, lp0))]:
        opt[name] = minimize_scalar(fn, bounds=(np.log(1e-4), np.log(100.0)), method="bounded", options=dict(xatol=1e-4)).x
    ratio_conv.append(np.exp(opt["conv"] - opt["exact"])); ratio_free.append(np.exp(opt["free"] - opt["exact"]))
ratio_conv, ratio_free = np.array(ratio_conv), np.array(ratio_free)
print(f"alpha^2 chosen by the conventional procedure / exact: median {np.median(ratio_conv):.2f}, 10-90 % "
      f"{np.percentile(ratio_conv, 10):.2f}-{np.percentile(ratio_conv, 90):.2f}; larger (less smoothing) in {np.sum(ratio_conv > 1)} of 40")
print(f"alpha^2 chosen ignoring positivity / exact:          median {np.median(ratio_free):.2f}, 10-90 % "
      f"{np.percentile(ratio_free, 10):.2f}-{np.percentile(ratio_free, 90):.2f}; larger in {np.sum(ratio_free > 1)} of 40  ({time.time() - t0:.0f} s)")

# %%
fig, axs = plt.subplots(1, 3, figsize=(15, 4.3), gridspec_kw=dict(width_ratios=[1.1, 1, 0.9]))
ax = axs[0]
for name, F, col, ls_ in [("conventional (NNLS in A7)", conv, C_CONV, "-"), ("exact with positivity", exact, C_FULL, "-"),
                          ("without positivity", free, C_ABIC, (0, (3, 2)))]:
    pa = knob_posterior(F).sum(0); dx = la2p[1] - la2p[0]
    ax.plot(la2p / np.log(10), pa / (dx / np.log(10)), color=col, lw=1.8, ls=ls_, label=name)
h, b = np.histogram(np.log10(smp[:, Mp + 1]), bins=np.arange(la2p[0], la2p[-1], 0.1) / np.log(10), density=True)
ax.bar(0.5 * (b[1:] + b[:-1]), h, width=b[1] - b[0], color="#d9d8d1", lw=0, label="exact Gibbs sampling", zorder=0)
ax.set_xlabel("log10 α²"); ax.set_ylabel("posterior density"); ax.set_xlim(-1.6, 2.6)
ax.set_title("Posterior of the smoothing knob", fontsize=10.5); ax.legend(fontsize=8)
ax = axs[1]
i_c, j_c = np.unravel_index(np.argmax(conv), conv.shape)
s2c, a2c = np.exp(ls2p[i_c]), np.exp(la2p[j_c])
s_conv = s_nnls(s2c, a2c, d_p); sd_conv = np.sqrt(np.diag(np.linalg.inv(A_p(s2c, a2c))))   # covariance of their eq. A9
zpc = 0.5 * (zp[1:] + zp[:-1])
ax.errorbar(s_conv, zpc + 0.25, xerr=1.645 * sd_conv, fmt="o", color=C_CONV, ms=5, capsize=2, lw=1.1,
            label="conventional ABIC: NNLS ± 90 % (eq. A9)")
qlo, qmed, qhi = np.percentile(smp[:, :Mp], [5, 50, 95], axis=0)
ax.errorbar(qmed, zpc - 0.25, xerr=[qmed - qlo, qhi - qmed], fmt="s", color=C_FULL, ms=5, capsize=2, lw=1.1,
            label="fully Bayesian: median, 5-95 %")
ax.plot(smp[:, :Mp].mean(0), zpc - 0.25, "x", color=C_FULL, ms=6, label="fully Bayesian mean")
ax.plot(np.repeat(s_true_p, 2), np.c_[zp[:-1], zp[1:]].ravel(), color=C_TRUE, lw=1.6, label="true slip")  # constant per patch
ax.axvline(0, color=S.AXIS, lw=0.8)
ax.invert_yaxis(); ax.set_xlabel("slip (m)"); ax.set_ylabel("depth (km)")
ax.set_title("Slip: plug-in vs fully Bayesian", fontsize=10.5); ax.legend(fontsize=7.4, loc="center right")
ax = axs[2]
allr = np.r_[ratio_conv, ratio_free]
bins_r = np.arange(min(0.95, allr.min() - 0.025), allr.max() + 0.05, 0.025)
ax.hist(ratio_conv, bins=bins_r, color=C_CONV, alpha=0.8, label="conventional / exact")
ax.hist(ratio_free, bins=bins_r, color=C_ABIC, alpha=0.5, label="without positivity / exact")
ax.axvline(1, color=S.INK, lw=1)
ax.text(1.005, ax.get_ylim()[1] * 0.97, "same choice\nas exact", fontsize=8, color=S.INK2, va="top")
ax.set_xlabel("α² chosen / exact choice"); ax.set_ylabel("noise realizations (of 40)")
ax.set_title("40 noise realizations (σ known)", fontsize=10.5); ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
bk.savefig(fig, "2-30_positivity")
plt.show()
print("90 %% interval width, mean over the 8 patches: conventional (NNLS ± 1.645 sd from eq. A9) %.3f m;"
      " fully Bayesian (5-95 %% of the Gibbs samples) %.3f m" % (np.mean(2 * 1.645 * sd_conv), np.mean(qhi - qlo)))
print("conventional intervals reaching below zero: %d of %d patches" % (np.sum(s_conv - 1.645 * sd_conv < 0), Mp))
print("posterior mean at the four zero-slip patches (Gibbs): " + ", ".join(f"{v:.3f}" for v in smp[:, 4:Mp].mean(0)) + " m")

# %% [markdown]
# **How to read it.** Left: the posterior of the smoothing knob under four computations. The exact positivity marginal
# likelihood (violet) and the exact Gibbs sampler of the full model (grey bars) agree, which validates the orthant formula;
# the conventional procedure (orange) and the formula without positivity (dashed blue) sit at larger α², that is, less
# smoothing (all numbers printed above). Middle: the conventional answer is a single NNLS solution whose error bars come
# from the unconstrained Gaussian covariance (their eq. A9), so they extend into negative slip that the model forbids. The
# fully Bayesian intervals respect positivity and are asymmetric near zero; the conventional intervals are also wider on
# average (printed). Right: over 40 noise realizations the
# conventional procedure chooses less smoothing than the exact answer every time (ratios printed above), the same direction
# as Fukuda & Johnson's Chi-Chi application, where the ABIC slip was "significantly rougher" than the fully Bayesian one and
# its uncertainty "significantly larger" (2008, p. 1137). The size of the effect here is specific to this small toy. The
# true slip is a sharp step that a smoothing prior does not expect, so neither answer recovers it exactly at the step;
# this comparison is about the choice of α² and the treatment of the bound, not about which interval covers better.
#
# **Common misreadings.**
# * "With positivity the prior's normalising constant depends on α²." Not for a zero-mean prior: the prior orthant
#   probability printed above is the same for all three α². The correction lives entirely in the posterior.
# * "A patch with zero true slip should have a posterior mean of zero." Under positivity the mean of a patch that the data
#   cannot see is pulled above zero (the crosses), because the allowed values are all positive. Report intervals or medians.
# * "Impossible" in Fukuda & Johnson (2008) means not in closed form. The orthant probabilities are computable for a few
#   patches (as here); for hundreds of patches they are not practical, which is why sampling is the general answer.
#
# **In practice.** An ABIC slip model computed with positivity (non-negative least squares inside the ABIC loop) uses
# the formula this section shows to be wrong, so its smoothing weight and its error bars need that caveat. In a sampler
# the bound is simply part of the model and needs no special formula: notebook 2.33 (section 6) samples a positive slip
# model with NUTS.

# %% [markdown]
# ## 4. Data weighting: several data sets with unknown noise levels
#
# **In plain words.** Combining GPS and InSAR raises the same question: how much should each data set count? If both get
# the same weight, the 300 InSAR points outweigh the 25 more precise GPS points. In the hierarchical model each data set
# gets its own noise scale σ_k, and the data estimate them along with everything else.
#
# **Precisely.** Fukuda & Johnson (2010) treat "multiple data sets with unknown relative weights, model parameters that are
# related linearly or non-linearly through theoretic models to observations" (p. 1441). With slip rates $\dot s_k$ entering
# linearly and locking depths $D_k$ nonlinearly, the slip rates can be integrated out in closed form for every
# $(D_1, D_2, \sigma_1, \sigma_2)$, so only four parameters need sampling: the *mixed linear–nonlinear* method. Their
# synthetic test has "25 observations with the smaller errors (data set 1) and 300 observations with the larger errors
# (data set 2). These two data sets might represent GPS and InSAR data, for example" (p. 1450), with noise sd 1.5 and
# 4.5 mm/yr. The geometry and true values below are chosen here (faults at −50 and 0 km, $\dot s = 25$ and 15 mm/yr,
# $D = 15$ and 10 km), not copied from the paper. Priors: $D_k$ uniform on 0.5–40 km, $\dot s_k \sim N(0, 50^2)$ mm/yr,
# $\ln\sigma_k$ uniform on (−3, 4).

# %%
x1f, x2f = -50.0, 0.0
truth4 = dict(s1=25.0, s2=15.0, D1=15.0, D2=10.0, sig1=1.5, sig2=4.5)
rng_pos = np.random.default_rng(5)
xg = np.sort(rng_pos.uniform(-250, 250, 25))
xi = np.sort(rng_pos.uniform(-150, 150, 300))
TAU = 50.0
Xall = jnp.asarray(np.r_[xg, xi])
grp = jnp.asarray(np.r_[np.zeros(25), np.ones(300)])
ramp = 0.02 * xi + 1.0                                         # used in section 5: an InSAR offset and orbital ramp (mm/yr)


def vel(x, s1, s2, D1, D2):
    return s1 / np.pi * np.arctan((x - x1f) / D1) + s2 / np.pi * np.arctan((x - x2f) / D2)


def make_data(seed, with_ramp=False):
    r_ = np.random.default_rng(seed)
    dg_ = vel(xg, truth4["s1"], truth4["s2"], truth4["D1"], truth4["D2"]) + truth4["sig1"] * r_.standard_normal(25)
    di_ = vel(xi, truth4["s1"], truth4["s2"], truth4["D1"], truth4["D2"]) + truth4["sig2"] * r_.standard_normal(300)
    if with_ramp:
        di_ = di_ + ramp
    return np.r_[dg_, di_]


def G_lin(D1, D2, with_ramp=False):
    cols = [jnp.arctan((Xall - x1f) / D1) / jnp.pi, jnp.arctan((Xall - x2f) / D2) / jnp.pi]
    if with_ramp:
        cols += [grp, grp * Xall / 100.0]                      # InSAR offset and ramp per 100 km, linear nuisances
    return jnp.stack(cols, 1)


def log_marg_lin(D1, D2, sig1, sig2, d, with_ramp=False):
    """ln p(d | D, sigma) with the linear parameters (slip rates, ramp) integrated out analytically."""
    G = G_lin(D1, D2, with_ramp)
    cinv = jnp.where(grp == 0, 1.0 / sig1 ** 2, 1.0 / sig2 ** 2)
    A = G.T @ (G * cinv[:, None]) + jnp.eye(G.shape[1]) / TAU ** 2
    b = G.T @ (cinv * d)
    c = jnp.linalg.cholesky(A)
    y = jax.scipy.linalg.solve_triangular(c, b, lower=True)
    return -0.5 * (d.size * jnp.log(2 * jnp.pi) - jnp.sum(jnp.log(cinv)) + 2 * jnp.sum(jnp.log(jnp.diag(c)))
                   + G.shape[1] * jnp.log(TAU ** 2) + jnp.sum(cinv * d ** 2) - y @ y)


def model_mixed(d, with_ramp=False, weights="separate", fixed_sig=3.0):
    D1 = numpyro.sample("D1", dist.Uniform(0.5, 40.0))
    D2 = numpyro.sample("D2", dist.Uniform(0.5, 40.0))
    if weights == "fixed":
        sig1 = sig2 = fixed_sig
    else:
        ls1 = numpyro.sample("log_sig1", dist.Uniform(-3.0, 4.0))
        ls2 = ls1 if weights == "common" else numpyro.sample("log_sig2", dist.Uniform(-3.0, 4.0))
        sig1, sig2 = jnp.exp(ls1), jnp.exp(ls2)
    numpyro.factor("marginal", log_marg_lin(D1, D2, sig1, sig2, d, with_ramp))


def model_joint(d):
    D1 = numpyro.sample("D1", dist.Uniform(0.5, 40.0)); D2 = numpyro.sample("D2", dist.Uniform(0.5, 40.0))
    s1 = numpyro.sample("s1", dist.Normal(0.0, TAU)); s2 = numpyro.sample("s2", dist.Normal(0.0, TAU))
    ls1 = numpyro.sample("log_sig1", dist.Uniform(-3.0, 4.0)); ls2 = numpyro.sample("log_sig2", dist.Uniform(-3.0, 4.0))
    mu_ = s1 * jnp.arctan((Xall - x1f) / D1) / jnp.pi + s2 * jnp.arctan((Xall - x2f) / D2) / jnp.pi
    numpyro.sample("obs", dist.Normal(mu_, jnp.where(grp == 0, jnp.exp(ls1), jnp.exp(ls2))), obs=d)


def run_nuts(model, seed=0, **kw):
    """4 NUTS chains. Returns the samples, the largest split R-hat and the number of divergent transitions."""
    mc = MCMC(NUTS(model), num_warmup=800, num_samples=1500, num_chains=4, chain_method="parallel", progress_bar=False)
    mc.run(jax.random.PRNGKey(seed), extra_fields=("diverging",), **kw)
    grouped = mc.get_samples(group_by_chain=True)
    rhat = max(float(np.max(split_gelman_rubin(np.asarray(v)))) for v in grouped.values())
    ndiv = int(np.sum(np.asarray(mc.get_extra_fields()["diverging"])))
    return {k: np.asarray(v) for k, v in mc.get_samples().items()}, rhat, ndiv


def sigmas_of(smp_, weights="separate", fixed_sig=3.0):
    n = smp_["D1"].size
    if weights == "fixed":
        return np.full(n, fixed_sig), np.full(n, fixed_sig)
    s1s = np.exp(smp_["log_sig1"])
    return s1s, (s1s if weights == "common" else np.exp(smp_["log_sig2"]))


def slip_rate_draws(smp_, d, weights="separate", fixed_sig=3.0, seed=0):
    """Draw the integrated-out slip rates from their Gaussian conditional, one draw per posterior sample."""
    r_ = np.random.default_rng(seed)
    s1s, s2s = sigmas_of(smp_, weights, fixed_sig)
    X = np.asarray(Xall); g = np.asarray(grp); dd_ = np.asarray(d)
    G = np.stack([np.arctan((X[None, :] - x1f) / smp_["D1"][:, None]),
                  np.arctan((X[None, :] - x2f) / smp_["D2"][:, None])], -1) / np.pi       # (samples, data, 2)
    cinv = np.where(g[None, :] == 0, 1 / s1s[:, None] ** 2, 1 / s2s[:, None] ** 2)
    A = np.einsum("kni,kn,knj->kij", G, cinv, G) + np.eye(2) / TAU ** 2
    b = np.einsum("kni,kn->ki", G, cinv * dd_[None, :])
    Ci = np.linalg.inv(A)
    return np.einsum("kij,kj->ki", Ci, b) + np.einsum("kij,kj->ki", np.linalg.cholesky(Ci), r_.standard_normal((len(s1s), 2)))


def q3(v):
    return np.percentile(v, [5, 50, 95])


t0 = time.time()
d4 = jnp.asarray(make_data(100))
sm, rh_m, dv_m = run_nuts(model_mixed, d=d4)
sj, rh_j, dv_j = run_nuts(model_joint, d=d4)
sc, rh_c, dv_c = run_nuts(model_mixed, d=d4, weights="common")
rates = slip_rate_draws(sm, d4)
print(f"{'parameter':10s} {'true':>6s} | {'mixed: 5 / 50 / 95 %':>24s} | {'full joint NUTS: 5 / 50 / 95 %':>30s}")
rows4 = [("D1 (km)", truth4["D1"], sm["D1"], sj["D1"]), ("D2 (km)", truth4["D2"], sm["D2"], sj["D2"]),
         ("s1 (mm/yr)", truth4["s1"], rates[:, 0], sj["s1"]), ("s2 (mm/yr)", truth4["s2"], rates[:, 1], sj["s2"]),
         ("sigma1", truth4["sig1"], np.exp(sm["log_sig1"]), np.exp(sj["log_sig1"])),
         ("sigma2", truth4["sig2"], np.exp(sm["log_sig2"]), np.exp(sj["log_sig2"]))]
for name, tv, a, b in rows4:
    qa, qb = q3(a), q3(b)
    print(f"{name:10s} {tv:6.2f} | {qa[0]:7.2f} {qa[1]:7.2f} {qa[2]:7.2f}  | {qb[0]:9.2f} {qb[1]:9.2f} {qb[2]:9.2f}")
print("one common sigma for both data sets: median %.2f mm/yr" % np.median(np.exp(sc["log_sig1"])))
for name, rh, dv in [("mixed", rh_m, dv_m), ("joint", rh_j, dv_j), ("common sigma", rh_c, dv_c)]:
    print(f"convergence, {name:12s}: largest split R-hat {rh:.3f}, divergent transitions {dv} (4 chains x 1500 draws)")
print("NUTS runs done in %.0f s" % (time.time() - t0))

# %% [markdown]
# **Worked example (integrating the slip rates out by hand).** For one setting of $(D_1, D_2, \sigma_1, \sigma_2)$ the
# closed form needs only a 2 × 2 matrix $\mathbf A = \mathbf G^\top\mathbf C_d^{-1}\mathbf G + \mathbf I/\tau^2$ and the vector
# $\mathbf b = \mathbf G^\top\mathbf C_d^{-1}\mathbf d$. The cell prints them and checks the closed form against a brute-force
# sum of $p(\mathbf d\mid\dot s_1,\dot s_2)\,p(\dot s_1,\dot s_2)$ over a fine grid of slip rates.

# %%
D1w, D2w, sg1w, sg2w = 15.0, 10.0, 1.5, 4.5
Gw = np.asarray(G_lin(D1w, D2w)); cinvw = np.where(np.asarray(grp) == 0, 1 / sg1w ** 2, 1 / sg2w ** 2); dw = np.asarray(d4)
Aw = Gw.T @ (Gw * cinvw[:, None]) + np.eye(2) / TAU ** 2; bw = Gw.T @ (cinvw * dw)
print("A =\n", np.round(Aw, 4)); print("b =", np.round(bw, 4))
print("conditional mean of the slip rates A^-1 b =", np.round(np.linalg.solve(Aw, bw), 3), "mm/yr")
cf_val = float(log_marg_lin(D1w, D2w, sg1w, sg2w, d4))
g1 = np.linspace(20, 32, 401); g2 = np.linspace(9, 23, 401); dg1, dg2 = g1[1] - g1[0], g2[1] - g2[0]
S1g, S2gr = np.meshgrid(g1, g2, indexing="ij")
pred = S1g[..., None] * Gw[:, 0] + S2gr[..., None] * Gw[:, 1]
loglik = -0.5 * np.sum(cinvw * (dw - pred) ** 2, -1) - 0.5 * dw.size * np.log(2 * np.pi) + 0.5 * np.sum(np.log(cinvw))
logprior = -0.5 * (S1g ** 2 + S2gr ** 2) / TAU ** 2 - np.log(2 * np.pi * TAU ** 2)
mx = (loglik + logprior).max()
brute = mx + np.log(np.sum(np.exp(loglik + logprior - mx)) * dg1 * dg2)
print(f"closed-form ln p(d | D, sigma) = {cf_val:.4f};  brute-force sum over slip rates = {brute:.4f}")

# %%
fig, axs = plt.subplots(1, 3, figsize=(15, 4.2), gridspec_kw=dict(width_ratios=[1.4, 1, 1]))
ax = axs[0]
xx = np.linspace(-250, 250, 500)
idx = np.random.default_rng(0).choice(len(rates), 60, replace=False)
for k in idx:
    ax.plot(xx, vel(xx, rates[k, 0], rates[k, 1], np.asarray(sm["D1"])[k], np.asarray(sm["D2"])[k]), color=C_FULL, lw=0.5, alpha=0.35)
ax.plot(xi, np.asarray(d4)[25:], ".", color="#a3a29b", ms=3, label="InSAR-like (300, σ 4.5)")
ax.plot(xg, np.asarray(d4)[:25], "o", color=S.INK, ms=4.5, label="GPS-like (25, σ 1.5)")
ax.set_xlabel("distance from fault 2 (km)"); ax.set_ylabel("velocity (mm/yr)")
ax.set_title("Data and 60 posterior velocity profiles", fontsize=10.5); ax.legend(fontsize=8, loc="upper left")
ax = axs[1]
for k_, (lab, col) in enumerate([("log_sig1", S.INK), ("log_sig2", "#7c7b75")]):
    v = np.exp(np.asarray(sm[lab]))
    ax.hist(v, bins=60, density=True, color=col, alpha=0.75, label=f"σ{k_ + 1} (true {truth4['sig%d' % (k_ + 1)]})")
ax.hist(np.exp(np.asarray(sc["log_sig1"])), bins=40, density=True, color=C_CONV, alpha=0.6, label="one common σ")
ax.axvline(1.5, color=S.INK, lw=0.8); ax.axvline(4.5, color=S.INK, lw=0.8)
ax.set_xlabel("data noise scale σ (mm/yr)"); ax.set_ylabel("posterior density")
ax.set_title("Estimated data weights", fontsize=10.5); ax.legend(fontsize=8)
ax = axs[2]
ext4 = [[0.5, 40.0], [0.5, 40.0]]
for s_, col in [(sc, C_CONV), (sm, C_FULL)]:
    H, xe, ye = np.histogram2d(s_["D1"], s_["D2"], bins=80, range=ext4)
    H = gaussian_filter(H, 1.5)                                   # smoothed sample density
    lev = sorted(set(P.hdr_levels(H, (0.5, 0.9)).values()))
    ax.contour(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), H.T, levels=lev, colors=col, linewidths=[1.0, 1.9])
ax.plot(truth4["D1"], truth4["D2"], "*", color=S.INK, ms=13, mec=S.SURFACE, zorder=3)
ax.set_xlim(4, 30); ax.set_ylim(0.5, 36)
ax.set_xlabel("locking depth D1 (km)"); ax.set_ylabel("locking depth D2 (km)")
ax.legend(handles=BP.legend_handles([("separate σ1, σ2", C_FULL, "line"), ("one common σ", C_CONV, "line")]), fontsize=8,
          title="50 % (thick) and 90 % regions", title_fontsize=8)
ax.set_title("Locking depths", fontsize=10.5)
fig.tight_layout()
bk.savefig(fig, "2-30_data_weights")
plt.show()
for name, s_ in [("separate sigma_k", sm), ("one common sigma", sc)]:
    w1, w2_ = (np.subtract(*np.percentile(s_[k], [95, 5])) for k in ("D1", "D2"))
    print(f"width of the 90 % interval with {name:16s}: D1 {w1:.1f} km, D2 {w2_:.1f} km")

# %% [markdown]
# **How to read it.** The table above shows that integrating the slip rates out (mixed method, four sampled parameters)
# gives the same posterior as sampling all six parameters, and the worked example confirms the closed form against a
# brute-force sum. The estimated noise scales land near the values used to make the data. Forcing one common σ makes
# each GPS point count only as much as an InSAR point, so the locking depths spread more (right panel; widths printed).
#
# One realization again; the next cell repeats the inversion for new noise and counts how often the 90 % intervals of the
# four physical parameters contain the true values, for three weighting choices: separate σ_k estimated, one common σ
# estimated, and both σ fixed by hand at 3 mm/yr (between the two true values).

# %%
t0 = time.time()
R4 = 12
choices = [("separate σ_k (estimated)", "separate"), ("one common σ (estimated)", "common"), ("both fixed at 3 mm/yr", "fixed")]
hits = {c: np.zeros(4) for c, _ in choices}
widths = {c: [] for c, _ in choices}
worst = [1.0, 0]                                        # largest R-hat and total divergences over all runs
for rep in range(R4):
    dd = jnp.asarray(make_data(300 + rep))
    for cname, wkind in choices:
        s_, rh_, dv_ = run_nuts(model_mixed, seed=rep, d=dd, weights=wkind)
        worst = [max(worst[0], rh_), worst[1] + dv_]
        rt = slip_rate_draws(s_, dd, weights=wkind, seed=rep)
        vals = [s_["D1"], s_["D2"], rt[:, 0], rt[:, 1]]
        tv = [truth4["D1"], truth4["D2"], truth4["s1"], truth4["s2"]]
        w_ = []
        for k, (v, t_) in enumerate(zip(vals, tv)):
            lo, hi = np.percentile(v, [5, 95]); hits[cname][k] += (lo <= t_ <= hi); w_.append(hi - lo)
        widths[cname].append(w_)
print(f"{'weighting':28s} {'D1':>6s} {'D2':>6s} {'s1':>6s} {'s2':>6s}   (share of {R4} realizations whose 90 % interval holds the truth)")
for cname, _ in choices:
    print(f"{cname:28s} " + " ".join(f"{h / R4:6.2f}" for h in hits[cname]) +
          "   median 90 % widths: " + ", ".join(f"{v:.1f}" for v in np.median(np.array(widths[cname]), 0)))
print("over all %d runs: largest split R-hat %.3f, divergent transitions %d  (%.0f s)" % (3 * R4, worst[0], worst[1], time.time() - t0))
print("binomial standard error of a share near 0.90 with %d realizations: %.2f" % (R4, np.sqrt(0.9 * 0.1 / R4)))

# %% [markdown]
# With only 12 realizations each share scatters by the binomial error printed above, so only large departures mean
# something. Read the table in two steps. The hand-fixed weights give the narrowest intervals and miss the truth far
# more often than one time in ten: σ = 3 mm/yr over-trusts the 300 InSAR points (true scatter 4.5) and under-trusts the
# 25 GPS points (true scatter 1.5), and the intervals do not know it. The common σ is estimated, so its intervals are
# honest about the total misfit, but every GPS point now counts only as much as an InSAR point, and the intervals are
# wider than with separate σ_k. The separate σ_k let each data set count as much as its scatter justifies.
#
# **Common misreadings.**
# * "Weighting by the number of points is the fix." The number of points is already in the likelihood; what is unknown is
#   each set's noise level, and that is what σ_k estimates.
# * "The estimated σ_k is the instrument noise." Only if the model is right. Section 5.
#
# **In practice.** Joint inversions of GNSS, InSAR and other data sets estimate exactly these σ_k. Fukuda & Johnson
# (2010) sampled the nonlinear parameters and the σ_k with a Metropolis random walk (2.16) and integrated the linear ones
# out, as here; with NUTS (2.18) the same closed-form marginal enters as one `numpyro.factor` term (code above), which
# simply adds that log marginal likelihood to the log density the sampler explores.

# %% [markdown]
# ## 5. Estimated weights measure misfit, not instrument quality
#
# **In plain words.** The data weight is estimated from how badly the model misses each data set. Any miss counts: noise,
# but also anything the model cannot represent. If the InSAR data contain an orbital ramp that the model lacks, the
# shared fault parameters bend to absorb part of it, then miss the GPS data, and the GPS weight drops.
#
# **Precisely.** Add an offset and linear ramp, $1 + 0.02\,x$ mm/yr ($x$ in km), to the InSAR-like data only. Invert
# (a) without modelling it and (b) with offset and ramp as two extra linear parameters for the InSAR set (integrated out
# like the slip rates). Ten noise realizations.

# %%
t0 = time.time()
print("ramp over the InSAR points: rms %.2f mm/yr about its mean, %.2f mm/yr including its offset (InSAR noise sd 4.5)"
      % (np.sqrt(np.mean((ramp - ramp.mean()) ** 2)), np.sqrt(np.mean(ramp ** 2))))


def residual_rms(smp_, d, with_ramp):
    """GPS and InSAR residual rms at the posterior medians of D and sigma, linear parameters at their conditional mean."""
    s1m, s2m = np.exp(np.median(smp_["log_sig1"])), np.exp(np.median(smp_["log_sig2"]))
    G = np.asarray(G_lin(np.median(smp_["D1"]), np.median(smp_["D2"]), with_ramp)); g = np.asarray(grp); dd_ = np.asarray(d)
    cinv = np.where(g == 0, 1 / s1m ** 2, 1 / s2m ** 2)
    A = G.T @ (G * cinv[:, None]) + np.eye(G.shape[1]) / TAU ** 2
    r = dd_ - G @ np.linalg.solve(A, G.T @ (cinv * dd_))
    return np.sqrt(np.mean(r[g == 0] ** 2)), np.sqrt(np.mean(r[g == 1] ** 2))


res5, rms5, worst5 = [], [], [1.0, 0]
for rep in range(10):
    dd = jnp.asarray(make_data(200 + rep, with_ramp=True))
    row, rrow = [], []
    for wr in [False, True]:
        s_, rh_, dv_ = run_nuts(model_mixed, seed=rep, d=dd, with_ramp=wr)
        worst5 = [max(worst5[0], rh_), worst5[1] + dv_]
        row.append([np.median(np.exp(s_["log_sig1"])), np.median(np.exp(s_["log_sig2"])), np.median(s_["D1"]), np.median(s_["D2"])])
        rrow.append(residual_rms(s_, dd, wr))
    res5.append(row); rms5.append(rrow)
res5, rms5 = np.array(res5), np.array(rms5)
for k, lab in enumerate(["ramp ignored", "ramp modelled"]):
    med = np.median(res5[:, k], 0)
    print(f"{lab:14s}: median over 10 realizations of the posterior medians: sigma1 {med[0]:.2f}, sigma2 {med[1]:.2f} mm/yr,"
          f" D1 {med[2]:.1f} km, D2 {med[3]:.1f} km   (true 1.5, 4.5, 15, 10)")
    print(f"{'':14s}  residual rms (median over realizations): GPS {np.median(rms5[:, k, 0]):.2f}, InSAR {np.median(rms5[:, k, 1]):.2f} mm/yr")
print("InSAR residual rms if the whole ramp had stayed in the residuals: %.2f mm/yr"
      % np.sqrt(np.median(rms5[:, 1, 1]) ** 2 + np.mean(ramp ** 2)))
print("realizations where ignoring the ramp raised the GPS sigma1: %d of 10; deepened D1: %d of 10"
      % (np.sum(res5[:, 0, 0] > res5[:, 1, 0]), np.sum(res5[:, 0, 2] > res5[:, 1, 2])))
print("over all 20 runs: largest split R-hat %.3f, divergent transitions %d  (%.0f s)" % (worst5[0], worst5[1], time.time() - t0))

# %%
fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.0))
lab_ = ["ramp ignored", "ramp modelled"]
panels = [(0, "GPS noise scale σ1 / true value", 1.5), (1, "InSAR noise scale σ2 / true value", 4.5), (2, "locking depth D1 (km)", None)]
for k_, (j_, name, tv) in enumerate(panels):
    ax = axs[k_]
    y = res5[:, :, j_] / tv if tv else res5[:, :, j_]           # the two noise scales as ratios on one common axis
    for i_ in range(10):
        ax.plot([0, 1], y[i_], color=S.AXIS, lw=0.8)
    ax.plot(np.zeros(10), y[:, 0], "o", color=C_CONV, ms=6, mec=S.SURFACE)
    ax.plot(np.ones(10), y[:, 1], "o", color=C_FULL, ms=6, mec=S.SURFACE)
    ax.axhline(1.0 if tv else truth4["D1"], color=S.INK, lw=0.9)
    ax.text(1.38, 1.0 if tv else truth4["D1"], "true", fontsize=8, color=S.INK2, va="bottom", ha="right")
    ax.set_xticks([0, 1]); ax.set_xticklabels(lab_); ax.set_xlim(-0.4, 1.4)
    ax.set_title(name, fontsize=10.5)
lo_ = min(axs[0].get_ylim()[0], axs[1].get_ylim()[0]); hi_ = max(axs[0].get_ylim()[1], axs[1].get_ylim()[1])
axs[0].set_ylim(lo_, hi_); axs[1].set_ylim(lo_, hi_)
fig.suptitle("Ten noise realizations of data that contain an InSAR ramp (lines join the same realization)", fontsize=11, y=1.02)
fig.tight_layout()
bk.savefig(fig, "2-30_weights_absorb_misfit")
plt.show()

# %% [markdown]
# **How to read it.** Each line is one realization; the two noise scales are drawn as ratios to their true values on one
# axis. Ignoring the ramp (orange) raises the estimated GPS noise scale well above its true value and pushes the locking
# depth of fault 1 deeper, while the InSAR scale, whose data contain the ramp, hardly moves (medians printed above).
# Modelling the ramp (violet) brings all three back. The printed residuals show the mechanism. With the ramp ignored, the
# InSAR residual grows far less than it would if the ramp had stayed in the residuals, so the fault parameters have
# absorbed most of the ramp; the GPS residual grows, because the shifted fault parameters no longer fit the GPS points.
# The GPS weight is then lowered to accommodate a misfit that the GPS data did not cause.
#
# So an estimated σ_k is "how far the model misses this data set", which equals the instrument noise only when the model
# can represent everything in the data. This is the rule notebook 2.13 found from the other side: a noise scale fitted to
# the residuals responds to how large the misfit is, not to how the model error affects the parameters. Two remedies:
# model the systematic part explicitly (here the ramp), or describe it with a correlated prediction-error covariance
# $\mathbf C_p$ (notebook 2.13; Duputel et al. 2014), which a single scale factor per data set cannot represent.
#
# **In practice.** Many InSAR inversions estimate an offset and a planar ramp per interferogram as extra linear
# parameters, which is the first remedy shown here. Read the posterior σ_k as the misfit level of each data set given
# the model, not as data quality. A large σ_k for one
# data set can come from a model error that enters through another data set, and any relative weight formed from two
# σ_k inherits that.

# %% [markdown]
# ## 6. The hyperprior is part of the model
#
# **In plain words.** "Let the data decide" still needs a prior on the knobs, and "uniform" depends on the scale you call
# uniform. A prior that is flat in α² from 10⁻⁵ to 10⁵ puts most of its probability in the top decade (printed below), a
# strong preference for weak smoothing. A prior flat in ln α² treats every decade the same.
#
# **Precisely.** Three hyperpriors on the data scale: uniform in σ² (Fukuda & Johnson 2008: "we assume that p(σ², α²) is
# a uniform PDF", p. 1131; 2010: "We thus assume that p(σ, α) is constant if all the components of σ and α are positive",
# p. 1446, where their σ is the vector of the σ_k²), log-uniform in σ (scale-invariant: the same probability for every
# decade, so the answer does not depend on the units), and HalfCauchy(1) on the factor σ/σ_nominal (the positive half of
# a Cauchy distribution, 2.03: centred at zero with a heavy tail that still allows large values; a common weakly
# informative choice, Gelman 2006; σ_nominal = 2 mm here). Two on the smoothing knob over [10⁻⁵, 10⁵]: uniform in α², as in Zhao & Jiang (2026): "We set a
# uniform prior of [10⁻⁵, 10⁵] for α²" (p. 5), and uniform in ln α². Test: the antiplane fault in depth with 8 and with 40
# stations; one realization for the posteriors, then 200 realizations for coverage.

# %%
print("prior probability of alpha^2 > 1e4 under a uniform prior on [1e-5, 1e5]: %.3f" % ((1e5 - 1e4) / (1e5 - 1e-5)))
ls2_6 = np.linspace(np.log(0.01), np.log(400.0), 90)
la2_6 = np.linspace(np.log(1e-5), np.log(1e5), 161)
S2_6, A2_6 = make_grid(ls2_6, la2_6)
sig_nom = 2.0


def log_hyperprior(kind_s, kind_a):
    if kind_s == "uniform in σ²":
        lp_s = ls2_6[:, None] * 1.0
    elif kind_s == "log-uniform":
        lp_s = np.zeros((ls2_6.size, 1))
    else:                                                   # HalfCauchy(1) on k = sigma / sigma_nominal
        k = np.sqrt(np.exp(ls2_6)) / sig_nom
        lp_s = (np.log(k) - np.log1p(k ** 2))[:, None]
    lp_a = la2_6[None, :] * 1.0 if kind_a == "uniform in α²" else np.zeros((1, la2_6.size))
    return lp_s + lp_a


kinds_s = ["uniform in σ²", "log-uniform", "HalfCauchy(1) on σ/2 mm"]
kinds_a = ["uniform in α²", "uniform in ln α²"]
post6 = {}
for name, xs in cases.items():
    G = screw_G(xs, ze); mu, V = simdiag(G, DD)
    e6 = sig_true * np.random.default_rng(7).standard_normal(xs.size)
    d = G @ s_true2 + e6
    print(f"{name} (noise rms of this realization {np.sqrt(np.mean(e6 ** 2)):.2f} mm)")
    for ks in kinds_s:
        for ka in kinds_a:
            o = analyse(d, G, mu, V, S2_6, A2_6, log_prior=log_hyperprior(ks, ka), s_true=s_true2)
            pa, ps = o["w"].sum(0), o["w"].sum(1)
            qa = np.exp(grid_quantiles(pa, la2_6))
            qs = np.exp(0.5 * grid_quantiles(ps, ls2_6)[1])
            post6[name, ks, ka] = pa
            print(f"   σ prior {ks:24s} α² prior {ka:17s}: α² median {qa[1]:.3g} [90 %: {qa[0]:.2g}, {qa[2]:.2g}],"
                  f" σ median {qs:.2f} mm, mean slip sd {np.sqrt(o['vmix']).mean():.3f} m")

t0 = time.time()
cov6 = {}
for name, xs in cases.items():
    G = screw_G(xs, ze); mu, V = simdiag(G, DD); rng6 = np.random.default_rng(21)
    for ka in kinds_a:
        cv, sdv = [], []
        rng6 = np.random.default_rng(21)
        for rep in range(200):
            st = Lc_prior @ rng6.standard_normal(M); dd = G @ st + sig_true * rng6.standard_normal(xs.size)
            o = analyse(dd, G, mu, V, S2_6, A2_6, log_prior=log_hyperprior("log-uniform", ka), s_true=st, wfloor=1e-6)
            cv.append(o["cover_full"]); sdv.append(np.sqrt(o["vmix"]).mean())
        cov6[name, ka] = (np.mean(cv), np.mean(sdv))
        print(f"{name:12s} α² prior {ka:17s}: coverage of 90 % intervals {np.mean(cv):.3f}, mean slip sd {np.mean(sdv):.3f} m")
print("(%.0f s)" % (time.time() - t0))

# %% [markdown]
# **Weak data: where does the prior stop?** A log-uniform prior is improper at zero (it cannot be normalised; reminder:
# improper priors, 2.09 and 2.11): its integral over ln α² diverges as
# α² → 0, and only the end of the grid keeps it finite. That is harmless when the data rule out tiny α², but not when the
# data could be pure noise. The next cell takes the weakest set-up of section 2 (along-strike profile, 6 stations, 30 km
# kernel) and moves the lower end of the α² range down, for three hyperpriors: log-uniform; flat in α²; and a
# HalfCauchy prior on the scale α (not α²), which is proper at zero. Its scale is put at the value used to make the data,
# an oracle choice that no real inversion has, so scales ten times smaller and larger are printed too. The upper end
# stays where section 2 put it.

# %%
t0 = time.time()
xs_w = np.linspace(0.5, Ms - 0.5, 6)
K_w = np.exp(-0.5 * ((xs_w[:, None] - xa[None, :]) / 30.0) ** 2); K_w /= K_w.sum(1, keepdims=True)
mu_w, V_w = simdiag(K_w, DDs)
Lc_w = np.linalg.cholesky(a2s * np.linalg.inv(DDs) + 1e-12 * np.eye(Ms))
ls2_w = np.linspace(np.log(1e-4), np.log(10.0), 81)
lowers = [7, 10, 14]                                 # lower end of the alpha^2 range, in ln-units below the true value


def weak_coverage(lower, prior, scale=1.0, R=100, seed=300):
    la2_w = np.arange(np.log(a2s) - lower, np.log(a2s) + 7 + 1e-9, 14 / 110)
    S2w, A2w = make_grid(ls2_w, la2_w)
    if prior == "log-uniform":
        lp = 0.0
    elif prior == "flat in α²":
        lp = la2_w[None, :] * 1.0
    else:                                            # HalfCauchy on alpha, written as a density in ln alpha^2
        lp = (0.5 * la2_w - np.log1p(np.exp(la2_w) / (scale ** 2 * a2s)))[None, :]
    r_ = np.random.default_rng(seed)
    cf, cp, edge, low = [], [], [], []
    for _ in range(R):
        st = Lc_w @ r_.standard_normal(Ms)
        o_ = analyse(K_w @ st + 0.5 * r_.standard_normal(6), K_w, mu_w, V_w, S2w, A2w, log_prior=lp, s_true=st, wfloor=1e-6)
        cf.append(o_["cover_full"]); cp.append(o_["cover_plug"]); edge.append(o_["ij"][1] == 0)
        low.append(o_["w"].sum(0)[la2_w < np.log(a2s) - 4].sum())     # posterior share far below the true alpha^2
    return np.mean(cf), np.mean(cp), np.mean(edge), np.mean(low)


weak = {p: [weak_coverage(lo, p) for lo in lowers] for p in ["log-uniform", "flat in α²", "HalfCauchy on α"]}
for p, rows in weak.items():
    print(f"{p:16s}: coverage of the fully Bayesian 90 % intervals for lower ends " +
          ", ".join(f"-{lo}: {r[0]:.3f}" for lo, r in zip(lowers, rows)) +
          "; posterior share more than 4 ln-units below the true alpha^2: " + ", ".join(f"{r[3]:.2f}" for r in rows))
print("ABIC plug-in for the same lower ends: coverage " + ", ".join(f"{r[1]:.3f}" for r in weak["log-uniform"]) +
      "; share of realizations whose ABIC optimum is the lowest alpha^2: " + ", ".join(f"{r[2]:.2f}" for r in weak["log-uniform"]))
print("HalfCauchy on alpha with its scale 10 times smaller / larger (lower end -7): coverage %.3f / %.3f"
      % (weak_coverage(7, "HalfCauchy on α", 0.1)[0], weak_coverage(7, "HalfCauchy on α", 10.0)[0]))
G5 = screw_G(XS_AP[5], ze); mu5, V5 = simdiag(G5, DD)
for lo_s2 in [1e-2, 1e-4]:                           # the same effect for sigma^2 when the slip can fit the data exactly
    S2s, A2s_ = make_grid(np.arange(np.log(lo_s2), np.log(400.0) + 1e-9, (np.log(400.0) - np.log(1e-4)) / 80), la2)
    r_ = np.random.default_rng(5); meds = []
    for _ in range(200):
        st = Lc_prior @ r_.standard_normal(M)
        o_ = analyse(G5 @ st + sig_true * r_.standard_normal(5), G5, mu5, V5, S2s, A2s_, wfloor=1e-6)
        meds.append(np.exp(0.5 * grid_quantiles(o_["w"].sum(1), np.log(S2s[:, 0]))[1]))
    print(f"antiplane fault, 5 stations, log-uniform prior on sigma^2 down to {lo_s2:g} mm^2: median of the posterior medians of sigma {np.median(meds):.2f} mm (true 2)")
print("(%.0f s)" % (time.time() - t0))

# %%
fig, axs = plt.subplots(1, 4, figsize=(17, 4.0), gridspec_kw=dict(width_ratios=[1, 1, 0.75, 0.85]))
dx6 = (la2_6[1] - la2_6[0]) / np.log(10)
for ax, name in zip(axs[:2], cases):
    for ka, col in [("uniform in α²", C_CONV), ("uniform in ln α²", C_FULL)]:
        pa = post6[name, "log-uniform", ka]
        ax.plot(la2_6 / np.log(10), pa / dx6, color=col, lw=1.8, label=f"hyperprior {ka}")
    ax.axvline(np.log10(a2_true), color=S.INK, lw=0.9, label="value used for the data")
    ax.set_xlim(np.log10(a2_true) - 3, np.log10(a2_true) + 3)
    ax.set_xlabel("log10 α²"); ax.set_ylabel("posterior density")
    ax.set_title(f"{name}: posterior of α²", fontsize=10.5); ax.legend(fontsize=8, loc="upper left")
ax = axs[2]
xb = np.arange(2)
for k_, (ka, col) in enumerate([("uniform in α²", C_CONV), ("uniform in ln α²", C_FULL)]):
    ax.bar(xb + (k_ - 0.5) * 0.36, [cov6[n, ka][0] for n in cases], width=0.34, color=col, label=ka)
ax.axhline(0.9, color=S.INK, lw=0.9)
ax.set_xticks(xb); ax.set_xticklabels(list(cases)); ax.set_ylim(0.7, 1.06)
ax.set_ylabel("share inside the 90 % interval"); ax.set_title("Coverage, 200 realizations", fontsize=10.5)
ax.legend(fontsize=7.5, loc="upper center", ncol=2)
ax = axs[3]
lines_d = [(weak["log-uniform"], 0, C_FULL, "-o", "log-uniform"), (weak["flat in α²"], 0, C_CONV, "-s", "flat in α²"),
           (weak["HalfCauchy on α"], 0, S.AQUA, "-D", "HalfCauchy on α"), (weak["log-uniform"], 1, C_ABIC, ":x", "ABIC plug-in")]
for rows, col_i, col, fmt, lab in lines_d:
    yv = [r[col_i] for r in rows]
    ax.plot(lowers, yv, fmt, color=col, lw=1.6, ms=6, **({} if "x" in fmt else dict(mec=S.SURFACE)))
    ax.text(lowers[-1] + 0.4, yv[-1], lab, color=col, fontsize=8, va="center")         # label at the line's end
ax.axhline(0.9, color=S.INK, lw=0.9)
ax.set_xlim(6.3, 19.5)
ax.set_xticks(lowers); ax.set_xticklabels([f"−{lo}" for lo in lowers])
ax.set_xlabel("lower end of the α² range (ln-units below the true α²)"); ax.set_ylabel("share inside the 90 % interval")
ax.set_title("Weakest set-up: 6 stations, 30 km kernel", fontsize=10.5)
fig.tight_layout()
bk.savefig(fig, "2-30_hyperpriors")
plt.show()

# %% [markdown]
# **How to read it.** (a, b) The data-noise prior matters only when data are few (σ medians printed above). The
# smoothing-knob prior matters even with 40 stations: a prior flat in α² drags the posterior towards weaker smoothing and
# inflates the slip uncertainty (printed). The reason is that the data constrain α² only loosely (the 90 % ranges printed
# above span large factors), and across such a range a prior flat in α² has a density proportional to α² on the
# logarithmic axis, which tilts the posterior towards the upper end. (c) In this test, where the true slip is drawn from
# the smoothing prior, the log-uniform choice gives close to nominal coverage and the flat-in-α² choice over-covers.
# Over-coverage means intervals that are too wide, which is safer than too narrow but still a statement about the prior,
# not the data.
#
# (d) With the weakest data the log-uniform coverage keeps falling as the lower end of its range moves down: the
# posterior piles up at tiny α² (the printed share far below the true value grows), that is, at nearly zero slip with a
# small uncertainty. ABIC fails in its own way here: in a share of the realizations (printed) its optimum is the lowest
# α² on offer, which means nearly zero slip with nearly zero uncertainty. The flat prior ignores the lower end but over-covers strongly. The HalfCauchy prior does not depend on the
# lower end, and at the oracle scale its coverage is close to 90 %, but it moves far from 90 % when the scale is ten times
# off (printed). With data this weak no hyperprior is neutral: the answer is mostly the prior. The same happens to σ²
# when there are fewer data than slip patches. The slip can then fit the data exactly, the likelihood stays finite as
# σ² → 0, and under a log-uniform prior the posterior of σ depends on where its range stops (printed for 5 stations).
#
# **Common misreadings.**
# * "A uniform hyperprior is uninformative." Uniform in which variable? Over ten decades, uniform in α² is very
#   informative about α².
# * "The hyperprior washes out with enough data." Only for knobs the data pin down. The smoothing knob of a slip model is
#   informed by the handful of resolved slip combinations (section 2), so its prior keeps mattering.
# * "A grid or sampler range is a technical detail." For a prior that is improper at one end, the end of the range *is*
#   the prior there, and with weak data it decides the answer (panel d).
#
# **In practice.** Fukuda & Johnson (2008, 2010) use priors flat in σ² and α²; Zhao & Jiang (2026) use the prior flat
# in α² quoted above (notebook 2.31 examines their method). A log-normal prior on a scale is proper at both ends, so the
# range problem of panel (d) cannot arise, but its median and width are choices a swap test should check. In this test
# HalfCauchy(1) on σ/σ_nominal behaves like the log-uniform prior for σ (σ medians printed above). Gelman (2006)
# discusses the failure of log-uniform priors for hierarchical scale parameters and recommends half-Cauchy priors on the
# scale instead. Whichever is used, state it and check whether the answer moves when it is swapped, as done here.

# %% [markdown]
# ## 7. What "objective" buys and what it does not
#
# | approach | chosen by the analyst | estimated from the data | uncertainty of the knobs |
# |---|---|---|---|
# | hand-tuned / L-curve (2.29) | λ (by eye), smoothing operator, noise model | nothing | ignored |
# | ABIC (empirical Bayes) | smoothing operator, noise model | σ², α² as single best values | ignored; wrong formula under positivity |
# | fully Bayesian (hierarchical) | smoothing operator, noise model, **hyperpriors** | σ², α² as distributions | propagated into the slip |
# | + data weights σ_k | grouping of the data, independence of errors | one scale per data set | propagated, but σ_k absorb model error |
#
# The fully Bayesian approach is the most complete of these, but every row still rests on choices: the form of the smoothing
# prior (is slip really smooth?), the error model (independent Gaussian? correlated? model errors?), and the hyperpriors.
# How much those choices matter depends on the data. Where the data resolve many slip combinations, ABIC and the fully
# Bayesian answer agree (section 2). Where they resolve few, the plug-in is overconfident and the fully Bayesian answer
# leans on the hyperprior (section 6), until with the weakest data the answer is mostly the prior; objectivity is then
# something the data cannot provide. The next notebook asks the first of these questions directly: should slip be
# smoothed at all?
#
# **In practice.**
# * Sampling the smoothing scale (often non-centered, $\mathbf m = \alpha\mathbf u$) with one sampled σ_k per data set is the
#   fully Bayesian row with data weights. Notebook 2.33 adds a caution: with strong data the centered form of the scale
#   can be the better-behaved one, so check divergences in both forms.
# * A prior whose length scale and smoothness are held fixed in a run (for example a Matérn prior) leaves their
#   uncertainty out of the error bars. Choosing them from a grid of runs by the evidence is the ABIC row of the table;
#   choosing them by eye is the hand-tuned row.
# * Read every estimated weight as a misfit level (section 5), and compare fits per data set by WRMS and χ² (2.29,
#   section 7).
#
# Next: **2.31 · To smooth or not**.
