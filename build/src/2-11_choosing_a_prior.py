# %% [markdown]
# # 2.11 · Choosing a prior: families, shapes and prior-predictive checks
#
# **Before this:** 2.03 (distributions), 2.08 (prior × likelihood), 2.09 (smoothing priors).
# **You will be able to:** match a prior family to what you actually know (sign, scale, tails, a fixed total,
# smoothness); see why a "flat" prior is flat only in one parameterisation; handle a parameter with two bounds
# through the logit transform; check a prior before fitting by simulating
# data from it, and revise it; tell, patch by patch, whether an answer comes from the data or from the prior.
# **Optional sections:** 7 (improper priors).
#
# A prior is part of the model, like the forward model and the noise model. It is where everything known before the
# data enters: that slip on a right-lateral fault is positive, that a viscosity spans orders of magnitude, that a
# seismic moment fixes the total, that slip varies smoothly. 2.09 showed priors acting as regularisers; this notebook
# shows the menu of prior families, what each one assumes, and the habit that makes priors defensible: simulate from
# the prior before looking at the fit.
#
# **Words this notebook defines:** prior family, Gaussian (normal), Laplace, Student-t, uniform, half-normal,
# truncated normal, log-normal, Dirichlet, concentration parameter, parameterisation, Jacobian, logit (log-odds),
# sigmoid (logistic function), logistic distribution, logit-normal, prior-predictive
# distribution, prior-predictive check, shrinkage, improper prior.

# %% [markdown]
# **Problems in this notebook**
#
# | problem | unknowns | data | forward model | linear? |
# |---|---|---|---|---|
# | one number (sections 1, 3) | 1 (a slip value or a viscosity) | none: prior only | none | n/a |
# | slip profile, prior draws (sections 2, 4) | 30 slip patches, 1 km each, 0-30 km | none: prior only | none | n/a |
# | antiplane fault (sections 5-7, as in 2.07) | 30 slip patches | 24 stations, σ = 3 mm | screw-dislocation Green's functions, $\mathbf d = \mathbf G\mathbf s$ | linear |

# %%
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path.cwd().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats as st
import bayeskit as bk
from bayeskit import diagnostics as D, style as S
import numpyro
numpyro.set_host_device_count(4)                 # 4 NUTS chains in parallel (before any JAX computation)
import jax, jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

T0 = time.time()
np.set_printoptions(precision=4, suppress=True)
rng = np.random.default_rng(2)

# %% [markdown]
# ## 1. One number, many priors
#
# **In plain words.** A prior family is a shape for "what values are plausible before the data". The families below
# differ in three things that matter in practice: whether negative values are allowed, how fast very large values
# become implausible (the **tails**), and whether the plausible range is set in absolute terms ("about 1 m, give or take
# 1 m") or in ratios ("somewhere between 0.1 and 10 m").
#
# **Precisely.** For one number $x$ (the **support** of a distribution is the set of values it allows; "up to a
# constant" means the normalising factor that makes the total probability 1 is left out):
#
# | family | density (up to a constant) | support | use it when you know |
# |---|---|---|---|
# | **Gaussian** = **normal** $\mathcal N(\mu,\sigma^2)$ | $e^{-(x-\mu)^2/2\sigma^2}$ | all $x$ | a typical value and a typical deviation; large deviations very unlikely |
# | **Laplace** | $e^{-\lvert x-\mu\rvert/b}$ | all $x$ | most values small, a few large (its MAP gives L1/sparsity, 2.09) |
# | **Student-t** ($\nu$ degrees of freedom) | $(1 + (x-\mu)^2/\nu s^2)^{-(\nu+1)/2}$ | all $x$ | a typical scale, but occasional much larger values (heavy tails; robust) |
# | **uniform** $\mathcal U(a,b)$ | 1 inside $[a,b]$ | $[a,b]$ | hard bounds and nothing else |
# | **half-normal** | $e^{-x^2/2\sigma^2}$ | $x\ge 0$ | the sign, and a scale |
# | **truncated normal** | $e^{-(x-\mu)^2/2\sigma^2}$ | $x\ge 0$ (or any bounds) | a sign, a typical value and a deviation |
# | **log-normal** | $\frac1x e^{-(\ln x-\mu)^2/2s^2}$ | $x>0$ | an order of magnitude: $\ln x$ is Gaussian |
#
# "Gaussian" and "normal" are two names for the same distribution (after Gauss; "normal" became common later). There is
# no difference between them. The Student-t's **degrees of freedom** $\nu$ set how heavy its tails are (heavy tails,
# 2.03: large values far more likely than a Gaussian allows): small $\nu$ gives very heavy tails, and as $\nu$ grows it
# approaches the Gaussian. The log-normal's **log-sd** is the standard deviation of $\ln x$: a log-sd of 1 means a
# typical spread of a factor $e\approx 2.7$ either side of the median.
#
# **Worked example.** Symmetric families scaled to the same standard deviation (1), and positive families scaled to
# the same median (1), so that only the shape differs. The code prints the tail probabilities.

# %%
x = np.linspace(-6, 6, 2401)
sym = {"Gaussian (normal)": st.norm(0, 1), "Laplace": st.laplace(0, 1 / np.sqrt(2)),
       "Student-t, ν = 3": st.t(3, 0, 1 / np.sqrt(3)), "uniform": st.uniform(-np.sqrt(3), 2 * np.sqrt(3))}
xp = np.linspace(1e-4, 6, 2401)
hn_scale = 1 / st.halfnorm.median()
pos = {"half-normal": st.halfnorm(0, hn_scale), "truncated normal N(1, 1), x > 0": st.truncnorm(-1, np.inf, 1, 1),
       "log-normal (median 1, log-sd 1)": st.lognorm(1.0, 0, 1.0)}
print("symmetric families, all with standard deviation 1:")
for k, dd in sym.items():
    print(f"  {k:24s} sd {dd.std():.3f}   P(|x| > 2) = {2 * dd.sf(2):.4f}   P(|x| > 3) = {2 * dd.sf(3):.5f}   P(|x| > 5) = {2 * dd.sf(5):.2e}")
print("positive families:")
for k, dd in pos.items():
    print(f"  {k:34s} median {dd.median():.3f}   P(x < 0.1) = {dd.cdf(0.1):.4f}   P(x > 3) = {dd.sf(3):.4f}   P(x > 10) = {dd.sf(10):.2e}")

# %%
cols = [S.INK, S.BLUE, S.ORANGE, S.AQUA]
fig, axs = plt.subplots(1, 3, figsize=(15.5, 3.9))
for (k, dd), c in zip(sym.items(), cols):
    axs[0].plot(x, dd.pdf(x), color=c, lw=1.8, label=k)
    axs[1].semilogy(x, np.maximum(dd.pdf(x), 1e-12), color=c, lw=1.8, label=k)
axs[0].set_xlim(-4.5, 4.5); axs[0].set_xlabel("x (standard deviations)"); axs[0].set_ylabel("density")
axs[0].set_title("(a) Same spread, different shapes"); axs[0].legend(fontsize=7.6, loc="upper right")
axs[1].set_ylim(1e-8, 2); axs[1].set_xlim(0, 6); axs[1].set_xlabel("x (standard deviations)"); axs[1].set_ylabel("density (log scale)")
axs[1].set_title("(b) The tails decide what 'large' means")
for (k, dd), c in zip(pos.items(), [S.BLUE, S.ORANGE, S.VIOLET]):
    axs[2].plot(xp, dd.pdf(xp), color=c, lw=1.8, label=k)
axs[2].axvline(1, color=S.AXIS, lw=0.8, ls=":")
axs[2].set_xlim(0, 5); axs[2].set_ylim(0, 1.6)
axs[2].set_xlabel("x (positive quantity)"); axs[2].set_ylabel("density")
axs[2].set_title("(c) Positive quantities (same median, dotted)"); axs[2].legend(fontsize=7.6, loc="upper right")
fig.tight_layout()
bk.savefig(fig, "2-11_families_1d")
plt.show()

# %% [markdown]
# **How to read it.** (a) With the same standard deviation the four symmetric shapes look similar near zero. (b) On a
# log scale they are not: the Gaussian tail falls fastest, Laplace falls as a straight line, the Student-t tail stays
# high, and the uniform has no tail at all (printed: the probability of a value beyond 5 standard deviations differs
# by many orders of magnitude). A Gaussian prior says a 5-sd value is essentially impossible; a Student-t says it is
# merely unusual. (c) Positive families with the same median: the half-normal is largest at zero, the truncated normal
# has a preferred value, the log-normal is skewed with a long right tail.
#
# **Common misreadings.**
# * *"Gaussian and normal priors are different choices."* Same distribution, two names.
# * *"If two priors have the same mean and standard deviation, they say the same thing."* They agree on typical values
#   and disagree, sometimes by orders of magnitude, on large ones (b), and large values are what outliers and sharp
#   slip features are made of.

# %% [markdown]
# ## 2. The same families on a slip profile
#
# **In plain words.** Put one prior on each of 30 patches, independently, and draw whole slip profiles. The pictures
# show what each family "believes" a fault looks like before any data. A smooth prior (correlated patches, 2.09, 2.12)
# is added for comparison.
#
# **Precisely.** Independent priors on each patch: $p(\mathbf s) = \prod_j p(s_j)$. The smooth prior is a Gaussian
# with covariance $C_{ij} = \sigma^2\exp(-\tfrac12(z_i-z_j)^2/\ell^2)$ (squared exponential, $\ell$ = 5 km; 2.12
# explains the family). All have a typical slip of order 1 m.

# %%
Nf = 30
z_edges = np.linspace(0.0, 30.0, Nf + 1)
zc = 0.5 * (z_edges[1:] + z_edges[:-1])
wz = np.diff(z_edges)                                                       # patch widths (km)


def sq_exp_cov(ell, sd):
    dz = zc[:, None] - zc[None, :]
    return sd ** 2 * np.exp(-0.5 * (dz / ell) ** 2) + 1e-9 * np.eye(Nf)


ndraw = 4
r2 = np.random.default_rng(5)
draws2 = {
    "Gaussian, sd 1 m": r2.normal(0, 1, (ndraw, Nf)),
    "Laplace, sd 1 m": r2.laplace(0, 1 / np.sqrt(2), (ndraw, Nf)),
    "Student-t ν = 3, scale 0.58 m": st.t(3, 0, 1 / np.sqrt(3)).rvs((ndraw, Nf), random_state=r2),
    "half-normal, scale 1 m": np.abs(r2.normal(0, 1, (ndraw, Nf))),
    "log-normal, median 1 m": np.exp(r2.normal(0, 1, (ndraw, Nf))),
    "smooth Gaussian, sd 1 m, ℓ = 5 km": r2.multivariate_normal(np.zeros(Nf), sq_exp_cov(5.0, 1.0), ndraw),
}
big = {k: r2.normal(0, 1, (20000, Nf)) for k in ["Gaussian"]}
lap = r2.laplace(0, 1 / np.sqrt(2), (20000, Nf))
print("share of patch values within 0.05 sd of zero: Gaussian %.3f, Laplace %.3f (a sparse vector would have most of them there)"
      % (np.mean(np.abs(big["Gaussian"]) < 0.05), np.mean(np.abs(lap) < 0.05)))
t3 = st.t(3, 0, 1 / np.sqrt(3)).rvs((20000, Nf), random_state=r2)
print("largest |slip| in a 30-patch draw, median over 20,000 draws: Gaussian %.2f m, Laplace %.2f m, Student-t %.2f m"
      % (np.median(np.abs(big["Gaussian"]).max(1)), np.median(np.abs(lap).max(1)), np.median(np.abs(t3).max(1))))


def step(ax, m, **kw):
    ax.plot(np.repeat(m, 2), np.c_[z_edges[:-1], z_edges[1:]].ravel(), **kw)


# %%
fig, axs = plt.subplots(1, 6, figsize=(17, 4.0), sharey=True)
shades = [S.INK, S.BLUE, S.ORANGE, S.AQUA]
for ax, (k, X) in zip(axs, draws2.items()):
    for i in range(ndraw):
        step(ax, X[i], color=shades[i], lw=1.0, alpha=0.85)
    ax.axvline(0, color=S.AXIS, lw=0.8)
    ax.set_title(k, fontsize=9)
    ax.set_xlabel("slip (m)")
axs[0].invert_yaxis(); axs[0].set_ylabel("depth (km)")
for ax in axs:
    ax.set_xlim(-4, 6)
fig.suptitle("Four draws from each prior: what each one believes a fault looks like before the data", fontsize=11, y=1.02)
fig.tight_layout()
bk.savefig(fig, "2-11_families_profiles")
plt.show()

# %% [markdown]
# **How to read it.** Independent priors give jagged profiles, because nothing ties one patch to the next. The Laplace
# draws are not sparse: the share of values near zero is only somewhat larger than for the Gaussian (printed); the
# sparse solutions of 2.09 come from the MAP, not from the prior's draws. The Student-t has occasional big spikes
# (printed largest values). Half-normal and log-normal are positive everywhere; the log-normal has rare large patches.
# Only the smooth Gaussian looks like the slip models in papers, and that is a modelling decision, not a fact about
# faults.
#
# **Common misreadings.**
# * *"A Laplace prior believes slip is sparse."* Its draws are not sparse (printed). Its MAP is.
# * *"Independent priors are the neutral choice."* They are a strong statement: neighbouring patches have nothing to
#   do with each other.

# %% [markdown]
# ## 3. Flat is not neutral: priors depend on the parameterisation
#
# **In plain words.** "I know nothing, so I use a uniform prior" sounds neutral, but uniform in *what*? A viscosity
# between $10^{18}$ and $10^{21}$ Pa s that is uniform in the viscosity itself puts almost all its probability in the
# top decade. Uniform in the logarithm puts equal probability in each decade. They are different beliefs, and with weak
# data they give different answers.
#
# **Precisely.** If $x$ has density $p_x(x)$ and $y = g(x)$ is a one-to-one transformation, then
# $p_y(y) = p_x(g^{-1}(y))\,\lvert dg^{-1}/dy\rvert$. The factor $\lvert dg^{-1}/dy\rvert$ is the **Jacobian**. For
# $u = \log_{10}\eta$ and $\eta$ uniform on $[a,b]$: $p_u(u) = \ln(10)\,10^{u}/(b-a)$, which rises steeply with $u$.
# The same rule is why a MAP changes when you change variables (2.08): densities, and hence their peaks, move.
#
# **Worked example.** 200,000 draws of a viscosity, uniform in η and uniform in log η, both on $[10^{18}, 10^{21}]$ Pa s.

# %%
a_eta, b_eta = 1e18, 1e21
n3 = 200_000
eta_lin = np.random.default_rng(3).uniform(a_eta, b_eta, n3)
eta_log = 10 ** np.random.default_rng(4).uniform(18, 21, n3)
for nm, e_ in [("uniform in eta", eta_lin), ("uniform in log10 eta", eta_log)]:
    shares = [np.mean((e_ >= 10 ** k) & (e_ < 10 ** (k + 1))) for k in (18, 19, 20)]
    print(f"{nm:22s}: share per decade 1e18-1e19 {shares[0]:.4f}, 1e19-1e20 {shares[1]:.4f}, 1e20-1e21 {shares[2]:.4f};"
          f"  median {np.median(e_):.2e} Pa s")
print("exact share of the top decade under uniform-in-eta: (1e21 - 1e20) / (1e21 - 1e18) = %.4f" % ((1e21 - 1e20) / (1e21 - 1e18)))

# %%
u = np.linspace(18, 21, 400)
fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.8))
ax = axs[0]
bins = np.linspace(18, 21, 61)
ax.hist(np.log10(eta_lin), bins=bins, density=True, color=S.ORANGE, alpha=0.45, label="draws: uniform in η")
ax.plot(u, np.log(10) * 10 ** u / (b_eta - a_eta), color=S.ORANGE, lw=2, label="Jacobian formula ln(10)·10ᵘ/(b − a)")
ax.hist(np.log10(eta_log), bins=bins, density=True, color=S.BLUE, alpha=0.45, label="draws: uniform in log₁₀ η")
ax.axhline(1 / 3, color=S.BLUE, lw=2, label="1/3 per unit of log₁₀ η")
ax.set_xlabel("log₁₀ viscosity (Pa s)"); ax.set_ylabel("density of log₁₀ η")
ax.set_title("(a) Two 'flat' priors, seen on the log axis")
ax.legend(fontsize=7.5, loc="upper left")
ax = axs[1]
dec = np.arange(3)
w_ = 0.36
lin_sh = [np.mean((eta_lin >= 10.0 ** (18 + k)) & (eta_lin < 10.0 ** (19 + k))) for k in dec]   # 10.0: integer 10**19 overflows
log_sh = [np.mean((eta_log >= 10.0 ** (18 + k)) & (eta_log < 10.0 ** (19 + k))) for k in dec]
ax.bar(dec - w_ / 2, lin_sh, w_, color=S.ORANGE, label="uniform in η")
ax.bar(dec + w_ / 2, log_sh, w_, color=S.BLUE, label="uniform in log₁₀ η")
ax.set_xticks(dec); ax.set_xticklabels(["10¹⁸–10¹⁹", "10¹⁹–10²⁰", "10²⁰–10²¹"])
ax.set_xlabel("viscosity decade (Pa s)"); ax.set_ylabel("prior probability")
ax.set_title("(b) Probability per decade")
ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
bk.savefig(fig, "2-11_flat_not_neutral")
plt.show()

# %% [markdown]
# **How to read it.** (a) On a log axis the uniform-in-η prior is not flat at all: its density rises tenfold per
# decade, and the histogram of draws follows the Jacobian formula. The uniform-in-log prior is flat on this axis.
# (b) The uniform-in-η prior puts most of its probability (printed) in the top decade, so it is a strong belief in
# high viscosity, not ignorance. For a quantity known only to an order of magnitude, a log-uniform or log-normal prior
# says what you mean.
#
# **Common misreadings.**
# * *"A uniform prior is uninformative."* It is uniform in one parameterisation and strongly informative in others.
# * *"The prior only matters if the data are weak."* True, and weakly constrained parameters (deep slip, viscosity,
#   friction) are exactly where the choice shows (section 6).

# %% [markdown]
# ### Two-sided bounds: the logit transform
#
# **In plain words.** The logarithm handles a quantity with one bound (positive). Many parameters have two: a fraction
# between 0 and 1 (the share of the plate rate a fault is locked, a probability, a share of a fixed total), a slip
# between zero and a physical maximum, a rake within a range. The **logit** stretches such an interval onto the whole
# number line, and its inverse, the **sigmoid** (also called the **logistic function**), squeezes the line back into the
# interval. Samplers like NUTS, ensemble Kalman updates, optimisers and neural networks all take steps in unbounded
# space, so they can step outside a bound. Working in the stretched variable instead, every step lands inside the bounds.
#
# **Precisely.** For $a < x < b$, the logit (the logarithm of the **odds**, $\frac{x-a}{b-x}$; for $a=0$, $b=1$ the odds
# of a probability) and the sigmoid are
# $$\xi = \ln\frac{x-a}{b-x}, \qquad x = a + (b-a)\,\sigma(\xi), \qquad \sigma(\xi) = \frac{1}{1+e^{-\xi}} .$$
# The midpoint of the interval maps to $\xi = 0$, the bounds to $\pm\infty$. The Jacobian (above) is
# $dx/d\xi = (b-a)\,\sigma(\xi)\,[1-\sigma(\xi)] = (x-a)(b-x)/(b-a)$, so
# $$p_\xi(\xi) = p_x(x)\,\frac{(x-a)(b-x)}{b-a}, \qquad p_x(x) = p_\xi(\xi)\,\frac{b-a}{(x-a)(b-x)} .$$
# Two consequences, checked below. A **uniform** prior on $(a,b)$ becomes the standard **logistic distribution** in
# $\xi$, a bell with standard deviation $\pi/\sqrt3$. And a **Gaussian** prior on $\xi$, $\xi \sim \mathcal N(0, s^2)$,
# becomes a **logit-normal** distribution on $x$ whose shape depends on $s$: one peak in the middle while $s^2 < 2$, two
# peaks pressed against the bounds once $s^2 > 2$.
#
# **Worked example.** A fraction $x$ on $(0, 1)$: 200,000 uniform draws taken to $\xi$, and Gaussian priors on $\xi$ with
# three spreads taken back to $x$.

# %%
logit = lambda x: np.log(x) - np.log1p(-x)
sigmoid = lambda z: 1.0 / (1.0 + np.exp(-z))
x_u = np.random.default_rng(31).uniform(size=200_000)
xi_u = logit(x_u)
print(f"uniform x on (0, 1) -> xi: sd {xi_u.std():.3f} (logistic theory pi/sqrt(3) = {np.pi / np.sqrt(3):.3f}), "
      f"median {np.median(xi_u):.3f}")
S_LN = [0.5, 1.4, 3.0]
xg = np.linspace(1e-6, 1 - 1e-6, 200_001)
for s_ in S_LN:
    dens = st.norm(0, s_).pdf(logit(xg)) / (xg * (1 - xg))                      # logit-normal density on (0, 1)
    peak_mid = np.isclose(dens[len(xg) // 2], dens[(xg > 0.01) & (xg < 0.99)].max(), rtol=1e-6)
    xs_ = sigmoid(np.random.default_rng(32).normal(0, s_, 200_000))
    print(f"xi ~ N(0, {s_}^2) (s^2 = {s_ * s_:.2f}): x has {'one peak at 0.5' if peak_mid else 'two peaks near 0 and 1'}; "
          f"share of x within 0.05 of a bound {np.mean((xs_ < 0.05) | (xs_ > 0.95)):.3f} (uniform: 0.100)")
q_ = np.random.default_rng(33).normal(0.7, 1.2, 200_000)                      # some posterior in xi
print(f"median: sigmoid(median xi) = {sigmoid(np.median(q_)):.4f}, median of sigmoid(xi) = {np.median(sigmoid(q_)):.4f}; "
      f"mean: sigmoid(mean xi) = {sigmoid(q_.mean()):.4f}, mean of sigmoid(xi) = {sigmoid(q_).mean():.4f}")

# %%
fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.0))
ax = axs[0]
z_ = np.linspace(-7, 7, 400)
ax.plot(z_, sigmoid(z_), color=S.INK, lw=2)
for yb in (0, 1):
    ax.axhline(yb, color=S.MUTED, lw=0.8, ls=":")
ax.axvline(0, color=S.AXIS, lw=0.8); ax.axhline(0.5, color=S.AXIS, lw=0.8)
ax.annotate("the whole line ξ ...", (4.5, 0.18), ha="center", fontsize=8.5, color=S.INK2)
ax.annotate("... squeezed into (0, 1)", (-3.6, 0.82), ha="center", fontsize=8.5, color=S.INK2)
ax.set_xlabel("ξ = logit(x)  (unbounded)"); ax.set_ylabel("x = sigmoid(ξ)  (bounded)")
ax.set_title("(a) The sigmoid and its inverse, the logit")
ax = axs[1]
ax.hist(xi_u, bins=np.linspace(-8, 8, 81), density=True, color=S.BLUE, alpha=0.45, label="uniform x, seen in ξ")
ax.plot(z_, st.logistic.pdf(z_), color=S.BLUE, lw=2, label="logistic density")
ax.plot(z_, st.norm(0, np.pi / np.sqrt(3)).pdf(z_), color=S.MUTED, lw=1.2, ls="--", label="Gaussian, same sd")
ax.set_xlabel("ξ"); ax.set_ylabel("density"); ax.set_xlim(-8, 8)
ax.set_title("(b) A uniform prior on x is a bell in ξ")
ax.legend(fontsize=7.5, loc="upper left")
ax = axs[2]
xp = np.linspace(0.002, 0.998, 2000)                    # plot just inside the bounds (the density is extreme there)
for s_, col in zip(S_LN, [S.AQUA, S.ORANGE, S.BAD]):
    dens = st.norm(0, s_).pdf(logit(xp)) / (xp * (1 - xp))
    ax.plot(xp, dens, color=col, lw=2, label=f"ξ ~ N(0, {s_}²)")
ax.axhline(1, color=S.BLUE, lw=1.2, ls="--", label="uniform on (0, 1)")
ax.set_ylim(0, 3.7); ax.set_xlim(0, 1)
ax.set_xlabel("x (bounded quantity)"); ax.set_ylabel("density of x")
ax.set_title("(c) A Gaussian in ξ, seen in x")
ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=2)
fig.tight_layout()
bk.savefig(fig, "2-11_logit")
plt.show()

# %% [markdown]
# **How to read it.** (a) The sigmoid is steep near the middle and flat near the bounds: equal steps in ξ move x a lot in
# the middle and very little near 0 or 1. (b) Uniform draws of x, taken to ξ, follow the logistic bell (its standard
# deviation printed above matches $\pi/\sqrt3$). It looks Gaussian but has heavier tails than the Gaussian with the same
# spread (dashed). (c) The reverse direction: a narrow Gaussian in ξ (s = 0.5) says "x is near the middle"; at s = 1.4,
# just below $s^2 = 2$, x is broad and nearly flat in the middle but still thinner than uniform near the bounds; at
# s = 3 the probability piles up against the two bounds (the shares within 0.05 of a bound are printed above). A "wide, vague" Gaussian in logit space is therefore a strong
# belief that x is close to 0 or to 1.
#
# **When you need it.**
# * **Samplers (NUTS, HMC).** NumPyro applies this map automatically for a prior with two bounds, including the Jacobian,
#   so you write the prior on x and it samples ξ (2.17). Writing your own sampler, you must add the Jacobian term
#   $\ln[(x-a)(b-x)/(b-a)]$ to the log density, or the prior silently changes.
# * **Ensemble Kalman methods and optimisers.** The update happens in ξ, so the members never leave the bounds (2.24).
#   The prior the ensemble starts from is the Gaussian in ξ, so choose its spread knowing panel (c).
# * **Neural networks.** A network outputs an unbounded number; a sigmoid on the output keeps a predicted fraction or
#   bounded parameter legal (3.07).
# * **Reporting.** Transform the draws back to x and report quantiles of x. Quantiles (the median, the 5 % and 95 %
#   points) pass through the transform unchanged, because the sigmoid is monotonic; means and the MAP do not (printed
#   above, and 2.08).
#
# **Common misreadings.**
# * *"A wide Gaussian prior in logit space is uninformative."* Above $s^2 = 2$ it piles probability against the bounds.
# * *"The logit only changes coordinates, so it changes nothing."* The parameter's distribution is unchanged only if the
#   Jacobian is included. A prior that looks the same in ξ and in x is two different priors.
# * *"The mean in ξ, transformed, is the mean of x."* No; only quantiles transform this way.

# %% [markdown]
# ## 4. The Dirichlet prior: sharing a fixed total
#
# **In plain words.** Sometimes you know a total better than its parts. The **seismic moment**
# $M_0 = \mu\times\text{area}\times\text{average slip}$ ($\mu$ the shear modulus, the rock's stiffness), measured
# from seismic waves, is an example: it fixes the total slip times area (the **potency**; on this 2-D fault, the sum
# of slip × patch width, in m·km) much better than a geodetic network constrains each patch. A **Dirichlet** prior describes how a total of 1 is shared among $K$ parts: every share is
# non-negative and the shares add to one. Multiply by the total and you have a positive slip profile with a known sum.
#
# **Precisely.** $\mathbf w\sim\text{Dirichlet}(\alpha_1,\dots,\alpha_K)$ has density
# $\propto\prod_k w_k^{\alpha_k - 1}$ on the simplex $\{w_k\ge 0,\ \sum_k w_k = 1\}$. With all $\alpha_k = \alpha$:
# the mean share is $1/K$, and the **concentration parameter** $\alpha$ sets the spread,
# $\mathrm{Var}(w_k) = \frac{(K-1)}{K^2(K\alpha + 1)}$. $\alpha = 1$ is uniform over the simplex; $\alpha < 1$ prefers
# a few large shares (concentrated slip); $\alpha > 1$ prefers even sharing. A Dirichlet draw is $K$ independent
# Gamma($\alpha$, 1) numbers divided by their sum, which is how the code below draws it (the **Gamma** distribution
# is a family of positive numbers; with shape $\alpha$ small, most draws are near zero and a few are large). The set of
# all possible share vectors, $\{w_k\ge 0,\ \sum_k w_k = 1\}$, is called the **simplex**; for $K = 3$ it is a
# triangle.
#
# **Worked example.** Three parts (drawn in a triangle) and 30 patches sharing a potency of 20 m·km, for three values
# of $\alpha$; the variance formula is checked against the draws.

# %%
def dirichlet_draws(alpha, K, n, seed):
    g = np.random.default_rng(seed).gamma(alpha, 1.0, (n, K))
    return g / g.sum(1, keepdims=True)


alphas = [0.3, 1.0, 5.0]
P0 = 20.0                                                                   # potency (m km), the total to be shared
for a_ in alphas:
    W = dirichlet_draws(a_, Nf, 50_000, 10)
    print(f"alpha = {a_:3.1f}, K = {Nf}: mean share {W.mean():.4f} (1/K = {1 / Nf:.4f}); var of a share {W[:, 0].var():.2e} "
          f"(formula {(Nf - 1) / (Nf ** 2 * (Nf * a_ + 1)):.2e}); largest share in a draw, median {np.median(W.max(1)):.3f}; "
          f"sum of shares {W.sum(1).min():.6f} to {W.sum(1).max():.6f}")


def tern(W):                                                                # barycentric -> 2-D triangle coordinates
    return W[:, 1] + 0.5 * W[:, 2], np.sqrt(3) / 2 * W[:, 2]


# %%
fig, axs = plt.subplots(2, 3, figsize=(14.5, 7.4), gridspec_kw=dict(height_ratios=[1, 1.05]))
for j, a_ in enumerate(alphas):
    ax = axs[0, j]
    W3 = dirichlet_draws(a_, 3, 3000, 20 + j)
    tx, ty = tern(W3)
    ax.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3) / 2, 0], color=S.INK2, lw=1)
    ax.scatter(tx, ty, s=2.5, color=S.VIOLET, alpha=0.35)
    for (px, py, lab) in [(0, 0, "all in part 1"), (1, 0, "all in part 2"), (0.5, np.sqrt(3) / 2, "all in part 3")]:
        ax.text(px, py + (0.04 if py > 0 else -0.06), lab, ha="center", fontsize=7.8, color=S.INK2)
    ax.set_xlim(-0.12, 1.12); ax.set_ylim(-0.12, 0.98); ax.set_aspect("equal"); ax.axis("off")
    ax.set_title(f"α = {a_:g}, K = 3: " + {0.3: "a few parts take most", 1.0: "uniform over all sharings",
                                         5.0: "near-even sharing"}[a_], fontsize=9.5)
    ax = axs[1, j]
    Wk = dirichlet_draws(a_, Nf, 4, 30 + j)
    for i in range(4):
        step(ax, P0 * Wk[i] / wz, color=shades[i], lw=1.1)
    ax.invert_yaxis(); ax.set_xlabel("slip (m)  = potency × share / width")
    ax.set_xlim(0, None)
    if j == 0:
        ax.set_ylabel("depth (km)")
    ax.set_title(f"α = {a_:g}: four slip profiles with potency {P0:g} m·km", fontsize=9.5)
fig.tight_layout()
bk.savefig(fig, "2-11_dirichlet")
plt.show()

# %% [markdown]
# **How to read it.** Top: each dot is one way of sharing a total among three parts; a corner means everything in one
# part. With α = 0.3 the dots crowd the corners and edges, with α = 1 they fill the triangle evenly, with α = 5 they
# gather in the middle. Bottom: 30-patch slip profiles with the same total potency. Small α gives a few strong
# patches, large α gives slip spread evenly over the whole fault; all are positive and all have exactly the same total
# (printed sums).
#
# **Common misreadings.**
# * *"Dirichlet with α = 1 is uninformative."* It is uniform over the ways of sharing, but it still fixes the total
#   and the sign. Those are strong, and here wanted, constraints.
# * *"The total must then be known exactly."* No: put a prior on the total (for example log-normal around a seismic
#   moment) and a Dirichlet on the shares, as section 6 does.

# %% [markdown]
# ## 5. Prior-predictive checks: simulate before you fit
#
# **In plain words.** A prior on 30 slip values is hard to judge by staring at it. Its consequences are easy to judge:
# draw slip profiles from the prior, run them through the forward model, and look at the totals and the predicted
# surface motion. If the prior makes earthquakes ten times too big, or half of them slip the wrong way, it does not say
# what you know. Revise it, and check again. Doing this before fitting keeps the data from leaking into the choice.
#
# **Precisely.** The **prior-predictive distribution** is
# $p(\mathbf d) = \int p(\mathbf d\mid\mathbf s)\,p(\mathbf s)\,d\mathbf s$: the data you would expect before
# observing anything. A **prior-predictive check** compares draws from it, or summaries of them, with what is
# physically plausible. Here the assumed background knowledge is: the fault is right-lateral (slip $\ge 0$), and a
# seismic moment puts the potency near 20 m·km (in this synthetic example, the true value). The forward model is the
# 30-patch antiplane fault of 2.07 with 24 stations.
#
# **Worked example.** Four priors, each checked by 4,000 draws:
# (A) "vague": independent $\mathcal N(0, 10^2)$ m per patch;
# (B) independent half-normal, scale 2 m per patch (looks reasonable patch by patch);
# (C) independent half-normal, with the scale revised so that the expected potency is 20 m·km;
# (D) potency log-normal (median 20 m·km, log-sd 0.1) times Dirichlet(α = 1) shares.

# %%
xs = np.array([1, 2, 3, 4, 6, 9, 13, 18, 25, 35, 50, 70.0])
xs = np.sort(np.r_[-xs, xs])                                                # 24 stations (km)


def patch_disp(x, z1, z2):
    return (np.arctan(z2 / x) - np.arctan(z1 / x)) / np.pi


Gf = 1000.0 * np.array([[patch_disp(x, z_edges[j], z_edges[j + 1]) for j in range(Nf)] for x in xs])   # mm per m
s_true = 2.0 * np.exp(-0.5 * ((zc - 12.0) / 4.0) ** 2)                                                  # m
sig_f = 3.0
d_f = Gf @ s_true + sig_f * np.random.default_rng(42).standard_normal(xs.size)
pot = lambda S_: S_ @ wz                                                                                 # m km
print(f"true potency {pot(s_true):.2f} m km;  largest observed |displacement| {np.abs(d_f).max():.0f} mm")
hn_rev = P0 / (Nf * np.sqrt(2 / np.pi))
print(f"revised half-normal scale so that E[potency] = {P0:g} m km: {hn_rev:.3f} m per patch")
n5 = 4000
r5 = np.random.default_rng(6)
tot_D = np.exp(np.log(P0) + 0.1 * r5.standard_normal(n5))
prior5 = {
    "(A) vague N(0, 10²)": r5.normal(0, 10, (n5, Nf)),
    "(B) half-normal 2 m": np.abs(r5.normal(0, 2, (n5, Nf))),
    "(C) half-normal, revised": np.abs(r5.normal(0, hn_rev, (n5, Nf))),
    "(D) potency × Dirichlet(1)": tot_D[:, None] * dirichlet_draws(1.0, Nf, n5, 7) / wz,
}
pp = {}
print(f"\n{'prior':28s} {'potency 5-50-95 % (m km)':>28s} {'P(potency < 0)':>15s} {'max|u| 5-50-95 % (mm)':>26s} {'share of observed max|d| quantile':>34s}")
for k, S_ in prior5.items():
    U_ = S_ @ Gf.T
    mx = np.abs(U_).max(1)
    pq = np.percentile(pot(S_), [5, 50, 95])
    mq = np.percentile(mx, [5, 50, 95])
    pp[k] = dict(U=U_, pot=pot(S_), mx=mx)
    print(f"{k:28s} {pq[0]:8.1f} {pq[1]:8.1f} {pq[2]:8.1f}   {np.mean(pot(S_) < 0):13.3f}   {mq[0]:7.0f} {mq[1]:7.0f} {mq[2]:7.0f}"
          f"   {np.mean(mx < np.abs(d_f).max()):20.3f}")

# %%
pcol = {k: c for k, c in zip(prior5, [S.MUTED, S.ORANGE, S.BLUE, S.VIOLET])}
fig, axs = plt.subplots(1, 3, figsize=(16, 4.1), gridspec_kw=dict(width_ratios=[1, 1, 1.25]))
ax = axs[0]
bins_p = np.linspace(-150, 150, 121)
for k in prior5:
    ax.hist(np.clip(pp[k]["pot"], -150, 150), bins=bins_p, density=True, histtype="step", lw=1.6, color=pcol[k], label=k)
ax.axvline(P0, color=S.INK, lw=1.2, ls="--", label="potency from the seismic moment")
ax.axvline(0, color=S.AXIS, lw=0.8)
ax.set_xlabel("potency of a prior draw (m·km), clipped at ±150"); ax.set_ylabel("density")
ax.set_title("(a) What each prior implies for the total")
ax.legend(fontsize=7.2, loc="upper left")
ax = axs[1]
bins_m = np.logspace(0.5, 4.5, 60)
for k in prior5:
    ax.hist(pp[k]["mx"], bins=bins_m, density=True, histtype="step", lw=1.6, color=pcol[k], label=k)
ax.axvline(np.abs(d_f).max(), color=S.INK, lw=1.2, ls="--", label="observed (for reference)")
ax.set_xscale("log"); ax.set_xlabel("largest |surface displacement| of a draw (mm)"); ax.set_ylabel("density")
ax.set_title("(b) What each prior implies for the data")
ax.legend(fontsize=7.2, loc="upper right")
ax = axs[2]
for k in ["(A) vague N(0, 10²)", "(B) half-normal 2 m", "(D) potency × Dirichlet(1)"]:
    lo_, hi_ = np.percentile(pp[k]["U"], [5, 95], axis=0)
    ax.fill_between(xs, lo_, hi_, color=pcol[k], alpha=0.22, lw=0, label=f"{k}: 90 % of draws")
ax.plot(xs, d_f, "o", color=S.INK, ms=3.5, label="observed data (for reference)")
ax.set_xscale("symlog", linthresh=1); ax.set_ylim(-2500, 2500)
tk = [-50, -10, -2, 0, 2, 10, 50]
ax.set_xticks(tk); ax.set_xticklabels([str(t_) for t_ in tk]); ax.minorticks_off()
ax.set_xlabel("station position across the fault (km; stretched near 0)"); ax.set_ylabel("surface displacement (mm)")
ax.set_title("(c) Prior-predictive bands of the surface displacement")
ax.legend(fontsize=7.2, loc="upper left")
fig.tight_layout()
bk.savefig(fig, "2-11_prior_predictive")
plt.show()

# %% [markdown]
# **How to read it.** (a) The vague prior (A) implies potencies spread far beyond the plausible total, half of them
# negative (printed share): it says a left-lateral earthquake ten times too big is as likely as the real one. Prior (B)
# looks sensible patch by patch, yet its implied total is far above 20 m·km: 30 patches of about a metre each add up.
# The revised half-normal (C) centres the total, and (D) controls it directly. (b) The same story in data space: under
# (A) and (B) the observed displacements would be an unusually small outcome (printed quantile of the observed maximum
# within each prior-predictive distribution); under (C) and (D) they are an ordinary one. (c) The bands of predicted
# surface displacement: (A) is symmetric about zero and huge, (B) is one-signed but too large, (D) brackets the data.
#
# The check used only background knowledge (sign and potency); the observed data are drawn for reference. Choosing a
# prior *because* it brackets these particular data would use the data twice.
#
# **Common misreadings.**
# * *"A wide prior is a safe prior."* (A) is wide, and it is confidently wrong about the size and sign of the event.
# * *"A prior that is reasonable for each parameter is reasonable for the model."* (B) is the counterexample: what
#   looks modest per patch adds up over 30 patches. Always check the implied totals and predicted data.

# %% [markdown]
# ## 6. Same data, different priors: where the data decide and where the prior does
#
# **In plain words.** Fit the same 24 data with several priors. Where the stations see the fault well (the shallow
# part), every prior gives nearly the same slip: the data decide. Where they see it poorly (deep), each prior returns
# something close to its own preference. Measuring this patch by patch tells you which parts of a result to trust.
#
# **Precisely.** For a Gaussian prior with standard deviation $\sigma_{\rm prior}$ the **shrinkage ratio**
# $\sigma_{\rm post}/\sigma_{\rm prior}$ is near 0 where the data dominate and near 1 where the prior dominates. For
# non-Gaussian priors there is no closed form; we draw samples from the posterior with **NUTS**, a sampling algorithm
# that 2.16-2.19 explain (2.18 is NUTS itself). For now it is a black box that returns posterior draws; the printed
# checks (**R-hat** close to 1, many **effective samples**, no **divergences**) say that its four independent runs agree
# and that it reported no numerical trouble (2.19 explains them). Priors compared: (A) vague, (C) revised half-normal,
# (D) potency × Dirichlet, and a smooth Gaussian (sd 2 m, ℓ = 5 km). Gaussian cases are exact (closed form, 2.08).

# %%
Cd = sig_f ** 2 * np.eye(xs.size)


def gauss_post(Cm, m0=None):
    m0 = np.zeros(Nf) if m0 is None else m0
    Pm = np.linalg.inv(Cm)
    A_ = Gf.T @ np.linalg.solve(Cd, Gf) + Pm
    C_ = np.linalg.inv(A_)
    return C_ @ (Gf.T @ np.linalg.solve(Cd, d_f) + Pm @ m0), C_


postA = gauss_post(100.0 * np.eye(Nf))
postS = gauss_post(sq_exp_cov(5.0, 2.0))
Gj, dj = jnp.asarray(Gf), jnp.asarray(d_f)


def model_C():
    s = numpyro.sample("s", dist.HalfNormal(hn_rev).expand([Nf]))
    numpyro.sample("d", dist.Normal(Gj @ s, sig_f), obs=dj)


def model_D():
    tot = numpyro.sample("tot", dist.LogNormal(np.log(P0), 0.1))
    w = numpyro.sample("w", dist.Dirichlet(jnp.ones(Nf)))
    s = numpyro.deterministic("s", tot * w / jnp.asarray(wz))
    numpyro.sample("d", dist.Normal(Gj @ s, sig_f), obs=dj)


post_s = {}
for nm, mdl in [("(C) half-normal, revised", model_C), ("(D) potency × Dirichlet(1)", model_D)]:
    t0 = time.time()
    mc = MCMC(NUTS(mdl, target_accept_prob=0.9), num_warmup=1000, num_samples=1000, num_chains=4,
              chain_method="parallel", progress_bar=False)
    mc.run(jax.random.PRNGKey(1), extra_fields=("diverging",))
    sch = np.asarray(mc.get_samples(group_by_chain=True)["s"])
    post_s[nm] = sch.reshape(-1, Nf)
    print(f"{nm}: NUTS {time.time() - t0:.1f} s, max R-hat {max(D.rhat(sch[:, :, j]) for j in range(Nf)):.3f}, "
          f"min ESS {min(D.ess(sch[:, :, j]) for j in range(Nf)):.0f}, divergences {int(np.sum(mc.get_extra_fields()['diverging']))}")
sdA = np.sqrt(np.diag(postA[1]))
shal, deep = zc < 8, zc > 22
print(f"\n(A) vague prior: shrinkage sd_post / sd_prior, median over patches shallower than 8 km {np.median(sdA[shal] / 10):.3f}, "
      f"deeper than 22 km {np.median(sdA[deep] / 10):.3f}")
sdS = np.sqrt(np.diag(postS[1]))
print(f"smooth prior: shrinkage, shallower than 8 km {np.median(sdS[shal] / 2):.3f}, deeper than 22 km {np.median(sdS[deep] / 2):.3f}")
means = {"(A) vague N(0, 10²)": postA[0], "smooth Gaussian": postS[0],
         **{k: v.mean(0) for k, v in post_s.items()}}
mm = np.array(list(means.values()))
print("spread of the posterior means across the four priors (max - min), median over patches: "
      f"shallower than 8 km {np.median(np.ptp(mm[:, shal], axis=0)):.3f} m, deeper than 22 km {np.median(np.ptp(mm[:, deep], axis=0)):.3f} m")
for k, v in means.items():
    print(f"  {k:28s} rms error vs true slip {np.sqrt(np.mean((v - s_true) ** 2)):.3f} m;  potency of the mean {pot(v):6.2f} m km")
for k, v in post_s.items():
    print(f"  {k:28s} posterior mean: peak {v.mean(0).max():.2f} m (true {s_true.max():.2f}); median deeper than 22 km "
          f"{np.median(v.mean(0)[deep]):.2f} m (true {np.median(s_true[deep]):.3f} m)")

# %%
fig, axs = plt.subplots(1, 3, figsize=(16, 4.6), gridspec_kw=dict(width_ratios=[1, 1, 0.9]))
TRUE_KW = dict(color="#b5b4ad", lw=4.5, solid_capstyle="round", zorder=1, label="true slip")
ax = axs[0]
ax.plot(s_true, zc, **TRUE_KW)
for k, (m_, C_), c in [("(A) vague N(0, 10²)", postA, S.MUTED), ("smooth Gaussian", postS, S.AQUA)]:
    sd_ = np.sqrt(np.diag(C_))
    ax.fill_betweenx(zc, m_ - 1.645 * sd_, m_ + 1.645 * sd_, color=c, alpha=0.18, lw=0)
    ax.plot(m_, zc, color=c, lw=1.8, label=f"{k}: mean and 90 %")
ax.invert_yaxis(); ax.set_xlim(-6, 6); ax.set_xlabel("slip (m)"); ax.set_ylabel("depth (km)")
ax.set_title("(a) Gaussian priors (exact posteriors)")
ax.legend(fontsize=7.4, loc="lower left")
ax = axs[1]
ax.plot(s_true, zc, **TRUE_KW)
for k, c in [("(C) half-normal, revised", S.BLUE), ("(D) potency × Dirichlet(1)", S.VIOLET)]:
    lo_, hi_ = np.percentile(post_s[k], [5, 95], axis=0)
    ax.fill_betweenx(zc, lo_, hi_, color=c, alpha=0.18, lw=0)
    ax.plot(post_s[k].mean(0), zc, color=c, lw=1.8, label=f"{k}: mean and 90 %")
ax.invert_yaxis(); ax.set_xlim(-0.2, 4); ax.set_xlabel("slip (m)")
ax.set_title("(b) Positive priors (NUTS)")
ax.legend(fontsize=7.4, loc="lower right")
ax = axs[2]
ax.plot(sdA / 10.0, zc, "o-", color=S.MUTED, ms=3, lw=1.4, label="(A) vague, sd 10 m")
ax.plot(sdS / 2.0, zc, "s-", color=S.AQUA, ms=3, lw=1.4, label="smooth Gaussian, sd 2 m")
ax.axvline(1, color=S.AXIS, lw=0.8, ls=":")
ax.invert_yaxis(); ax.set_xlim(0, 1.05)
ax.set_xlabel("posterior sd / prior sd"); ax.set_ylabel("depth (km)")
ax.set_title("(c) Shrinkage: 0 = data decide, 1 = prior decides")
ax.legend(fontsize=7.4, loc="lower left")
fig.tight_layout()
bk.savefig(fig, "2-11_data_vs_prior")
plt.show()

# %% [markdown]
# **How to read it.** (a) The vague prior lets slip below a few kilometres swing over metres in both directions, with
# a 90 % band wider than the plot; the smooth prior ties neighbouring patches together and follows the true slip.
# (b) The two positive priors follow the true slip near the surface. Below 22 km, where the true slip is close to zero,
# both return a mean of a few tenths of a metre with a wide band (printed): a positive quantity that the data cannot pin
# down has a positive mean, because the prior forbids the negative half. The Dirichlet prior also puts the peak higher
# than the half-normal (printed peaks). (c) The shrinkage ratio. For the vague prior it is small only in the top few
# kilometres and close to 1 below: there the answer is the prior's. The smooth prior shrinks far more at every depth
# (printed medians), because each patch borrows information from its neighbours; that borrowing is itself an
# assumption. The printed spread of the posterior means across the four priors tells the same story in metres: smaller
# near the surface than at depth.
#
# **Common misreadings.**
# * *"The posterior is what the data say."* It is what the data and the prior say together; (c) shows, patch by patch,
#   how much is which.
# * *"If two priors give the same answer, the answer is data-driven."* Only where the shrinkage ratio is small. Two
#   priors with similar preferences can agree at depth for reasons that have nothing to do with the data.

# %% [markdown]
# ## 7. Improper priors (optional)
#
# **In plain words.** A "prior" that is flat over all real numbers cannot be normalised: its total probability is
# infinite. Such an **improper prior** is sometimes harmless, because multiplying by the likelihood can still give a
# proper posterior. That happens only if the data constrain every direction of the model. When some slip patterns are
# invisible to the stations, the posterior is improper too, and no sampler can give a meaningful answer.
#
# **Precisely.** The **precision** matrix is the inverse of a covariance (2.08): large precision, small uncertainty.
# With a flat prior on $\mathbf s$, the posterior is Gaussian with precision $\mathbf G^\top\mathbf C_d^{-1}\mathbf G$.
# It is proper only if this matrix has no zero eigenvalue (eigenvalue, 2.06): a zero eigenvalue is a direction with
# zero precision, so infinite variance. Its eigenvalues are the squared singular values of the whitened forward
# operator $\mathbf G/\sigma$ (SVD and rank, 2.06), which we compute from the SVD because forming
# $\mathbf G^\top\mathbf G$ first loses half of the significant digits (2.06, section 6). For the 24-station antiplane
# fault the stations come in mirror pairs, so the rank is 12 (as in 2.07): 18 directions have zero precision. Any
# independent proper prior with standard deviation $\sigma_{\rm prior}$ adds $\mathbf I/\sigma_{\rm prior}^2$ and lifts
# every eigenvalue to at least $1/\sigma_{\rm prior}^2$.

# %%
sv_w = np.linalg.svd(Gf / sig_f, compute_uv=False)
rank_f = int(np.sum(sv_w > sv_w[0] * 1e-10))                       # the same threshold as 2.07
ev_flat = np.zeros(Nf)
ev_flat[:rank_f] = sv_w[:rank_f] ** 2                               # nonzero eigenvalues; the rest are exactly zero
ev_A = ev_flat + 1.0 / 100.0
print(f"whitened G: {sv_w.size} singular values, rank {rank_f}; posterior precision with a flat prior has {rank_f} nonzero "
      f"eigenvalues (smallest {ev_flat[rank_f - 1]:.2e} per m^2) and {Nf - rank_f} zero ones")
print(f"singular values beyond the rank: largest {sv_w[rank_f:].max():.1e} (round-off)")
print(f"with the vague N(0, 10^2) prior the smallest eigenvalue is {ev_A.min():.4f} = 1/10^2: every direction proper")

# %%
fig, ax = plt.subplots(figsize=(6.8, 3.8))
floor = 1e-14
ax.semilogy(np.arange(1, rank_f + 1), ev_flat[:rank_f], "o-", color=S.BAD, ms=4, lw=1.2, label="flat prior: Gᵀ C_d⁻¹ G")
ax.semilogy(np.arange(rank_f + 1, Nf + 1), np.full(Nf - rank_f, floor), "o", color=S.BAD, ms=4, mfc="none",
            label="exactly zero (drawn at the bottom)")
ax.semilogy(np.arange(1, Nf + 1), ev_A, "s-", color=S.MUTED, ms=4, lw=1.2, label="vague prior: + I / 10²")
ax.axhline(1 / 100.0, color=S.AXIS, lw=0.8, ls=":")
ax.axvline(rank_f + 0.5, color=S.INK2, lw=0.8, ls="--")
ax.text(rank_f + 1, 1e-7, f"{Nf - rank_f} directions the data\ncannot see (rank {rank_f})", fontsize=8, color=S.INK2)
ax.set_ylim(floor / 5, ev_flat.max() * 5)
ax.set_xlabel("eigenvalue number"); ax.set_ylabel("posterior precision (1/m²)")
ax.set_title("Eigenvalues of the posterior precision")
ax.legend(fontsize=8, loc="upper right")
fig.tight_layout()
bk.savefig(fig, "2-11_improper")
plt.show()

# %% [markdown]
# **How to read it.** With a flat prior (red) the eigenvalues fall steeply and are exactly zero after the rank: in
# those directions the posterior has no curvature and no finite variance; the last nonzero ones are so small (printed)
# that those directions are almost as unconstrained. A proper prior (grey) puts a floor under every
# eigenvalue, so every direction has a finite variance; in the invisible directions that variance is simply the prior's.
#
# **Common misreadings.**
# * *"A sampler will tell me if the posterior is improper."* Often it just wanders without converging, or a bounded
#   box quietly makes it proper; the eigenvalue check is direct.

# %% [markdown]
# ## Summary: what you know → which prior
#
# | what you know | prior | section |
# |---|---|---|
# | a typical value and a typical deviation | Gaussian (normal) | 1 |
# | most values small, a few large | Laplace or Student-t | 1, 2 |
# | only hard bounds | uniform on the bounds, in the parameterisation you mean | 3 |
# | the sign and a scale | half-normal; truncated normal if there is a preferred value | 1, 5 |
# | an order of magnitude | log-normal or log-uniform | 3 |
# | a total, shared among parts | (prior on the total) × Dirichlet on the shares | 4, 5 |
# | smoothness / correlation between neighbours | correlated Gaussian, GP (2.12), smoothing (2.09) | 2, 6 |
#
# Whatever the choice: simulate from it before fitting (section 5), and after fitting check where the answer is
# data-driven (section 6).
#
# **The image to keep.** A prior is a set of possible faults before the data. Draw some, push them through the forward
# model, and ask whether they look like earthquakes you could believe in.
#
# Next: **2.12 · Gaussian processes: priors on whole functions**.

# %%
print(f"notebook run time: {time.time() - T0:.0f} s")
