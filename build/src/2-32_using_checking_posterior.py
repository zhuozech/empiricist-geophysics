# %% [markdown]
# # 2.32 · Using and checking a posterior
#
# **Before this:** 2.08 (Bayesian linear inversion), 2.16 (Metropolis), 2.29 (AIC and BIC)  
# **You will be able to:** compute derived quantities per sample; choose and name a point summary and an interval;
# make joint rather than per-parameter statements; run a posterior predictive check and simulation-based calibration and
# say what each can and cannot detect; compare models by LOO or WAIC  
# **Optional sections:** 6 (LOO and WAIC)
#
# A sampler, an EnKI run or a closed-form formula hands you a cloud of models. The work is not done: you still have to
# turn the cloud into the numbers you report (a moment, a peak slip, an overlap with afterslip), say which summary you
# used, and check two different things: that the *model* can reproduce the data, and that the *code* returns the
# posterior it claims to. This notebook covers both, then compares the predictive model-comparison tools (LOO-CV and
# WAIC) with AIC and BIC from 2.29.
#
# **Words this notebook defines:** derived quantity, Jensen's inequality, plug-in estimate, posterior mean, median and
# MAP, equal-tailed interval, highest-posterior-density (HPD) interval, pointwise and simultaneous bands, posterior
# predictive distribution, replicated data, test statistic, posterior predictive p-value, simulation-based calibration
# (SBC), rank statistic, leave-one-out cross-validation (LOO-CV), expected log predictive density (elpd), WAIC,
# effective number of parameters ($p_\text{loo}$, $p_\text{waic}$), importance-sampling LOO.

# %% [markdown]
# **Problems in this notebook**
#
# | problem | unknowns | data | forward model | posterior |
# |---|---|---|---|---|
# | coseismic slip on an antiplane fault (sections 1, 3, 4, 5) | 30 patches, 0-30 km | 24 stations, σ = 3 mm | screw-dislocation Green's functions $\mathbf d = \mathbf G\mathbf m$ (as in 2.13) | exactly Gaussian (linear model, Gaussian prior) |
# | afterslip on the same fault (section 1) | 30 patches | 24 stations, σ = 3 mm | same | exactly Gaussian |
# | one positive parameter; one two-peaked parameter (section 2) | 1 | none (posteriors written down) | none | log-normal; mixture of two Gaussians |
# | how many patches? (section 6) | 3 to 30 equal patches | 24 stations, σ = 3 mm | same Green's functions at each resolution | exactly Gaussian per model |
#
# Priors: coseismic slip Gaussian with mean 1 m, sd 1 m and Matérn correlation (ν = 1.5, ℓ = 5 km), as in 2.13;
# afterslip mean 0, sd 0.5 m, same correlation; section 6 uses independent $N(1, 2^2)$ m per patch. The Gaussian priors
# allow negative slip; 2.30 treats positivity. Every posterior here is exact, so every check below tests the check
# itself, not a sampler (except where a sampler is the point, section 5).

# %%
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))
import numpy as np
import matplotlib.pyplot as plt
import bayeskit as bk
from bayeskit import problems as P, samplers as SM, diagnostics as D, plotting as BP, style as S
import time
from scipy.special import kv, gamma, logsumexp
from scipy.stats import norm, chisquare, binom, lognorm
from scipy.linalg import solve_triangular

T0 = time.time()
N = 30
z_edges = np.linspace(0.0, 30.0, N + 1)
zc = 0.5 * (z_edges[1:] + z_edges[:-1])
xs = np.array([1, 2, 3, 4, 6, 9, 13, 18, 25, 35, 50, 70.0])
xs = np.sort(np.r_[-xs, xs])
SIG = 3.0
ND = xs.size
MU_SHEAR = 30e9                                           # shear modulus (Pa)


def patch_disp(x, z1, z2):
    return (np.arctan(z2 / x) - np.arctan(z1 / x)) / np.pi


def Gmat(scale=1.0, x=xs, edges=z_edges):
    """Green's functions (mm per m of slip); scale stretches all depths (the 2.13 structural error)."""
    ze = edges * scale
    return 1000.0 * np.array([[patch_disp(xx, ze[j], ze[j + 1]) for j in range(len(ze) - 1)] for xx in x])


def matern(dist, ell, nu):
    dist = np.maximum(dist, 1e-12)
    a = np.sqrt(2 * nu) * dist / ell
    c = (2 ** (1 - nu) / gamma(nu)) * a ** nu * kv(nu, a)
    c[dist < 1e-10] = 1.0
    return c


R5 = matern(np.abs(zc[:, None] - zc[None, :]), 5.0, 1.5)
Cm = R5 + 1e-9 * np.eye(N)
m0 = np.ones(N)
G0 = Gmat()


def posterior(d, G=G0, m0=m0, Cm=Cm, sig=SIG):
    """Exact linear-Gaussian posterior (Kalman-gain form)."""
    Sm = G @ Cm @ G.T + sig ** 2 * np.eye(G.shape[0])
    K = np.linalg.solve(Sm, G @ Cm).T
    mean = m0 + K @ (d - G @ m0)
    C = Cm - K @ G @ Cm
    return mean, 0.5 * (C + C.T)


def draw(mean, C, n, rng):
    L = np.linalg.cholesky(C + 1e-10 * np.eye(len(mean)))
    return mean[None, :] + (L @ rng.standard_normal((len(mean), n))).T


def shear_stress_row(z_km, edges=z_edges, mu=MU_SHEAR / 1e6):
    """Shear-stress change (MPa per m of slip) at depth z on the fault from each patch (same formula as
    bayeskit.afterslip.stress_kernel; lengths in metres)."""
    z, a, b = z_km * 1e3, edges[:-1] * 1e3, edges[1:] * 1e3
    return mu / (2 * np.pi) * (1 / (a - z) + 1 / (a + z) - 1 / (b - z) - 1 / (b + z))


# %% [markdown]
# ## 1. Derived quantities: compute them per sample
#
# **In plain words.** You rarely report the slip on patch 17. You report a moment, a peak slip, the depth range that
# slipped more than a metre, how much afterslip overlaps the coseismic rupture, or the stress change at a point. Each is
# a function of the whole slip model. The posterior of that number is obtained by computing it for *every* sample in
# the cloud and looking at the resulting cloud of numbers. Computing it once from the average slip model gives a
# different number whenever the function is not a straight line.
#
# **Precisely.** For a derived quantity $q = f(\mathbf m)$ the posterior is the distribution of $f(\mathbf m^{(s)})$ over the
# samples $\mathbf m^{(s)}$, with mean $E[f(\mathbf m)]\approx\frac1S\sum_s f(\mathbf m^{(s)})$. The **plug-in** value $f(E[\mathbf m])$ equals it
# only if $f$ is linear (affine). Otherwise Jensen's inequality (reminder: 2.28) says that for a convex $f$, one that
# curves upward like a bowl (the maximum of several numbers is an example), $E[f(\mathbf m)] \ge f(E[\mathbf m])$: the
# average of the function is at least the function of the average. Thresholds and ratios have no fixed sign, but still
# differ. The mean over samples carries a Monte Carlo standard error (reminder: 2.05 and 2.19). Five quantities:
#
# * seismic moment per metre along strike, $M_0' = \mu\sum_j s_j\,\Delta z_j$, the size of the earthquake: shear modulus
#   μ times slip times fault area (linear);
# * shear-stress change on the fault at 17.5 km depth, below the rupture, $\Delta\tau = \sum_j K_j s_j$ (linear; in
#   this antiplane model the normal stress does not change, so this is also the Coulomb stress change, the combination of
#   shear and normal stress change that brings a fault closer to failure);
# * peak slip, $\max_j s_j$ (convex);
# * the depth range with slip above 1 m (a threshold count, 1 km per patch);
# * coseismic-afterslip overlap: the depth range where coseismic slip exceeds 1 m *and* afterslip exceeds 0.2 m,
#   computed from one coseismic sample paired with one afterslip sample (the two posteriors are independent).

# %%
rng = np.random.default_rng(3)
s_true = 3.0 * np.exp(-0.5 * ((zc - 9) / 3.5) ** 2)                 # coseismic slip (m)
a_true = 0.6 * np.exp(-0.5 * ((zc - 16) / 4.0) ** 2)                # afterslip (m)
d_co = G0 @ s_true + SIG * rng.standard_normal(ND)
d_af = G0 @ a_true + SIG * rng.standard_normal(ND)
mu_co, C_co = posterior(d_co)
Cm_af = 0.5 ** 2 * R5 + 1e-9 * np.eye(N)
mu_af, C_af = posterior(d_af, m0=np.zeros(N), Cm=Cm_af)
NS = 4000
S_co = draw(mu_co, C_co, NS, rng)
S_af = draw(mu_af, C_af, NS, rng)
K175 = shear_stress_row(17.5)

QTY = {
    "moment per metre of strike (10¹⁴ N·m/m)": lambda s, a: MU_SHEAR * s.sum(-1) * 1e3 / 1e14,
    "shear stress change at 17.5 km (MPa)": lambda s, a: s @ K175,
    "peak slip (m)": lambda s, a: s.max(-1),
    "depth range with slip > 1 m (km)": lambda s, a: (s > 1.0).sum(-1) * 1.0,
    "co/afterslip overlap (km)": lambda s, a: ((s > 1.0) & (a > 0.2)).sum(-1) * 1.0,
}
print(f"{'quantity':42s} {'mean of f':>10s} {'(MC s.e.)':>10s} {'f(mean)':>9s} {'true':>8s} {'posterior sd':>13s}")
res1 = {}
for name, f in QTY.items():
    v = f(S_co, S_af)
    res1[name] = (v, f(mu_co, mu_af), f(s_true, a_true))
    print(f"{name:42s} {v.mean():10.3f} {v.std() / np.sqrt(NS):10.3f} {float(f(mu_co, mu_af)):9.3f} "
          f"{float(f(s_true, a_true)):8.3f} {v.std():13.3f}")
j175 = int(np.argmin(np.abs(zc - 17.5)))
print(f"stress kernel at 17.5 km (MPa per m of slip): own patch {K175[j175]:+.1f}, neighbours {K175[j175 - 1]:+.1f} and "
      f"{K175[j175 + 1]:+.1f}, patches 5 km away {K175[j175 - 5]:+.2f} and {K175[j175 + 5]:+.2f}; "
      f"posterior sd of slip at 17.5 km {np.sqrt(C_co[j175, j175]):.2f} m")

fig, axs = plt.subplots(2, 3, figsize=(15.5, 7.6))
ax = axs[0, 0]
for k in range(40):
    ax.plot(S_co[k], zc, color=S.MUTED, lw=0.5, alpha=0.6)
ax.plot(mu_co, zc, color=S.BLUE, lw=2.0, label="posterior mean (coseismic)")
ax.plot(s_true, zc, color=S.INK, lw=1.4, ls=(0, (4, 2)), label="true coseismic slip")
for k in range(40):
    ax.plot(S_af[k], zc, color=S.AQUA, lw=0.4, alpha=0.35)
ax.plot(mu_af, zc, color=S.AQUA, lw=2.0, label="posterior mean (afterslip)")
ax.plot(a_true, zc, color=S.INK2, lw=1.2, ls=(0, (1, 1.5)), label="true afterslip")
ax.plot([], [], color=S.MUTED, lw=0.8, label="40 posterior samples each")
ax.axvline(1.0, color=S.AXIS, lw=0.8); ax.axvline(0.2, color=S.AXIS, lw=0.8)
ax.invert_yaxis(); ax.set_xlabel("slip (m)"); ax.set_ylabel("depth (km)")
ax.set_title("(a) Samples, means and the true slip")
ax.legend(fontsize=7.5, loc="lower right")
for ax, (name, (v, plug, tru)) in zip(axs.ravel()[1:], res1.items()):
    discrete = "range" in name or "overlap" in name
    bins = np.arange(v.min() - 0.5, v.max() + 1.5, 1.0) if discrete else 40
    ax.hist(v, bins=bins, color="#d9d8d1", edgecolor=S.SURFACE, lw=0.4)
    ax.axvline(v.mean(), color=S.INK, lw=1.6, ls=(0, (4, 2)), label=f"mean over samples: {v.mean():.2f}")
    ax.axvline(float(plug), color=S.ORANGE, lw=1.8, label=f"computed from the mean model: {float(plug):.2f}")
    ax.plot(float(tru), 0, "*", color=S.INK, ms=13, mec=S.SURFACE, clip_on=False, zorder=5,
            label=f"true value: {float(tru):.2f}")
    ax.set_xlabel(name); ax.set_yticks([])
    ax.legend(fontsize=7.6, loc="upper right")
axs[0, 1].set_title("(b) Linear: plug-in = mean")
axs[0, 2].set_title("(c) Linear: plug-in = mean")
axs[1, 0].set_title("(d) Maximum: plug-in too low")
axs[1, 1].set_title("(e) Threshold: plug-in differs")
axs[1, 2].set_title("(f) Overlap: plug-in differs")
for ax in axs.ravel()[1:]:
    ax.set_ylim(0, ax.get_ylim()[1] * 1.35)
fig.tight_layout()
bk.savefig(fig, "2-32_derived_quantities")
plt.show()

# %% [markdown]
# **How to read it.** (a) The cloud: 40 samples of each posterior, their means and the true slip; the thin vertical
# lines at 0.2 m and 1 m are the thresholds used in (e) and (f). (b, c) For the two linear quantities the value computed from the mean
# model (orange) agrees with the mean of the per-sample values (dashed) to within the Monte Carlo standard error printed
# in the table: for these, either way is right. The stress at a single point is very uncertain (its posterior sd is
# printed), because it is dominated by the slip on the patch at that depth and its neighbours (several MPa per metre of
# slip, printed), which the surface data resolve poorly at 17.5 km. (d) Peak slip: every sample has its own peak, often at a slightly different depth, so
# the average of the peaks is larger than the peak of the average model, which is smoothed by averaging (Jensen). The
# true peak is printed for comparison. (e, f) Threshold quantities: the mean model gives one number, the samples give a
# distribution, and the two disagree (printed). For the overlap the posterior also says how uncertain the number is,
# which the mean model cannot. Note that the posterior here overstates the overlap relative to the true slip models
# (printed): the data resolve depth only coarsely, and both posteriors are smoother than the truth.
#
# **Worked example.** Two patches with independent slip, each $N(1, 0.5^2)$ m. The peak of the mean model is
# $\max(1, 1) = 1$ m. The mean of the peak has the closed form $E[\max(s_1, s_2)] = \mu + \sigma/\sqrt\pi$ for two
# independent $N(\mu,\sigma^2)$ variables. The steps: (1) the plug-in value; (2) the closed form; (3) a Monte Carlo
# check with 10⁶ draws.

# %%
mu_w, sd_w = 1.0, 0.5
rng_w = np.random.default_rng(0)
s12 = mu_w + sd_w * rng_w.standard_normal((1_000_000, 2))
print(f"step 1  max of the means = max({mu_w}, {mu_w}) = {mu_w:.4f} m")
print(f"step 2  E[max] = {mu_w} + {sd_w} / sqrt(pi) = {mu_w + sd_w / np.sqrt(np.pi):.4f} m")
print(f"step 3  Monte Carlo mean of max over 10^6 draws = {s12.max(1).mean():.4f} m")

# %% [markdown]
# **Common misreadings.**
# * "The posterior mean model is the best model, so compute everything from it." It is the best single model under
#   squared error for the *slip*, not for a nonlinear function of the slip.
# * "The mean model is a typical sample." It is smoother than any sample (panel a), because averaging cancels the
#   wiggles that each sample has.
#
# **In practice.** Overlap between coseismic slip and afterslip is one of the claims 2.31 discussed; it is a threshold
# quantity, so it belongs in the per-sample column. A good reporting habit is to say how each number was obtained: the
# moment and stress above are derived per sample (exactly as good as sampled), the plug-in values are not.

# %% [markdown]
# ## 2. One number and one interval: mean, median, MAP; equal-tailed and HPD
#
# **In plain words.** If you must reduce a posterior to one number, there are three common choices: the average
# (mean), the middle value (median, half the probability on each side) and the most probable value (MAP, the peak).
# They agree for a symmetric single-peaked posterior and drift apart for a skewed one. Intervals have the same choice:
# cut equal probability off each tail (equal-tailed), or keep the narrowest region holding the stated probability
# (highest posterior density, HPD). For a two-peaked posterior the HPD region can be two separate pieces.
#
# **Precisely.** For a 1-D posterior $p(\theta)$: mean $E[\theta]$; median $F^{-1}(0.5)$; MAP $\arg\max p$. A $100(1-\alpha)\,\%$
# equal-tailed interval is $[F^{-1}(\alpha/2), F^{-1}(1-\alpha/2)]$; the HPD region is $\{\theta : p(\theta)\ge c\}$ with $c$
# chosen so it holds $1-\alpha$. Under a change of variables $\phi = g(\theta)$ (monotone), quantiles transform
# ($\text{median}_\phi = g(\text{median}_\theta)$, and equal-tailed intervals map onto equal-tailed intervals), but the
# MAP and the HPD region do **not**, because the density picks up the Jacobian $|d\theta/d\phi|$ (the factor by which
# the change of variables stretches or squeezes the axis; reminder: 2.11). HPD is the highest-density region of 2.01.
#
# Two examples: a positive parameter (think of a viscosity) whose logarithm is Gaussian, $\log_{10}\eta\sim N(19, 0.3^2)$
# with η in Pa s; and a two-peaked posterior, a mixture $0.6\,N(-1, 0.4^2) + 0.4\,N(1.8, 0.5^2)$.

# %%
def hpd_grid(x, p, mass):
    """HPD region on a grid: the highest-density cells holding `mass`; returns the list of intervals."""
    dx = x[1] - x[0]
    order = np.argsort(p)[::-1]
    cum = np.cumsum(p[order]) * dx
    keep = np.zeros_like(p, bool)
    keep[order[:np.searchsorted(cum, mass) + 1]] = True
    edges = np.flatnonzero(np.diff(np.r_[0, keep.astype(int), 0]))
    return [(x[a], x[b - 1]) for a, b in zip(edges[::2], edges[1::2])]


LMU, LSD = 19.0, 0.3
lam = np.log(10)
eta = np.linspace(1e17, 1.2e20, 400001)                          # eta in Pa s (linear grid)
p_eta = lognorm.pdf(eta, s=LSD * lam, scale=10 ** LMU)
le = np.linspace(LMU - 6 * LSD, LMU + 6 * LSD, 200001)            # log10 eta grid
p_le = norm.pdf(le, LMU, LSD)
summ_eta = dict(mean=10 ** LMU * np.exp((LSD * lam) ** 2 / 2), median=10 ** LMU, MAP=10 ** LMU * np.exp(-(LSD * lam) ** 2))
et_eta = 10 ** norm.ppf([0.05, 0.95], LMU, LSD)
hpd_eta = hpd_grid(eta, p_eta, 0.9)
hpd_le = hpd_grid(le, p_le, 0.9)
print("viscosity example (eta in 1e19 Pa s):")
for k, v in summ_eta.items():
    print(f"  {k:6s} in eta: {v / 1e19:.3f};   10^({k} of log10 eta) = {10 ** LMU / 1e19:.3f}")
print(f"  90% equal-tailed: [{et_eta[0] / 1e19:.3f}, {et_eta[1] / 1e19:.3f}] (same whether computed in eta or log10 eta)")
print(f"  90% HPD in eta: [{hpd_eta[0][0] / 1e19:.3f}, {hpd_eta[0][1] / 1e19:.3f}];  HPD in log10 eta mapped back: "
      f"[{10 ** hpd_le[0][0] / 1e19:.3f}, {10 ** hpd_le[0][1] / 1e19:.3f}]")

xb = np.linspace(-4, 5, 90001)
pb = 0.6 * norm.pdf(xb, -1, 0.4) + 0.4 * norm.pdf(xb, 1.8, 0.5)
Fb = np.cumsum(pb) * (xb[1] - xb[0])
summ_b = dict(mean=0.6 * -1 + 0.4 * 1.8, median=xb[np.searchsorted(Fb, 0.5)], MAP=xb[np.argmax(pb)])
et_b = xb[np.searchsorted(Fb, [0.05, 0.95])]
hpd_b = hpd_grid(xb, pb, 0.9)
print("two-peaked example:")
print("  " + ", ".join(f"{k} {v:+.3f}" for k, v in summ_b.items()))
print(f"  90% equal-tailed: [{et_b[0]:+.3f}, {et_b[1]:+.3f}];  90% HPD: " + " and ".join(f"[{a:+.3f}, {b:+.3f}]" for a, b in hpd_b))
gap = (xb > -0.2) & (xb < 0.8)
print(f"  density at the mean relative to the higher peak: {np.interp(summ_b['mean'], xb, pb) / pb.max():.3f}")

fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.0))
C3 = {"mean": S.BLUE, "median": S.ORANGE, "MAP": S.AQUA}
for ax, x, p, summ, et, hpd, unit, ttl in [
        (axs[0], eta / 1e19, p_eta * 1e19, {k: v / 1e19 for k, v in summ_eta.items()}, et_eta / 1e19,
         [(a / 1e19, b / 1e19) for a, b in hpd_eta], "viscosity η (10¹⁹ Pa s)", "(a) Skewed: a positive parameter"),
        (axs[1], le, p_le, {"mean": LMU, "median": LMU, "MAP": LMU}, np.log10(et_eta), hpd_le, "log₁₀ η (Pa s)",
         "(b) The same posterior in log₁₀ η"),
        (axs[2], xb, pb, summ_b, et_b, hpd_b, "θ", "(c) Two peaks")]:
    sel = p > p.max() * 1e-4
    ax.fill_between(x[sel], p[sel], color="#ecebe6", lw=0)
    ax.plot(x[sel], p[sel], color=S.INK, lw=1.2)
    top = p.max()
    for j, (k, v) in enumerate(summ.items()):
        ax.axvline(v, color=C3[k], lw=1.6, ls=["-", (0, (4, 2)), (0, (1, 1.5))][j], label=f"{k} {v:.2f}")
    ax.plot(et, [-0.08 * top] * 2, color=S.INK2, lw=3, solid_capstyle="butt", label="90 % equal-tailed")
    for i, (a, b) in enumerate(hpd):
        ax.plot([a, b], [-0.16 * top] * 2, color=S.VIOLET, lw=3, solid_capstyle="butt",
                label="90 % HPD" if i == 0 else None)
    ax.set_ylim(-0.24 * top, 1.12 * top); ax.set_yticks([])
    ax.set_xlabel(unit); ax.set_title(ttl)
    ax.legend(fontsize=7.8, loc="upper right")
axs[0].set_xlim(0, 4.5)
fig.tight_layout()
bk.savefig(fig, "2-32_point_summaries")
plt.show()

# %% [markdown]
# **How to read it.** (a) A skewed posterior: the MAP is lowest, the median in the middle, the mean highest (printed).
# The HPD interval (violet) is shifted toward the peak relative to the equal-tailed one (grey). (b) The same
# information in $\log_{10}\eta$, where it is Gaussian: all three summaries coincide, and the HPD equals the equal-tailed
# interval. The MAP of η is not 10 to the MAP of $\log_{10}\eta$ (printed), while the median and the equal-tailed
# interval map across exactly. (c) Two peaks: the mean falls in the valley between them, at a value the posterior
# considers unlikely (density printed); the equal-tailed interval covers the valley, the HPD region is two pieces.
#
# **Worked example.** For a log-normal with $\ln\eta\sim N(\mu_\ell, s^2)$, $\mu_\ell = 19\ln 10$ and $s = 0.3\ln 10$:
# mean $= e^{\mu_\ell + s^2/2}$, median $= e^{\mu_\ell}$, mode $= e^{\mu_\ell - s^2}$. Each step is printed below.

# %%
s_ln = LSD * lam
print(f"s = 0.3 ln 10 = {s_ln:.4f};  s^2 = {s_ln ** 2:.4f}")
print(f"mean   = 1e19 x exp(+s^2/2) = 1e19 x {np.exp(s_ln ** 2 / 2):.4f}")
print(f"median = 1e19 x exp(0)      = 1e19 x 1.0000")
print(f"mode   = 1e19 x exp(-s^2)   = 1e19 x {np.exp(-s_ln ** 2):.4f}")

# %% [markdown]
# **Common misreadings.**
# * "The MAP is the most likely value, so it is the natural summary." It depends on the parameterisation (panel b), and
#   in high dimensions the peak can sit where almost no posterior mass is.
# * "Mean ± 2 sd is the 95 % interval." Only for a Gaussian. For a skewed posterior quote quantiles, and say which kind
#   of interval it is.

# %% [markdown]
# ## 3. Marginal and joint statements
#
# **In plain words.** A per-patch band ("slip at 10 km is 2.2-3.1 m with 90 % probability") is a statement about one
# patch at a time. It is not a statement that the whole profile lies inside the band: with 30 patches, most samples
# poke out somewhere. And two patches whose intervals overlap may still be ordered with near certainty, because their
# errors are correlated. Questions about several patches at once must be answered with the joint samples.
#
# **Precisely.** The pointwise 90 % band holds each marginal at 90 %. A **simultaneous** band holds the whole
# profile with probability 90 %: scale the pointwise half-widths by $k$, the 90 % quantile over samples of
# $\max_j |s_j - \bar s_j|/\sigma_j$. Probabilities of joint events, such as $P(s_i > s_j\mid\mathbf d)$, are fractions of
# samples.

# %%
lo, hi = np.percentile(S_co, [5, 95], axis=0)
inside_pw = np.mean(np.all((S_co >= lo) & (S_co <= hi), axis=1))
sd_co = S_co.std(0)
zmax = np.max(np.abs(S_co - mu_co) / sd_co, axis=1)
k_sim = np.quantile(zmax, 0.9)
lo_s, hi_s = mu_co - k_sim * sd_co, mu_co + k_sim * sd_co
inside_sim = np.mean(np.all((S_co >= lo_s) & (S_co <= hi_s), axis=1))
print(f"fraction of whole sampled profiles inside the pointwise 90% band: {inside_pw:.3f}")
print(f"simultaneous band: half-width = {k_sim:.2f} sd (pointwise Gaussian 90%: {norm.ppf(0.95):.3f} sd); "
      f"profiles inside: {inside_sim:.3f}")
i_a, i_b = 4, 5
pa = np.mean(S_co[:, i_a] > S_co[:, i_b])
ci_a, ci_b = np.percentile(S_co[:, i_a], [5, 95]), np.percentile(S_co[:, i_b], [5, 95])
corr_ab = np.corrcoef(S_co[:, i_a], S_co[:, i_b])[0, 1]
print(f"patch at {zc[i_a]:.1f} km: 90% interval [{ci_a[0]:.2f}, {ci_a[1]:.2f}] m; patch at {zc[i_b]:.1f} km: "
      f"[{ci_b[0]:.2f}, {ci_b[1]:.2f}] m (they overlap)")
print(f"correlation {corr_ab:+.2f};  P(slip at {zc[i_a]:.1f} km > slip at {zc[i_b]:.1f} km | d) = {pa:.4f}")
diff = S_co[:, i_b] - S_co[:, i_a]
print(f"difference (deeper minus shallower): 90% interval [{np.percentile(diff, 5):.3f}, {np.percentile(diff, 95):.3f}] m")

fig, axs = plt.subplots(1, 2, figsize=(13, 4.6), gridspec_kw=dict(width_ratios=[1.15, 1]))
ax = axs[0]
ax.fill_betweenx(zc, lo_s, hi_s, color="#ecebe6", lw=0, label=f"simultaneous 90 % band ({inside_sim:.2f} of profiles inside)")
ax.fill_betweenx(zc, lo, hi, color="#c3c2b7", lw=0, label=f"pointwise 90 % band ({inside_pw:.2f} of profiles inside)")
outs = np.flatnonzero(~np.all((S_co >= lo) & (S_co <= hi), axis=1))[:4]
for k in outs:
    ax.plot(S_co[k], zc, color=S.BLUE, lw=0.8, alpha=0.9)
ax.plot([], [], color=S.BLUE, lw=0.8, label="4 samples that leave the pointwise band")
ax.plot(mu_co, zc, color=S.INK, lw=1.6, label="posterior mean")
ax.invert_yaxis(); ax.set_xlabel("slip (m)"); ax.set_ylabel("depth (km)")
ax.set_title("(a) A band for each patch is not a band for the profile")
ax.legend(fontsize=7.8, loc="lower right")
ax = axs[1]
ax.plot(S_co[:2000, i_a], S_co[:2000, i_b], ".", color=S.BLUE, ms=2, alpha=0.4)
ax.add_patch(plt.Rectangle((ci_a[0], ci_b[0]), ci_a[1] - ci_a[0], ci_b[1] - ci_b[0], fill=False, ec=S.INK2, lw=1.2,
                           ls=(0, (4, 2)), label="box of the two 90 % intervals"))
lim = [np.percentile(S_co[:, [i_a, i_b]], 0.1) - 0.05, np.percentile(S_co[:, [i_a, i_b]], 99.9) + 0.05]
ax.plot(lim, lim, color=S.ORANGE, lw=1.3, label="equal slip on both patches")
ax.plot(s_true[i_a], s_true[i_b], "*", color=S.INK, ms=12, mec=S.SURFACE, label="true value")
ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
ax.set_xlabel(f"slip at {zc[i_a]:.1f} km (m)"); ax.set_ylabel(f"slip at {zc[i_b]:.1f} km (m)")
ax.set_title("(b) Overlapping intervals, yet ordered")
ax.legend(fontsize=7.8, loc="lower right")
fig.tight_layout()
bk.savefig(fig, "2-32_marginal_joint")
plt.show()

# %% [markdown]
# **How to read it.** (a) The darker band is the usual per-patch 90 % band; only the printed fraction of whole sampled
# profiles stays inside it everywhere (blue: four that leave it). The light band is wide enough to contain 90 % of whole
# profiles. (b) Two neighbouring patches. Their 90 % intervals overlap (dashed box), which is often read as "cannot
# tell which slipped more". The joint samples (blue) lie almost entirely on one side of the equal-slip line, because
# the two are positively correlated, so the ordering is nearly certain (probability printed).
#
# **Common misreadings.**
# * "Overlapping error bars mean no significant difference." Not even for independent estimates (computed below: two
#   independent estimates 0 and 3, each with standard error 1, have overlapping 95 % intervals although their difference
#   is 2.1 standard errors). With correlation, as here, the overlap says even less. Compute the probability of the
#   difference from the joint samples; that handles dependence directly.
# * "The true profile lies in the ±2σ band." Per patch, roughly; for the whole profile at once, much less often.

# %%
z95 = norm.ppf(0.975)
dz = 3.0 / np.sqrt(2.0)                                           # difference of two independent estimates, each se 1
print(f"estimate 1: 0 +- 1, 95 % interval [{-z95:.2f}, {z95:.2f}];  estimate 2: 3 +- 1, 95 % interval [{3 - z95:.2f}, {3 + z95:.2f}]")
print(f"the intervals overlap ({3 - z95:.2f} < {z95:.2f}), yet the difference 3 has standard error sqrt(2) = {np.sqrt(2):.3f}: "
      f"z = {dz:.3f}, two-sided p = {2 * norm.sf(dz):.4f}")
print(f"Bayesian reading (flat priors, Gaussian likelihoods): P(theta_2 > theta_1 | data) = {norm.cdf(dz):.4f}")

# %% [markdown]
# ## 4. Posterior predictive checks: can the model reproduce the data?
#
# **In plain words.** Take a model from the posterior, let it predict the data, add the noise the likelihood assumes,
# and you have a fake data set the model thinks is typical. Do this thousands of times. If the real data look unusual
# among the fakes in some respect you care about (a misfit, the largest residual, residuals that run in streaks), the
# model is missing something. If they look typical, the check has found nothing; that is not proof the model is right.
#
# **Precisely.** The posterior predictive distribution is $p(\mathbf d^\text{rep}\mid\mathbf d) = \int p(\mathbf d^\text{rep}\mid\mathbf m)\,p(\mathbf m\mid\mathbf d)\,d\mathbf m$.
# For a test statistic $T(\mathbf d, \mathbf m)$, the posterior predictive p-value is
# $p_B = P\big(T(\mathbf d^\text{rep},\mathbf m)\ge T(\mathbf d,\mathbf m)\mid\mathbf d\big)$, estimated by drawing $\mathbf m^{(s)}$ and
# $\mathbf d^{\text{rep}(s)}$ (Gelman et al., Bayesian Data Analysis): the probability, under the fitted model, that
# replicated data look at least as extreme as the real data in the chosen statistic. Values near 0 or 1 flag a misfit. Three statistics of the
# residuals $r_i = (d_i - \mathbf g_i^\top\mathbf m)/\sigma$: $\chi^2 = \sum r_i^2$, $\max_i|r_i|$, and the lag-one correlation of the
# residuals ordered west to east (streaks).
#
# Three data sets, all inverted with the same correct-looking model (σ = 3 mm, the half-space Green's functions):
# 1. **well specified**: data from the model plus 3 mm noise;
# 2. **two offsets**: the same, plus unmodelled offsets of +15 mm and −15 mm at the stations 35 km west and 50 km east
#    (instrument changes that were not removed);
# 3. **wrong Earth**: the data come from an Earth 15 % deeper than the model's (the 2.13 depth-scale error, θ = 0.15).

# %%
def ppc(d, ns=4000, seed=0):
    rg = np.random.default_rng(seed)
    mu_, C_ = posterior(d)
    Sm = draw(mu_, C_, ns, rg)
    pred = Sm @ G0.T
    yrep = pred + SIG * rg.standard_normal(pred.shape)

    def stats(Rm):
        Rc = Rm - Rm.mean(1, keepdims=True)
        return {"χ²": (Rm ** 2).sum(1), "max |r|": np.abs(Rm).max(1),
                "lag-one correlation": (Rc[:, 1:] * Rc[:, :-1]).sum(1) / (Rc ** 2).sum(1)}
    T_obs, T_rep = stats((d[None, :] - pred) / SIG), stats((yrep - pred) / SIG)
    pv = {k: np.mean(T_rep[k] >= T_obs[k]) for k in T_obs}
    return pv, T_obs, T_rep, mu_, C_


i_w, i_e = np.flatnonzero(xs == -35)[0], np.flatnonzero(xs == 50)[0]


def make_cases(seed):
    e = SIG * np.random.default_rng(seed).standard_normal(ND)
    d1 = G0 @ s_true + e
    d2 = d1.copy()
    d2[i_w] += 15.0
    d2[i_e] -= 15.0
    d3 = Gmat(1.15) @ s_true + e
    return {"well specified": d1, "two offsets": d2, "wrong Earth (15 % deeper)": d3}


CASES = make_cases(100)
out4 = {k: ppc(d) for k, d in CASES.items()}
for k, (pv, *_rest) in out4.items():
    print(f"{k:26s} p-values: " + ", ".join(f"{s} {v:.3f}" for s, v in pv.items()))

fig, axs = plt.subplots(3, 4, figsize=(16, 10.5), gridspec_kw=dict(width_ratios=[1.1, 1, 1, 1]))
for row, (case, (pv, To, Tr, mu_, C_)) in enumerate(out4.items()):
    ax = axs[row, 0]
    sd_ = np.sqrt(np.diag(C_))
    ax.fill_betweenx(zc, mu_ - 1.645 * sd_, mu_ + 1.645 * sd_, color="#d9d8d1", lw=0, label="posterior 90 % band")
    ax.plot(mu_, zc, color=S.BLUE, lw=1.6, label="posterior mean")
    ax.plot(s_true, zc, color=S.INK, lw=1.3, ls=(0, (4, 2)), label="true slip")
    ax.invert_yaxis(); ax.set_xlabel("slip (m)"); ax.set_ylabel("depth (km)")
    ax.set_title(f"({'abc'[row]}1) {case}: slip", fontsize=10)
    if row == 0:
        ax.legend(fontsize=7.8, loc="lower right")
    for col, stat in enumerate(["χ²", "max |r|", "lag-one correlation"], start=1):
        ax = axs[row, col]
        ax.plot(To[stat][:1500], Tr[stat][:1500], ".", color=S.BLUE, ms=2, alpha=0.35)
        lim = [min(To[stat].min(), Tr[stat][:1500].min()), max(To[stat].max(), Tr[stat][:1500].max())]
        ax.plot(lim, lim, color=S.INK, lw=1.0)
        ok = 0.025 < pv[stat] < 0.975
        S.badge(ax, f"p = {pv[stat]:.3f}", ok=ok, loc="upper left")
        ax.set_xlabel(f"{stat} of the real data"); ax.set_ylabel(f"{stat} of replicated data")
        ax.set_title(f"({'abc'[row]}{col + 1}) {stat}", fontsize=10)
fig.tight_layout()
bk.savefig(fig, "2-32_ppc")
plt.show()

# %% [markdown]
# **How to read it.** Each dot in columns 2-4 is one posterior draw: horizontal, the statistic for the real data with
# that draw's model; vertical, the statistic for data replicated from that model. If the real data are typical the dots
# straddle the diagonal ($p$ between 0.025 and 0.975, ✓). Row a: nothing flagged. Row b: the χ² and the largest
# residual of the real data exceed almost every replicate, so the offsets are caught. Row c: every statistic looks
# normal, yet the slip (c1) is pushed deeper than the truth: the slip absorbed the wrong Earth, exactly as in 2.13, so
# the predictions fit and no residual statistic can see it.
#
# How reliable is this? The same three cases with 20 independent noise draws: the fraction of draws in which each
# check flags a problem ($p < 0.025$ or $p > 0.975$), and how often the 90 % band of each patch contains the true slip.

# %%
NSEED = 20
flag = {k: {s: 0 for s in ["χ²", "max |r|", "lag-one correlation"]} for k in CASES}
cover = {k: [] for k in CASES}
for sd in range(NSEED):
    cs = make_cases(200 + sd)
    for k, d in cs.items():
        pv, To, Tr, mu_, C_ = ppc(d, ns=2000, seed=sd)
        for s_, v in pv.items():
            flag[k][s_] += int(v < 0.025 or v > 0.975)
        cover[k].append(np.mean(np.abs(mu_ - s_true) < 1.645 * np.sqrt(np.diag(C_))))
print(f"{'case':26s} " + "  ".join(f"{s:>20s}" for s in flag["well specified"]) + "   90% band covers the true slip (patches)")
for k in CASES:
    print(f"{k:26s} " + "  ".join(f"{flag[k][s] / NSEED:20.2f}" for s in flag[k]) + f"   {np.mean(cover[k]):.2f}")

# %% [markdown]
# The table (fractions of the 20 noise draws) confirms the single example: the offsets are flagged by χ² and the largest
# residual in most draws, the well-specified case in none, and the wrong Earth no more often than the well-specified
# case, while its 90 % bands cover the true slip on fewer patches (last column). For one fixed true slip the patch
# coverage need not equal 0.90 (the 90 % refers to the prior's population of slip models); here it is above 0.90 for the
# well-specified case because the deep part of the posterior is wide around a true slip near zero. Compare the cases
# with each other, not with 0.90.
#
# **Common misreadings.**
# * "The posterior predictive check passed, so the model is right." It passed for the statistics you chose. A model
#   error that the parameters can absorb leaves no trace in the residuals (row c); only other information (a second
#   data type, a different Earth model, $\mathbf C_p$ from 2.13) can reveal it.
# * "$p_B$ is a p-value of a significance test." It uses the data twice (to fit and to check), so it is conservative:
#   values near 0.5 are common even for imperfect models. Read extreme values as alarms, not middle values as approval.

# %% [markdown]
# ## 5. Simulation-based calibration: does the code return the right posterior?
#
# **In plain words.** A posterior predictive check asks whether the model fits the data. A different question is
# whether the *software* computes the posterior of the model you wrote down. Test it on problems where you know the
# answer: draw a "true" model from the prior, simulate data from it, run the full pipeline, and see where the true
# model falls among the posterior samples. Over many repeats, if the pipeline is right, the true model is equally
# likely to fall anywhere in its own posterior: below 10 % of the samples, in the middle, above 90 %. A pipeline that is
# too confident puts the truth in the tails too often; a biased one puts it on one side.
#
# **Precisely.** If $\mathbf m^*\sim p(\mathbf m)$, $\mathbf d\sim p(\mathbf d\mid\mathbf m^*)$ and $\mathbf m^{(1..L)}\sim p(\mathbf m\mid\mathbf d)$, then
# the rank of $f(\mathbf m^*)$ among $f(\mathbf m^{(1)}),\dots,f(\mathbf m^{(L)})$ is uniform on $\{0,\dots,L\}$ for any scalar $f$
# (Talts et al. 2018), and also for any scalar $f(\mathbf m, \mathbf d)$ that depends on the data as well (Modrák et al.
# 2023; neither paper is quoted here, only the idea). The test needs the prior used to generate the truths to be the
# prior in the pipeline. Uniform ranks are what a correct pipeline produces; the converse does not hold, so a passed
# test means "no departure detected for these quantities with this many repeats", never "correct". Here: 1000 repeats,
# $L$ = 99 draws, four quantities: slip at 3.5 km, which the data constrain well; slip at 24.5 km, which they hardly
# constrain; the total slip, proportional to the moment; and the log-likelihood of the simulated data,
# $-\tfrac12\lVert\mathbf d - \mathbf G\mathbf m\rVert^2/\sigma^2$, a data-dependent quantity. Six pipelines:
#
# | pipeline | what it does |
# |---|---|
# | exact | the closed-form Gaussian posterior |
# | MCMC, long | random-walk Metropolis from the prior mean, 10,000 burn-in steps, then 99 draws thinned by 100 |
# | σ typed as 1.5 mm | exact posterior, but the likelihood uses half the true noise |
# | prior mean typed as 0 | exact posterior, but the inference uses prior mean 0 m (the truths come from mean 1 m) |
# | MCMC, short | the same sampler stopped after 400 steps; the last 99 steps are kept |
# | returns the prior | ignores the data and returns 99 draws from the prior |

# %%
RSBC, LD = 1000, 99
rng5 = np.random.default_rng(7)
Lm = np.linalg.cholesky(Cm)
M_true = m0[None, :] + (Lm @ rng5.standard_normal((N, RSBC))).T
D_sim = M_true @ G0.T + SIG * rng5.standard_normal((RSBC, ND))


def exact_draws(sig_l=SIG, m0_l=m0):
    Sm = G0 @ Cm @ G0.T + sig_l ** 2 * np.eye(ND)
    K = np.linalg.solve(Sm, G0 @ Cm).T
    C = Cm - K @ G0 @ Cm
    Lp = np.linalg.cholesky(0.5 * (C + C.T) + 1e-10 * np.eye(N))
    means = m0_l[None, :] + (D_sim - (G0 @ m0_l)[None, :]) @ K.T
    return means[:, None, :] + np.einsum("ij,rlj->rli", Lp, rng5.standard_normal((RSBC, LD, N)))


def rwm_draws(n_steps, thin, burn):
    """Random-walk Metropolis run on all 1000 data sets at once; proposal = posterior covariance x 2.38^2 / N."""
    Sm = G0 @ Cm @ G0.T + SIG ** 2 * np.eye(ND)
    K = np.linalg.solve(Sm, G0 @ Cm).T
    C = Cm - K @ G0 @ Cm
    Lprop = np.linalg.cholesky(0.5 * (C + C.T) + 1e-10 * np.eye(N)) * 2.38 / np.sqrt(N)

    def logp(X):
        r = D_sim - X @ G0.T
        zz = solve_triangular(Lm, (X - m0).T, lower=True)
        return -0.5 * (r ** 2).sum(1) / SIG ** 2 - 0.5 * (zz ** 2).sum(0)
    X = np.tile(m0, (RSBC, 1))
    lp = logp(X)
    out, acc = [], 0.0
    for it in range(n_steps):
        Y = X + (Lprop @ rng5.standard_normal((N, RSBC))).T
        ly = logp(Y)
        a = np.log(rng5.random(RSBC)) < ly - lp
        X[a], lp[a] = Y[a], ly[a]
        acc += a.mean()
        if it >= burn and (it - burn) % thin == 0:
            out.append(X.copy())
    return np.stack(out, 1)[:, -LD:], acc / n_steps


t5 = time.time()
mc_long, acc_long = rwm_draws(20000, thin=100, burn=10001)
mc_short, acc_short = rwm_draws(400, thin=1, burn=301)
PIPES = {"exact": exact_draws(), "MCMC, long": mc_long, "σ typed as 1.5 mm": exact_draws(sig_l=SIG / 2),
         "prior mean typed as 0": exact_draws(m0_l=np.zeros(N)), "MCMC, short": mc_short,
         "returns the prior": m0[None, None, :] + np.einsum("ij,rlj->rli", Lm, rng5.standard_normal((RSBC, LD, N)))}


def loglik_sim(X):
    """log-likelihood of each repeat's simulated data at the model(s) X: (RSBC, N) or (RSBC, L, N)."""
    d = D_sim if X.ndim == 2 else D_sim[:, None, :]
    return -0.5 * ((d - X @ G0.T) ** 2).sum(-1) / SIG ** 2


print(f"pipelines run in {time.time() - t5:.1f} s; Metropolis acceptance: long {acc_long:.2f}, short {acc_short:.2f}")
FQ = {"slip at 3.5 km": lambda X: X[..., 3], "slip at 24.5 km": lambda X: X[..., 24], "total slip": lambda X: X.sum(-1),
      "log-likelihood": loglik_sim}
NB = 20
edges = np.arange(0, LD + 2, (LD + 1) // NB) - 0.5
lo_b, hi_b = binom.ppf([0.005, 0.995], RSBC, 1 / NB)
sbc = {}
for pn, Dr in PIPES.items():
    for qn, f in FQ.items():
        rk = (f(Dr) < f(M_true)[:, None]).sum(1)
        h = np.histogram(rk, bins=edges)[0]
        sbc[(pn, qn)] = (h, chisquare(h).pvalue)
print(f"chi-square test of uniform ranks ({NB} bins); 99% band per bin [{lo_b:.0f}, {hi_b:.0f}] around {RSBC / NB:.0f}")
for pn in PIPES:
    print(f"  {pn:24s} " + "   ".join(f"{qn}: p = {sbc[(pn, qn)][1]:.2g}" for qn in FQ))

fig, axs = plt.subplots(4, 6, figsize=(19.5, 10.6))
for j, pn in enumerate(PIPES):
    for i, qn in enumerate(FQ):
        ax = axs[i, j]
        h, pval = sbc[(pn, qn)]
        ax.axhspan(lo_b, hi_b, color="#ecebe6", lw=0)
        ax.axhline(RSBC / NB, color=S.AXIS, lw=0.8)
        ax.bar(np.arange(NB), h, width=0.85, color=S.BLUE if pval > 0.01 else S.ORANGE)
        ax.set_ylim(0, max(230, 1.12 * h.max()))
        S.badge(ax, "no departure detected" if pval > 0.01 else "departure detected", ok=pval > 0.01, loc="upper left", fs=7.4)
        ax.set_xticks([0, NB - 1]); ax.set_xticklabels(["truth below\nall draws", "truth above\nall draws"], fontsize=7.2)
        if i == 0:
            ax.set_title(pn, fontsize=10)
        if j == 0:
            ax.set_ylabel(f"{qn}\nrepeats per bin")
fig.tight_layout()
bk.savefig(fig, "2-32_sbc")
plt.show()

# %% [markdown]
# **How to read it.** Each panel is a histogram of the rank of the true value among 99 posterior draws, over 1000
# repeats; the grey band is where 99 % of bins of a uniform histogram fall (the y-axis grows where a histogram is very
# peaked). The χ² test of uniformity compares the 20 bin counts with the 50 expected per bin; its p-value is the chance
# of a mismatch at least this large if the ranks really were uniform. Blue panels show no departure at the 1 % level,
# orange ones do (p-values printed).
# * **exact** and **MCMC, long**: no departure detected in any of the four quantities. That is consistent with a correct
#   pipeline; it does not prove one, since a small enough error would not show with 1000 repeats.
# * **σ typed as 1.5 mm**: a U shape (the truth is too often outside the posterior: overconfident) for the well-constrained
#   quantities. The slip at 24.5 km, which the data barely constrain, passes the 1 % test (p printed) with only slightly
#   raised end bins: the bug is much weaker where the prior dominates.
# * **prior mean typed as 0**: the shallow, data-dominated patch passes; the deep patch and the total slip show a tilt
#   (the truth sits systematically on one side). This bug hides where the data dominate.
# * **MCMC, short**: a deep U in every quantity. 99 consecutive steps of a chain that has not converged are too close
#   together, so the truth almost always lies outside them.
# * **returns the prior**: the three parameter rows show no departure, because a truth drawn from the prior is ranked
#   among other draws from the same prior. Only the log-likelihood row catches it: the truth explains its own data far
#   better than prior draws do, so its rank piles up at the top. A pipeline that ignores the data entirely passes
#   ordinary parameter-rank SBC.
#
# The lesson: SBC checks a pipeline, one quantity at a time, and different bugs show in different quantities. Run it
# for several, including the ones you will report and at least one that depends on the data, and read a pass as "no
# departure detected", not as "calibrated".
#
# **Common misreadings.**
# * "SBC tests whether my model is right." It tests whether the code computes the posterior of the model it was given.
#   It says nothing about the Earth; that is the job of section 4 and of independent data.
# * "One synthetic test with a known answer is enough." One test is one draw from this histogram. A pipeline that is
#   overconfident by a factor of two passes a single test about half the time.
# * "SBC passed, so the posterior is right." It passed for these quantities and this many repeats. The prior-returning
#   pipeline passes three of the four.

# %% [markdown]
# ## 6. Comparing models by prediction: LOO-CV and WAIC vs AIC and BIC
#
# **In plain words.** 2.29 compared models with AIC and BIC, which count parameters. A Bayesian model with a prior does
# not use all its parameters freely: the prior holds some of them back, so counting them overstates its flexibility.
# Leave-one-out cross-validation asks directly: fit the model without station $i$, predict station $i$, score the
# prediction, repeat for every station. WAIC estimates the same score from a single fit.
#
# **Precisely.** The target is the expected log predictive density for new data, elpd $=\sum_i E_{\tilde d_i}\big[\ln p(\tilde d_i\mid\mathbf d)\big]$.
# Estimates (higher is better, all on the same scale):
# * exact LOO: $\text{elpd}_\text{loo} = \sum_i \ln p(d_i\mid\mathbf d_{-i})$, here by refitting without each station (24 refits);
# * importance-sampling LOO from one set of posterior samples: reweight the full-data posterior samples so that they
#   represent the posterior without datum $i$ (importance weights, reminder: 2.20), with weights
#   $1/p(d_i\mid\mathbf m^{(s)})$ truncated at $\bar w\sqrt S$ to stop a few huge weights from dominating (Ionides 2008).
#   Vehtari, Gelman & Gabry (2017) stabilise these weights with a Pareto fit, a heavy-tailed distribution fitted to the
#   largest weights, whose shape also warns when the estimate cannot be trusted (PSIS-LOO,
#   used by ArviZ and the R package loo); neither paper nor ArviZ is available here, so the simpler truncation is used
#   and graded against exact LOO;
# * WAIC: $\text{lppd} - p_\text{waic}$, with $\text{lppd}=\sum_i\ln\frac1S\sum_s p(d_i\mid\mathbf m^{(s)})$ and $p_\text{waic}=\sum_i\mathrm{Var}_s\ln p(d_i\mid\mathbf m^{(s)})$;
# * AIC and BIC from 2.29 on the same scale: $\ln\hat L - k$ and $\ln\hat L - \tfrac k2\ln N_d$.
#
# $p_\text{loo} = \text{lppd} - \text{elpd}_\text{loo}$ and $p_\text{waic}$ are **effective numbers of parameters**. The models: slip on
# $n$ equal patches between 0 and 30 km, $n$ = 3 to 30, prior $N(1, 2^2)$ m per patch, 24 stations. The true slip is
# smooth and defined on a 60-patch grid, so no candidate is exactly right. Because the stations sit symmetrically on
# both sides of the fault and antiplane displacement is antisymmetric, the 24 stations give only 12 independent
# combinations. Above $n$ = 12 a maximum-likelihood (least-squares) fit still exists and its predictions are unique,
# but the parameters are not: any slip pattern in the null space of $\mathbf G$ can be added without changing the fit.
# The textbook AIC and BIC count $k = n$ parameters and assume an identifiable model, so they are computed only up to
# $n$ = 12, where that formula applies. The referee is the true elpd, computed exactly for this data set from the known
# truth.

# %%
zf = np.linspace(0, 30, 61)
s_fine = 3.0 * np.exp(-0.5 * ((0.5 * (zf[1:] + zf[:-1]) - 9) / 3.5) ** 2)
f_true = Gmat(1.0, edges=zf) @ s_fine
NMOD = [3, 4, 5, 6, 8, 10, 12, 15, 20, 30]
GS = {n: Gmat(1.0, edges=np.linspace(0, 30, n + 1)) for n in NMOD}
PRIOR_M, PRIOR_SD = 1.0, 2.0
s2 = SIG ** 2
print("rank of G for each n:", {n: int(np.linalg.matrix_rank(G)) for n, G in GS.items()})


def gpost(G, y):
    n = G.shape[1]
    C = np.linalg.inv(np.eye(n) / PRIOR_SD ** 2 + G.T @ G / s2)
    return C @ (np.full(n, PRIOR_M) / PRIOR_SD ** 2 + G.T @ y / s2), C


def criteria(y, rg, S_=4000):
    out = {}
    for n, G in GS.items():
        mu_, C_ = gpost(G, y)
        loo_i = []
        for i in range(ND):
            k = np.arange(ND) != i
            mi, Ci = gpost(G[k], y[k])
            v = G[i] @ Ci @ G[i] + s2
            loo_i.append(-0.5 * np.log(2 * np.pi * v) - 0.5 * (y[i] - G[i] @ mi) ** 2 / v)
        loo_i = np.array(loo_i)
        Ms = mu_[None, :] + (np.linalg.cholesky(C_ + 1e-12 * np.eye(n)) @ rg.standard_normal((n, S_))).T
        ll = -0.5 * np.log(2 * np.pi * s2) - 0.5 * (y[None, :] - Ms @ G.T) ** 2 / s2
        lppd_i = logsumexp(ll, 0) - np.log(S_)
        pw = ll.var(0, ddof=1)
        isl = []
        for i in range(ND):
            lw = -ll[:, i] - (-ll[:, i]).max()
            w = np.exp(lw)
            w = np.minimum(w, w.mean() * np.sqrt(S_))
            isl.append(logsumexp(ll[:, i], b=w) - np.log(w.sum()))
        v = np.einsum("ij,jk,ik->i", G, C_, G) + s2
        true = np.sum(-0.5 * np.log(2 * np.pi * v) - (s2 + (f_true - G @ mu_) ** 2) / (2 * v))
        r_ = dict(loo=loo_i.sum(), loo_i=loo_i, isloo=np.sum(isl), waic=np.sum(lppd_i - pw), p_waic=pw.sum(),
                  p_loo=lppd_i.sum() - loo_i.sum(), true=true, aic=np.nan, bic=np.nan)
        if np.linalg.matrix_rank(G) == n:
            mh = np.linalg.lstsq(G, y, rcond=None)[0]
            lL = np.sum(-0.5 * np.log(2 * np.pi * s2) - 0.5 * (y - G @ mh) ** 2 / s2)
            r_.update(aic=lL - n, bic=lL - 0.5 * n * np.log(ND))
        out[n] = r_
    return out


rng6 = np.random.default_rng(5)
y6 = f_true + SIG * rng6.standard_normal(ND)
c6 = criteria(y6, rng6)
print(f"{'n':>3s} {'true elpd':>10s} {'exact LOO':>10s} {'IS-LOO':>8s} {'WAIC':>8s} {'AIC/-2':>8s} {'BIC/-2':>8s} {'p_loo':>7s} {'p_waic':>7s}")
for n in NMOD:
    r_ = c6[n]
    print(f"{n:3d} {r_['true']:10.2f} {r_['loo']:10.2f} {r_['isloo']:8.2f} {r_['waic']:8.2f} {r_['aic']:8.2f} {r_['bic']:8.2f} "
          f"{r_['p_loo']:7.2f} {r_['p_waic']:7.2f}")
best_loo = max(NMOD, key=lambda n: c6[n]["loo"])
for n in [10, 12]:
    dif = c6[best_loo]["loo_i"] - c6[n]["loo_i"]
    print(f"LOO difference, n = {best_loo} minus n = {n}: {dif.sum():.2f} +/- {np.sqrt(ND * dif.var()):.2f} (standard error)")

# %%
REP6 = 200
rng6b = np.random.default_rng(11)
CRIT = ["loo", "isloo", "waic", "aic", "bic"]
regret = {c: [] for c in CRIT}
chosen = {c: [] for c in CRIT + ["true"]}
for _ in range(REP6):
    cc = criteria(f_true + SIG * rng6b.standard_normal(ND), rng6b, S_=2000)
    tb = max(NMOD, key=lambda n: cc[n]["true"])
    chosen["true"].append(tb)
    for c in CRIT:
        cand = [n for n in NMOD if np.isfinite(cc[n][c])]
        pick = max(cand, key=lambda n: cc[n][c])
        chosen[c].append(pick)
        regret[c].append(cc[tb]["true"] - cc[pick]["true"])
NAMES6 = {"loo": "exact LOO", "isloo": "IS-LOO", "waic": "WAIC", "aic": "AIC", "bic": "BIC", "true": "true elpd"}
print(f"over {REP6} data sets: model chosen (counts) and mean loss of true elpd versus the best model")
print(f"{'criterion':>10s} " + " ".join(f"n={n:<3d}" for n in NMOD) + "   mean loss")
for c in CRIT + ["true"]:
    cnt = [chosen[c].count(n) for n in NMOD]
    loss = f"{np.mean(regret[c]):.2f}" if c != "true" else "0"
    print(f"{NAMES6[c]:>10s} " + " ".join(f"{k:<5d}" for k in cnt) + f"   {loss}")

fig, axs = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw=dict(width_ratios=[1.25, 1, 1]))
ax = axs[0]
show = [n for n in NMOD if n >= 6]
COL6 = {"loo": S.BLUE, "isloo": S.BLUE, "waic": S.AQUA, "aic": S.ORANGE, "bic": S.VIOLET}
ax.plot(show, [c6[n]["true"] for n in show], "o-", color=S.INK, lw=2.0, ms=4, label="true elpd (referee)")
for c in CRIT:
    vals = [c6[n][c] for n in show]
    ax.plot(show, vals, "o", ls=(0, (4, 2)) if c == "isloo" else "-", color=COL6[c], lw=1.3, ms=3.5,
            label=NAMES6[c] + (" (max ln L − k)" if c == "aic" else " (max ln L − ½k ln N)" if c == "bic" else ""))
ax.set_xscale("log"); ax.set_xticks(show); ax.set_xticklabels([str(n) for n in show]); ax.minorticks_off()
ax.set_xlabel("number of patches n"); ax.set_ylabel("expected log predictive density")
ax.set_title("(a) Estimates of predictive accuracy, one data set")
ax.legend(fontsize=7.6, loc="lower center", ncol=2)
ax = axs[1]
ax.plot(NMOD, NMOD, color=S.AXIS, lw=1.0, ls=(0, (4, 2)), label="k = n (parameter count)")
ax.axhline(12, color=S.MUTED, lw=0.8, ls=(0, (1, 1.5)), label="12 independent data combinations")
ax.plot(NMOD, [c6[n]["p_loo"] for n in NMOD], "o-", color=S.BLUE, ms=4, label="$p_{loo}$")
ax.plot(NMOD, [c6[n]["p_waic"] for n in NMOD], "s-", color=S.AQUA, ms=3.5, label="$p_{waic}$")
ax.set_xscale("log"); ax.set_yscale("log")
ax.set_xticks(NMOD); ax.set_xticklabels([str(n) for n in NMOD]); ax.minorticks_off()
ax.set_xlabel("number of patches n"); ax.set_ylabel("effective number of parameters")
ax.set_title("(b) Effective vs counted parameters")
ax.legend(fontsize=7.8, loc="upper right")
ax = axs[2]
yy = np.arange(len(CRIT))[::-1]
ax.barh(yy, [np.mean(regret[c]) for c in CRIT], color=[COL6[c] for c in CRIT], height=0.6)
for y_, c in zip(yy, CRIT):
    ax.text(np.mean(regret[c]) + 0.03, y_, f"{np.mean(regret[c]):.2f}", va="center", fontsize=8.5)
ax.set_yticks(yy); ax.set_yticklabels([NAMES6[c] + (" (n ≤ 12)" if c in ("aic", "bic") else "") for c in CRIT])
ax.set_xlabel("mean loss of true elpd vs the best model")
ax.set_title(f"(c) Cost of trusting each criterion ({REP6} data sets)")
fig.tight_layout()
bk.savefig(fig, "2-32_loo_waic")
plt.show()

# %% [markdown]
# **How to read it.** (a) One data set. The true elpd (black) rises steeply while the patches are too coarse to
# represent the slip, then flattens: from about 10 patches on, the prior keeps the extra patches in check and
# predictions barely change. Exact LOO, IS-LOO and WAIC track that curve within a few units (table above). AIC and BIC
# (orange, violet) stop at 12 patches and keep charging for every parameter, so they turn down earlier. The standard
# errors printed above show that differences of a unit or two between the larger models are within the noise of LOO
# itself. (b) The effective number of parameters: for coarse models it is far *above* the parameter count, a sign that
# the model cannot fit the data (LOO and WAIC then report a misfit, not flexibility); for fine models it levels off well
# below both the count and the 12 independent data combinations, because the prior holds the rest. (c) Over 200 data
# sets: how much predictive accuracy is lost on average by picking the model each criterion prefers (0 = always the
# best). The printed table gives the choices themselves. One more thing the first table shows: for the badly
# misspecified coarse models, IS-LOO departs from exact LOO by 84 units (n = 3) and 9 units (n = 4), because a few samples dominate
# the importance weights. That is the failure the Pareto diagnostic of PSIS-LOO is designed to flag; exact refits (or
# that diagnostic) are needed before trusting IS-LOO on a poorly fitting model.
#
# **Common misreadings.**
# * "A model with more parameters is penalised in LOO." Only to the extent that its extra parameters change the
#   predictions. With a prior, $p_\text{loo}$ counts what the data actually used.
# * "LOO works for any data." Leaving out one datum assumes the data split into conditionally independent pieces
#   (independent once the model parameters are known). For
#   correlated errors (2.14), leave out whole blocks (an interferogram, a station's time series) instead.
# * "The best LOO wins." A difference smaller than about two standard errors (printed) does not separate the models.

# %% [markdown]
# ## 7. Reporting checklist
#
# | report | how | why |
# |---|---|---|
# | every derived number | computed per sample, then summarised; say "sampled" or "derived per sample" vs "from the mean model" | Jensen (section 1) |
# | the summary used | mean, median or MAP, and in which parameterisation | they differ for skewed posteriors (section 2) |
# | intervals | equal-tailed or HPD, pointwise or simultaneous | section 2 and 3 |
# | comparisons between parameters | probabilities from joint samples | overlapping intervals mislead (section 3) |
# | fit per data set | WRMS and χ² per data set, and the residual plot | a single number hides which data misfit (2.29) |
# | model adequacy | posterior predictive checks with statistics aimed at plausible failures | and say what they cannot see (section 4) |
# | sampler health | R-hat, ESS, divergences, per-chain energy (notebooks 2.18-2.19) | a converged-looking run can still be wrong |
# | pipeline correctness | SBC on the quantities you report plus a data-dependent one, once per new pipeline; report "no departure detected" with the number of repeats | section 5 |
# | model choice | LOO or WAIC with standard errors; AIC/BIC only when k is well defined | section 6, 2.29 |
# | figures | the posterior itself (samples or bands), not only the numbers | look at the picture |
#
# **In practice.**
# * Reporting WRMS and χ² per data set (2.29) describes the fit; section 6 adds a predictive alternative to the evidence
#   for model choice. It needs pointwise log-likelihoods and conditionally independent pieces of data.
# * A single synthetic test (invert synthetic data made with the same forward code, check that the truth comes back)
#   catches gross mismatches such as data and predictions stored in different orders. SBC is the systematic version:
#   many synthetic truths drawn from the prior, each ranked within the pipeline's own posterior.
# * Sampler diagnostics (R-hat, ESS, divergences, per-chain energy; row 7) check the sampler; sections 4 and 5 are the
#   two checks they cannot do.
#
# Next: **2.33 · Capstone: one fault-slip inversion, every choice**.

# %%
print(f"notebook run time: {time.time() - T0:.0f} s")
