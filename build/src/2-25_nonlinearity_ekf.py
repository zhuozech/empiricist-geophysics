# %% [markdown]
# # 2.25 · Nonlinearity: how to measure it, and how EKF, EnKI and the samplers cope
#
# **Before this:** 2.17 (Jacobian, linearisation), 2.23 (Kalman filter), 2.24 (EnKI); 2.18-2.20 for the samplers.
# **You will be able to:** measure how nonlinear a forward model is over the region that matters, in noise units;
# explain what the extended Kalman filter does and how its covariance differs from the Laplace one; predict how the
# EKF, EnKI and the samplers degrade as nonlinearity grows; see why a filter that linearises each datum once is biased.
# **Optional sections:** 5 (the sequential EKF day by day).
#
# A forward model is *linear* when doubling the parameters doubles the predictions. With a Gaussian prior and
# Gaussian noise the posterior is then Gaussian and the Kalman formula is exact. Real geodetic problems are partly nonlinear (afterslip that grows with
# $\log(1 + t/\tau)$, a source depth, a rupture time, a jointly estimated hyperparameter), and every Kalman-type
# method has to approximate the nonlinearity somehow. This notebook makes "how nonlinear" a number, turns it up with
# one knob, and grades every method as it rises, including the **extended Kalman filter (EKF)** that Qiu et al.
# (2018) used in the form of the Extended Network Inversion Filter.
#
# **The problems here are small on purpose (2 unknowns)**, so the exact posterior can be computed on a grid and every
# method can be graded against it. The lessons carry over to large problems; the grading does not.
#
# **Words this notebook defines:** linear (superposition), tangent / linearisation, Jacobian, degree of
# nonlinearity, statistical linearisation, extended Kalman filter (classic, iterated, tempered), Network Inversion
# Filter, stranded ensemble members.

# %%
import sys, pathlib, time, inspect
sys.path.insert(0, str(pathlib.Path.cwd().parent))
import numpy as np
import matplotlib.pyplot as plt
import jax, jax.numpy as jnp
import bayeskit as bk
from bayeskit import problems as P, samplers as SM, diagnostics as D, plotting as BP, style as S

rng = np.random.default_rng(10)
EKF_C = S.INK2                                     # EKF: dark grey (ink family), EnKI stays violet

# %% [markdown]
# ## 1. The test problem and its knob
#
# | | |
# |---|---|
# | unknowns | 2: epicentre east and north (units of 10 km) |
# | data | 1: the distance from one station, noise σ = 0.1 |
# | forward model | $d = \lvert \mathbf m - \mathbf s \rvert$ (nonlinear: a distance) |
# | prior | Gaussian, mean (0.5, 0.3), sd 1 in each direction |
# | **knob** | the station's distance $r$: it sits at $(0, -r)$ and the datum says "distance = $r$", so the data circle passes through (0, 0) |
#
# A datum constrains the parameters to a band (notebook 2.02). Here the band is a circle of radius $r$ around the
# station. **Far station (large $r$): across the prior the circle is almost a straight line, so the problem is almost
# linear. Close station: the circle bends inside the prior, and the posterior becomes a banana, then a ring.**
# The black line in each panel is the circle's tangent at the posterior mean: what any linearised method believes
# the datum says.

# %%
R_LIST = [100, 30, 10, 5, 2.5, 1.5, 1.0, 0.6]
SHOW = [100, 10, 2.5, 1.0]
probs = {r: P.location(kind="knob", r=r) for r in R_LIST}
WIDE = (-4.5, 4.5, -4.5, 4.5)       # compute on a wider window than we plot, so no posterior tail is cut off
grids = {r: P.grid(probs[r], extent=WIDE, n=600) for r in R_LIST}
print("largest probability on the edge of the computation grid: %.1e" % max(
    (g.P[0].sum() + g.P[-1].sum() + g.P[:, 0].sum() + g.P[:, -1].sum()) * g.dx * g.dy for g in grids.values()))

fig, axs = plt.subplots(1, len(SHOW), figsize=(14, 3.7))
th = np.linspace(0, 2 * np.pi, 600)
for ax, r in zip(axs, SHOW):
    pr, g = probs[r], grids[r]
    BP.density(ax, g)
    for k in (-2, 2):                                                     # the data band: circle +- 2 sigma
        ax.plot((r + k * 0.1) * np.cos(th), -r + (r + k * 0.1) * np.sin(th), color=S.MUTED, lw=0.6)
    mref = g.mean
    Jm, gm = pr.jacobian(mref)[0], pr.forward(mref)[0]
    # tangent line: points where G(mref) + J (m - mref) = r
    tdir = np.array([-Jm[1], Jm[0]]) / np.linalg.norm(Jm)
    base = mref + (r - gm) * Jm / (Jm @ Jm)
    ss = np.linspace(-6, 6, 2)
    ax.plot(base[0] + ss * tdir[0], base[1] + ss * tdir[1], color=S.INK, lw=1.0)
    ax.add_patch(plt.Circle(pr.m0, 2.0, fill=False, ls=(0, (1, 2)), color=S.INK2, lw=0.8))
    BP.frame(ax, pr, equal=True)
    ax.set_title(f"station at distance r = {r:g}", fontsize=10)
axs[0].text(0.03, 0.03, "dotted: prior 2 sd\ngrey lines: data band ±2σ\nblack: tangent (linearised datum)",
            transform=axs[0].transAxes, fontsize=7.5, color=S.INK2, va="bottom")
fig.tight_layout()
bk.savefig(fig, "2-25_knob")
plt.show()

# %% [markdown]
# ## 2. Measuring the degree of nonlinearity
#
# "Nonlinear" is not yes-or-no. What matters is how far the forward model departs from a straight line **over the
# region the method has to explore, measured against the data noise**. Three numbers:
#
# 1. **Taylor index $\mathrm{NL}$** (what a *local* linearisation misses: EKF, Laplace, Gauss-Newton). Over a set of
#    parameter values (prior draws or posterior draws), take the tangent of $G$ at their mean and compute the RMS of
#    $\big(G(\mathbf m) - \text{tangent}(\mathbf m)\big)/\sigma$. Below about 0.1 the problem is linear for all
#    practical purposes; around 1 the curvature is as large as the noise; well above 1 it dominates.
# 2. **Ensemble index** (what an *ensemble* Kalman update misses: EnKF, EKI, EnKI). Fit the best straight-line
#    relation between parameters and predictions across an ensemble, and measure the RMS residual, in noise units,
#    on members that were *not* used for the fit (cross-validation, below). It can be computed from the members and
#    predictions of any saved EnKI iteration, but only when the members clearly outnumber the parameters.
# 3. **Non-Gaussianity of the exact posterior**: the Kullback-Leibler divergence (nats; in brief in 2.15 and 2.21, fully in 2.28) between the posterior and the
#    Gaussian with the same mean and covariance. This is the *outcome*; the first two are the *cause*.
#
# (Bates & Watts 1980 formalised curvature measures of this kind for nonlinear regression; the indices here are the
# noise-unit versions you can compute for any forward model.)

# %%
rows = []
for r in R_LIST:
    pr, g = probs[r], grids[r]
    Xpost, Xprior = g.sample(20000, rng), pr.sample_prior(rng, 20000)      # 20,000 draws: Monte Carlo spread ~1 %
    rows.append(dict(r=r,
                     nl_prior=D.nonlinearity_taylor(pr, Xprior, m_ref=pr.m0)["nl"],
                     nl_post=D.nonlinearity_taylor(pr, Xpost)["nl"],
                     nl_ens=D.nonlinearity_ensemble(pr, Xprior)["nl"],
                     kl=D.gaussianity(g)["kl"]))
print(f"{'r':>6s} {'NL over prior':>14s} {'NL over posterior':>18s} {'ensemble index (prior draws)':>29s} {'KL (nats)':>10s}")
for w in rows:
    print(f"{w['r']:6g} {w['nl_prior']:14.2f} {w['nl_post']:18.2f} {w['nl_ens']:29.2f} {w['kl']:10.3f}")
NL = {w["r"]: w["nl_post"] for w in rows}

fig, axs = plt.subplots(1, 2, figsize=(11, 3.6))
rr = np.array(R_LIST)
ax = axs[0]
for lo, hi, lab in [(1e-3, 0.1, "linear"), (0.1, 1, "mildly nonlinear"), (1, 100, "strongly nonlinear")]:
    ax.axhspan(lo, hi, color=["#f3f2ee", "#e6e5df", "#d8d7d0"][["linear", "mildly nonlinear", "strongly nonlinear"].index(lab)], lw=0)
    ax.text(250, np.sqrt(lo * hi) if lo > 1e-3 else 0.05, lab, fontsize=8, color=S.INK2, va="center")
ax.loglog(rr, [w["nl_prior"] for w in rows], "o-", color=S.INK2, label="Taylor index over the prior")
ax.loglog(rr, [w["nl_post"] for w in rows], "s-", color=S.INK, label="Taylor index over the posterior")
ax.loglog(rr, [w["nl_ens"] for w in rows], "^-", color=S.COLORS["enki"], label="ensemble index (prior draws)")
ax.set_xlim(0.5, 300); ax.set_ylim(0.02, 30); ax.invert_xaxis()
ax.set_xlabel("station distance r  (closer →)"); ax.set_ylabel("degree of nonlinearity (noise units)")
ax.legend(loc="lower right", fontsize=8)
ax.set_title("Turning the knob")
ax = axs[1]
ax.semilogx([w["nl_post"] for w in rows], [w["kl"] for w in rows], "o-", color=S.INK)
for w in rows:
    ax.annotate(f"r={w['r']:g}", (w["nl_post"], w["kl"]), textcoords="offset points", xytext=(4, -10), fontsize=7.5, color=S.INK2)
ax.set_xlabel("Taylor index over the posterior"); ax.set_ylabel("KL(posterior ‖ its Gaussian), nats")
ax.set_title("Cause and outcome")
fig.tight_layout()
bk.savefig(fig, "2-25_nonlinearity_index")
plt.show()

# %% [markdown]
# **The image to keep: nonlinearity is a property of the forward model *over the region you are uncertain about*,
# measured against the noise.** The same distance function is almost linear for a far station and strongly
# nonlinear for a close one. Here the index over the prior and over the posterior are similar (first two columns)
# for every station but the closest, because a single datum narrows the posterior only across the circle, not along
# it: the region that matters for curvature stays about as wide as the prior. (For the closest station the prior
# cloud reaches across the whole tight circle, which inflates the prior index.) When data do shrink the posterior in every direction, the index falls with
# it; section 5 shows that happening day by day. That is the logic behind tempered methods (EnKI's θ steps and their
# cap, 2.24): each step only has to be linear over the current spread. KL rises with the index and falls again
# for the closest station, where the posterior turns into a fairly symmetric ring.

# %% [markdown]
# **Why the ensemble index is measured on held-out members.** With $J$ members and $d$ parameters, the straight line
# has $d + 1$ coefficients. When $J \le d + 1$ it can pass exactly through every member, so the residual *on the
# members used for the fit* is zero for any forward model, however nonlinear. Above, $J$ = 20,000 and $d$ = 2, so this
# never bites. Large inversions are the opposite case (hundreds of members for thousands of unknowns). Test: a strongly
# nonlinear forward model, $G(\mathbf x) = \sum_i x_i^2$ with 30 parameters, noise 1, seen by 20 and by 200 members.

# %%
import warnings
from types import SimpleNamespace
rng_cv = np.random.default_rng(20261011)
quad = SimpleNamespace(forward_batch=lambda X: np.sum(np.atleast_2d(X) ** 2, axis=1, keepdims=True), sigma=np.ones(1))
X20, X200, Xnew = rng_cv.standard_normal((20, 30)), rng_cv.standard_normal((200, 30)), rng_cv.standard_normal((2000, 30))
with warnings.catch_warnings(record=True) as w_:
    warnings.simplefilter("always")
    r20 = D.nonlinearity_ensemble(quad, X20)
r200 = D.nonlinearity_ensemble(quad, X200)
mu20 = X20.mean(0)                                          # the straight line fitted to the 20 members ...
c20 = np.linalg.lstsq(np.column_stack([np.ones(20), X20 - mu20]), quad.forward_batch(X20), rcond=None)[0]
miss_new = np.sqrt(np.mean((quad.forward_batch(Xnew) - np.column_stack([np.ones(2000), Xnew - mu20]) @ c20) ** 2))
G_lin = rng_cv.standard_normal((5, 30))                     # ... and a truly linear model for comparison
lin_model = SimpleNamespace(forward_batch=lambda X: np.atleast_2d(X) @ G_lin.T + 1.0, sigma=np.ones(5))
r200_lin = D.nonlinearity_ensemble(lin_model, X200)
print(f"20 members, 30 parameters, quadratic model: in-sample residual {r20['nl_train']:.1e} noise units "
      f"(looks perfectly linear); held-out index: {r20['nl']} (underdetermined = {r20['underdetermined']})")
print(f"   warning issued: {str(w_[0].message) if w_ else 'none'}")
print(f"   the same straight line, tested on 2,000 new draws, misses by {miss_new:.1f} noise units (RMS)")
print(f"200 members, 30 parameters, quadratic model: in-sample {r200['nl_train']:.2f}, held-out {r200['nl']:.2f} noise units")
print(f"200 members, 30 parameters, linear model:    in-sample {r200_lin['nl_train']:.1e}, held-out {r200_lin['nl']:.1e} noise units")

# %% [markdown]
# **How to read it.** With 20 members for 30 parameters the in-sample residual is at round-off level although the
# model is a pure quadratic; the function refuses to report an index (NaN, with a warning), and the same straight
# line misses new draws by many noise units (printed). With 200 members the held-out index is large for the quadratic
# model and at round-off for the linear one, as it should be. So an in-sample fit can never certify that an ensemble
# Kalman update is adequate, and with fewer members than parameters the ensemble index cannot be measured at all;
# check nonlinearity along a few directions with the Taylor index instead (a tangent needs one Jacobian-vector
# product per direction, 2.22).

# %% [markdown]
# ## 3. The extended Kalman filter
#
# The EKF replaces the forward model by its tangent at a chosen point,
# $G(\mathbf m) \approx G(\mathbf m_\ast) + \mathbf H(\mathbf m - \mathbf m_\ast)$ with $\mathbf H = \partial G/\partial\mathbf m$
# (the **Jacobian**, the matrix of derivatives of each predicted datum with respect to each parameter, pictured in
# 2.17; here from JAX), and then applies the ordinary Kalman update. It is a *gradient-based* Kalman method, where EnKI
# is a *derivative-free* one (the ensemble's average slope replaces $\mathbf H$). The EKF returns one Gaussian; EnKI
# returns an ensemble moved by mean-and-covariance updates. Three ways to choose where to linearise:
#
# * **classic EKF**: once, at the prior mean (what a filter does with each new data batch);
# * **iterated EKF** (Bell & Cathey 1993): re-linearise at the latest estimate until it stops moving. This is
#   undamped Gauss-Newton (2.10): when it converges it stops at a local MAP, but convergence is not guaranteed. Its
#   covariance is the Gauss-Newton one, $(\mathbf P_0^{-1} + \mathbf H^\top\mathbf R^{-1}\mathbf H)^{-1}$, which is the
#   Laplace covariance (2.08) only when the forward model's curvature, weighted by the residuals, is negligible (tested
#   below);
# * **tempered EKF**: assimilate the data in equal fractions and re-linearise before each one, like EnKI's θ steps.
#
# **The EKF in Qiu et al. (2018).** They used a modified Extended Network Inversion Filter (ENIF; Segall & Matthews
# 1997, McGuire & Segall 2003), a Kalman filter that steps through the postseismic GPS time series epoch by epoch.
# Their state holds afterslip on the megathrust and viscous strain in the mantle wedge; the spatial and temporal
# smoothing hyperparameters are estimated together with them, and afterslip is kept positive. "Extended" means the
# filter linearises whatever is not linear in its state, exactly as in this section. Section 5 builds a
# two-parameter version of the time-stepping idea.

# %%
print(inspect.getsource(SM.ekf))

# %%
r0 = 2.5
pr, g = probs[r0], grids[r0]
res3 = {mode: SM.ekf(pr, mode=mode) for mode in ("classic", "iterated", "tempered")}
for mode, e in res3.items():
    print(f"EKF {mode:9s}: mean {np.round(e['mean'], 3)}  sd {np.round(np.sqrt(np.diag(e['cov'])), 3)}  forward runs {e['n_forward']}")
print(f"exact posterior: mean {np.round(g.mean, 3)}  sd {np.round(np.sqrt(np.diag(g.cov)), 3)}")
en = SM.enki(pr, J=1000, rng=rng)
fig, axs = plt.subplots(1, 2, figsize=(9.5, 4.0))
e = res3["iterated"]
ax = axs[0]
BP.density(ax, g)
for q in (0.5, 0.9):
    BP.ellipse(ax, e["mean"], e["cov"], nsig=np.sqrt(-2 * np.log(1 - q)), color=EKF_C, lw=1.4)
ax.plot(*pr.m0, "P", color=S.INK, ms=8, mec=S.SURFACE)
ax.plot([0, pr.m0[0] * 1.6], [-r0, -r0 + (pr.m0[1] + r0) * 1.6], color=S.MUTED, lw=0.7)
Sd = rng.multivariate_normal(e["mean"], e["cov"], 4000)
S.badge(ax, f"{100 * g.inside(Sd):.0f} % of draws inside exact 90 % region", ok=g.inside(Sd) > 0.85, loc="lower left", fs=8)
BP.frame(ax, pr, equal=True, labels=False)
ax.set_title("EKF (classic = iterated = tempered here)", fontsize=10)
ax = axs[1]
BP.density(ax, g)
BP.samples(ax, en["samples"], S.COLORS["enki"], n_max=1500, s=3, alpha=0.5)
S.badge(ax, f"{100 * g.inside(en['samples']):.0f} % inside exact 90 % region", ok=g.inside(en["samples"]) > 0.85, loc="lower left", fs=8)
BP.frame(ax, pr, equal=True, labels=False)
ax.set_title(f"EnKI ensemble (J = 1000, {en['n_forward']} forward runs)", fontsize=10)
fig.tight_layout()
bk.savefig(fig, "2-25_ekf_variants")
plt.show()

# %% [markdown]
# Ellipses: the EKF Gaussian (50 % and 90 %); cross: prior mean; grey line: the ray from the station through the
# prior mean. **Here the three EKF variants give the same answer** (printed above), and the reason is worth a
# sentence: a distance is exactly linear along any ray from the station, the round prior moves the estimate along the
# gradient, and the gradient at the prior mean points along that ray, so the very first update lands on the MAP.
# Section 5 has a forward model where the linearisation point does matter. Either way the Gaussian is straight where
# the exact posterior is curved, and the badge counts how many of its draws land where the exact posterior actually has
# probability. The EnKI cloud bends partway (notebook 2.24 explains how) at the price of a thousand forward runs per
# step instead of one forward run and one Jacobian.

# %% [markdown]
# **Gauss-Newton covariance or Laplace covariance?** The Laplace approximation (2.08) uses the full curvature (Hessian,
# 2.17) of $-\log p$ at the peak. With residual $\mathbf r = G(\mathbf m) - \mathbf d$, that Hessian is
# $\mathbf P_0^{-1} + \mathbf H^\top\mathbf R^{-1}\mathbf H + \sum_i (\mathbf R^{-1}\mathbf r)_i\,\nabla^2 G_i$; the
# Gauss-Newton form used by the iterated EKF drops the last term, the second derivatives of the forward model weighted
# by the residuals. For $r$ = 2.5, at the iterated EKF's answer, both are computed below (Hessian by JAX).

# %%
m_gn = jnp.asarray(res3["iterated"]["mean"])
P0i = jnp.asarray(np.linalg.inv(pr.P))
y_, s_ = jnp.asarray(pr.y), jnp.asarray(pr.sigma)
neg_log_post = lambda m: 0.5 * (m - pr.m0) @ P0i @ (m - pr.m0) + 0.5 * jnp.sum(((pr.forward_fn(m) - y_) / s_) ** 2)
grad_norm = float(jnp.linalg.norm(jax.grad(neg_log_post)(m_gn)))
C_lap = np.linalg.inv(np.asarray(jax.hessian(neg_log_post)(m_gn)))
C_gn = np.asarray(res3["iterated"]["cov"])
print(f"gradient of -log p at the iterated EKF answer: {grad_norm:.1e} (a stationary point)")
print(f"residual there: {float(pr.forward_fn(m_gn)[0] - pr.y[0]):.4f} (noise sd {pr.sigma[0]:g})")
print("eigenvalues of the Gauss-Newton (EKF) covariance:", np.round(np.linalg.eigvalsh(C_gn), 6))
print("eigenvalues of the full Laplace covariance:      ", np.round(np.linalg.eigvalsh(C_lap), 6))
print(f"exact posterior sd {np.round(np.sqrt(np.diag(g.cov)), 3)}; Gauss-Newton sd {np.round(np.sqrt(np.diag(C_gn)), 3)}; "
      f"Laplace sd {np.round(np.sqrt(np.diag(C_lap)), 3)}")

fig, ax = plt.subplots(figsize=(5.6, 4.6))
BP.density(ax, g)
for C_, col, ls, lab in [(C_gn, EKF_C, "-", "Gauss-Newton (iterated EKF)"), (C_lap, S.BAD, "--", "full Laplace")]:
    BP.ellipse(ax, np.asarray(m_gn), C_, nsig=np.sqrt(-2 * np.log(0.1)), color=col, lw=1.6, ls=ls, label=f"{lab}, 90 %")
ax.plot(*np.asarray(m_gn), "*", color=S.INK, ms=12, mec=S.SURFACE, label="MAP (iterated EKF)")
BP.frame(ax, pr, equal=True, labels=True)
ax.set_xlim(float(m_gn[0]) - 2.6, float(m_gn[0]) + 2.6); ax.set_ylim(float(m_gn[1]) - 2.2, float(m_gn[1]) + 2.2)
ax.set_title("Same peak, two covariances (r = 2.5)", fontsize=10)
ax.legend(fontsize=7.5, loc="lower right")
fig.tight_layout()
bk.savefig(fig, "2-25_gn_vs_laplace")
plt.show()

# %% [markdown]
# **How to read it.** Both ellipses sit at the same stationary point (gradient printed) and agree across the circle,
# the direction the datum constrains (equal small eigenvalue). They differ along the circle. There the datum adds no
# Gauss-Newton precision at all, so the Gauss-Newton variance is the prior's (eigenvalue 1). The residual at the peak
# is small (printed, a few hundredths of σ), but in the dropped term it is divided by $\sigma^2$ and multiplied by the
# curvature of the data circle (1/radius), which together is a noticeable fraction of the prior precision along the
# circle; so the Laplace variance in that direction is smaller (eigenvalues printed). Neither ellipse can follow the curved exact posterior (grey); the Laplace one
# is the exact local curvature at the peak, the Gauss-Newton one an approximation to it that is good when residuals
# are small or the model is nearly linear.

# %% [markdown]
# ## 4. Every method against the knob
#
# Seven methods on all eight knob settings, each graded by the same three numbers against the exact posterior (grid): the share
# of its samples inside the exact 90 % region (0.90 if right), the error of its mean (in posterior standard
# deviations), and its wall-clock time. Budgets: EKF one Jacobian per iteration; EnKI 1,000 members; NUTS 4 chains
# × (500 warm-up + 1,000 draws); SMC 2,000 particles; Metropolis 4 chains × 5,000 steps with a step tuned during
# warm-up. Wall-clock depends on implementation (this teaching code runs NUTS in a Python loop and SMC vectorised);
# the evaluation counts do not.

# %%
METHODS = ["EKF (classic)", "EKF (iterated)", "EnKI", "EnKI (cap 0.1)", "NUTS", "SMC", "Metropolis"]
MCOL = {"EKF (classic)": "#7c7b75", "EKF (iterated)": EKF_C, "EnKI": S.COLORS["enki"], "EnKI (cap 0.1)": "#8f86d6",
        "NUTS": S.COLORS["nuts"], "SMC": S.COLORS["smc"], "Metropolis": S.COLORS["rwm"]}


def run_all(pr, rng):
    out = {}

    def timed(name, fn):
        pr.reset_counts()
        t0 = time.perf_counter()
        Smp, cost = fn()
        out[name] = dict(S=Smp, sec=time.perf_counter() - t0, cost=cost)

    def ekf(mode):
        e = SM.ekf(pr, mode=mode)
        return rng.multivariate_normal(e["mean"], e["cov"], 4000), f"{e['n_forward']} fwd + {e['n_jacobian']} Jac"

    def enki(cap):
        e = SM.enki(pr, J=1000, rng=rng, max_dtheta=cap)
        return e["samples"], f"{e['n_forward']} fwd"

    def nuts():
        ch = [SM.nuts(pr, x0, 1000, n_warmup=500, rng=rng)["chain"] for x0 in pr.sample_prior(rng, 4)]
        return np.vstack(ch), f"{pr.n_grad} grad"

    def smc():
        s = SM.smc(pr, n=2000, rng=rng)
        return s["samples"], f"{s['n_logp']} log p"

    def rwm():
        X, step = pr.sample_prior(rng, 4), 0.5
        for _ in range(20):
            r = SM.rwm(pr, X, 50, step, rng=rng)
            step *= np.exp(2 * (r["accept_rate"] - 0.3))
            X = r["samples"][:, -1]
        r = SM.rwm(pr, X, 5000, step, rng=rng)
        return r["samples"].reshape(-1, 2), f"{pr.n_logp} log p"

    timed("EKF (classic)", lambda: ekf("classic"))
    timed("EKF (iterated)", lambda: ekf("iterated"))
    timed("EnKI", lambda: enki(1.0))
    timed("EnKI (cap 0.1)", lambda: enki(0.1))
    timed("NUTS", nuts)
    timed("SMC", smc)
    timed("Metropolis", rwm)
    return out


run_all(probs[5], np.random.default_rng(0))                   # warm-up: JAX compiles once, not inside the timings
results = {r: run_all(probs[r], rng) for r in R_LIST}
score = {}
for r in R_LIST:
    g = grids[r]
    sd = np.sqrt(np.diag(g.cov))
    for m, v in results[r].items():
        score[r, m] = dict(inside=g.inside(v["S"]), err=float(np.max(np.abs(v["S"].mean(0) - g.mean) / sd)),
                           sec=v["sec"], cost=v["cost"])
print(f"{'r':>5s} {'NL':>5s} | " + " | ".join(f"{m:>16s}" for m in METHODS))
for r in R_LIST:
    print(f"{r:5g} {NL[r]:5.2f} | " + " | ".join(f"{score[r, m]['inside']:4.2f} / {score[r, m]['err']:4.2f} sd" for m in METHODS))
print("cells: share inside the exact 90 % region / error of the mean (posterior sd)")
print("median wall-clock per run (ms): " + ", ".join(f"{m} {1e3 * np.median([score[r, m]['sec'] for r in R_LIST]):.2f}" for m in METHODS))
print("cost per run at r = 2.5: " + ", ".join(f"{m}: {score[2.5, m]['cost']}" for m in METHODS))

# %%
COLS = [30, 5, 1.5, 0.6]
fig, axs = plt.subplots(len(METHODS), len(COLS), figsize=(11.5, 2.55 * len(METHODS)))
for i, m in enumerate(METHODS):
    for j, r in enumerate(COLS):
        ax = axs[i, j]
        BP.density(ax, grids[r])
        BP.samples(ax, results[r][m]["S"], MCOL[m], n_max=1500, s=2, alpha=0.45)
        BP.frame(ax, probs[r], equal=True, labels=False)
        ax.set_xticks([]); ax.set_yticks([])
        sc = score[r, m]
        S.badge(ax, f"{100 * sc['inside']:.0f} % inside", ok=sc["inside"] >= 0.85, loc="lower right", fs=7.5)
        if i == 0:
            ax.set_title(f"r = {r:g}   (NL {NL[r]:.2f})", fontsize=10)
        if j == 0:
            ax.set_ylabel(m, fontsize=9.5, fontweight="bold")
fig.tight_layout()
bk.savefig(fig, "2-25_sweep_panels")
plt.show()

# %%
fig, axs = plt.subplots(1, 3, figsize=(15, 4.0))
xs = np.array([NL[r] for r in R_LIST])
order = np.argsort(xs)
for m in METHODS:
    ins = np.array([score[r, m]["inside"] for r in R_LIST])[order]
    err = np.array([score[r, m]["err"] for r in R_LIST])[order]
    mk = "s" if m.startswith("EKF") else ("^" if m.startswith("EnKI") else "o")
    axs[0].semilogx(xs[order], ins, mk + "-", color=MCOL[m], label=m, ms=5)
    axs[1].semilogx(xs[order], err, mk + "-", color=MCOL[m], label=m, ms=5)
axs[0].axhline(0.9, color=S.MUTED, lw=0.8)
axs[0].set_ylabel("share of samples inside the exact 90 % region")
axs[0].set_title("Shape: does it put samples where the exact posterior is?")
axs[1].set_ylabel("error of the mean (posterior sd)")
axs[1].set_title("Location: is the centre right?")
for ax in axs[:2]:
    ax.set_xlabel("degree of nonlinearity (Taylor index over the posterior)")
    ax.axvspan(1, 20, color="#f3f2ee", lw=0, zorder=0)
axs[0].legend(fontsize=7.8, loc="lower left")
med = [max(np.median([score[r, m]["sec"] for r in R_LIST]), 1e-4) for m in METHODS]   # floor 0.1 ms for the log axis
axs[2].barh(range(len(METHODS)), med, color=[MCOL[m] for m in METHODS], height=0.6)
axs[2].set_xscale("log")
axs[2].set_yticks(range(len(METHODS))); axs[2].set_yticklabels(METHODS)
axs[2].invert_yaxis()
axs[2].set_xlabel("median wall-clock per run (s, log scale)")
axs[2].set_title("Time (this teaching code)")
for k, v in enumerate(med):
    axs[2].text(v * 1.15, k, ("< 1 ms" if v <= 1e-3 else f"{v:.3g} s"), va="center", fontsize=8, color=S.INK2)
fig.tight_layout()
bk.savefig(fig, "2-25_degradation")
plt.show()

# %% [markdown]
# **How to read it** (all numbers are in the table above):
# * **Below an index of about 0.1 every method is right**, including the one-Jacobian EKF, which is then the
#   cheapest by far. That is the linear-Gaussian world of notebook 2.02.
# * **Kalman-type methods degrade as the index rises.** The EKF returns one Gaussian, which sits on the tangent;
#   EnKI's ensemble, moved by mean-and-covariance updates, bends partway but, once the posterior wraps around (closest
#   stations), it too misses most of the probability. Capping Δθ does not rescue them here: the limitation is the update
#   itself, not the step size (notebook 2.24 shows where the cap does matter).
# * **The samplers (NUTS, SMC, Metropolis) stayed right at every setting here**, because they never linearise. In two
#   dimensions the budgets are generous; in hundreds of dimensions Metropolis would fall behind (notebook 2.16) and NUTS
#   needs a differentiable model (notebook 2.17).
# * **Time** follows cost per answer: one Jacobian (EKF) is nearly free, a thousand-member ensemble costs a thousand
#   forward runs per step, and the samplers need thousands of evaluations. With an expensive forward model (PyLith,
#   an afterslip ODE) the evaluation counts decide what is affordable.

# %% [markdown]
# ## 5. The sequential EKF, Qiu-style: a postseismic time series day by day
#
# A two-parameter cousin of the Extended Network Inversion Filter. One GPS station records postseismic displacement
# $u(t) = a\,\log(1 + t/\tau)$, one value per day for 120 days with noise σ = 2 mm. The unknowns are
# $\mathbf m = (\ln a, \ln\tau)$, so the model is nonlinear in both. The state is the parameter vector itself (no
# dynamics), and each day the filter assimilates that day's datum:
#
# * **EKF**: linearise $u(t_k;\mathbf m)$ at the current estimate with the JAX gradient, Kalman update (`SM.ekf_step`);
# * **EnKF**: a 500-member ensemble, one square-root Kalman update per day (`SM.etkf_update`, same algebra as EnKI);
# * **exact**: the posterior after $k$ days on a 2-D grid, for comparison.

# %%
a_true, tau_true, sig_u = 30.0, 3.0, 2.0                       # mm, days, mm
t_days = np.arange(1, 121, 1.0)
rng_s = np.random.default_rng(7)
u_obs = a_true * np.log1p(t_days / tau_true) + sig_u * rng_s.standard_normal(t_days.size)
m0_s, P0_s = np.array([np.log(20.0), np.log(10.0)]), np.diag([0.7 ** 2, 1.2 ** 2])
print("problem: 2 unknowns (ln a, ln tau), 120 daily data, noise 2 mm; true values ln a = %.3f, ln tau = %.3f"
      % (np.log(a_true), np.log(tau_true)))

u_fn = lambda m, tk: jnp.exp(m[0]) * jnp.log1p(tk / jnp.exp(m[1]))
u_j, du_j = jax.jit(u_fn), jax.jit(jax.grad(u_fn))


def batch_problem(d):
    """The posterior after d days as an ordinary inverse problem (all data so far at once)."""
    tk = jnp.asarray(t_days[:d])
    return P.InverseProblem(lambda m: jnp.exp(m[0]) * jnp.log1p(tk / jnp.exp(m[1])), u_obs[:d], sig_u,
                            m0=m0_s, P=P0_s, name=f"after day {d}", labels=("ln a", "ln τ"))


def exact_grid(d, center, half, n=260):
    """Exact posterior after d days on a grid zoomed onto it (so even day 120's narrow posterior is resolved)."""
    a = np.linspace(center[0] - half[0], center[0] + half[0], n)
    b = np.linspace(center[1] - half[1], center[1] + half[1], n)
    A, T = np.meshgrid(a, b)
    lp = -0.5 * ((A - m0_s[0]) ** 2 / P0_s[0, 0] + (T - m0_s[1]) ** 2 / P0_s[1, 1])
    pred = np.exp(A)[..., None] * np.log1p(t_days[:d][None, None, :] / np.exp(T)[..., None])
    lp = lp - 0.5 * np.sum(((u_obs[:d] - pred) / sig_u) ** 2, axis=-1)
    Pd = np.exp(lp - lp.max())
    dx, dy = a[1] - a[0], b[1] - b[0]
    Pd /= Pd.sum() * dx * dy
    return P.Grid(A, T, Pd, dx, dy, P.hdr_levels(Pd))


DAYS = sorted(set(np.unique(np.round(np.logspace(0, np.log10(120), 30)).astype(int)).tolist()) | {3, 10, 40, 120})
exact = {}
center, half = m0_s.copy(), 5 * np.sqrt(np.diag(P0_s))
for d in DAYS:
    for _ in range(6):                                   # widen the window until the edges hold no probability
        g = exact_grid(d, center, half)
        edge = (g.P[0].sum() + g.P[-1].sum()) * g.dx * g.dy + (g.P[:, 0].sum() + g.P[:, -1].sum()) * g.dx * g.dy
        if edge < 1e-4:
            break
        half = half * 1.6
    exact[d] = g
    sd = np.sqrt(np.diag(g.cov))
    center, half = g.mean, np.maximum(7 * sd, 0.02)

# the two filters, one day at a time
m_e, P_e = m0_s.copy(), P0_s.copy()
J_ens = 500
X_e = m0_s + rng_s.standard_normal((J_ens, 2)) @ np.linalg.cholesky(P0_s).T
SNAP = [3, 10, 40, 120]
filt = {}
for k, tk in enumerate(t_days):
    m_e, P_e = SM.ekf_step(m_e, P_e, np.array([u_obs[k]]), sig_u,
                           lambda mm: np.array([float(u_j(jnp.asarray(mm), tk))]),
                           lambda mm: np.array(du_j(jnp.asarray(mm), tk))[None, :])
    Yk = np.exp(X_e[:, 0]) * np.log1p(tk / np.exp(X_e[:, 1]))
    X_e = SM.etkf_update(X_e.T, Yk[None, :], np.array([u_obs[k]]), np.array([sig_u]), alpha=1.0).T
    if k + 1 in DAYS:
        filt[k + 1] = dict(ekf=(m_e.copy(), P_e.copy()), ens=X_e.copy())

for d in SNAP:
    g = exact[d]
    em, es = g.mean, np.sqrt(np.diag(g.cov))
    me, Pe = filt[d]["ekf"]
    Xd = filt[d]["ens"]
    far = np.mean(np.abs(Xd[:, 0] - em[0]) > 5 * es[0])
    rob = 1.4826 * np.median(np.abs(Xd[:, 0] - np.median(Xd[:, 0])))
    print(f"day {d:3d}: exact ln a {em[0]:.3f} ± {es[0]:.3f} | EKF {me[0]:.3f} ± {np.sqrt(Pe[0, 0]):.3f}"
          f" | EnKF median {np.median(Xd[:, 0]):.3f}, robust sd {rob:.3f}, plain sd {Xd[:, 0].std():.3f};"
          f" members more than 5 exact sd away: {100 * far:.0f} %")

# how nonlinear is the problem over the current posterior?
nl_day = {d: D.nonlinearity_taylor(batch_problem(d), exact[d].sample(3000, rng))["nl"] for d in SNAP}
print("degree of nonlinearity over the exact posterior: " + ", ".join(f"day {d}: {v:.3f}" for d, v in nl_day.items()))

# %%
dd = np.array(DAYS)
fig, axs = plt.subplots(1, 3, figsize=(15, 3.9))
ax = axs[0]
ax.plot(t_days, u_obs, "o", color=S.INK, ms=2.5, label="daily GPS")
ax.plot(t_days, a_true * np.log1p(t_days / tau_true), color=S.MUTED, lw=1.5, label="true curve")
for d in (3, 10):
    me, _ = filt[d]["ekf"]
    ax.plot(t_days, np.exp(me[0]) * np.log1p(t_days / np.exp(me[1])), color=EKF_C, lw=1, alpha=0.5 + 0.3 * (d == 10))
    ax.axvline(d, color=S.AXIS, lw=0.6)
ax.set_xlabel("days after the earthquake"); ax.set_ylabel("displacement (mm)")
ax.set_title("The data, and the EKF's fit after day 3 and day 10", fontsize=10)
ax.legend(fontsize=8, loc="lower right")
for j, name in enumerate(["ln a", "ln τ"]):
    ax = axs[j + 1]
    em = np.array([exact[d].mean[j] for d in DAYS]); es = np.array([np.sqrt(exact[d].cov[j, j]) for d in DAYS])
    km = np.array([filt[d]["ekf"][0][j] for d in DAYS]); ks = np.array([np.sqrt(filt[d]["ekf"][1][j, j]) for d in DAYS])
    nm = np.array([np.median(filt[d]["ens"][:, j]) for d in DAYS]); ns = np.array([filt[d]["ens"][:, j].std() for d in DAYS])
    ax.fill_between(dd, em - 2 * es, em + 2 * es, color="#d9d8d1", lw=0, label="exact ± 2 sd")
    ax.plot(dd, km, color=EKF_C, lw=1.6, label="EKF mean ± 2 sd")
    ax.plot(dd, km - 2 * ks, color=EKF_C, lw=0.7); ax.plot(dd, km + 2 * ks, color=EKF_C, lw=0.7)
    ax.plot(dd, nm, color=S.COLORS["enki"], lw=1.6, label="EnKF median ± 2 sd")
    ax.plot(dd, nm - 2 * ns, color=S.COLORS["enki"], lw=0.7); ax.plot(dd, nm + 2 * ns, color=S.COLORS["enki"], lw=0.7)
    ax.axhline(np.log(a_true) if j == 0 else np.log(tau_true), color=S.INK, lw=0.6)
    ax.set_xscale("log"); ax.set_xlabel("day (log scale)"); ax.set_ylabel(name)
    ax.set_title(f"{name}: estimate ± 2 sd as data arrive", fontsize=10)
axs[1].legend(fontsize=8, loc="upper right")
fig.tight_layout()
bk.savefig(fig, "2-25_sequential_tracks")
plt.show()

# %%
fig, axs = plt.subplots(1, len(SNAP), figsize=(15, 3.8))
for ax, d in zip(axs, SNAP):
    g = exact[d]
    BP.density(ax, g)
    Xd = filt[d]["ens"]
    ax.scatter(Xd[:, 0], Xd[:, 1], s=3, color=S.COLORS["enki"], alpha=0.45, lw=0)
    me, Pe = filt[d]["ekf"]
    for q in (0.5, 0.9):
        BP.ellipse(ax, me, Pe, nsig=np.sqrt(-2 * np.log(1 - q)), color=EKF_C, lw=1.2)
    ax.plot(np.log(a_true), np.log(tau_true), "*", color=S.INK, ms=10, mec=S.SURFACE)
    ax.set_xlim(g.X.min(), g.X.max()); ax.set_ylim(g.Y.min(), g.Y.max())
    inside_win = np.mean((Xd[:, 0] > g.X.min()) & (Xd[:, 0] < g.X.max()) & (Xd[:, 1] > g.Y.min()) & (Xd[:, 1] < g.Y.max()))
    S.badge(ax, f"{100 * (1 - inside_win):.0f} % of EnKF members outside this window", loc="lower left", fs=7.5)
    ax.set_xlabel("ln a"); ax.set_title(f"after day {d}   (nonlinearity {nl_day[d]:.2f})", fontsize=10)
axs[0].set_ylabel("ln τ")
fig.tight_layout()
bk.savefig(fig, "2-25_sequential_shapes")
plt.show()

# %% [markdown]
# Grey: exact posterior (50 / 90 / 99 %), each panel zoomed onto it; dark ellipses: EKF; violet dots: EnKF members;
# star: true value. Panel titles give the degree of nonlinearity over the exact posterior on that day.
#
# * **In the first days the uncertainty is broad and the model is curved across it** (the banana on day 3). The EKF
#   linearises at a point the data have not pinned down yet; by day 10 its Gaussian is shifted and too narrow
#   (numbers printed above). As data accumulate, the posterior shrinks into a region where $\log(1 + t/\tau)$ is
#   nearly straight: the degree of nonlinearity falls day by day, and by day 40 the EKF agrees with the exact
#   answer. That is the general EKF behaviour: wrong while the curvature is large compared with the current
#   uncertainty, fine once it is small.
# * **The EnKF keeps the right centre but its spread stops shrinking.** Members that the early, strongly nonlinear
#   updates left far out on the curved trade-off never come back: the gain of each update is computed from the whole
#   ensemble, and once the bulk has converged the gain is tiny. An ensemble Kalman filter has no step that removes
#   bad members (SMC's resampling does exactly that). The printed share of members far from the exact posterior grows
#   with time while the median and the robust spread follow the bulk.
# * **The fix used in practice** is to stop filtering and iterate over all the data at once, in small tempered steps:
#   that is EnKI (and the iterative ensemble smoothers). The next cell does that for each snapshot day, and also
#   shows where a single linearisation goes wrong.

# %%
print(f"{'day':>4s} | {'exact ln a':>15s} | {'EKF at prior mean':>18s} | {'iterated EKF (MAP)':>19s} | {'EnKI, all data':>15s} | share inside exact 90 %: EKF-1 / IEKF / EnKI")
batch = {}
for d in SNAP:
    prb, g = batch_problem(d), exact[d]
    e1, ei = SM.ekf(prb, mode="classic"), SM.ekf(prb, mode="iterated")
    eb = SM.enki(prb, J=1000, rng=rng, max_dtheta=0.1)
    ins = [g.inside(rng.multivariate_normal(e1["mean"], e1["cov"], 4000)),
           g.inside(rng.multivariate_normal(ei["mean"], ei["cov"], 4000)), g.inside(eb["samples"])]
    batch[d] = dict(e1=e1, ei=ei, eb=eb, ins=ins)
    print(f"{d:4d} | {g.mean[0]:7.3f} ± {np.sqrt(g.cov[0, 0]):.3f} | {e1['mean'][0]:9.3f} ± {np.sqrt(e1['cov'][0, 0]):.3f}"
          f" | {ei['mean'][0]:10.3f} ± {np.sqrt(ei['cov'][0, 0]):.3f} | {eb['samples'][:, 0].mean():6.3f} ± {eb['samples'][:, 0].std():.3f}"
          f" | {ins[0]:.2f} / {ins[1]:.2f} / {ins[2]:.2f}")

fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.0))
for ax, d in zip(axs, (3, 10)):
    g, b_ = exact[d], batch[d]
    BP.density(ax, g)
    BP.samples(ax, b_["eb"]["samples"], S.COLORS["enki"], n_max=1500, s=3, alpha=0.4)
    for e_, ls_, lab in [(b_["e1"], (0, (2, 2)), "EKF linearised at the prior mean"), (b_["ei"], "-", "iterated EKF (at the MAP)")]:
        BP.ellipse(ax, e_["mean"], e_["cov"], nsig=np.sqrt(-2 * np.log(0.1)), color=EKF_C, lw=1.4, ls=ls_, label=lab)
    ax.plot(*m0_s, "P", color=S.INK, ms=8, mec=S.SURFACE)
    ax.set_xlim(g.X.min(), g.X.max()); ax.set_ylim(g.Y.min(), g.Y.max())
    ax.set_xlabel("ln a"); ax.set_ylabel("ln τ")
    ax.set_title(f"all data up to day {d}: where you linearise matters", fontsize=10)
axs[0].legend(fontsize=7.8, loc="upper left")
fig.tight_layout()
bk.savefig(fig, "2-25_linearisation_point")
plt.show()

# %% [markdown]
# Ellipses are 90 % regions; the cross is the prior mean. A single linearisation at the prior mean (dashed) puts the
# Gaussian where the tangent drawn at the *prior* says the data point. Here the prior mean is far from the answer,
# and the tangent there is wrong in direction as well as position, so none of its draws land in the exact 90 % region
# (table). Iterating the linearisation (solid) walks it to the MAP, and on every snapshot day the result is close to
# exact, because over the posterior this problem is at most mildly nonlinear (indices printed above).
# Re-assimilating all data so far with EnKI in small tempered steps (violet) is close to exact on every day too.
#
# **Why the sequential EKF was biased while the batch iterated EKF is not.** A filter linearises each datum once, at
# the estimate it has when that datum arrives, and never revisits it. The early data were linearised around poor
# early estimates, and those errors stay in the filter's mean and covariance until later data outweigh them. The
# batch iterated EKF re-linearises every datum at the final estimate. That is the same reason the EnKF strands
# members while the iterated EnKI does not.

# %% [markdown]
# ## 6. Report card
#
# What each method did in this notebook and in notebooks 2.15, 2.18, 2.20 and 2.24: a summary of these examples and
# budgets, not a guarantee.
#
# | method | needs | exact when linear-Gaussian | strong nonlinearity (this notebook) | two separated peaks (2.15, 2.20) | cost per answer |
# |---|---|---|---|---|---|
# | **EKF** (classic / iterated) | forward model + Jacobian | yes | no: one Gaussian on the tangent (iterated: at a local MAP, with the Gauss-Newton covariance) | no | 1 forward run + 1 Jacobian per iteration |
# | **EnKF / EnKI** | forward runs only | yes (J large) | partly: bends a little, then fails; a sequential EnKF can strand members | no | J forward runs per update |
# | **NUTS / HMC** | log p and its gradient (differentiable model) | yes | yes | no: one chain stays on one peak | thousands of gradients |
# | **SMC** (tempering) | log p (prior and likelihood separately) | yes | yes | **yes**, with the right weights | tens of thousands of evaluations |
# | **Metropolis** | log p only | yes | yes in 2-D (slow in high dimension, 2.16) | no: one chain stays on one peak | thousands of evaluations, more in high dimension |
#
# The measured numbers behind the "strong nonlinearity" column are the table in section 4; the two-peak column is
# the map in notebook 2.15 and the comparison in notebook 2.20.

# %% [markdown]
# **In practice.**
# * **Measuring the nonlinearity of a saved ensemble run.** If every iteration's members and predictions are saved,
#   the ensemble index of section 2 is one least-squares fit per iteration (in the variables the ensemble actually
#   updates), evaluated on held-out members, with the residual in units of the data sigmas. It can be measured only
#   when the members clearly outnumber the parameters; otherwise use the Taylor index along a few directions.
# * **EKF or EnKI.** The EKF returns one Gaussian, needs the Jacobian of the forward model (cheap for a
#   Green's-function slip model; an adjoint or a differentiable rewrite for an ODE solver, 2.22) and costs one forward
#   run per step; EnKI needs no derivatives and costs J forward runs per step. A filter in time (such as the ENIF of
#   Qiu et al. 2018) linearises each datum once; iterating over all data at once (iterated EKF, EnKI) is the remedy
#   section 5 shows for biased early linearisations and stranded members.
# * **Before choosing a method, compute the index over the prior and over a pilot posterior.** If both are well
#   below 1, a Kalman method is accurate and cheap; if the prior index is large but the posterior one is small,
#   tempering is enough; if the posterior index is large, only a sampler gives the right shape.
#
# Next: **2.26 · Localized ensemble Kalman inversion (optional)**.
