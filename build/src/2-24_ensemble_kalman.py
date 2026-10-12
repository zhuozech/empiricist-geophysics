# %% [markdown]
# # 2.24 · Ensemble Kalman methods: EnKF, EKI, EnKI
#
# **Before this:** 2.23 (Kalman filter and gain), 2.20 (tempering); 2.06 for covariance matrices.
# **You will be able to:** explain how an ensemble replaces covariance matrices and gradients in the Kalman update;
# tell EnKF, classic EKI and tempered EnKI apart by how they use the data; predict where the mean-and-covariance update
# fails (strong nonlinearity, separate modes, too few members); read θ and WRMS as health monitors; know what
# localisation and inflation are for.
# **Optional sections:** 4 (EnKF in time), 8 (capping the step).
#
# SMC (notebook 2.20) moves a population of particles from prior to posterior using weights. The ensemble Kalman
# family moves a population too, but with no weights and no accept/reject: every member is *shifted* by a Kalman
# update computed from the ensemble itself. No gradients are needed, only forward runs, which is why these
# methods are popular for expensive, non-differentiable physics (stress-driven afterslip models, for example).
#
# **Three names, one update.**
#
# | name | what it is for | how the data are used |
# |---|---|---|
# | **EnKF**, ensemble Kalman filter (Evensen 1994) | a *state that evolves in time* (weather, a slow-slip transient) | each new observation is assimilated once, when it arrives |
# | **EKI**, ensemble Kalman inversion (Iglesias, Law & Stuart 2013) | *static* parameters | the same data are assimilated again at every iteration; behaves like an optimiser |
# | **EnKI** (Fukuda & Barbot 2025) | static parameters, as an approximate sampler | the data are split into fractions $\Delta\theta$ that add up to 1 (tempering, θ: 0 → 1) |
#
# Fukuda & Barbot write that they abbreviate their method EnKI "to distinguish it from the original EKI".
# An **ensemble** here is a set of $J$ parameter vectors (**members**) that together stand for a distribution: their
# mean and covariance replace the mean and covariance matrix of the exact Kalman filter.
#
# **Words this notebook defines:** Kalman gain, ensemble, cross-covariance, derivative-free, statistical
# linearisation, forecast and analysis, ETKF square root, θ and $\Delta\theta$, $\Delta\theta$ cap, ensemble collapse,
# subspace property, localisation, inflation.

# %% [markdown]
# **Problems in this notebook** (size and forward model; notebook 2.25 explains the degree of nonlinearity)
#
# | problem | unknowns | data | forward model | linear? |
# |---|---|---|---|---|
# | trade-off | 2 | 2 | $\mathbf d = \mathbf G\mathbf m$ | linear |
# | banana, two modes, ring, locking depth | 2 | 1 to 5 | distances / $\frac{s}{\pi}\arctan(x/D)$ | nonlinear |
# | slow-slip filter (4) | 2 per epoch (displacement, velocity) | 1 every 5 days | linear dynamics and observation | linear |
# | log-afterslip, Δθ cap (8) | 2 | 100 | $A\log(1 + t/t_c)$ | nonlinear |
# | 1-D field, subspace property (10) | 100 | 40 | point observations, GP prior | linear |

# %%
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))
import inspect
import numpy as np
import matplotlib.pyplot as plt
import jax.numpy as jnp
from scipy.special import kv, gamma
import bayeskit as bk
from bayeskit import problems as P, samplers as SM, diagnostics as D, plotting as BP, style as S

np.seterr(all="ignore")
EN = S.COLORS["enki"]
EN_LIGHT = "#b9b3e3"                      # a lighter step of the same violet, for "before"


def stats(X):
    return X.mean(0), np.cov(X.T)


# %% [markdown]
# ## 1. The Kalman update is Bayes' rule for a linear-Gaussian problem
#
# Notebook 2.02 showed that a datum which sees $m_1 + m_2$ is a band, and that prior × band = a narrower ellipse.
# The Kalman update is the formula for that ellipse:
#
# $$\bar{\mathbf m}_{\text{post}} = \bar{\mathbf m} + \mathbf K\,(\mathbf d - \mathbf G\bar{\mathbf m}), \qquad
# \mathbf K = \mathbf P\mathbf G^\top(\mathbf G\mathbf P\mathbf G^\top + \mathbf R)^{-1}, \qquad
# \mathbf C_{\text{post}} = \mathbf P - \mathbf K\mathbf G\mathbf P .$$
#
# Read $\mathbf K$ (the **Kalman gain** of 2.04, 2.08 and 2.23) as: *how far, and in which direction, to move the
# model per unit of misfit*. It is large where the prior is uncertain and the data are precise.

# %%
G1 = np.array([[1.0, 1.0]])
P0, m0 = np.eye(2), np.zeros(2)
d1, s1 = np.array([1.0]), 0.15
mp, Cp = P.linear_gaussian_posterior(G1, d1, s1, m0, P0)
K = P0 @ G1.T @ np.linalg.inv(G1 @ P0 @ G1.T + s1 ** 2)
fig, ax = plt.subplots(figsize=(5.2, 4.6))
xx = np.linspace(-3, 3, 2)
ax.fill_between(xx, 1 - xx - 2 * s1, 1 - xx + 2 * s1, color="#e1e0d9", lw=0, label="datum: $m_1+m_2 = 1 \\pm 2\\sigma$")
BP.ellipse(ax, m0, P0, nsig=2, color=S.MUTED, lw=1.2, label="prior (2σ)")
BP.ellipse(ax, mp, Cp, nsig=2, color=S.INK, lw=1.6, label="posterior (2σ)")
ax.annotate("", xy=mp, xytext=m0, arrowprops=dict(arrowstyle="-|>", color=S.INK, lw=1.4))
ax.annotate("$\\mathbf{K}(\\mathbf{d}-\\mathbf{G}\\bar{\\mathbf{m}})$", xy=0.5 * mp, xytext=(0.45, -1.55), fontsize=10,
            arrowprops=dict(arrowstyle="-", color=S.INK2, lw=0.6))
ax.set_xlim(-3, 3); ax.set_ylim(-3, 3); ax.set_aspect("equal")
ax.set_xlabel("$m_1$"); ax.set_ylabel("$m_2$")
ax.set_title("Kalman update = prior × band")
ax.legend(loc="lower left", fontsize=8)
fig.tight_layout()
bk.savefig(fig, "2-24_kalman_update")
plt.show()
print("Kalman gain K =", K.ravel().round(3), "| posterior mean", mp.round(3), "| posterior sd along the band %.2f, across %.3f"
      % tuple(np.sqrt(np.linalg.eigvalsh(Cp))[::-1]))

# %% [markdown]
# The mean moves straight towards the band, perpendicular to it, because the datum only has an opinion about
# $m_1 + m_2$. Along the band nothing changes except what the prior says.

# %% [markdown]
# ## 2. The ensemble version: let a cloud of members carry the covariance
#
# For a big model we cannot store or invert $\mathbf P$, and for a nonlinear model there is no $\mathbf G$. The ensemble
# trick: draw $J$ members from the prior, run the forward model for each, and estimate everything the gain needs from
# the cloud:
#
# $$\mathbf P\mathbf G^\top \;\to\; \mathbf C_{mG} = \operatorname{cov}(\mathbf m_j, \mathbf G(\mathbf m_j)), \qquad
# \mathbf G\mathbf P\mathbf G^\top \;\to\; \mathbf C_{GG} = \operatorname{cov}(\mathbf G(\mathbf m_j)).$$
#
# $\mathbf C_{mG}$ is a **cross-covariance**: how the members' parameters and their predicted data vary together
# across the ensemble (covariance, 2.03 and 2.06, between two different sets of quantities).
#
# Then every member is shifted. The original EnKF shifts each member by the gain times its own (perturbed) misfit.
# This notebook uses the **ETKF square-root** form (ensemble transform Kalman filter): move the ensemble mean by the
# gain, and shrink the spread around it by a matrix $\mathbf T^{1/2}$, with no random perturbations. $\mathbf T^{1/2}$
# is a matrix square root (a matrix whose square is $\mathbf T$), computed here from the eigen-decomposition of
# 2.06, section 4; the Cholesky factor of 2.06, section 5 is another kind of square root. Here is the update as it
# runs below:

# %%
print(inspect.getsource(SM.etkf_update))

# %%
cor = P.correlated()
m_ex, C_ex = cor.exact
J = 50
X0 = cor.sample_prior(np.random.default_rng(2), J)                    # (J, 2) prior members
Y0 = cor.forward_batch(X0)                                             # (J, M) one forward run per member
X1 = SM.etkf_update(X0.T, Y0.T, cor.y, cor.sigma, alpha=1.0).T          # one update with the full data
g_cor = P.grid(cor, n=400)
fig, ax = plt.subplots(figsize=(5.6, 5.0))
for a, b in zip(X0, X1):
    ax.plot([a[0], b[0]], [a[1], b[1]], color=EN_LIGHT, lw=0.5, zorder=1)
ax.scatter(*X0.T, s=14, color=EN_LIGHT, lw=0, label=f"prior members (J = {J})", zorder=2)
ax.scatter(*X1.T, s=14, color=EN, lw=0, label="after one ETKF update", zorder=3)
BP.ellipse(ax, m_ex, C_ex, nsig=2, color=S.INK, lw=1.2, label="exact posterior (2σ)")
BP.ellipse(ax, *stats(X1), nsig=2, color=EN, lw=1.2, label="ensemble (2σ)")
BP.frame(ax, cor)
ax.set_title("Every member moves; nobody is rejected")
ax.legend(loc="lower left", fontsize=8)
fig.tight_layout()
bk.savefig(fig, "2-24_ensemble_update")
plt.show()
me, Ce = stats(X1)
print("ensemble mean", me.round(3), " exact", m_ex.round(3))
print("ensemble cov ", Ce.ravel().round(4), " exact", C_ex.ravel().round(4))

# %% [markdown]
# Thin lines join each member's position before and after. The cloud is squeezed onto the narrow posterior in one
# step. With 50 members the ensemble covariance is a noisy estimate (compare the printed numbers); section 10 shows
# what that noise becomes in high dimensions.

# %% [markdown]
# ## 3. Derivative-free: the ensemble's slope instead of the gradient
#
# The only place the forward model enters is through $\mathbf C_{mG}$ and $\mathbf C_{GG}$. For a linear model,
# $\mathbf C_{mG} \approx \mathbf P\mathbf G^\top$ exactly (up to sampling noise). For a nonlinear model the ensemble
# effectively fits a straight line (a regression) through the forward runs of all members: a **statistical
# linearisation**. It is the *average* slope over the cloud, not the slope at any point. That is why EnKI needs no
# gradient, and also why it can be misled when the slope changes a lot across the cloud.
#
# Example: the one-station ring problem (notebook 2.02). The datum is the distance to the station. The local gradient of
# the distance always has length 1 and points away from the station. A cloud that surrounds the station averages slopes
# that point in all directions.

# %%
for J_ in [50, 1000]:
    Xs = cor.sample_prior(np.random.default_rng(3), J_)
    Ys = cor.forward_batch(Xs)
    C_mG = np.cov(np.column_stack([Xs, Ys]).T)[:2, 2:]
    print(f"J = {J_:4d}: ensemble C_mG = {C_mG.ravel().round(3)}   exact P G^T = {(cor.P @ cor.G.T).ravel().round(3)}")

ring = P.location(kind="ring")
st_ring = np.array([-0.6, 0.0])
fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.6))
for ax, scale, title in zip(axs, [1.0, 0.15], ["wide cloud (prior draws)", "narrow cloud (spread × 0.15)"]):
    Xr = ring.sample_prior(np.random.default_rng(0), 2000) * scale
    Gr = ring.forward_batch(Xr)[:, 0]
    H = np.cov(np.column_stack([Xr, Gr]).T)[2, :2] @ np.linalg.inv(np.cov(Xr.T))     # regression slope
    xm = Xr.mean(0)
    grad = (xm - st_ring) / np.linalg.norm(xm - st_ring)
    sc = ax.scatter(*Xr.T, c=Gr, s=4, cmap="Purples", lw=0, alpha=0.6)
    ax.plot(*st_ring, "^", color=S.INK, ms=9)
    L_ = 6.0 if scale == 1.0 else 0.6                                          # arrow scale (for visibility only)
    ax.plot(*xm, "o", color=S.INK, ms=3.5, zorder=6)
    ax.annotate("", xy=xm + L_ * grad, xytext=xm, arrowprops=dict(arrowstyle="-|>", color=S.COLORS["nuts"], lw=2))
    ax.annotate("", xy=xm + L_ * H, xytext=xm, arrowprops=dict(arrowstyle="-|>", color=EN, lw=2.6), zorder=7)
    ax.set_aspect("equal")
    lim = 7 if scale == 1.0 else 1.1
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.set_title(title)
    S.badge(ax, f"slope length: local gradient {np.linalg.norm(grad):.2f}, ensemble {np.linalg.norm(H):.2f}",
            loc="lower left", fs=8)
    print(f"{title}: ensemble slope {H.round(3)}, local gradient at the ensemble mean {grad.round(3)}")
fig.colorbar(sc, ax=axs, label="predicted distance to the station", shrink=0.8)
fig.legend(handles=BP.legend_handles([("local gradient at the ensemble mean", S.COLORS["nuts"], "line"),
                                      ("ensemble (statistical) slope", EN, "line")]), loc="upper center", ncol=2)
bk.savefig(fig, "2-24_statistical_slope")
plt.show()

# %% [markdown]
# Arrows start at the ensemble mean (black dot); both arrows in a panel use the same scale, chosen per panel for
# visibility, and the badges give the actual lengths. In the wide cloud the members surround the station, the slopes cancel, and the ensemble "sees" a much
# weaker dependence of the datum on the parameters than the local gradient says. As the cloud shrinks, the
# statistical slope approaches the local one. This is the mechanism behind section 8: a big update taken while the
# cloud is wide uses a slope that is wrong for the final, narrow posterior.

# %% [markdown]
# ## 4. EnKF: tracking a state that evolves in time
#
# The filter alternates two steps:
# * **forecast:** push every member through the dynamics (and add *process noise*: how much the model is allowed to
#   be wrong per step). The spread grows.
# * **analysis:** when an observation arrives, apply the Kalman update. The spread shrinks.
#
# Toy problem: a slow-slip transient seen by one GPS component, observed every 5 days with 2 mm noise. The state
# is (displacement, velocity); the forecast model is "velocity stays the same, displacement integrates it". The
# process noise $q$ on the velocity says how fast the slip rate may change, and it is the knob we turn.
# (The Network Inversion Filter of Segall & Matthews 1997 is a Kalman filter of this kind for fault slip.)

# %%
nd = 120
t = np.arange(nd)
v_true = 0.6 * np.exp(-0.5 * ((t - 55) / 10) ** 2)          # mm/day
u_true = np.cumsum(v_true)
sig_obs, every = 2.0, 5
y_obs = u_true + sig_obs * np.random.default_rng(0).standard_normal(nd)
obs_days = t[t % every == 0]


def enkf(q_sd, J=50, seed=0):
    rng = np.random.default_rng(seed)
    X = np.vstack([rng.normal(0, 3, J), rng.normal(0, 0.3, J)])
    out = dict(mean=[], sd=[], t=[])
    for k in range(nd):
        X[0] += X[1]                                          # forecast: integrate velocity
        X[1] += q_sd * rng.standard_normal(J)                 # process noise on velocity
        out["t"].append(k); out["mean"].append(X.mean(1)); out["sd"].append(X.std(1, ddof=1))
        if k % every == 0:                                   # analysis: Kalman update with today's observation
            X = SM.etkf_update(X, X[:1], np.array([y_obs[k]]), np.array([sig_obs]))
            out["t"].append(k + 0.001); out["mean"].append(X.mean(1)); out["sd"].append(X.std(1, ddof=1))
    return {k: np.array(v) for k, v in out.items()}


fig, axs = plt.subplots(2, 2, figsize=(13, 5.6), sharex=True)
for j, q_sd in enumerate([0.01, 0.05]):
    r = enkf(q_sd)
    for i, (lab, truth) in enumerate([("displacement (mm)", u_true), ("velocity (mm/day)", v_true)]):
        ax = axs[i, j]
        ax.fill_between(r["t"], r["mean"][:, i] - 2 * r["sd"][:, i], r["mean"][:, i] + 2 * r["sd"][:, i],
                        color=EN, alpha=0.18, lw=0, label="ensemble mean ± 2 sd")
        ax.plot(r["t"], r["mean"][:, i], color=EN, lw=1.4)
        ax.plot(t, truth, color=S.INK, lw=1.2, label="true state")
        if i == 0:
            ax.plot(obs_days, y_obs[obs_days], "o", color=S.INK2, ms=3.5, label="GPS (every 5 days)")
        ax.set_ylabel(lab)
        tg = np.interp(t, r["t"], r["mean"][:, i])
        sg = np.interp(t, r["t"], r["sd"][:, i])
        cover = np.mean(np.abs(tg - truth) < 2 * sg)
        S.badge(ax, f"true state inside ±2 sd on {cover:.0%} of days", ok=cover > 0.85, loc="upper left", fs=8)
    axs[0, j].set_title(f"process noise q = {q_sd} mm/day per day")
    axs[1, j].set_xlabel("day")
axs[0, 0].legend(loc="lower right", fontsize=8)
fig.tight_layout()
bk.savefig(fig, "2-24_enkf_slow_slip")
plt.show()
for q_sd in [0.01, 0.05]:
    r = enkf(q_sd)
    tg = np.array([np.interp(t, r["t"], r["mean"][:, i]) for i in range(2)]).T
    print(f"q = {q_sd}: rms error displacement {np.sqrt(np.mean((tg[:, 0] - u_true) ** 2)):.2f} mm (GPS noise {sig_obs} mm), "
          f"velocity {np.sqrt(np.mean((tg[:, 1] - v_true) ** 2)):.3f} mm/day")

# %% [markdown]
# **The image to keep:** the band breathes. Between observations the forecast widens it, at each observation the
# analysis pinches it (the saw-tooth). With a process noise that is too small (left) the filter believes the slip rate
# cannot change, so it lags behind the transient and its band is too narrow: the true state leaves the band (badges). With
# a larger $q$ (right) it follows the transient and the band is honest. In a filter, $q$ plays the role that the
# prior plays in an inversion.

# %% [markdown]
# ## 5. Classic EKI: the same data, again and again, until the cloud collapses
#
# For *static* parameters there is no time. The original EKI (Iglesias, Law & Stuart 2013) iterates anyway: run the
# forward model for every member, apply the update with the full data, repeat. For a linear problem the first iteration
# already gives the posterior. The second assimilates the same data again, as if you had measured everything twice,
# and so on. Here bayeskit's `eki_classic` applies the same ETKF update as above with the full data at every iteration
# (the original used perturbed observations; the collapse is the same).

# %%
ec = SM.eki_classic(cor, J=500, n_iter=10, rng=np.random.default_rng(4))
snap = [0, 1, 2, 5, 10]
fig, axs = plt.subplots(1, len(snap) + 1, figsize=(17, 3.1))
for ax, k in zip(axs, snap):
    Xk = ec["history"][k]["X"]
    BP.density(ax, g_cor)
    BP.samples(ax, Xk, EN, s=3, alpha=0.4)
    BP.frame(ax, cor, labels=False)
    ax.set_title(f"iteration {k}  (data used {k}×)", fontsize=9.5)
tr = [np.trace(np.cov(h["X"].T)) for h in ec["history"]]
axs[-1].semilogy(range(len(tr)), tr, "o-", color=EN, ms=5)
axs[-1].axhline(np.trace(C_ex), color=S.INK, lw=1)
axs[-1].text(4.5, np.trace(C_ex) * 1.2, "exact posterior", fontsize=8, color=S.INK2)
axs[-1].set_xlabel("iteration"); axs[-1].set_title("ensemble spread (trace of cov)", fontsize=9.5)
fig.tight_layout()
bk.savefig(fig, "2-24_eki_collapse")
plt.show()
print("trace of ensemble covariance / exact posterior trace, by iteration:", np.round(np.array(tr) / np.trace(C_ex), 3))

# %% [markdown]
# After iteration 1 the cloud matches the grey exact posterior; after that it keeps shrinking towards a single point (the
# best-fitting model). Classic EKI is therefore an **optimiser**: it is stopped early (for example when the misfit
# reaches the noise level) and its spread is not an uncertainty.

# %% [markdown]
# ## 6. EnKI: assimilate the data in fractions that add up to one
#
# Split the likelihood into pieces: $L = L^{\Delta\theta_1} L^{\Delta\theta_2}\cdots$ with $\sum_i \Delta\theta_i = 1$. Each
# piece is the same likelihood with the noise covariance inflated by $\alpha_i = 1/\Delta\theta_i$, so each step is a
# gentle, tempered update (the same idea as SMC's β, notebook 2.20). For a linear-Gaussian problem *any* split gives the
# exact posterior in the limit of a very large ensemble (with $J$ members, up to their sampling error); for a nonlinear
# problem small steps keep each statistical linearisation local. The Iglesias & Yang (2021) rule chooses the next step
# from the current misfits $\Phi_j = \tfrac12\|\mathbf d - \mathbf G(\mathbf m_j)\|^2_{\mathbf R}$:
#
# $$\Delta\theta = \min\!\Big(\max\big(\mathrm{WRMS}^{-2},\ \sqrt{M/(2\operatorname{Var}_j\Phi_j)}\big),\ \Delta\theta_{\max},\ 1-\theta\Big).$$
#
# While the ensemble misfits are large, steps are tiny; as the members approach the data, steps grow. The run stops
# when θ reaches 1. The code:

# %%
print(inspect.getsource(SM.enki))

# %%
en = SM.enki(cor, J=1000, rng=np.random.default_rng(5))
H = en["history"]
ncol = int(np.ceil((len(H) + 1) / 2))                          # two rows so each panel stays readable
fig, axs = plt.subplots(2, ncol, figsize=(2.9 * ncol, 6.0))
axs = axs.ravel()
for ax, h in zip(axs, H):
    BP.density(ax, g_cor)
    BP.samples(ax, h["X"], EN, s=2, alpha=0.35)
    BP.frame(ax, cor, labels=False)
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"iteration {h['iter']}: θ = {h['theta']:.3g}", fontsize=9.5)
axw = axs[len(H)]
axw.semilogy([h["iter"] for h in H], [h["wrms"] for h in H], "o-", color=EN, ms=5)
axw.axhline(1.0, color=S.MUTED, lw=0.8)
axw.set_xlabel("iteration"); axw.set_title("WRMS (ensemble)", fontsize=9.5)
for ax in axs[len(H) + 1:]:
    ax.axis("off")
fig.tight_layout()
bk.savefig(fig, "2-24_enki_linear")
plt.show()
print("iter   theta    dtheta_next   WRMS(ensemble)  WRMS(mean prediction)")
for h in H:
    print(f"{h['iter']:4d}  {h['theta']:.4f}   {h.get('dtheta_next', float('nan')):9.4f}   {h['wrms']:12.3f}   {h['wrms_of_mean']:12.3f}")
me, Ce = stats(en["samples"])
print("final ensemble mean", me.round(3), " exact", m_ex.round(3))
print("final ensemble cov ", Ce.ravel().round(4), " exact", C_ex.ravel().round(4))

# %% [markdown]
# The printed log is what an EnKI run typically reports: θ creeps first (thousandths), then accelerates and reaches 1
# within a handful of iterations, while the ensemble WRMS (weighted root-mean-square misfit, 2.15) falls towards 1 (the
# noise level). The final cloud matches the exact posterior mean and covariance to within the sampling error of its
# members (printed), unlike classic EKI, which would keep shrinking.

# %% [markdown]
# ## 7. Nonlinear problems through a mean-and-covariance update
#
# Every EnKI step moves a member by the same gain matrix times that member's own misfit. The gain is built only from
# ensemble means and covariances (the statistical slope), while each member's misfit comes from its own nonlinear
# forward run. This does not force the cloud to stay Gaussian (the banana below bends), but the update has no
# mechanism that sends members to two separated islands in proportion to their probability, and curvature enters only
# through the members' individual misfits. What does that give on the banana and the two-island problems of notebook
# 2.02? (J = 1000.)

# %%
ban, bim = P.location(kind="banana"), P.location(kind="bimodal")
g_ban, g_bim = P.grid(ban, n=450), P.grid(bim, n=450)
e_ban = SM.enki(ban, J=1000, rng=np.random.default_rng(6))
e_bim = SM.enki(bim, J=1000, rng=np.random.default_rng(6))
fig, axs = plt.subplots(1, 2, figsize=(11, 4.3))
for ax, pr, g, e, t_ in [(axs[0], ban, g_ban, e_ban, "Banana"), (axs[1], bim, g_bim, e_bim, "Two islands")]:
    BP.density(ax, g)
    BP.samples(ax, e["samples"], EN, s=3, alpha=0.4)
    BP.ellipse(ax, g.mean, g.cov, nsig=2, color=S.INK, lw=1.0, label="2σ ellipse of the exact posterior")
    BP.frame(ax, pr, equal=True)
    ax.set_title(f"{t_}: EnKI ensemble (J = 1000) vs exact posterior")
axs[0].legend(loc="lower left", fontsize=8)
nb_e = np.mean(e_bim["samples"][:, 1] > 0)
nb_t = g_bim.mass(lambda X, Y: Y > 0)
ipk = np.unravel_index(np.argmax(g_bim.P), g_bim.P.shape)
lp_peak = bim.logp_batch(np.array([[g_bim.X[ipk], g_bim.Y[ipk]]]))[0]
far = np.mean(bim.logp_batch(e_bim["samples"]) < lp_peak + np.log(0.01))
S.badge(axs[1], f"{far:.0%} of members where density < 1 % of the peak", ok=False, loc="lower left", fs=8)
in_arc = g_ban.P[np.clip(np.searchsorted(g_ban.Y[:, 0], e_ban["samples"][:, 1]), 0, g_ban.P.shape[0] - 1),
                 np.clip(np.searchsorted(g_ban.X[0], e_ban["samples"][:, 0]), 0, g_ban.P.shape[1] - 1)] >= g_ban.levels[0.9]
S.badge(axs[0], f"{in_arc.mean():.0%} of members inside the exact 90 % region", loc="upper right", fs=8)
fig.tight_layout()
bk.savefig(fig, "2-24_gaussian_lens")
plt.show()
for name, e, g in [("banana", e_ban, g_ban), ("two islands", e_bim, g_bim)]:
    me, Ce = stats(e["samples"])
    print(f"{name:11s}: EnKI mean {me.round(3)} exact {g.mean.round(3)} | EnKI sd {np.sqrt(np.diag(Ce)).round(3)} "
          f"exact {np.sqrt(np.diag(g.cov)).round(3)} | iterations {len(e['theta']) - 1}, final WRMS {e['wrms'][-1]:.2f}")
print("two islands: north share EnKI %.3f, exact %.3f; members where the density is below 1%% of the peak: %.0f%%"
      % (nb_e, nb_t, 100 * far))
print("two islands: final theta %.3f after %d iterations (limit 20), final WRMS %.2f"
      % (e_bim["theta"][-1], len(e_bim["theta"]) - 1, e_bim["wrms"][-1]))
print("banana: fraction of members inside the exact 90 %% region %.2f (a perfect sampler: 0.90)" % in_arc.mean())

# %% [markdown]
# Watch the banana run: in EnKI every member keeps its identity from iteration to iteration (no resampling, unlike
# SMC), so we can let each member glide from its position at one iteration to the next.

# %%
frames = []
Hb = e_ban["history"]
gb_small = P.grid(ban, n=250)
for k in range(len(Hb) - 1):
    A_, B_ = Hb[k]["X"], Hb[k + 1]["X"]
    for f in np.linspace(0, 1, 7):
        fig, ax = plt.subplots(figsize=(5.6, 4.2), dpi=100)
        BP.density(ax, gb_small)
        Xf_ = (1 - f) * A_ + f * B_
        ax.scatter(*Xf_[:600].T, s=5, color=EN, alpha=0.45, lw=0)
        ax.plot(0, -2.5, "^", color=S.INK, ms=8)
        ax.set_xlim(-3.2, 3.2); ax.set_ylim(-3.0, 3.0); ax.set_aspect("equal")
        th = (1 - f) * Hb[k]["theta"] + f * Hb[k + 1]["theta"]
        ax.set_title(f"EnKI on the banana: iteration {k} → {k + 1},  θ = {th:.3f}", fontsize=10)
        fig.tight_layout()
        frames.append(S.fig_to_rgb(fig))
        if k == len(Hb) - 2 and f == 1.0:
            final_png = bk.savefig(fig, "2-24_enki_banana_final")      # the last frame as a 300-dpi still
        plt.close(fig)
gif = S.frames_to_gif(frames, S.FIG / "2-24_enki_banana.gif", fps=10, hold_last=20)
print("GIF size %.1f MB" % (gif.stat().st_size / 1e6))
S.show_gif(gif, width=500)
from IPython.display import Image as _Image, display as _display
print("The last frame at full resolution (300 dpi):")
_display(_Image(filename=str(final_png), width=500))

# %% [markdown]
# Compare each cloud with the grey exact posterior and with the black ellipse (the Gaussian that has the exact
# mean and covariance).
# * **Banana.** The cloud does bend along the arc, because each member moves by an amount that depends on its own
#   predicted datum, and its mean and standard deviations come out close to the exact values (printed). It is more diffuse
#   than the exact posterior, though: fewer members fall inside the exact 90 % region than the 90 % a perfect sampler would put
#   there (printed), and stragglers sit off the arc.
# * **Two islands.** The cloud becomes one long streak through both islands *and* the sea between them, much like the
#   black ellipse. Most members sit where the posterior density is negligible (printed), so the north share, although
#   numerically close to the exact value here, means nothing. EnKI's own monitor raises the alarm: θ had not reached 1
#   after the 20-iteration limit and the WRMS stayed far above 1 (printed; compare section 9).
#
# In these runs the update bent the cloud but did not split it into the two islands. For multimodal problems a method
# built to weigh separate modes, such as SMC (notebook 2.20), is the safer choice; for this curved but unimodal one EnKI
# gave a fair summary at a small cost in forward runs.

# %% [markdown]
# ## 8. Capping $\Delta\theta$: do not take a big step while the cloud is still wide
#
# Section 3 showed that the ensemble slope is an average over the cloud. A big tempering step taken while the cloud is
# much wider than the posterior moves every member with a slope that is wrong for the final posterior. The adaptive
# rule can do exactly that when the data are many and each fits moderately: WRMS is a per-datum average, so it can
# look modest while the total information is huge.
#
# Toy problem built to test this: a log-afterslip time series $u(t) = A\log(1 + t/t_c)$ sampled 100 times in one year
# (1 mm noise), with $m_1 = \log A$ and $m_2 = -\log t_c$. The prediction depends on $m_2$ through $e^{m_2}$, as
# stress-driven afterslip depends on $\log v_0$. We compare the uncapped rule with $\Delta\theta_{\max} = 0.1$, six seeds each, against
# the exact posterior (grid).

# %%
t_as = np.linspace(0.01, 1.0, 100)
t_j = jnp.asarray(t_as)
fwd_as = lambda m: jnp.exp(m[0]) * jnp.log1p(t_j * jnp.exp(m[1]))
m_true = np.array([0.0, 3.0])
y_as = np.asarray(fwd_as(jnp.asarray(m_true))) + 0.1 * np.random.default_rng(2).standard_normal(t_as.size)
aft = P.InverseProblem(fwd_as, y_as, 0.1, m0=[0.2, 2.5], P=np.diag([0.3 ** 2, 0.8 ** 2]),
                       labels=("$m_1 = \\log A$", "$m_2 = -\\log t_c$"))


def autogrid(pr, ext0, n=450):
    g0 = P.grid(pr, extent=ext0, n=n)
    c, s = g0.mean, np.sqrt(np.diag(g0.cov))
    return P.grid(pr, extent=(c[0] - 7 * s[0], c[0] + 7 * s[0], c[1] - 7 * s[1], c[1] + 7 * s[1]), n=n)


def score(X, g):
    ev, U = np.linalg.eigh(g.cov)
    me = np.abs((X.mean(0) - g.mean) @ U / np.sqrt(ev)).max()
    vr = np.diag(U.T @ np.cov(X.T) @ U) / ev
    return me, vr[0], vr[1]


g_aft = autogrid(aft, (-2, 2, 0, 6))
aft.extent = (g_aft.X.min(), g_aft.X.max(), g_aft.Y.min(), g_aft.Y.max())
runs = {cap: [SM.enki(aft, J=1000, rng=np.random.default_rng(s), max_dtheta=cap) for s in range(6)] for cap in [1.0, 0.1]}

fig, axs = plt.subplots(1, 4, figsize=(16, 3.7), gridspec_kw=dict(width_ratios=[1.15, 0.9, 1, 1]))
ax = axs[0]
for X in runs[1.0][0]["samples"][:40]:
    ax.plot(t_as * 365, np.asarray(fwd_as(jnp.asarray(X))), color=EN, lw=0.4, alpha=0.4)
ax.plot(t_as * 365, y_as, "o", color=S.INK, ms=2.5)
ax.set_xlabel("days after the earthquake"); ax.set_ylabel("displacement (cm)")
ax.set_title("data and 40 uncapped members")
for cap, ls, mk in [(1.0, "-", "o"), (0.1, "-", "s")]:
    th = runs[cap][0]["theta"]
    axs[1].plot(range(len(th)), th, ls, marker=mk, color=EN if cap == 1.0 else S.INK2, ms=5,
                label="uncapped" if cap == 1.0 else "Δθ ≤ 0.1")
axs[1].set_xlabel("iteration"); axs[1].set_ylabel("θ"); axs[1].set_title("tempering schedule (seed 0)")
axs[1].legend(fontsize=8)
for ax, cap, title in [(axs[2], 1.0, "uncapped: final ensemble"), (axs[3], 0.1, "Δθ ≤ 0.1: final ensemble")]:
    BP.density(ax, g_aft)
    X = runs[cap][0]["samples"]
    BP.samples(ax, X, EN, s=3, alpha=0.35)
    BP.ellipse(ax, *stats(X), nsig=2, color=EN, lw=1.2)
    BP.ellipse(ax, g_aft.mean, g_aft.cov, nsig=2, color=S.INK, lw=1.0)
    BP.frame(ax, aft)
    sc_ = score(X, g_aft)
    S.badge(ax, f"mean off by {sc_[0]:.2f} sd; variance × {sc_[1]:.2f} / {sc_[2]:.2f}", ok=sc_[0] < 0.15 and sc_[1] < 1.3,
            loc="upper right", fs=7.5)
    ax.set_title(title)
fig.tight_layout()
bk.savefig(fig, "2-24_dtheta_cap")
plt.show()
print("seed  cap    steps  final WRMS   mean error (post. sd)   variance ratio short / long axis")
for cap in [1.0, 0.1]:
    for s, e in enumerate(runs[cap]):
        sc_ = score(e["samples"], g_aft)
        print(f"{s:4d}  {cap:3.1f}  {len(e['theta']) - 1:6d}  {e['wrms'][-1]:9.3f}   {sc_[0]:10.3f}              {sc_[1]:.2f} / {sc_[2]:.2f}")
print("uncapped theta (seed 0):", np.round(runs[1.0][0]["theta"], 3))

# %% [markdown]
# Read the printed table seed by seed. The uncapped rule takes a few small steps and then one big jump in θ (the
# schedule panel), a pattern to look for in the log of a real run. Both versions end with the same WRMS, so the fit alone does
# not reveal the problem. Against the exact posterior (grid), however, the uncapped ensemble is off-centre and too wide along the
# short (well-determined) axis in every seed, while the capped one is much closer (still somewhat too wide).
#
# **When does the cap matter?** The same comparison on the other test problems:

# %%
tests = {"banana": (ban, (-2.6, 2.6, -2.9, 1.1)), "locking depth": (P.screw_dislocation(), (0.0, 80, 5, 80)),
         "log-afterslip (above)": (aft, (-2, 2, 0, 6))}
s_ = np.linspace(1, 10, 20)
s_j = jnp.asarray(s_)
fwd_pl = lambda m: jnp.exp(m[0]) * s_j ** m[1]
y_pl = np.asarray(fwd_pl(jnp.asarray([-2.0, 3.0])))
sig_pl = 0.05 * y_pl.mean()
tests["power law exp(m1)·s^m2"] = (P.InverseProblem(fwd_pl, y_pl + sig_pl * np.random.default_rng(1).standard_normal(s_.size),
                                                          sig_pl, m0=[-1.0, 2.5], P=np.diag([1.5 ** 2, 0.8 ** 2])), (-8, 4, -1, 6))
print("problem                    cap   median mean error (sd)  [range]       median short-axis variance ratio [range]   steps")
summary = {}
for name, (pr, ext0) in tests.items():
    g = autogrid(pr, ext0)
    for cap in [1.0, 0.1]:
        es = [SM.enki(pr, J=1000, rng=np.random.default_rng(s), max_dtheta=cap) for s in range(6)]
        sc = np.array([score(e["samples"], g) for e in es])
        summary[(name, cap)] = sc
        print(f"{name:26s} {cap:4.1f}   {np.median(sc[:, 0]):6.3f} [{sc[:, 0].min():.2f}, {sc[:, 0].max():.2f}]"
              f"          {np.median(sc[:, 1]):6.2f} [{sc[:, 1].min():.2f}, {sc[:, 1].max():.2f}]"
              f"                    {int(np.median([len(e['theta']) - 1 for e in es]))}")

# %% [markdown]
# The cap matters only where the uncapped rule takes a large step while the forward model is still strongly
# nonlinear across the cloud: here the log-afterslip and power-law problems. On the banana and the locking-depth
# problem the two versions agree (printed). The price of the cap is more iterations, each one a full ensemble of
# forward runs. A cheap check: if the WRMS rises after an update, if the best members fit far better
# than the median, or if member fit correlates with one parameter, rerun with $\Delta\theta_{\max} = 0.1$.

# %% [markdown]
# ## 9. θ as a health monitor
#
# A crawling θ is an alarm, not a diagnosis: section 7 already showed it with a correctly specified two-island
# problem. Its possible causes include data the forward model cannot produce, a strongly non-Gaussian posterior,
# too few members and plain bugs. This section shows the first cause.
#
# On synthetic data generated by the same forward model, θ should reach 1 in a handful of iterations. If the data are
# something the forward model cannot produce for *any* parameters, the misfits never drop, the rule keeps taking tiny
# steps, and θ crawls. Here: three-component GPS time series (E, N, U; 40 epochs) from a log-afterslip model. The
# forward model returns rows in (epoch, component) order. The "bug" writes the same synthetic data column by column
# (all E, then all N, then all U), a common reshape-order mistake when synthetic data are written to a file.

# %%
tt = np.linspace(0.02, 1.0, 40)
tt_j = jnp.asarray(tt)


def fwd_enu(m):                    # m = (log A_E, log A_N, log A_U, -log t_c); rows ordered (epoch, component)
    A = jnp.exp(m[:3])
    return (jnp.log1p(tt_j * jnp.exp(m[3]))[:, None] * A[None, :]).ravel()


m_enu = np.array([np.log(30.0), np.log(12.0), np.log(8.0), 3.0])
clean = np.asarray(fwd_enu(jnp.asarray(m_enu))).reshape(40, 3) + np.random.default_rng(3).standard_normal((40, 3))
data = {"consistent (row order)": clean.ravel(), "scrambled (column order)": clean.ravel(order="F")}
fig, axs = plt.subplots(1, 2, figsize=(11, 3.4))
for (lab, d), col, mk in zip(data.items(), [EN, S.BAD], ["o", "s"]):
    pr = P.InverseProblem(fwd_enu, d, 1.0, m0=[3.0, 2.5, 2.0, 2.5], P=np.diag([0.7 ** 2] * 3 + [0.8 ** 2]))
    e = SM.enki(pr, J=500, rng=np.random.default_rng(0))
    axs[0].plot(range(len(e["theta"])), e["theta"], "-", marker=mk, color=col, ms=4, label=lab)
    axs[1].semilogy(range(len(e["wrms"])), e["wrms"], "-", marker=mk, color=col, ms=4, label=lab)
    print(f"{lab}:\n   theta {np.round(e['theta'], 4)}\n   WRMS  {np.round(e['wrms'], 2)}")
axs[0].set_title("θ per iteration"); axs[0].set_xlabel("iteration"); axs[0].legend(fontsize=8)
axs[1].set_title("WRMS (ensemble) per iteration"); axs[1].set_xlabel("iteration")
axs[1].axhline(1, color=S.MUTED, lw=0.8)
fig.tight_layout()
bk.savefig(fig, "2-24_theta_monitor")
plt.show()

# %% [markdown]
# With consistent data θ shoots to 1 and WRMS falls to about 1. With scrambled rows θ crawls for many iterations and
# WRMS stalls far above 1: the inversion cannot fit data that no parameter set produces. Watch the first 3-4
# iterations of θ in the log; this pattern is visible long before the run ends.

# %% [markdown]
# ## 10. The subspace property: the answer lives in the span of the initial ensemble
#
# The ETKF update writes every new member as the ensemble mean plus a *linear combination of the current anomalies*,
# and those are combinations of the initial ones. So the final mean is always initial mean + (combination of the $J$
# initial anomalies), whatever the data say (Iglesias et al. 2013; Tong & Morzfeld 2023). With fewer members than the
# number of directions the data can resolve, some features of the exact posterior are simply unreachable.
#
# Test: a 1-D field with 100 unknowns, a Gaussian-process (Matérn) prior, 40 noisy point observations, linear forward
# model, so the exact posterior is known. Two priors: smooth (long correlation length) and rough (short). For each,
# J = 10, 50, 200, 2000, two seeds each.

# %%
n_f = 100
xg = np.linspace(0, 1, n_f)


def matern(d, L, nu, s=1.0):
    d = np.maximum(d, 1e-12)
    z = np.sqrt(2 * nu) * d / L
    c = s ** 2 * 2 ** (1 - nu) / gamma(nu) * z ** nu * kv(nu, z)
    c[d <= 1e-12] = s ** 2
    return c


obs_idx = np.sort(np.random.default_rng(6).choice(n_f, 40, replace=False))
Gf = np.zeros((40, n_f))
Gf[np.arange(40), obs_idx] = 1.0
priors = {"smooth prior (L = 0.12, ν = 1.5)": (0.12, 1.5), "rough prior (L = 0.05, ν = 0.5)": (0.05, 0.5)}
Js = [10, 50, 200, 2000]
fig, axs = plt.subplots(2, 4, figsize=(16, 5.6), sharex=True)
err_tab = {}
for i, (pname, (L, nu)) in enumerate(priors.items()):
    Cf = matern(np.abs(xg[:, None] - xg[None, :]), L, nu) + 1e-8 * np.eye(n_f)
    ev = np.linalg.eigvalsh(Cf)[::-1]
    n95 = int(np.searchsorted(np.cumsum(ev) / ev.sum(), 0.95) + 1)
    f_true = np.linalg.cholesky(Cf) @ np.random.default_rng(5).standard_normal(n_f)
    yf = Gf @ f_true + 0.1 * np.random.default_rng(7).standard_normal(40)
    lin = P.linear(Gf, yf, 0.1, np.zeros(n_f), Cf)
    mx, Cx = lin.exact
    sx = np.sqrt(np.diag(Cx))
    for j, J_ in enumerate(Js):
        ax = axs[i, j]
        ax.fill_between(xg, mx - 2 * sx, mx + 2 * sx, color="#d9d8d1", lw=0, label="exact ±2 sd")
        ax.plot(xg, mx, color=S.INK, lw=1.2, label="exact mean")
        rows = []
        for seed, ls in [(0, "-"), (1, "--")]:
            e = SM.enki(lin, J=J_, rng=np.random.default_rng(seed))
            Xe = e["samples"]
            mE, sE = Xe.mean(0), Xe.std(0, ddof=1)
            ax.plot(xg, mE, color=EN, lw=1.1, ls=ls, label=f"EnKI mean, seed {seed}")
            ax.plot(xg, mE + 2 * sE, color=EN, lw=0.5, ls=ls, alpha=0.6)
            ax.plot(xg, mE - 2 * sE, color=EN, lw=0.5, ls=ls, alpha=0.6)
            X0 = e["history"][0]["X"]
            A = (X0 - X0.mean(0)).T
            r_ = mE - X0.mean(0)
            coef = np.linalg.lstsq(A, r_, rcond=None)[0]
            rows.append((np.sqrt(np.mean(((mE - mx) / sx) ** 2)), np.median(sE / sx),
                         np.linalg.norm(A @ coef - r_) / np.linalg.norm(r_), mE))
        diff = np.sqrt(np.mean(((rows[0][3] - rows[1][3]) / sx) ** 2))
        err_tab[(pname, J_)] = (rows[0][0], rows[1][0], diff, rows[0][1], rows[0][2], rows[1][2])
        ax.plot(xg[obs_idx], yf, "o", color=S.INK2, ms=2)
        ax.set_title(f"J = {J_}", fontsize=10)
        S.badge(ax, f"mean error {rows[0][0]:.2f} / {rows[1][0]:.2f} sd", ok=max(rows[0][0], rows[1][0]) < 0.5,
                loc="lower left", fs=7.5)
    axs[i, 0].set_ylabel(f"{pname}\n({n95} prior modes hold 95 %)", fontsize=9)
axs[0, 0].legend(loc="upper left", fontsize=6.5, ncol=1)
for ax in axs[1]:
    ax.set_xlabel("position along the fault (normalised)")
fig.tight_layout()
bk.savefig(fig, "2-24_subspace")
plt.show()
print("prior                              J    mean error (sd) seed0/seed1   seed-to-seed (sd)   sd ratio   out-of-span residual")
for (pname, J_), v in err_tab.items():
    print(f"{pname:34s} {J_:5d}   {v[0]:6.2f} / {v[1]:5.2f}           {v[2]:6.2f}          {v[3]:5.2f}      {max(v[4], v[5]):.1e}")

# %% [markdown]
# **How to read it.** Black line and grey band: the exact posterior. Violet: EnKI mean and ±2 sd, solid and dashed for
# two seeds. "Mean error" is the RMS difference from the exact mean in units of the exact posterior sd (badge ✓ when
# both seeds are within 0.5 sd).
# * The last column of the table is the size of the part of (final mean − initial mean) that lies *outside* the span
#   of the initial ensemble. It is zero to machine precision for every run: the subspace property, measured.
# * With too few members the mean is wrong by several posterior sd, the band is too narrow (sd ratio < 1), and the
#   two seeds disagree with each other by more than the error bars. That disagreement is what a **two-seed test**
#   detects: run the same inversion twice with different random seeds and compare (2.26 measures how well it works).
# * The rough prior needs many more members than the smooth one, although both have 100 unknowns. What matters is
#   how many independent directions the prior and the data can excite (the printed "prior modes"), not the raw
#   dimension. Changing L or ν changes the draws, hence the subspace and what EnKI can reach.
#
# The standard cures are **localisation** and **inflation** (2.26 tests both). Localisation multiplies the ensemble
# covariances by a taper that goes to zero with distance, so the spurious long-range correlations that a small
# ensemble produces by chance are ignored; it also breaks the subspace restriction (Tong & Morzfeld 2023 prove
# convergence for localised EKI). Inflation multiplies the spread by a factor slightly above one at each step, to
# counter the shrinking of small ensembles.

# %% [markdown]
# ## 11. Summary
#
# | method | data use | gradient? | output | fails when |
# |---|---|---|---|---|
# | EnKF | each observation once, as time goes on | no | state estimate with spread, over time | process noise too small (lag, overconfidence) |
# | classic EKI | full data every iteration | no | a best-fitting model (optimiser) | used as an uncertainty estimate (collapse) |
# | EnKI (tempered) | fractions Δθ adding to 1 | no | an ensemble ≈ posterior (exact for linear-Gaussian problems as J grows) | multimodal posteriors; big Δθ on strongly nonlinear forwards; J too small for the directions that matter |
#
# **In practice.** Three habits follow from sections 7-10:
# 1. Read θ in the log during the first iterations. A crawling θ with WRMS stuck far above 1 is an alarm with several
#    possible causes (data the forward model cannot produce, section 9; a posterior far from Gaussian, section 7; too
#    few members; a bug), not slow convergence.
# 2. When the forward model is strongly nonlinear across the ensemble (stress-driven afterslip depends roughly
#    exponentially on $\log v_0$ and $1/a\sigma$), cap $\Delta\theta$ (section 8). The same final WRMS does not mean
#    the same posterior.
# 3. With fewer members than the directions the data resolve (for example several hundred members for thousands of
#    unknowns), the answer is confined to the span of the prior draws (section 10). Run the two-seed test, and consider
#    localisation (2.26), before trusting the error bars.
#
# Next: **2.25 · Nonlinearity and the extended Kalman filter**.
