# %% [markdown]
# # 2.09 · Regularisation and smoothing are priors
#
# 2.08 showed that the MAP of a linear-Gaussian posterior is a regularised least-squares solution. This notebook runs
# that dictionary the other way: every common regulariser (damping, Laplacian smoothing, L1 sparsity, total
# variation, a Matérn covariance; all defined below) is a prior, and each one prefers a different kind of slip model. Seeing what each
# prior believes, and what its MAP looks like next to its full posterior, is the best protection against reading a
# regulariser's taste as a property of the Earth.
#
# **Words this notebook defines:** regulariser, penalty, regularisation weight λ, constrained vs penalised form,
# L2 (ridge, damping), L1 (lasso, sparsity), shrinkage, soft-thresholding, first- and second-difference smoothing,
# improper prior, null space of a smoothing operator, total variation, Matérn covariance, discrepancy principle,
# SPDE link, Bayesian lasso.
#
# **Before this:** 2.07, 2.08. · **You will be able to:** translate a regulariser into the prior it implies and back;
# predict what kind of slip model each prior prefers; explain why an ill-posed problem needs one; tell an L1 MAP's
# sparsity from what the posterior says. · **Optional sections:** 5 (Laplacian and Matérn) on a first reading.

# %% [markdown]
# **Problems in this notebook**
#
# | problem | unknowns | data | forward model | linear? |
# |---|---|---|---|---|
# | geometry of L2 and L1 (section 1) | 2 | 2 | misfit with correlated parameters | linear |
# | orthogonal design (section 2) | 2 | 2 | $d_i = m_i$ | linear |
# | antiplane fault (sections 3, 4, 6; as in notebook 2.33 with a different true slip) | 30 slip patches | 24 stations | $\mathbf d = \mathbf G\mathbf m$ | linear |
# | Matérn and Laplacian link (section 5) | 400 grid values | none | prior only | no data |

# %%
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path.cwd().parent))
import numpy as np
import matplotlib.pyplot as plt
from scipy.special import kv, gamma
from sklearn.linear_model import lars_path
import bayeskit as bk
from bayeskit import problems as P, samplers as SM, diagnostics as D, plotting as BP, style as S

import numpyro
numpyro.set_host_device_count(4)                 # 4 NUTS chains in parallel (before any JAX computation)
import jax, jax.numpy as jnp
import numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS

np.set_printoptions(precision=3, suppress=True)
TRUE_KW = dict(color="#b5b4ad", lw=4.5, solid_capstyle="round", zorder=1, label="true slip")
soft = lambda x, t: np.sign(x) * np.maximum(np.abs(x) - t, 0.0)


def step_profile(ax, m, z_edges, **kw):
    """Plot slip per patch as a vertical staircase (one constant value over each patch's depth range)."""
    ax.plot(np.repeat(m, 2), np.c_[z_edges[:-1], z_edges[1:]].ravel(), **kw)

# %% [markdown]
# ## 1. A regulariser is a prior, and its shape decides the answer
#
# **In plain words.** A regularised inversion minimises "misfit + λ × penalty". The penalty says which models you
# prefer when the data cannot decide. Writing the penalty as minus a log probability turns it into a prior, so every
# regularised inversion is the MAP of some Bayesian problem. Two penalties dominate: the sum of squares of the
# parameters (L2) and the sum of their absolute values (L1). They look similar but behave very differently: L2 shrinks
# every parameter a little, L1 sets some parameters exactly to zero.
#
# **Precisely.** With misfit $\Phi_d(\mathbf m) = \tfrac12(\mathbf d-\mathbf G\mathbf m)^\top\mathbf C_d^{-1}(\mathbf d-\mathbf G\mathbf m)$,
#
# $$\hat{\mathbf m} = \arg\min_{\mathbf m}\;\Phi_d(\mathbf m) + \lambda R(\mathbf m)
# \quad\Longleftrightarrow\quad \text{MAP with prior } p(\mathbf m) \propto e^{-\lambda R(\mathbf m)} .$$
#
# By a Lagrange-multiplier argument (the standard link between penalties and constraints in optimisation) the same
# solution solves the constrained problem $\min \Phi_d$ subject to $R(\mathbf m) \le t$ for some $t$.
#
# Notation for the table: $\lVert\mathbf m\rVert_2^2$ is the sum of squares of the parameters and $\lVert\mathbf m\rVert_1$
# the sum of their absolute values; $\mathbf D_1\mathbf m$ lists the differences between neighbouring patches and
# $\mathbf D_2\mathbf m$ the differences of those differences (a discrete second derivative, the Laplacian). A *random
# walk* is a sequence in which each value is the previous one plus an independent random step; an *integrated* random
# walk does this to the slope. A **Matérn** correlation decays with distance over a length scale ℓ, with a parameter ν
# that sets how smooth the field is, and a **Gaussian process** is a Gaussian prior on a whole function (both are
# explained fully in 2.12). The dictionary used in this notebook:
#
# | regulariser $\lambda R(\mathbf m)$ | prior | prefers |
# |---|---|---|
# | $\tfrac{\lambda}{2}\lVert\mathbf m\rVert_2^2$ (L2, ridge, damping) | independent Gaussian, variance $1/\lambda$ | small values everywhere |
# | $\lambda\lVert\mathbf m\rVert_1$ (L1, lasso) | independent Laplace, scale $1/\lambda$ | few nonzero values |
# | $\tfrac{\lambda}{2}\lVert\mathbf D_1\mathbf m\rVert_2^2$ (first-difference smoothing) | Gaussian random walk (improper) | small changes between neighbours |
# | $\tfrac{\lambda}{2}\lVert\mathbf D_2\mathbf m\rVert_2^2$ (second-difference, "Laplacian") | integrated random walk (improper) | small changes of slope |
# | $\lambda\lVert\mathbf D_1\mathbf m\rVert_1$ (total variation) | Laplace increments | piecewise-constant blocks |
# | $\tfrac{\lambda}{2}\mathbf m^\top\mathbf R^{-1}\mathbf m$, $\mathbf R$ a Matérn correlation | Gaussian process | smooth over a length scale ℓ |
#
# The picture below is the constrained form in two dimensions: the misfit's elliptical contours meet the region
# $R(\mathbf m) \le t$, a circle for L2 and a diamond for L1. The solution is where the smallest reachable ellipse
# touches the region. Diamonds have corners on the axes, and ellipses tend to touch them there.

# %%
m_ls = np.array([2.0, 0.6])                               # least-squares point (misfit minimum)
H2 = 2.0 * np.array([[1.0, 0.7], [0.7, 1.0]])             # misfit curvature: correlated parameters


def l2_sol(lam):
    return np.linalg.solve(H2 + lam * np.eye(2), H2 @ m_ls)


def l1_sol(lam, sweeps=2000):
    m = m_ls.copy()
    for _ in range(sweeps):                               # exact coordinate minimisation, 2 unknowns
        for j in range(2):
            k = 1 - j
            rhs = H2[j, j] * m_ls[j] + H2[j, k] * (m_ls[k] - m[k])
            m[j] = soft(rhs, lam) / H2[j, j]
    return m


lams = np.linspace(0, 5, 251)
path_l2 = np.array([l2_sol(l) for l in lams])
path_l1 = np.array([l1_sol(l) for l in lams])
lam_zero = lams[np.argmax(path_l1[:, 1] == 0)]
lam_pick = 1.6 * lam_zero
mr, ml = l2_sol(lam_pick), l1_sol(lam_pick)
print(f"L1: the second parameter becomes exactly zero at lambda = {lam_zero:.2f} and stays zero; the first reaches zero at lambda = {lams[np.argmax(path_l1[:, 0] == 0)]:.2f}")
print(f"at lambda = {lam_pick:.2f}:  L2 solution {np.round(mr, 3)} (both nonzero),  L1 solution {np.round(ml, 3)}")
print(f"L2 path: second parameter at lambda = 5 is still {path_l2[-1, 1]:.3f} (it only tends to zero as lambda grows without bound)")

# %%
Xg, Yg = np.meshgrid(np.linspace(-0.8, 2.8, 400), np.linspace(-1.2, 2.0, 400))
dX, dY = Xg - m_ls[0], Yg - m_ls[1]
Phi = 0.5 * (H2[0, 0] * dX ** 2 + 2 * H2[0, 1] * dX * dY + H2[1, 1] * dY ** 2)
fig, axs = plt.subplots(1, 3, figsize=(15, 4.3), gridspec_kw=dict(width_ratios=[1, 1, 1.15]))
for ax, msol, kind in [(axs[0], mr, "L2"), (axs[1], ml, "L1")]:
    ax.contour(Xg, Yg, Phi, levels=[0.05, 0.2, 0.5, 1.0, 1.6], colors=S.MUTED, linewidths=0.6)
    ax.contour(Xg, Yg, Phi, levels=[0.5 * (msol - m_ls) @ H2 @ (msol - m_ls)], colors=S.INK, linewidths=1.4)
    if kind == "L2":
        r = np.linalg.norm(msol)
        th = np.linspace(0, 2 * np.pi, 300)
        ax.fill(r * np.cos(th), r * np.sin(th), color=S.BLUE, alpha=0.15, lw=0)
        ax.plot(r * np.cos(th), r * np.sin(th), color=S.BLUE, lw=1.5)
        title = "(a) L2: the ellipse touches the circle\nanywhere; both parameters stay nonzero"
    else:
        r = np.abs(msol).sum()
        dia = np.array([[r, 0], [0, r], [-r, 0], [0, -r], [r, 0]])
        ax.fill(dia[:, 0], dia[:, 1], color=S.ORANGE, alpha=0.15, lw=0)
        ax.plot(dia[:, 0], dia[:, 1], color=S.ORANGE, lw=1.5)
        title = "(b) L1: the ellipse touches the diamond\nat a corner, so the second parameter is 0"
    ax.plot(*m_ls, "P", color=S.INK, ms=9, mec=S.SURFACE, label="least-squares point")
    ax.plot(*msol, "o", color=S.BLUE if kind == "L2" else S.ORANGE, ms=8, mec=S.SURFACE, label=f"{kind} solution")
    ax.axhline(0, color=S.AXIS, lw=0.6); ax.axvline(0, color=S.AXIS, lw=0.6)
    ax.set_aspect("equal"); ax.set_xlim(-0.8, 2.8); ax.set_ylim(-1.2, 2.0)
    ax.set_xlabel("$m_1$"); ax.set_ylabel("$m_2$"); ax.set_title(title, fontsize=10)
    ax.legend(fontsize=7.8, loc="upper left")
ax = axs[2]
for j, ls in [(0, "-"), (1, (0, (4, 2)))]:
    ax.plot(lams, path_l2[:, j], color=S.BLUE, lw=1.6, ls=ls, label=f"L2: $m_{j + 1}$")
    ax.plot(lams, path_l1[:, j], color=S.ORANGE, lw=1.6, ls=ls, label=f"L1: $m_{j + 1}$")
ax.axhline(0, color=S.AXIS, lw=0.6)
ax.axvline(lam_pick, color=S.MUTED, lw=0.8)
ax.set_xlabel("regularisation weight λ"); ax.set_ylabel("solution")
ax.set_title("(c) As λ grows: L2 shrinks, L1 switches off", fontsize=10)
ax.legend(fontsize=7.8, ncol=2)
fig.tight_layout()
bk.savefig(fig, "2-09_geometry")
plt.show()

# %% [markdown]
# **How to read it.** (a) and (b) use the same λ (grey line in (c)). The black ellipse is the misfit level reached by
# each solution. The circle has no corners, so the touching point has both coordinates nonzero. The diamond's corners
# stick out along the axes, so the ellipse meets it at a corner where $m_2 = 0$. (c) The full paths: with L2 the second
# parameter shrinks smoothly and never reaches zero; with L1 it hits zero at a finite λ and stays there (numbers
# printed above).
#
# **Common misreadings.**
# * *"L1 and L2 regularisation differ only in strength."* They differ in shape, and shape decides which parameters
#   survive.
# * *"A regulariser expresses no belief."* It is exactly a prior; choosing it is a modelling decision.

# %% [markdown]
# ## 2. Worked example: shrinkage versus soft-thresholding
#
# **In plain words.** When each datum measures one parameter directly, the two penalties have one-line answers. L2
# divides every measurement by the same factor, so everything shrinks toward zero but nothing reaches it. L1 subtracts
# a fixed amount from the size of every measurement, and anything smaller than that amount becomes exactly zero.
#
# **Precisely.** For $d_i = m_i + e_i$ with unit noise, minimising $\tfrac12(d_i - m_i)^2 + \tfrac{\lambda}{2}m_i^2$
# gives the **shrinkage** rule $\hat m_i = d_i/(1+\lambda)$; minimising $\tfrac12(d_i - m_i)^2 + \lambda\lvert m_i\rvert$
# gives the **soft-threshold** rule $\hat m_i = \operatorname{sign}(d_i)\max(\lvert d_i\rvert - \lambda, 0)$
# (set the derivative to zero on each side of $m_i = 0$; at zero the absolute value has a corner, and any
# $\lvert d_i\rvert \le \lambda$ lands there).
#
# **Worked example** (inputs: $\mathbf d$ = (2.0, 0.4), λ = 0.5):

# %%
d2, lam2 = np.array([2.0, 0.4]), 0.5
print(f"shrinkage       d / (1 + lambda) = {d2} / {1 + lam2} = {np.round(d2 / (1 + lam2), 4)}")
print(f"soft-threshold  sign(d) max(|d| - lambda, 0) = {np.round(soft(d2, lam2), 4)}   (|0.4| < 0.5, so the second is zero)")
dd = np.linspace(-3, 3, 601)
fig, ax = plt.subplots(figsize=(6.2, 3.6))
ax.plot(dd, dd, color=S.AXIS, lw=1, label="no regularisation (estimate = datum)")
ax.plot(dd, dd / (1 + lam2), color=S.BLUE, lw=2, label="L2: shrinkage d / (1 + λ)")
ax.plot(dd, soft(dd, lam2), color=S.ORANGE, lw=2, label="L1: soft-threshold")
ax.axvspan(-lam2, lam2, color=S.ORANGE, alpha=0.08, lw=0)
ax.text(0, -2.6, "|d| ≤ λ: L1 estimate is 0", ha="center", fontsize=8, color=S.INK2)
ax.plot(d2, soft(d2, lam2), "o", color=S.ORANGE, ms=6); ax.plot(d2, d2 / (1 + lam2), "o", color=S.BLUE, ms=6)
ax.set_xlabel("datum d"); ax.set_ylabel("estimate"); ax.set_title(f"One datum per parameter, λ = {lam2}")
ax.legend(fontsize=7.8, loc="upper left")
fig.tight_layout()
bk.savefig(fig, "2-09_shrink_vs_threshold")
plt.show()

# %% [markdown]
# **Common misreadings.**
# * *"Soft-thresholding removes noise only."* It also shrinks the large values by λ, so L1 estimates are biased toward
#   zero by a fixed amount.
#
# **In geodesy.** Sparse (L1) slip inversions exploit exactly this: weak patches are switched off and strong ones
# survive, minus λ.

# %% [markdown]
# ## 3. Six regularisers on the same fault
#
# **In plain words.** Same 24 stations, same noise, six different preferences. The true slip has two features: a
# compact 1.2 m block between 3 and 7 km and a smooth 1.6 m bump at 16 km. Each regulariser's weight is chosen by
# the same rule, the **discrepancy principle**: make the fit as good as the noise allows and no better, i.e. choose λ
# so that $\chi^2$ equals the number of data. What remains different between the six answers is purely the prior.
#
# **Precisely.** Quadratic penalties have closed-form MAPs, $(\mathbf G^\top\mathbf C_d^{-1}\mathbf G + \lambda\mathbf B)\hat{\mathbf m} = \mathbf G^\top\mathbf C_d^{-1}\mathbf d$
# with $\mathbf B = \mathbf I$, $\mathbf D_1^\top\mathbf D_1$, $\mathbf D_2^\top\mathbf D_2$ or $\mathbf R^{-1}$. The L1
# solution is computed exactly along the whole λ path (the LARS-lasso algorithm in scikit-learn; optimality conditions
# checked below), total variation with the alternating direction method of multipliers (ADMM), checked by its
# primal and dual residuals. These are standard optimisation algorithms; their inner workings do not matter here, only
# that each result is checked.

# %%
Nf = 30
z_edges = np.linspace(0.0, 30.0, Nf + 1)
zc = 0.5 * (z_edges[1:] + z_edges[:-1])
xs = np.array([1, 2, 3, 4, 6, 9, 13, 18, 25, 35, 50, 70.0])
xs = np.sort(np.r_[-xs, xs])


def patch_disp(x, z1, z2):
    return (np.arctan(z2 / x) - np.arctan(z1 / x)) / np.pi


Gf = 1000.0 * np.array([[patch_disp(x, z_edges[j], z_edges[j + 1]) for j in range(Nf)] for x in xs])
s_true = 1.6 * np.exp(-0.5 * ((zc - 16.0) / 3.5) ** 2) + 1.2 * ((zc >= 3) & (zc < 7))
sig_f = 3.0
d_f = Gf @ s_true + sig_f * np.random.default_rng(42).standard_normal(xs.size)
Nd = xs.size
A = Gf.T @ Gf / sig_f ** 2
b = Gf.T @ d_f / sig_f ** 2
chi2 = lambda m: np.sum(((d_f - Gf @ m) / sig_f) ** 2)
D1 = np.diff(np.eye(Nf), axis=0)
D2 = np.diff(np.eye(Nf), n=2, axis=0)


def matern_corr(z, ell, nu):
    r = np.abs(z[:, None] - z[None, :])
    x = np.sqrt(2 * nu) * r / ell
    with np.errstate(invalid="ignore", divide="ignore"):
        C = 2 ** (1 - nu) / gamma(nu) * x ** nu * kv(nu, x)
    C[r == 0] = 1.0
    return C + 1e-9 * np.eye(z.size)


Rm = matern_corr(zc, 5.0, 1.0)
quad = {"L2 ridge (damping)": np.eye(Nf), "first difference": D1.T @ D1, "second difference (Laplacian)": D2.T @ D2,
        "Matérn (ℓ = 5 km, ν = 1)": np.linalg.inv(Rm)}


def bisect_log(f, target, lo=-8.0, hi=8.0, n=80):
    """lambda with f(lambda) = target, for f increasing in lambda (chi^2 of the MAP grows with lambda)."""
    for _ in range(n):
        mid = 0.5 * (lo + hi)
        lo, hi = (mid, hi) if f(10 ** mid) < target else (lo, mid)
    return 10 ** (0.5 * (lo + hi))


sols, lam_of = {}, {}
for name, Bm in quad.items():
    solve = lambda lam, Bm=Bm: np.linalg.solve(A + lam * Bm, b)
    lam = bisect_log(lambda l: chi2(solve(l)), Nd)
    lam_of[name], sols[name] = lam, solve(lam)

# L1: exact lasso path (objective 1/2 chi^2 + lambda ||m||_1 = sklearn's with X = G / sigma, alpha = lambda / N_d)
alphas, _, coefs = lars_path(Gf / sig_f, d_f / sig_f, method="lasso", alpha_min=1e-10, eps=1e-15, max_iter=5000)
lam_path = alphas * Nd
chis = np.array([chi2(coefs[:, k]) for k in range(coefs.shape[1])])
kk = np.where((chis[:-1] >= Nd) & (chis[1:] < Nd))[0][0]
m_at = lambda lam: coefs[:, kk] + (lam_path[kk] - lam) / (lam_path[kk] - lam_path[kk + 1]) * (coefs[:, kk + 1] - coefs[:, kk])
lam_l1 = bisect_log(lambda l: chi2(m_at(l)), Nd, lo=np.log10(lam_path[kk + 1]), hi=np.log10(lam_path[kk]))
m_l1 = m_at(lam_l1)
g_l1 = A @ m_l1 - b
nz = m_l1 != 0
print(f"L1 optimality check: |grad + lambda sign(m)| on nonzero patches <= {np.abs(g_l1[nz] + lam_l1 * np.sign(m_l1[nz])).max():.1e};  "
      f"max(|grad| - lambda) on zero patches = {(np.abs(g_l1[~nz]) - lam_l1).max():.2f} (must be <= 0)")
lam_of["L1 (sparse)"], sols["L1 (sparse)"] = lam_l1, m_l1


def tv_admm(lam, rho=10.0, iters=200000, tol=1e-10):
    Mi = np.linalg.inv(A + rho * D1.T @ D1)
    z = np.zeros(Nf - 1); u = np.zeros(Nf - 1)
    for k in range(iters):
        m = Mi @ (b + rho * D1.T @ (z - u))
        Dm = D1 @ m
        z_old, z = z, soft(Dm + u, lam / rho)
        u = u + Dm - z
        if np.linalg.norm(Dm - z) < tol and rho * np.linalg.norm(D1.T @ (z - z_old)) < tol:
            break
    return m, z, k


lam_tv = bisect_log(lambda l: chi2(tv_admm(l)[0]), Nd, lo=-3, hi=4, n=50)
m_tv, z_tv, it_tv = tv_admm(lam_tv)
print(f"TV (ADMM) converged in {it_tv} iterations; primal residual {np.linalg.norm(D1 @ m_tv - z_tv):.1e}")
lam_of["total variation"], sols["total variation"] = lam_tv, m_tv

print(f"\n{'regulariser':32s} {'lambda':>10s} {'chi^2':>7s} {'rms error vs truth (m)':>23s} {'max slip (m)':>13s}  notes")
for name, m in sols.items():
    note = ""
    if name.startswith("L1"):
        note = f"{np.sum(m == 0)} of {Nf} patches exactly zero"
    if name == "total variation":
        note = f"{np.sum(z_tv != 0)} jumps between neighbours"
    print(f"{name:32s} {lam_of[name]:10.4g} {chi2(m):7.2f} {np.sqrt(np.mean((m - s_true) ** 2)):23.3f} {m.max():13.2f}  {note}")

# %%
order = ["L2 ridge (damping)", "first difference", "second difference (Laplacian)", "Matérn (ℓ = 5 km, ν = 1)", "L1 (sparse)", "total variation"]
cols = [S.INK, S.BLUE, S.AQUA, S.VIOLET, S.ORANGE, S.INK2]
fig, axs = plt.subplots(1, 6, figsize=(17, 4.2), sharey=True)
for ax, name, c in zip(axs, order, cols):
    m = sols[name]
    ax.plot(s_true, zc, **TRUE_KW)
    if name.startswith("L1"):
        step_profile(ax, m, z_edges, color=c, lw=1.4)
    elif name == "total variation":
        step_profile(ax, m, z_edges, color=c, lw=1.8)
    else:
        ax.plot(m, zc, color=c, lw=1.8)
    ax.set_title(name, fontsize=9.5)
    ax.set_xlabel("slip (m)")
    S.badge(ax, f"rms error {np.sqrt(np.mean((m - s_true) ** 2)):.2f} m", loc="lower right", fs=7.8)
axs[0].set_ylim(30, 0); axs[0].set_ylabel("depth (km)")
axs[0].legend(fontsize=7.5, loc="lower left")
fig.tight_layout()
bk.savefig(fig, "2-09_fault_six")
plt.show()

# %% [markdown]
# **How to read it.** All six models fit the data equally well ($\chi^2$ = number of data, printed), yet they disagree
# about the slip. Each one paints the truth in its own style:
# * **L2 damping** prefers small values everywhere. The cheapest way, by its measure, to produce the signal of the deep
#   bump is to spread it thinly over many deep patches, so the bump becomes a broad low tail reaching the bottom.
# * **First- and second-difference smoothing** and the **Matérn** prior draw one smooth hump. It sits between the true
#   block and the true bump and is lower than either: at these depths and with this smoothing strength, the two features
#   are not resolved separately (2.07, section 6).
# * **L1** keeps only a few patches (count printed) and concentrates the slip into spikes; the deep bump becomes a
#   single patch with several times the true slip. The data constrain the deep slip's total, and L1 puts that total
#   into as few patches as possible.
# * **Total variation** makes a few flat blocks (number of jumps printed): one plateau covers both the block and most of
#   the bump, and a lower level continues to the bottom. Its edges are sharp, but not where the true edges are.
#
# None of these is "the" answer: the data cannot distinguish them, and the differences are the priors talking. Their
# errors against the true slip are similar except for L1 (badges).
#
# **Common misreadings.**
# * *"The sharpest model is the best resolved."* Sharp edges in a TV or L1 model come from the prior's taste for
#   blocks and spikes, not from the data.
# * *"A good fit validates the slip model."* All six fit equally well.
#
# **In geodesy.** This is why slip models of the same earthquake from different groups disagree even with the same
# data, and why comparing a model with its prior draws (next section) belongs in every paper.

# %% [markdown]
# ## 4. What each prior believes before seeing data
#
# **In plain words.** A prior can be judged by drawing random slip models from it. Draws from the L2 prior are
# independent noise; difference priors make wandering curves; the Matérn prior makes smooth curves of a given wiggle
# length; L1 makes mostly small values with rare larger ones; total variation makes random walks with occasional jumps.
# The difference priors are **improper**: they say nothing about the overall level (first difference) or level and
# slope (second difference), so those components must be fixed to draw a picture.
#
# **Precisely.** The precision $\lambda\mathbf D_1^\top\mathbf D_1$ has the constant vector in its null space and
# $\lambda\mathbf D_2^\top\mathbf D_2$ has constants and linear trends; along those directions the prior density is
# flat, so it cannot be normalised. The posterior is still proper when the data constrain those directions, as they do
# here (the MAPs above are unique). Draws below fix the free components at zero at the surface: the first-difference
# draws start at zero, the second-difference draws start at zero with zero slope. Each prior uses the λ chosen in
# section 3.

# %%
rng4 = np.random.default_rng(8)
ndraw = 4
draws = {}
draws["L2 ridge (damping)"] = rng4.standard_normal((ndraw, Nf)) / np.sqrt(lam_of["L2 ridge (damping)"])
draws["first difference"] = np.cumsum(np.c_[np.zeros(ndraw), rng4.standard_normal((ndraw, Nf - 1)) / np.sqrt(lam_of["first difference"])], axis=1)
inc2 = rng4.standard_normal((ndraw, Nf - 2)) / np.sqrt(lam_of["second difference (Laplacian)"])
draws["second difference (Laplacian)"] = np.cumsum(np.cumsum(np.c_[np.zeros((ndraw, 2)), inc2], axis=1), axis=1)
draws["Matérn (ℓ = 5 km, ν = 1)"] = (np.linalg.cholesky(Rm / lam_of["Matérn (ℓ = 5 km, ν = 1)"]) @ rng4.standard_normal((Nf, ndraw))).T
draws["L1 (sparse)"] = rng4.laplace(0, 1 / lam_l1, (ndraw, Nf))
draws["total variation"] = np.cumsum(np.c_[np.zeros(ndraw), rng4.laplace(0, 1 / lam_tv, (ndraw, Nf - 1))], axis=1)
fig, axs = plt.subplots(1, 6, figsize=(17, 4.0), sharey=True)
shades = [S.INK, S.INK2, S.MUTED, "#b5b4ad"]
for ax, name, c in zip(axs, order, cols):
    for k in range(ndraw):
        if name in ("L1 (sparse)", "total variation"):
            step_profile(ax, draws[name][k], z_edges, color=shades[k], lw=1.0)
        else:
            ax.plot(draws[name][k], zc, color=shades[k], lw=1.1)
    ax.set_title(name, fontsize=9.5)
    ax.set_xlabel("prior draw (m); axes differ")
axs[0].set_ylim(30, 0); axs[0].set_ylabel("depth (km)")
fig.tight_layout()
bk.savefig(fig, "2-09_prior_draws")
plt.show()
print("typical size of a prior draw (rms over patches and draws), m:")
for name in order:
    print(f"  {name:32s} {np.sqrt(np.mean(draws[name] ** 2)):.3f}")
print(f"exact zeros in {ndraw * Nf} values drawn from the L1 (Laplace) prior: {np.sum(draws['L1 (sparse)'] == 0)};  in its MAP: {np.sum(m_l1 == 0)} of {Nf}")
print(f"for comparison, the true slip has rms {np.sqrt(np.mean(s_true ** 2)):.3f} m and a maximum of {s_true.max():.2f} m")
# one-knob counterfactual: the L2 prior with its standard deviation set to the size of the true slip
lam_match = 1 / np.mean(s_true ** 2)
m_match = np.linalg.solve(A + lam_match * np.eye(Nf), b)
print(f"L2 prior with sd = rms of the true slip (lambda = {lam_match:.2f} instead of {lam_of['L2 ridge (damping)']:.1f}): "
      f"chi^2 = {chi2(m_match):.2f} (below the discrepancy target {Nd}), rms error vs truth {np.sqrt(np.mean((m_match - s_true) ** 2)):.3f} m "
      f"(discrepancy choice: {np.sqrt(np.mean((sols['L2 ridge (damping)'] - s_true) ** 2)):.3f} m)")

# %% [markdown]
# **How to read it.** These are the beliefs the six answers of section 3 were built on, at the strengths the
# discrepancy rule chose (each panel has its own axis; the printed sizes compare them). Three things are worth a
# sentence each.
# * The tuned priors are much tighter than the true slip (compare the printed draw sizes with the true slip's size).
#   Asking the fit to be exactly as good as the noise over-regularises: the misfit of a good estimate is expected to be
#   somewhat *below* the noise level, because the estimate is fitted to the same data. The printed counterfactual sets
#   the L2 prior's width to the size of the true slip instead: the fit is better than the discrepancy target and the
#   error against the truth changes by the printed amount. 2.29 compares this rule with the L-curve, ABIC and sampling λ.
# * The difference priors wander slowly because their level and slope are free.
# * Draws from the Laplace (L1) prior contain no exact zeros at all (count printed), while the L1 MAP has many: the
#   sparsity is a property of the MAP, not of the prior's typical draws. Section 6 follows this thread.
#
# **Common misreadings.**
# * *"A smoothing prior says the slip is smooth."* It says neighbouring values are similar; how far the whole profile
#   may drift is left completely free (improper) unless a damping term or the data fix it.

# %% [markdown]
# ## 5. Laplacian smoothing and the Matérn prior are close relatives
#
# **In plain words.** Smoothing with a difference operator and a Matérn covariance look like two different schools:
# one penalises roughness, the other describes correlation. Lindgren, Rue & Lindström (2011) showed they are two
# descriptions of the same kind of random field (the "SPDE link", after the stochastic partial differential equation
# below). A Matérn field is what you get when you smooth white noise with the operator $(\kappa^2 - \nabla^2)$, so a
# Matérn prior has a *sparse* precision matrix (mostly zeros, which makes it cheap to store and solve with) built from
# the discrete Laplacian.
#
# **Precisely.** In $d$ dimensions the stationary solution of $(\kappa^2 - \Delta)^{\alpha/2} x = \mathcal W$ (white
# noise) is a Matérn field with $\nu = \alpha - d/2$ and length scale $\ell = \sqrt{2\nu}/\kappa$. In 1-D with $\alpha = 2$:
# $\nu = 3/2$, correlation $\rho(r) = (1 + \sqrt3\,r/\ell)\,e^{-\sqrt3 r/\ell}$, variance $1/(4\kappa^3)$. Discretised on
# a grid of spacing $h$, the precision is $\mathbf Q = h(\kappa^2\mathbf I - \boldsymbol\Delta_h)^2$, a banded matrix
# (five nonzero diagonals). Below: a 100 km line, 400 grid points, ℓ = 10 km, with reflecting (Neumann) ends.

# %%
n5, Lx, ell5 = 400, 100.0, 10.0
h = Lx / (n5 - 1)
x5 = np.linspace(0, Lx, n5)
kappa = np.sqrt(3.0) / ell5
Lap = (np.diag(-2.0 * np.ones(n5)) + np.diag(np.ones(n5 - 1), 1) + np.diag(np.ones(n5 - 1), -1))
Lap[0, 0] = Lap[-1, -1] = -1.0                               # reflecting ends
Lap /= h ** 2
Kop = kappa ** 2 * np.eye(n5) - Lap
Q5 = h * Kop @ Kop
C5 = np.linalg.inv(Q5)
var_theory = 1 / (4 * kappa ** 3)
ic = n5 // 2
r = np.abs(x5 - x5[ic])
sd5 = np.sqrt(np.diag(C5))
rho_spde = C5[ic] / (sd5[ic] * sd5)                          # correlation: divide by both standard deviations
rho_mat = (1 + np.sqrt(3) * r / ell5) * np.exp(-np.sqrt(3) * r / ell5)
inner = r < 3 * ell5
print(f"interior point (x = {x5[ic]:.1f} km): variance {C5[ic, ic]:.4f} vs Matérn theory 1/(4 kappa^3) = {var_theory:.4f}")
print(f"interior correlation: max |SPDE - Matérn| within 3 l = {np.abs(rho_spde - rho_mat)[inner].max():.4f}")
edge_ratio = np.diag(C5)[0] / var_theory
print(f"variance at the ends / interior theory = {edge_ratio:.2f}; variance is within 5 % of theory for x between "
      f"{x5[np.argmax(np.abs(np.diag(C5) / var_theory - 1) < 0.05)]:.1f} and {x5[n5 - 1 - np.argmax(np.abs(np.diag(C5)[::-1] / var_theory - 1) < 0.05)]:.1f} km")
ie = 8
rho_edge = C5[ie] / (sd5[ie] * sd5)
r_e = np.abs(x5 - x5[ie])
print(f"near the end (x = {x5[ie]:.1f} km): max |SPDE correlation - Matérn| = {np.abs(rho_edge - (1 + np.sqrt(3) * r_e / ell5) * np.exp(-np.sqrt(3) * r_e / ell5))[r_e < 3 * ell5].max():.3f}")
print(f"nonzero entries: precision Q {np.sum(np.abs(Q5) > 1e-12 * np.abs(Q5).max())} of {n5 * n5}; covariance {np.sum(np.abs(C5) > 1e-12 * np.abs(C5).max())} of {n5 * n5}")

# %%
fig, axs = plt.subplots(1, 3, figsize=(15, 3.9), gridspec_kw=dict(width_ratios=[1.2, 1.2, 1]))
ax = axs[0]
ax.plot(x5 - x5[ic], rho_mat, color=S.INK, lw=2.5, label="Matérn ν = 3/2")
ax.plot(x5 - x5[ic], rho_spde, color=S.VIOLET, lw=1.2, ls=(0, (4, 2)), label=f"SPDE, point at {x5[ic]:.0f} km")
ax.plot(x5 - x5[ie], rho_edge, color=S.ORANGE, lw=1.4, label=f"SPDE, point at {x5[ie]:.0f} km")
ax.set_xlim(-40, 40); ax.set_xlabel("distance r (km)"); ax.set_ylabel("correlation")
ax.set_title("Correlation with one point: interior matches,\nnear the end it is distorted", fontsize=10)
ax.legend(fontsize=7.4, loc="upper left")
ax = axs[1]
ax.plot(x5, np.diag(C5) / var_theory, color=S.VIOLET, lw=1.8)
ax.axhline(1, color=S.INK, lw=0.8)
ax.set_xlabel("position along the line (km)"); ax.set_ylabel("variance / Matérn theory")
ax.set_title("Variance: right in the interior, doubled at\nreflecting ends (boundary effect)", fontsize=10)
ax = axs[2]
blk = slice(0, 30)
ax.imshow(np.where(np.abs(Q5[blk, blk]) > 1e-12 * np.abs(Q5).max(), 1.0, 0.0), cmap=S.GREYS, vmin=0, vmax=1.2)
ax.set_title("Precision Q (first 30 × 30 entries):\nonly five diagonals are nonzero", fontsize=10)
ax.set_xticks([]); ax.set_yticks([])
fig.tight_layout()
bk.savefig(fig, "2-09_matern_spde")
plt.show()

# %% [markdown]
# **How to read it.** Left: away from the ends, the correlation implied by the banded Laplacian precision lies on the
# Matérn formula (the largest difference is printed above). Near an end the field is distorted. Middle: the variance
# matches the theory in the interior and rises toward twice the theory at the reflecting ends (ratio printed): a
# boundary effect, and the reason SPDE models are usually solved on a domain extended beyond the region of interest.
# Right: the precision is sparse, while the covariance it encodes is dense (counts printed). Sparse precision is what
# makes Matérn priors affordable on fine 3-D meshes.
#
# **Worked example.** With ℓ = 10 km, $\kappa = \sqrt3/\ell$, so the predicted interior variance $1/(4\kappa^3)$ and
# the computed one are compared in the first printed line above.
#
# **Common misreadings.**
# * *"Laplacian smoothing and a Matérn prior encode different physics."* Up to boundary handling and a damping term,
#   a Laplacian-type smoothing prior is a Matérn-type Gaussian process with a particular ν.
#
# **In practice.** Matérn priors are common for slip and friction fields on fault meshes, with a correlation length
# chosen relative to the size of the fault and the resolution of the data. The same family can be written as a sparse precision
# built from a discrete Laplacian on the mesh, which is what makes Matérn priors affordable on large meshes.

# %% [markdown]
# ## 6. L1: a sparse MAP, but a posterior that is not sparse
#
# **In plain words.** The L1 solution of section 3 puts most of the deep slip into one patch. Is that what the data and
# the Laplace prior believe? Sample the full posterior with the same prior and look. The samples are not sparse at all:
# the posterior spreads the deep slip over many neighbouring patches, and the posterior mean is smooth. The MAP is a
# single corner of a large set of nearly equally good models.
#
# **Precisely.** With prior $p(\mathbf m) \propto \exp(-\lambda\lVert\mathbf m\rVert_1)$ (independent Laplace, scale
# $1/\lambda$) the MAP is exactly the L1 solution. The **Bayesian lasso** (Park & Casella 2008) samples the whole
# posterior instead. Because the L1 penalty charges the same for slip of one sign whether it sits in one patch or is
# spread over ten, and the data cannot tell those arrangements apart at depth, the posterior gives weight to all of
# them; no posterior draw has an exact zero, since a single point has zero probability.

# %%
Gj, dj = jnp.asarray(Gf), jnp.asarray(d_f)


def model_lasso():
    m = numpyro.sample("m", dist.Laplace(0.0, 1.0 / lam_l1).expand([Nf]))
    numpyro.sample("d", dist.Normal(Gj @ m, sig_f), obs=dj)


t0 = time.time()
mc = MCMC(NUTS(model_lasso, target_accept_prob=0.9), num_warmup=2000, num_samples=2000, num_chains=4,
          chain_method="parallel", progress_bar=False)
mc.run(jax.random.PRNGKey(0), extra_fields=("diverging",))
mch = np.asarray(mc.get_samples(group_by_chain=True)["m"])
post = mch.reshape(-1, Nf)
print(f"NUTS: {time.time() - t0:.1f} s, max R-hat {max(D.rhat(mch[:, :, j]) for j in range(Nf)):.3f}, "
      f"min ESS {min(D.ess(mch[:, :, j]) for j in range(Nf)):.0f}, divergences {int(np.sum(mc.get_extra_fields()['diverging']))}")
lo6, hi6 = np.percentile(post, [2.5, 97.5], axis=0)
jmax = int(np.argmax(m_l1))
jzero = int(np.argmin(np.abs(zc - 20.5)))
print(f"share of posterior draws with |m| < 1e-6: {np.mean(np.abs(post) < 1e-6):.3f}")
print(f"patch at {zc[jmax]:.1f} km: MAP {m_l1[jmax]:.2f} m; posterior mean {post[:, jmax].mean():.2f} m, 95 % interval {lo6[jmax]:.2f} to {hi6[jmax]:.2f} m")
print(f"patch at {zc[jzero]:.1f} km: MAP {m_l1[jzero]:.2f} m; posterior mean {post[:, jzero].mean():.2f} m, 95 % interval {lo6[jzero]:.2f} to {hi6[jzero]:.2f} m")
lp = lambda m: -0.5 * chi2(m) - lam_l1 * np.abs(m).sum()
print(f"log posterior (up to a constant): at the MAP {lp(m_l1):.2f}, at the posterior mean {lp(post.mean(0)):.2f}, "
      f"median over the draws {np.median([lp(s_) for s_ in post[::20]]):.2f}")
print(f"chi^2 of the posterior-mean prediction: {chi2(post.mean(0)):.2f} for {Nd} data")
print(f"true slip inside the posterior 95 % interval at {np.mean((s_true >= lo6) & (s_true <= hi6)) * 100:.0f} % of the patches")

# %%
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.1), gridspec_kw=dict(width_ratios=[1.2, 1]))
ax = axs[0]
ax.plot(s_true, zc, **TRUE_KW)
ax.fill_betweenx(zc, lo6, hi6, color=S.BLUE, alpha=0.15, lw=0, label="posterior 95 % interval")
for k, s_ in enumerate(post[np.random.default_rng(1).choice(len(post), 6, replace=False)]):
    ax.plot(s_, zc, color=S.BLUE, lw=0.6, alpha=0.6, label="posterior draws" if k == 0 else None)
ax.plot(post.mean(0), zc, color=S.BLUE, lw=2, label="posterior mean")
step_profile(ax, m_l1, z_edges, color=S.ORANGE, lw=1.4, label="MAP (= L1 solution)")
ax.set_ylim(30, 0); ax.set_xlim(-0.5, 13.5)
ax.set_xlabel("slip (m)"); ax.set_ylabel("depth (km)")
ax.set_title("Same Laplace prior: MAP vs posterior")
ax.legend(fontsize=7.6, loc="lower right")
ax = axs[1]
bins = np.linspace(-0.3, 13.0, 90)
ax.hist(post[:, jmax], bins=bins, color=S.BLUE, alpha=0.6, density=True, label=f"patch at {zc[jmax]:.1f} km: posterior")
ax.axvline(m_l1[jmax], color=S.ORANGE, lw=1.8, label=f"its MAP value ({m_l1[jmax]:.1f} m)")
ax.hist(post[:, jzero], bins=bins, color=S.AQUA, alpha=0.5, density=True, label=f"patch at {zc[jzero]:.1f} km (MAP exactly 0)")
ax.set_xlabel("slip (m)"); ax.set_ylabel("density")
ax.set_title("Two patches: where the MAP falls in its own posterior")
ax.legend(fontsize=7.6, loc="upper right")
fig.tight_layout()
bk.savefig(fig, "2-09_l1_map_vs_posterior")
plt.show()

# %% [markdown]
# **How to read it.** Left: the orange MAP is a few spikes; the blue posterior mean and draws spread the deep slip over
# many patches, and the posterior mean is a broad hump. The 95 % band contains the true slip (the share of patches is
# printed). Right: for the patch where the MAP piles up its slip, the MAP value lies outside that patch's own posterior
# 95 % interval (printed); for a patch where the MAP is exactly zero, the posterior still spreads over a range of
# values, mostly positive. The MAP has the highest posterior density of all models
# (printed log posteriors), yet typical draws sit far below that density: in many dimensions the highest point is
# not where the probability is (notebook 2.01 calls it the difference between the peak and the crowd).
#
# **Common misreadings.**
# * *"An L1 inversion found sparse slip, so the slip is sparse."* The MAP is sparse by construction; the posterior under
#   the same prior is not.
# * *"The MAP is a typical model from the posterior."* Here it is far from typical. Report posterior summaries, or at
#   least check the MAP against posterior draws.
#
# **In practice.** Sampling the slip (and the smoothing scale) instead of reporting only a MAP is what protects
# against this. Choosing λ itself (here by the discrepancy principle) is the subject of 2.29, which compares the
# discrepancy rule, the L-curve, ABIC and sampling λ hierarchically (Fukuda & Johnson 2008), and 2.30 goes further.
#
# Next: **2.10 · Loss functions and optimisation**.
