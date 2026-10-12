# %% [markdown]
# # 3.03 · GP surrogates and Calibrate-Emulate-Sample
#
# *Topic 3 is under development: content and numbering may still change.*
#
# **Before this:** 2.12 (Gaussian processes), 2.13 (model error $\mathbf C_p$), 2.24 (EnKI), 3.01 (surrogate workflow,
# PCA) · **You will be able to:** build a GP emulator of a forward model and check its error bars; carry the emulator's
# variance into the likelihood; choose where to spend a run budget (prior-wide, CES, active learning); decide whether
# to emulate the data or the log-likelihood · **Optional sections:** 7 (kernels and cost), 8 (model discrepancy).
#
# When one forward run takes hours, a sampler that needs a hundred thousand runs is out of reach. A **Gaussian-process
# emulator** replaces the forward model by a cheap statistical fit to a few hundred runs, and it reports its own error
# bar at every input. **Calibrate-Emulate-Sample** (CES; Cleary et al. 2021, cited for the idea only) is a three-step
# recipe: a cheap ensemble method finds where the posterior is (calibrate), a GP is fitted to the runs made there
# (emulate), and an MCMC sampler runs on the emulated posterior (sample). This notebook asks three practical questions
# on the afterslip forward model of Topic 3.
# Where should the expensive runs go? What should the GP emulate: the predicted data, or the log-likelihood? And how does
# the emulator's own error enter the posterior?
#
# **Words this notebook defines:** emulator, design, run budget, prior-wide design, Latin hypercube, CES, active
# learning (sequential design), output compression (PCA), whitening, emulator variance $s^2$, calibration of the
# emulator's error bars, shared vs independent kernels, log-likelihood emulation, trust region, amortisation over data
# sets, ARD length scales, conditioning cost $O(N^3)$, model discrepancy.

# %% [markdown]
# **Problems in this notebook** (size and forward model; notebook 2.25 explains the degree of nonlinearity)
#
# | problem | unknowns | data | forward model | linear? |
# |---|---|---|---|---|
# | single-patch afterslip, Δτ known (sections 1-6) | 2: $\log a\sigma$, $\log v_0$ (uniform prior box) | 40 GPS epochs, day 1 to year 1, σ = 2 mm | ODE solve (diffrax), milliseconds per run | nonlinear |
# | single-patch afterslip, all three parameters (section 1 note, section 7) | 3: $\log a\sigma$, $\log v_0$, Δτ | 4,000 cached runs (no data needed) | same | nonlinear |
#
# The model is the one of `bayeskit/afterslip.py`: slip rate $v = 2v_0\sinh(\tau/a\sigma)$, stress
# $\dot\tau = -k v$, starting from the coseismic stress change $\tau(0) = \Delta\tau$; a GPS site sees $u(t) = g\,s(t)$.
# One run costs milliseconds here, so we can afford the exact answer and grade every emulator against it. We pretend
# that each run is expensive and **count runs**: the budget of a design is its number of forward runs.

# %%
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))
import numpy as np
import matplotlib.pyplot as plt
import bayeskit as bk
from bayeskit import problems as P, samplers as SM, diagnostics as D, plotting as BP, style as S

import time
T0 = time.time()
import jax, jax.numpy as jnp
from jax.scipy.stats import norm as jnorm
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.stats import chi2 as CHI2, binom
from matplotlib.colors import LinearSegmentedColormap, LogNorm, SymLogNorm
import matplotlib.patches as mpatches
from matplotlib.tri import Triangulation
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from bayeskit import afterslip as A

C_PW, C_CES, C_ACT = S.ORANGE, S.VIOLET, S.AQUA        # prior-wide, CES (EnKI-based), active learning
C_REF = S.COLORS["nuts"]                                # reference draws: NUTS through the true model
SEQ = LinearSegmentedColormap.from_list("bx_seq", ["#f4f3ef", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIV = LinearSegmentedColormap.from_list("bx_div", ["#0d366b", "#3987e5", "#f0efec", "#e34948", "#7a1f1f"])

# %% [markdown]
# ## 1. The black box, the data and the reference posterior
#
# **In plain words.** A fault patch is loaded by an earthquake and creeps afterwards; how fast it creeps and for how long
# depends on its friction ($a\sigma$) and on a reference creep rate ($v_0$). A GPS station records 40 positions in the
# first year. We want the posterior of $(a\sigma, v_0)$ from those 40 numbers.
#
# **Why only two unknowns.** The model has three parameters, but with these data only two combinations of them are
# pinned down: for $\Delta\tau \gg a\sigma$ the curve depends on $a\sigma$ and on $v_0 e^{\Delta\tau/a\sigma}$ only
# (the logarithmic afterslip law in the module docstring). The cell below measures this with the Laplace approximation
# of the three-parameter posterior (2.08): the posterior standard deviations along its principal axes.

# %%
SIG = 0.002                                            # GPS noise (m): 2 mm
TRUE3 = np.array([np.log(0.6), np.log(0.03), 3.0])     # log a_sigma (MPa), log v0 (m/yr), dtau (MPa) used for the data
J3 = np.asarray(jax.jacrev(lambda th: A.single_patch_series(th, A.T_OBS))(jnp.asarray(TRUE3)))
ev3, E3 = np.linalg.eigh(np.linalg.inv(J3.T @ J3 / SIG ** 2))
print("three parameters (log a_sigma, log v0, dtau), Laplace approximation at the values used for the data:")
print("  posterior sd along the principal axes:", np.array2string(np.sqrt(ev3), precision=4))
print("  the loosest axis points along", np.round(E3[:, -1], 3), " ; prior box widths:",
      np.round(A.SP_BOUNDS[:, 1] - A.SP_BOUNDS[:, 0], 2))

# %% [markdown]
# Two axes are pinned tightly; along the third the Laplace sd is comparable to the size of the prior box itself (the
# posterior is a thin filament through the box, mostly a trade-off between $\log v_0$ and $\Delta\tau$). We therefore hold $\Delta\tau$ = 3 MPa fixed, as if it came
# from the coseismic slip model, and keep two unknowns. Every posterior below can then be drawn in a plane and computed
# exactly. Section 7 returns to all three parameters.
#
# We work in **unit-box coordinates** $\mathbf u \in [0,1]^2$, $\log a\sigma = \log 0.2 + u_1 \log 10$ and
# $\log v_0 = \log 10^{-3} + u_2 \log 1000$; the prior is uniform on the box. The solver runs at relative tolerance
# $10^{-6}$; the cell checks that this changes the predictions by far less than the noise.

# %%
DTAU = 3.0
LO, HI = A.SP_BOUNDS[:2, 0], A.SP_BOUNDS[:2, 1]
TRUE = np.array([0.6, 0.03])                           # a_sigma (MPa), v0 (m/yr) used to make the data
u_true = (np.log(TRUE) - LO) / (HI - LO)


def phys(U):
    """unit box -> (a_sigma in MPa, v0 in m/yr)"""
    return np.exp(LO + (HI - LO) * np.asarray(U))


def fwd_u(u, rtol=1e-6):
    th = jnp.asarray(LO) + jnp.asarray(HI - LO) * u
    return A.single_patch_series(jnp.stack([th[0], th[1], jnp.asarray(DTAU)]), A.T_OBS, rtol=rtol, atol=rtol * 1e-2)


_fwd_batch = jax.jit(jax.vmap(fwd_u))
RUNS = {"n": 0}


def run(U):
    """The expensive black box: one forward run per row of U (unit box). Every call is counted."""
    U = np.atleast_2d(U)
    RUNS["n"] += len(U)
    return np.asarray(_fwd_batch(jnp.asarray(U)))


def lhs(rng, n, d=2):
    """Latin hypercube in the unit box: each axis cut into n strata, one point per stratum (3.01)."""
    return (np.argsort(rng.uniform(size=(n, d)), axis=0) + rng.uniform(size=(n, d))) / n


def plain_ticks(ax, x=None, y=None, fmt="{:g}"):
    """Readable tick labels on log axes: the given major ticks, no minor labels."""
    from matplotlib.ticker import NullFormatter
    if x is not None:
        ax.set_xticks(x); ax.set_xticklabels([fmt.format(v) for v in x]); ax.xaxis.set_minor_formatter(NullFormatter())
    if y is not None:
        ax.set_yticks(y); ax.set_yticklabels([fmt.format(v) for v in y]); ax.yaxis.set_minor_formatter(NullFormatter())


def box_axes(ax):
    """The whole prior box in physical units, log axes."""
    ax.set_xscale("log"); ax.set_yscale("log"); ax.set_xlim(0.2, 2); ax.set_ylim(1e-3, 1)
    plain_ticks(ax, x=[0.2, 0.5, 1, 2], y=[0.001, 0.01, 0.1, 1])
    ax.set_xlabel("aσ (MPa)"); ax.set_ylabel("v₀ (m/yr)")


Uchk = np.random.default_rng(9).uniform(size=(200, 2))
d_tol = np.abs(run(Uchk) - np.asarray(jax.vmap(lambda u: fwd_u(u, 1e-8))(jnp.asarray(Uchk))))
RUNS["n"] = 0                                         # the check does not count towards any budget
print(f"solver tolerance 1e-6 vs 1e-8 at 200 random inputs: largest difference {1000 * d_tol.max():.2e} mm "
      f"(noise sd {1000 * SIG:.0f} mm)")

clean = run(u_true)[0]
y = clean + SIG * np.random.default_rng(1).standard_normal(clean.size)
RUNS["n"] = 0
print(f"data: 40 epochs, displacement {1000 * y[0]:.1f} mm at day 1 to {1000 * y[-1]:.1f} mm at year 1")

# %% [markdown]
# **The reference posterior** is computed twice. First with NumPyro NUTS (2.18) through the diffrax solver (diffrax is a
# library of ODE solvers written in JAX; gradients by automatic differentiation through the ODE solve, 2.22), two chains
# started at random points of the box. Second, because there are only two unknowns, by brute force (quadrature: the
# posterior density summed over a fine grid of points): the true model on a 49 × 49 grid laid along the principal axes
# of the Laplace approximation, ±6 sd, which is exact up to the grid spacing. The two must agree before we use them as the
# yardstick.

# %%
yj = jnp.asarray(y)


def model_true():
    u = numpyro.sample("u", dist.Uniform(0.0, 1.0).expand([2]).to_event(1))
    numpyro.factor("loglik", -0.5 * jnp.sum(((yj - fwd_u(u)) / SIG) ** 2))


t0 = time.time()
mcmc = MCMC(NUTS(model_true, dense_mass=True), num_warmup=300, num_samples=500, num_chains=2,
            chain_method="sequential", progress_bar=False)
mcmc.run(jax.random.PRNGKey(0), extra_fields=("num_steps", "diverging"))
ref_c = np.asarray(mcmc.get_samples(group_by_chain=True)["u"])
ref = ref_c.reshape(-1, 2)
ex = mcmc.get_extra_fields()
t_nuts = time.time() - t0
n_grad = int(np.sum(ex["num_steps"]))
mref, sdref = ref.mean(0), ref.std(0)
print(f"NUTS through diffrax: 2 chains x 500 draws in {t_nuts:.0f} s; {n_grad} gradient evaluations in the sampling "
      f"phase (each = one forward run + its adjoint); divergences {int(np.sum(ex['diverging']))}; "
      f"split R-hat {D.rhat(ref_c[:, :, 0]):.3f}, {D.rhat(ref_c[:, :, 1]):.3f}; ESS {D.ess(ref_c[:, :, 0]):.0f}, "
      f"{D.ess(ref_c[:, :, 1]):.0f}")

# exact quadrature on a rotated grid
Jr = np.asarray(jax.jacrev(fwd_u)(jnp.asarray(mref)))
evr, Er = np.linalg.eigh(np.linalg.inv(Jr.T @ Jr / SIG ** 2))
XI = np.linspace(-6, 6, 49)
XIg = np.stack(np.meshgrid(XI, XI, indexing="ij"), -1).reshape(-1, 2)
Ugrid = mref + (XIg * np.sqrt(evr)) @ Er.T
lq = -0.5 * np.sum(((y - run(Ugrid)) / SIG) ** 2, 1)
RUNS["n"] = 0
wq = np.exp(lq - lq.max()); wq /= wq.sum()
mq = wq @ Ugrid; sdq = np.sqrt(wq @ (Ugrid - mq) ** 2)
corr_q = (wq @ ((Ugrid[:, 0] - mq[0]) * (Ugrid[:, 1] - mq[1]))) / sdq.prod()
edge = wq.reshape(49, 49)[[0, -1], :].sum() + wq.reshape(49, 49)[:, [0, -1]].sum()
print(f"exact quadrature (2,401 true runs): mean shift NUTS - grid = {np.round((mref - mq) / sdq, 3)} sd, "
      f"sd ratio NUTS / grid = {np.round(sdref / sdq, 3)}; posterior correlation {corr_q:.3f}; mass on the grid edge "
      f"{edge:.1e}")
print(f"posterior sd: a_sigma {np.std(phys(ref)[:, 0]):.4f} MPa ({100 * np.std(phys(ref)[:, 0]) / TRUE[0]:.2f} %), "
      f"v0 {np.std(phys(ref)[:, 1]):.5f} m/yr ({100 * np.std(phys(ref)[:, 1]) / TRUE[1]:.2f} %)")

# %%
fig, axs = plt.subplots(1, 3, figsize=(14.5, 4.0), gridspec_kw=dict(width_ratios=[1.15, 1, 1]))
ax = axs[0]
td = A.T_OBS * 365.25
ax.plot(td, 1000 * clean, color=S.INK, lw=1.2, label="true curve")
ax.errorbar(td, 1000 * y, yerr=1000 * SIG, fmt="o", ms=3.2, color=S.INK2, mfc=S.SURFACE, lw=0.8, label="data ± 2 mm")
ax.set_xscale("log"); ax.set_xlabel("days after the earthquake"); ax.set_ylabel("surface displacement (mm)")
ax.set_title("40 GPS epochs of afterslip"); ax.legend(loc="upper left")
ax = axs[1]
pr = phys(ref)
ax.scatter(pr[::4, 0], pr[::4, 1], s=4, color=C_REF, zorder=3, label="posterior (NUTS draws)")
zx, zy = np.percentile(pr[:, 0], [0, 100]), np.percentile(pr[:, 1], [0, 100])
ax.add_patch(plt.Rectangle((zx[0] * 0.97, zy[0] * 0.9), zx[1] * 1.03 - zx[0] * 0.97, zy[1] * 1.1 - zy[0] * 0.9,
                           fill=False, ec=S.INK, lw=0.8))
box_axes(ax); ax.set_title("The whole prior box")
ax.legend(loc="upper right")
ax = axs[2]
Pq = phys(Ugrid)
tri = Triangulation(XIg[:, 0], XIg[:, 1])
levs = sorted(set(P.hdr_levels(wq.reshape(49, 49), (0.5, 0.9, 0.99)).values()))
tri_p = Triangulation(Pq[:, 0], Pq[:, 1], tri.triangles)
ax.tricontourf(tri_p, wq, levels=levs + [wq.max() * 1.01], colors=["#ecebe6", "#d9d8d1", "#c3c2b7"], zorder=0)
ax.tricontour(tri_p, wq, levels=levs, colors=S.MUTED, linewidths=0.7, zorder=1)
ax.scatter(pr[::5, 0], pr[::5, 1], s=2, color=C_REF, alpha=0.6, zorder=3)
ax.plot(*TRUE, "*", color=S.INK, ms=11, mec=S.SURFACE, zorder=5)
ax.set_xlabel("aσ (MPa)"); ax.set_ylabel("v₀ (m/yr)"); ax.set_title("Zoom: NUTS draws on the exact posterior")
ax.legend(handles=[plt.Line2D([], [], marker="o", ls="", ms=4, color=C_REF, label="NUTS draws"),
                   plt.Line2D([], [], marker="*", ls="", ms=10, color=S.INK, label="values used for the data"),
                   mpatches.Patch(color="#d9d8d1", label="exact posterior, 50/90/99 %")],
          loc="upper left", fontsize=7.5)
fig.tight_layout()
bk.savefig(fig, "3-03_problem")
plt.show()

# %% [markdown]
# **How to read it.** Left: the data are 40 points on a logarithmic afterslip curve, with error bars too small to see.
# Middle: in the whole prior box the posterior is a speck (the small rectangle). Right: zoomed in, the NUTS draws fill
# the exact 50/90/99 % regions, a thin tilted ellipse. The agreement printed above (mean shifts and sd ratios close to
# 0 and 1) makes NUTS the reference for every grade below.
#
# **The image to keep.** The posterior occupies a tiny part of the prior box. Any emulator must be accurate *there*, to
# a fraction of the 2 mm noise; elsewhere it only has to be good enough to say "the posterior is not here".

# %% [markdown]
# ## 2. Anatomy of a GP emulator of the forward model
#
# **In plain words.** Run the model at $N$ inputs (the **design**), then fit a smooth function through the results that
# also says how unsure it is between the runs. The output is a curve of 40 numbers, so two tricks keep the fit small.
# First, **whiten**: divide by the noise sd, so one unit means "one σ" (whitening with a full covariance, 2.06). Second,
# **compress**: the 40 numbers of every run are nearly determined by a few shape coefficients (principal components,
# PCA, 3.01; SVD, 2.06). The GP emulates those few coefficients.
#
# **Precisely.** With training outputs $\mathbf u(\boldsymbol\theta_i)$, $i = 1..N$, whitened
# $\mathbf w_i = \mathbf u(\boldsymbol\theta_i)/\sigma$, mean $\bar{\mathbf w}$ and the first $K$ right singular vectors
# $\mathbf V_K$ of the centred matrix, every run is summarised by $\mathbf c_i = \mathbf V_K^\top(\mathbf w_i - \bar{\mathbf w})$.
# Each coefficient $c_k(\boldsymbol\theta)$ gets its own GP (2.12) with a Matérn-5/2 kernel and one length scale per input
# (ARD, *automatic relevance determination*: an input whose length scale grows large is judged irrelevant),
# hyperparameters by maximising the marginal likelihood (the evidence, 2.29). At a new input the
# GP returns a mean $\mu_k(\boldsymbol\theta)$ and an sd $s_k(\boldsymbol\theta)$; the emulated curve is
# $\sigma(\bar{\mathbf w} + \mathbf V_K\boldsymbol\mu)$. We choose $K$ so that dropping the remaining components costs
# less than 0.05 σ (rms) on the training runs.
#
# The GP below is written out in NumPy, about forty lines: the kernel, its derivative with respect to the length scales,
# the negative log marginal likelihood with its gradient for L-BFGS, and the predictive mean and variance.

# %%
def kern(X1, X2, ls, amp2, kind="m52", grad=False):
    """Stationary ARD kernels on inputs scaled by the length scales ls. Returns K (and dK/dlog ls_j if grad)."""
    if grad:
        D2 = ((X1[:, None, :] - X2[None, :, :]) / ls) ** 2          # (n1, n2, d) squared scaled distances per axis
        q = D2.sum(-1)
    else:                                                            # same q without the (n1, n2, d) array
        A1, A2 = X1 / ls, X2 / ls
        q = np.clip((A1 ** 2).sum(1)[:, None] + (A2 ** 2).sum(1)[None, :] - 2 * A1 @ A2.T, 0, None)
    if kind == "m52":
        r = np.sqrt(5 * q); e = np.exp(-r)
        K = amp2 * (1 + r + r ** 2 / 3) * e
        dK = (amp2 * e * (1 + r) * 5 / 3)[..., None] * D2 if grad else None
    elif kind == "m32":
        a = np.sqrt(3 * q); e = np.exp(-a)
        K = amp2 * (1 + a) * e
        dK = (amp2 * e * 3)[..., None] * D2 if grad else None
    else:                                                            # squared exponential
        K = amp2 * np.exp(-0.5 * q)
        dK = K[..., None] * D2 if grad else None
    return (K, dK) if grad else K


class GP:
    """GP regression with one kernel shared by the columns of Y (each column standardised). Hyperparameters:
    log length scales (d), log amplitude^2, log nugget; fitted by maximising the marginal likelihood (L-BFGS)."""

    def __init__(self, X, Y, kind="m52", starts=(0.2, 1.0)):
        X, Y = np.asarray(X, float), np.atleast_2d(np.asarray(Y, float).T).T
        self.X, self.kind = X, kind
        self.ym, self.ys = Y.mean(0), Y.std(0) + 1e-300
        Z = (Y - self.ym) / self.ys
        n, d = X.shape
        best = None
        for l0 in starts:
            p0 = np.r_[np.log(np.full(d, l0)), 0.0, np.log(1e-6)]
            r = minimize(self._nlml, p0, args=(X, Z), jac=True, method="L-BFGS-B",
                         bounds=[(-7, 3)] * d + [(-6, 8), (np.log(1e-9), np.log(1e-1))])
            if np.isfinite(r.fun) and (best is None or r.fun < best.fun):
                best = r
        self.p = best.x
        self.ls, self.amp2, self.nug = np.exp(best.x[:d]), np.exp(best.x[d]), np.exp(best.x[d + 1])
        self.cf = cho_factor(kern(X, X, self.ls, self.amp2, kind) + (self.nug + 1e-10) * np.eye(n), lower=True)
        self.alpha = cho_solve(self.cf, Z)

    def _nlml(self, p, X, Z):
        n, d = X.shape
        ls, amp2, nug = np.exp(p[:d]), np.exp(p[d]), np.exp(p[d + 1])
        K, dK = kern(X, X, ls, amp2, self.kind, grad=True)
        Kn = K + (nug + 1e-10) * np.eye(n)
        try:
            cf = cho_factor(Kn, lower=True)
        except np.linalg.LinAlgError:
            return np.inf, np.zeros_like(p)
        al = cho_solve(cf, Z)
        m = Z.shape[1]
        f = 0.5 * np.sum(Z * al) + m * np.sum(np.log(np.diag(cf[0])))
        W = m * cho_solve(cf, np.eye(n)) - al @ al.T                  # dNLML/dK = W / 2
        g = np.r_[[0.5 * np.sum(W * dK[..., j]) for j in range(d)], 0.5 * np.sum(W * K), 0.5 * nug * np.trace(W)]
        return f, g

    def predict(self, Xs, var=True):
        Ks = kern(np.atleast_2d(Xs), self.X, self.ls, self.amp2, self.kind)
        mu = Ks @ self.alpha * self.ys + self.ym
        if not var:
            return mu
        v = cho_solve(self.cf, Ks.T)
        s2 = np.clip(self.amp2 - np.sum(Ks.T * v, 0), 1e-16, None)    # nugget left out: it is jitter, not error
        return mu, np.sqrt(s2)[:, None] * self.ys

    def var_after(self, Xs, Xadd):
        """Predictive variance (standardised) at Xs if the GP were also conditioned on Xadd: needs no outputs."""
        X = np.vstack([self.X, Xadd])
        cf = cho_factor(kern(X, X, self.ls, self.amp2, self.kind) + (self.nug + 1e-10) * np.eye(len(X)), lower=True)
        Ks = kern(np.atleast_2d(Xs), X, self.ls, self.amp2, self.kind)
        return np.clip(self.amp2 - np.sum(Ks.T * cho_solve(cf, Ks.T), 0), 1e-16, None)


class Emulator:
    """Whitened PCA of the training outputs + a GP for each kept coefficient (shared=False) or one GP with a single
    kernel shared by all coefficients (shared=True, as in a multi-output sklearn GaussianProcessRegressor)."""

    def __init__(self, U, Yout, shared=False, kind="m52", tol=0.05):
        W = np.asarray(Yout) / SIG
        self.mean = W.mean(0)
        _, sv, Vt = np.linalg.svd(W - self.mean, full_matrices=False)
        res = [np.sqrt(np.mean(((W - self.mean) - (W - self.mean) @ Vt[:k].T @ Vt[:k]) ** 2)) for k in range(1, 16)]
        self.K = 1 + int(np.argmax(np.array(res) < tol)) if np.any(np.array(res) < tol) else 15
        self.V = Vt[:self.K].T
        C = (W - self.mean) @ self.V
        self.U, self.shared = np.asarray(U), shared
        self.gps = [GP(U, C, kind)] if shared else [GP(U, C[:, k], kind) for k in range(self.K)]

    def predict(self, Us):
        """GP means mu (n, K) and sds s (n, K) of the whitened coefficients."""
        out = [g.predict(Us) for g in self.gps]
        return np.hstack([o[0] for o in out]), np.hstack([o[1] for o in out])

    def coef(self, yobs):
        return (np.asarray(yobs) / SIG - self.mean) @ self.V

    def curve_mm(self, Us):
        mu, _ = self.predict(Us)
        return 1000 * SIG * (mu @ self.V.T + self.mean)

    def loglik(self, Us, yobs, use_var=True):
        """CES likelihood in whitened coefficients: -1/2 sum (c - mu)^2/(1+s^2) - 1/2 sum log(1+s^2)."""
        mu, s = self.predict(Us)
        v = 1 + s ** 2 if use_var else np.ones_like(s)
        return -0.5 * np.sum((self.coef(yobs) - mu) ** 2 / v + np.log(v), 1)

    def total_var(self, Us):
        return np.sum(self.predict(Us)[1] ** 2, 1)

# %% [markdown]
# A prior-wide design of 60 runs (Latin hypercube over the box, 3.01) and the anatomy of the emulator fitted to it.
# The error map uses 900 extra true runs on a 30 × 30 grid; those are for grading only and are not part of any budget.

# %%
rng = np.random.default_rng(0)
U60 = lhs(rng, 60)
Y60 = run(U60)
em60 = Emulator(U60, Y60)
print(f"design: {RUNS['n']} runs; kept K = {em60.K} principal components")
print("ARD length scales (unit-box units) of the first three coefficients:",
      [np.round(g.ls, 3).tolist() for g in em60.gps[:3]])

# truncation error on held-out runs (the grading grid) as a function of K
g30 = (np.arange(30) + 0.5) / 30
Ueval = np.stack(np.meshgrid(g30, g30, indexing="ij"), -1).reshape(-1, 2)
Yeval = run(Ueval)
RUNS["n"] = 0
W60 = Y60 / SIG; Wm = W60.mean(0)
_, sv60, Vt60 = np.linalg.svd(W60 - Wm, full_matrices=False)
trunc = [np.sqrt(np.mean(((Yeval / SIG - Wm) - (Yeval / SIG - Wm) @ Vt60[:k].T @ Vt60[:k]) ** 2)) for k in range(1, 13)]
trunc_tr = [np.sqrt(np.mean(((W60 - Wm) - (W60 - Wm) @ Vt60[:k].T @ Vt60[:k]) ** 2)) for k in range(1, 13)]
print("PCA truncation error (rms over epochs, in sigma units), K = 1..12")
print("   training runs:", np.array2string(np.array(trunc_tr), precision=3))
print("   900 held-out :", np.array2string(np.array(trunc), precision=3))
err_map = np.sqrt(np.mean((em60.curve_mm(Ueval) - 1000 * Yeval) ** 2, 1)).reshape(30, 30)
print(f"emulator error over the box (rms over epochs): median {np.median(err_map):.2f} mm, largest "
      f"{err_map.max():.1f} mm; at the posterior mean {np.sqrt(np.mean((em60.curve_mm(mref[None]) - 1000 * run(mref)) ** 2)):.3f} mm")
RUNS["n"] = 0

# %% [markdown]
# The last printed number is the one that matters for inversion: the emulator's error at the posterior, in mm, against
# a noise of 2 mm per epoch. (The 900-node map shows how the error varies over the box.)

# %%
# a slice through the posterior: a_sigma varies, v0 at its posterior mean
us = np.linspace(0, 1, 80)
Usl = np.column_stack([us, np.full_like(us, mref[1])])
Ysl = run(Usl); RUNS["n"] = 0
c_true_sl = (Ysl / SIG - em60.mean) @ em60.V
mu_sl, s_sl = em60.predict(Usl)

fig, axs = plt.subplots(1, 4, figsize=(17, 3.9), gridspec_kw=dict(width_ratios=[0.9, 1, 1.05, 1.15]))
ax = axs[0]
ax.semilogy(np.arange(1, 13), trunc_tr, "o-", color=S.INK2, ms=4, label="the 60 training runs")
ax.semilogy(np.arange(1, 13), trunc, "s--", color=S.MUTED, ms=4, mfc=S.SURFACE, label="900 held-out runs")
ax.legend(fontsize=7.5, loc="lower left")
ax.axhline(0.05, color=S.MUTED, lw=0.8, ls="--")
ax.axvline(em60.K, color=C_PW, lw=1.0)
ax.text(em60.K + 0.2, trunc[0], f"K = {em60.K} kept", color=C_PW, fontsize=8, va="top")
ax.text(12, 0.055, "0.05 σ", color=S.MUTED, fontsize=7.5, ha="right", va="bottom")
ax.set_xlabel("number of components K"); ax.set_ylabel("truncation error (σ units)")
ax.set_title("PCA: few shapes suffice")
ax = axs[1]
for k, c in zip(range(4), [S.INK, S.BLUE, S.ORANGE, S.AQUA]):
    ax.plot(td, Vt60[k], color=c, lw=1.5, label=f"component {k + 1}")
ax.set_xscale("log"); ax.set_xlabel("days after the earthquake"); ax.set_ylabel("weight per epoch")
lo_, hi_ = ax.get_ylim(); ax.set_ylim(lo_ - 0.3 * (hi_ - lo_), hi_)   # room below the curves for the legend
ax.set_title("The leading output shapes"); ax.legend(fontsize=7.5, ncol=2, loc="lower left")
ax = axs[2]
asl = phys(Usl)[:, 0]
for k, c in ((0, C_PW), (2, S.BLUE)):
    ax.plot(asl, mu_sl[:, k] - c_true_sl[:, k], color=c, lw=1.4, label=f"coefficient {k + 1}: GP mean − truth")
    ax.fill_between(asl, -2 * s_sl[:, k], 2 * s_sl[:, k], color=c, alpha=0.15, lw=0, label=f"coefficient {k + 1}: ± 2 s")
near_sl = np.abs(U60[:, 1] - mref[1]) < 0.08
ax.plot(phys(U60[near_sl])[:, 0], np.full(near_sl.sum(), 0.0), "|", color=S.INK, ms=10, mew=1.2,
        label="design runs near the slice")
ax.axvline(TRUE[0], color=C_REF, lw=0.9, ls=":")
ax.set_xscale("log"); plain_ticks(ax, x=[0.2, 0.5, 1, 2])
ax.set_xlabel("aσ (MPa), v₀ at its posterior mean"); ax.set_ylabel("σ units")
ax.set_title("GP error and its ± 2 s band, along a slice"); ax.legend(fontsize=7, loc="upper right")
ax = axs[3]
PE = phys(Ueval)
im = ax.pcolormesh(PE[:, 0].reshape(30, 30), PE[:, 1].reshape(30, 30), err_map, cmap=SEQ,
                   norm=LogNorm(vmin=max(err_map.min(), 1e-3), vmax=err_map.max()), shading="nearest")
cs = ax.contour(PE[:, 0].reshape(30, 30), PE[:, 1].reshape(30, 30), err_map, levels=[2.0], colors=S.INK, linewidths=0.9)
ax.clabel(cs, fmt="2 mm", fontsize=7)
p60 = phys(U60)
ax.scatter(p60[:, 0], p60[:, 1], s=9, color=S.SURFACE, ec=S.INK, lw=0.6, zorder=3, label="design runs")
ax.plot(*TRUE, "*", color=C_REF, ms=11, mec=S.SURFACE, zorder=4, label="posterior")
box_axes(ax); ax.set_title("Emulator error over the box")
fig.colorbar(im, ax=ax, label="rms error (mm)", fraction=0.05)
ax.legend(fontsize=7.5, loc="upper left", facecolor=S.SURFACE, framealpha=0.9, frameon=True, edgecolor=S.AXIS)
fig.tight_layout()
bk.savefig(fig, "3-03_emulator_anatomy")
plt.show()

# %% [markdown]
# **How to read it.** (a) A handful of components reproduce every run to a small fraction of σ (training runs solid,
# held-out runs dashed). (b) The leading components are smooth shapes over log time, overall amplitude first. (c) Along
# a slice through the posterior, the actual GP error (lines) and the GP's own ±2 s band (shading) for coefficients 1 and
# 3, in σ units; the ticks mark design runs close to the slice. The band is far wider than the actual error on
# coefficient 1: the GP is more careful than it needs to be (section 3 measures this). (d) The error of the emulated
# curve over the whole box: small around the posterior and among dense design runs, larger in a few patches with
# sparse runs and toward parts of the edge; the 2 mm contour is drawn.
#
# **Worked example (what the GP does at one new input).** At the posterior mean $\mathbf u^*$ the GP for coefficient
# $k$ computes the kernel vector $\mathbf k_* = k(\mathbf u^*, \mathbf u_i)$ to the $N$ runs, the weights
# $\boldsymbol\alpha = (\mathbf K + \epsilon\mathbf I)^{-1}\mathbf z$ (standardised training values $\mathbf z$), and returns
# $\mu = \mathbf k_*^\top\boldsymbol\alpha$ and $s^2 = k(\mathbf u^*,\mathbf u^*) - \mathbf k_*^\top(\mathbf K + \epsilon\mathbf I)^{-1}\mathbf k_*$,
# both rescaled to σ units. The cell prints each step for coefficient 1.

# %%
g1 = em60.gps[0]
ks = kern(mref[None], g1.X, g1.ls, g1.amp2)[0]
print(f"nearest design run to the posterior mean: {np.min(np.linalg.norm(U60 - mref, axis=1)):.3f} (unit-box distance)")
print(f"kernel vector: largest entries {np.round(np.sort(ks)[::-1][:4] / g1.amp2, 3)} x amp^2  (amp^2 = {g1.amp2:.3f})")
mu1 = ks @ g1.alpha[:, 0] * g1.ys[0] + g1.ym[0]
s1 = np.sqrt(g1.amp2 - ks @ cho_solve(g1.cf, ks)) * g1.ys[0]
print(f"mu = {mu1:.3f}, s = {s1:.3f} (sigma units); true coefficient {((run(mref)[0] / SIG - em60.mean) @ em60.V)[0]:.3f}")
RUNS["n"] = 0

# %% [markdown]
# **Common misreadings.**
# * "A small error map in mm means the emulator is good enough." What matters is the error where the posterior
#   lives, accumulated over all 40 epochs and compared with the noise. Section 3 shows a 30-run emulator whose posterior
#   has the right width but sits several posterior sd away from the reference.
# * "The GP sd s is the error." It is the GP's *estimate* of the error, and that estimate has to be checked (section 3).

# %% [markdown]
# ## 3. Carrying the emulator's error into the likelihood: the $1 + s^2$ term
#
# **In plain words.** Using the emulator instead of the model adds a second error on top of the GPS noise: the
# emulator's own. If we ignore it, a slightly wrong emulator gives a confidently wrong posterior. If we add it, the
# posterior widens where the emulator is unsure. This is the same idea as the prediction-error covariance $\mathbf C_p$
# of 2.13, with the GP supplying $\mathbf C_p$.
#
# **Precisely.** Project the whitened data on the same basis, $\hat{\mathbf c} = \mathbf V_K^\top(\mathbf d/\sigma - \bar{\mathbf w})$.
# If $\hat c_k = c_k(\boldsymbol\theta) + \varepsilon_k$ with $\varepsilon_k \sim \mathcal N(0, 1)$ (whitened noise; an
# orthonormal basis keeps it white) and the GP says $c_k(\boldsymbol\theta) \sim \mathcal N(\mu_k, s_k^2)$, then
# $\hat c_k \sim \mathcal N(\mu_k, 1 + s_k^2)$ and
# $$\ln p(\mathbf d\mid\boldsymbol\theta) \approx -\tfrac12\sum_{k=1}^K\frac{(\hat c_k - \mu_k(\boldsymbol\theta))^2}{1 + s_k^2(\boldsymbol\theta)} - \tfrac12\sum_{k=1}^K \ln\big(1 + s_k^2(\boldsymbol\theta)\big) + \text{const}.$$
# This is the usual CES likelihood for a whitened, orthonormal output basis. The data part outside the $K$ components is dropped as a constant, which is right when every model curve lies in the
# span of $\mathbf V_K$ (section 2, panel a). Two assumptions are worth naming: the coefficients' errors are treated as
# independent, and $s_k$ is taken at face value. Note also what the second term does: $-\tfrac12\sum\ln(1+s_k^2)$ is
# largest where the emulator is sure, which means near design runs.
#
# **Is $s_k$ right?** A GP's error bar is only as good as its kernel. The check is simple: run the true model at a few
# inputs where the posterior lives, and compare the actual coefficient errors with $s_k$. The ratio
# $|\text{error}|/s$ (rms over the check inputs) should be near 1: much smaller means the GP overstates its error, much
# larger means it understates it. We compare two ways of building the GP for $K$ coefficients:
# * **independent**: one GP per coefficient, each with its own length scales and amplitude (section 2);
# * **shared**: one kernel for all coefficients, each coefficient standardised. This is what a multi-output
#   scikit-learn `GaussianProcessRegressor` with `normalize_y=True` does by default, so it is a common choice in practice.

# %%
Ucheck = ref[::25]                                     # 40 check inputs drawn from the reference posterior
Ycheck = run(Ucheck); RUNS["n"] = 0


def calib(em):
    mu, s = em.predict(Ucheck)
    c = (Ycheck / SIG - em.mean) @ em.V
    return np.sqrt(np.mean(((c - mu) / s) ** 2, 0)), np.sqrt(np.mean((c - mu) ** 2, 0)), np.median(s, 0)


# ---------- emulated posteriors on a grid (two unknowns: a grid is exact up to its spacing, no sampler noise)
class GridPost:
    """Posterior of a 2-D log-likelihood on the unit box (uniform prior), computed on grids, no sampler noise:
    a coarse grid of the box, an axis-aligned zoom, then twice a grid laid along the principal axes of the current
    estimate (+-7 sd), so that even a thin tilted ellipse is resolved."""

    def __init__(self, logl_fn, n1=150, n2=81, n3=101, width=8.0, span=7.0):
        def norm_w(l):
            w = np.exp(l - l.max()); return w / w.sum()

        def moments(G, w):
            m = w @ G; return m, ((G - m) * w[:, None]).T @ (G - m)

        g = (np.arange(n1) + 0.5) / n1
        G = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
        m, C = moments(G, norm_w(logl_fn(G)))
        h = np.maximum(width * np.sqrt(np.diag(C)), 3.0 / n1)
        lo, hi = np.clip(m - h, 0, 1), np.clip(m + h, 0, 1)
        G = np.stack(np.meshgrid(np.linspace(lo[0], hi[0], n2), np.linspace(lo[1], hi[1], n2), indexing="ij"),
                     -1).reshape(-1, 2)
        m, C = moments(G, norm_w(logl_fn(G)))
        self.xi = np.linspace(-span, span, n3)
        XI = np.stack(np.meshgrid(self.xi, self.xi, indexing="ij"), -1).reshape(-1, 2)
        for _ in range(2):
            ev, E = np.linalg.eigh(C)
            ev = np.clip(ev, 1e-14, None)
            G = m + (XI * np.sqrt(ev)) @ E.T
            box = np.all((G >= 0) & (G <= 1), 1)
            l = np.full(len(G), -np.inf); l[box] = logl_fn(G[box])
            w = norm_w(l)
            self.m0, self.E, self.ev = m, E, ev
            m, C = moments(G, w)
        self.G, self.w, self.XI = G, w, XI
        self.W = w.reshape(n3, n3)
        self.mean, self.sd = m, np.sqrt(np.diag(C))
        self.lev90 = P.hdr_levels(self.W, (0.9,))[0.9]

    def inside90(self, Q):
        Q = np.atleast_2d(Q)
        X = ((Q - self.m0) @ self.E) / np.sqrt(self.ev)
        dx = self.xi[1] - self.xi[0]
        ia = np.round((X[:, 0] - self.xi[0]) / dx).astype(int)
        ib = np.round((X[:, 1] - self.xi[0]) / dx).astype(int)
        n = len(self.xi)
        ok = (ia >= 0) & (ia < n) & (ib >= 0) & (ib < n)
        out = np.zeros(len(Q), bool)
        out[ok] = self.W[ia[ok], ib[ok]] >= self.lev90
        return out

    def contour(self, ax, color, ls="-", lw=1.5, label=None):
        PG = phys(self.G)
        tri = Triangulation(PG[:, 0], PG[:, 1], Triangulation(self.XI[:, 0], self.XI[:, 1]).triangles)
        ax.tricontour(tri, self.w, levels=[self.lev90], colors=color, linestyles=ls, linewidths=lw)
        if label:
            ax.plot([], [], color=color, ls=ls, lw=lw, label=label)

    def extent(self):
        """physical bounding box of the 90 % region"""
        PG = phys(self.G[self.w >= self.lev90])
        return PG.min(0), PG.max(0)


def set_view(ax, posts, pad=0.25):
    """Axis limits (physical units) covering the reference draws and the 90 % regions of the given posteriors."""
    pr = phys(ref)
    lo, hi = pr.min(0), pr.max(0)
    for p in posts:
        a, b = p.extent(); lo, hi = np.minimum(lo, a), np.maximum(hi, b)
    d = hi - lo
    ax.set_xlim(lo[0] - pad * d[0], hi[0] + pad * d[0]); ax.set_ylim(lo[1] - pad * d[1], hi[1] + pad * d[1])


def grade(post):
    """Against the NUTS reference: largest |mean shift| in reference sd, geometric-mean sd ratio, and the share of
    reference draws inside the emulated 90 % region (0.9 = right; well below = biased or too narrow; 1 = too wide)."""
    return dict(shift=float(np.max(np.abs((post.mean - mref) / sdref))),
                sd_ratio=float(np.exp(np.mean(np.log(post.sd / sdref)))),
                in90=float(post.inside90(ref).mean()))


def fmt(g):
    return f"shift {g['shift']:6.2f} sd, sd ratio {g['sd_ratio']:6.2f}, reference draws inside its 90 % region {g['in90']:.2f}"


# a benchmark: how the exact quadrature posterior scores when graded like the others (against the same finite NUTS sample)
class _Exact:
    pass


ex_post = _Exact(); ex_post.mean, ex_post.sd = mq, sdq
lev_q = P.hdr_levels(wq.reshape(49, 49), (0.9,))[0.9]
xi_ref = ((ref - mref) @ Er) / np.sqrt(evr)
ia = np.round((xi_ref[:, 0] + 6) / (XI[1] - XI[0])).astype(int); ib = np.round((xi_ref[:, 1] + 6) / (XI[1] - XI[0])).astype(int)
ok = (ia >= 0) & (ia < 49) & (ib >= 0) & (ib < 49)
ex_post.inside90 = lambda Q, _v=np.where(ok, wq.reshape(49, 49)[ia.clip(0, 48), ib.clip(0, 48)] >= lev_q, False): _v
G_EXACT = grade(ex_post)
print("benchmark: the exact posterior itself scores", fmt(G_EXACT))

# %%
EMS = {}
for N in (30, 120):
    U_ = lhs(np.random.default_rng(100 + N), N); Y_ = run(U_)
    EMS[N] = {"independent": Emulator(U_, Y_), "shared": Emulator(U_, Y_, shared=True), "U": U_}
RUNS["n"] = 0
CAL, POSTS = {}, {}
for N in (30, 120):
    for kind in ("independent", "shared"):
        r, e, s = calib(EMS[N][kind]); CAL[(N, kind)] = r
        print(f"N = {N:3d}, {kind:11s} GP: |error|/s per coefficient {np.array2string(r, precision=3)}")
        print(f"{'':30s} actual rms error {np.array2string(e, precision=3)}  vs median s {np.array2string(s, precision=3)}")
    em_i, em_s = EMS[N]["independent"], EMS[N]["shared"]
    POSTS[N] = {"mean only (no s²)": GridPost(lambda U: em_i.loglik(U, y, False)),
                "with s², independent GPs": GridPost(lambda U: em_i.loglik(U, y, True)),
                "with s², one shared kernel": GridPost(lambda U: em_s.loglik(U, y, True))}
    for k, p in POSTS[N].items():
        print(f"   N = {N:3d} {k:28s} {fmt(grade(p))}")

# %%
VAR_STYLE = {"mean only (no s²)": (C_PW, "--"), "with s², independent GPs": (C_ACT, "-"),
             "with s², one shared kernel": (S.INK2, ":")}
fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.3), gridspec_kw=dict(width_ratios=[1.05, 1, 1]))
ax = axs[0]
for j, (N, kind, c) in enumerate([(30, "independent", C_ACT), (30, "shared", S.INK2), (120, "independent", C_ACT),
                                  (120, "shared", S.INK2)]):
    r = CAL[(N, kind)]
    ax.plot(np.arange(1, len(r) + 1), r, marker="o" if N == 30 else "s", ls="-" if N == 30 else "--", color=c, ms=5,
            mfc=c if N == 30 else S.SURFACE, label=f"{kind}, N = {N}")
ax.axhline(1, color=S.INK, lw=0.8)
ax.set_yscale("log"); ax.set_xlabel("coefficient k"); ax.set_ylabel("|actual error| / GP sd s  (rms)")
ax.text(0.98, 0.04, "below 1: GP overstates its error\nabove 1: GP understates it", transform=ax.transAxes,
        ha="right", va="bottom", fontsize=7.5, color=S.INK2)
ax.set_title("Are the GP error bars right?"); ax.legend(fontsize=7.5, loc="upper left")
for ax, N in zip(axs[1:], (30, 120)):
    pr = phys(ref)
    ax.scatter(pr[::3, 0], pr[::3, 1], s=3, color=C_REF, alpha=0.5, zorder=3)
    for k, p in POSTS[N].items():
        c, ls = VAR_STYLE[k]
        g = grade(p)
        p.contour(ax, c, ls, label=f"{k}: inside {g['in90']:.2f}, sd ×{g['sd_ratio']:.1f}")
    ax.plot(*TRUE, "*", color=S.INK, ms=10, mec=S.SURFACE, zorder=5)
    ax.plot([], [], "o", color=C_REF, ms=3, label="reference (NUTS)")
    ax.set_xlabel("aσ (MPa)"); ax.set_ylabel("v₀ (m/yr)")
    ax.set_title(f"Prior-wide design, N = {N}: 90 % regions")
    set_view(ax, [POSTS[N]["mean only (no s²)"], POSTS[N]["with s², independent GPs"]])
    ax.legend(fontsize=6.8, loc="upper left", frameon=True, facecolor=S.SURFACE, framealpha=0.92, edgecolor=S.SURFACE)
    if N == 30:                                         # the reference and the mean-only region are tiny here: inset
        axi = ax.inset_axes([0.55, 0.07, 0.42, 0.40])
        axi.scatter(pr[::3, 0], pr[::3, 1], s=2, color=C_REF, alpha=0.5, zorder=3)
        for k in ("mean only (no s²)", "with s², independent GPs"):
            POSTS[N][k].contour(axi, *VAR_STYLE[k])
        axi.plot(*TRUE, "*", color=S.INK, ms=8, mec=S.SURFACE, zorder=5)
        set_view(axi, [POSTS[N]["mean only (no s²)"]], pad=0.3)
        axi.tick_params(labelsize=6.5); axi.set_facecolor(S.SURFACE)
        ax.indicate_inset_zoom(axi, edgecolor=S.INK2)
fig.tight_layout()
bk.savefig(fig, "3-03_emulator_variance")
plt.show()

# %% [markdown]
# **How to read it.** Left: both versions of the GP overstate their error on the leading coefficients (ratios below 1,
# log axis) and understate it on one or two of the last ones (ratios above 1). At N = 120 the shared kernel is much
# further off on the leading coefficients: a coefficient that varies by hundreds of σ over the box gets the same
# *relative* uncertainty as the small ones, although it is predicted far better than that. Its actual errors are also
# larger at both sizes (printed). Middle (N = 30): without $s^2$ (orange dashed) the region has about the right size
# but sits off the reference cloud and holds almost none of its draws; with $s^2$ from independent GPs (aqua) it is many
# times wider and contains the cloud; with the shared kernel (dotted) it is wider still. The inset zooms on the
# reference cloud and the region without $s^2$. Right (N = 120): without $s^2$
# the region matches the reference; $s^2$ from independent GPs widens it somewhat; the shared kernel widens and moves
# it. The legends carry each region's share of reference draws and its sd ratio.
#
# **The image to keep.** The $1+s^2$ term is the right idea (it is $\mathbf C_p$ for the emulator) but it inherits the
# GP's honesty. Check $|\text{error}|/s$ per coefficient on a few true runs near the posterior before trusting it.
#
# **Common misreadings.**
# * "Adding $s^2$ only widens the posterior." An overstated $s$ weakens the data term so much that the centre of the
#   posterior moves as well (printed shifts). We checked that this is not the $\ln(1+s^2)$ term: removing that term
#   leaves shifts of the same order. The result covers the truth but says little about it.
# * "A wide posterior with $s^2$ is a property of the data." It is a property of the emulator; more runs in the right
#   place remove it (next section).

# %% [markdown]
# ## 4. Where to spend the runs: prior-wide vs CES vs active learning, same budget
#
# **In plain words.** The posterior is a speck, so runs spread over the whole box mostly teach the emulator about places
# the posterior is not. Two ways to put the runs where they matter:
# * **CES**: run an ensemble Kalman inversion first. Its members march from the prior toward the posterior, and every
#   member of every iteration is a training run. Here the calibrate step is the EnKI of 2.24 (with its adaptive
#   tempering step), with J = 10 members started from the prior. EnKI works with a Gaussian prior, so it runs in
#   $\mathbf z = \Phi^{-1}(\mathbf u)$, where $\Phi^{-1}$ is the inverse of the standard normal cdf (2.03); this maps
#   the uniform box exactly onto a standard normal.
# * **Active learning** (sequential design): start with a third of the budget prior-wide, then add batches of 10
#   runs where the current emulated posterior has mass *and* the emulator is unsure: draw candidates from the current
#   emulated posterior (with $s^2$), then pick, one at a time, the candidate with the largest total GP variance
#   $\sum_k s_k^2$, updating the variances after each pick (a GP's variance does not need the new outputs).
#
# The budget is set by CES (members × iterations); the other two designs get exactly as many runs. All three use the
# same emulator (independent GPs, with $s^2$).

# %%
PROB_Z = P.InverseProblem(lambda z: fwd_u(jnorm.cdf(z)), y, SIG, m0=np.zeros(2), P=np.eye(2))


def ces_design(yobs, seed, J=10):
    # EnKI reads the data from problem.y; one problem object is reused so that the solver is compiled only once
    PROB_Z.y = np.asarray(yobs, float).copy()
    out = SM.enki(PROB_Z, J=J, rng=np.random.default_rng(seed))
    Z = np.concatenate([h["X"] for h in out["history"]])
    return np.asarray(jnorm.cdf(jnp.asarray(Z))), out


def active_design(yobs, budget, seed, batch=10):
    rng = np.random.default_rng(seed)
    U = lhs(rng, budget // 3); Yo = run(U)
    while len(U) < budget:
        em = Emulator(U, Yo)
        post = GridPost(lambda X: em.loglik(X, yobs, True))
        cand = post.G[rng.choice(len(post.G), 300, p=post.w)]                  # draws from the emulated posterior,
        dx = post.xi[1] - post.xi[0]                                            # jittered within their grid cell
        cand = np.clip(cand + ((rng.uniform(-0.5, 0.5, cand.shape) * dx) * np.sqrt(post.ev)) @ post.E.T, 0, 1)
        pick = []
        for _ in range(min(batch, budget - len(U))):
            if pick:
                tv = sum(g.var_after(cand, np.array(pick)) * g.ys[0] ** 2 for g in em.gps)
            else:
                tv = em.total_var(cand)
            pick.append(cand[int(np.argmax(tv))])
        pick = np.array(pick)
        U, Yo = np.vstack([U, pick]), np.vstack([Yo, run(pick)])
    return U, Yo


t0 = time.time()
RUNS["n"] = 0
Uces, enki_out = ces_design(y, seed=0)
Yces = run(Uces)                       # the same inputs as the EnKI members; re-run here only to keep the outputs
NB = len(Uces)
RUNS["n"] = 0
print(f"CES calibrate step: EnKI with J = 10, {len(enki_out['history'])} ensembles (theta = "
      f"{np.array2string(enki_out['theta'], precision=3)}), budget = {NB} runs")
Upw = lhs(np.random.default_rng(1), NB); Ypw = run(Upw)
Uact, Yact = active_design(y, NB, seed=2)
print(f"runs used: prior-wide {len(Upw)}, CES {len(Uces)}, active {len(Uact)}")
DES = {"prior-wide": (Upw, Ypw, C_PW), "CES": (Uces, Yces, C_CES), "active learning": (Uact, Yact, C_ACT)}
DPOST = {}
for name, (U_, Y_, c) in DES.items():
    em = Emulator(U_, Y_)
    DPOST[name] = (GridPost(lambda X: em.loglik(X, y, True)), em)
    near_err = np.sqrt(np.mean((em.curve_mm(Ucheck) - 1000 * Ycheck) ** 2))
    print(f"{name:16s}: emulator error at 40 posterior draws {near_err:.3f} mm rms; {fmt(grade(DPOST[name][0]))}")
print(f"({time.time() - t0:.0f} s)")

# %%
fig, axs = plt.subplots(2, 3, figsize=(14.5, 8.2))
for j, (name, (U_, Y_, c)) in enumerate(DES.items()):
    ax = axs[0, j]
    pu = phys(U_)
    ax.scatter(pu[:, 0], pu[:, 1], s=10, color=c, ec=S.SURFACE, lw=0.3, zorder=3)
    ax.plot(*TRUE, "*", color=S.INK, ms=11, mec=S.SURFACE, zorder=4)
    box_axes(ax)
    ax.set_title(f"{name}: {len(U_)} runs")
    ax = axs[1, j]
    post, em = DPOST[name]
    pr = phys(ref)
    ax.scatter(pr[::3, 0], pr[::3, 1], s=3, color=C_REF, alpha=0.5, zorder=3)
    post.contour(ax, c, "-", 1.8)
    set_view(ax, [post])
    ax.scatter(*phys(U_).T, s=22, marker="D", color=c, ec=S.INK, lw=0.6, zorder=4)
    ax.plot(*TRUE, "*", color=S.INK, ms=10, mec=S.SURFACE, zorder=5)
    g = grade(post)
    ok = 0.8 <= g["in90"] <= 0.97 and 0.8 <= g["sd_ratio"] <= 1.25
    S.badge(ax, f"inside {g['in90']:.2f}, sd ratio {g['sd_ratio']:.2f}", ok=ok, loc="lower right", fs=8)
    ax.set_xlabel("aσ (MPa)"); ax.set_ylabel("v₀ (m/yr)")
    ax.set_title("emulated 90 % region vs reference")
axs[1, 0].legend(handles=[plt.Line2D([], [], marker="o", ls="", ms=3, color=C_REF, label="reference draws (NUTS)"),
                          plt.Line2D([], [], color=S.INK2, lw=1.8, label="emulated 90 % region"),
                          plt.Line2D([], [], marker="D", ls="", ms=4, color=S.MUTED, mec=S.INK, label="design runs in view")],
                 loc="upper left", fontsize=7.5)
fig.tight_layout()
bk.savefig(fig, "3-03_designs")
plt.show()

# %% [markdown]
# **How to read it.** Top row: where each design put its runs (log axes over the whole box; the star is where the
# posterior is). Bottom row: each design's emulated 90 % region (with $s^2$) on top of the reference draws, zoomed to the
# region. The badges carry the share of reference draws inside and the sd ratio (the emulated posterior's sd divided by
# the reference sd, 1 = right width); a tick means both are within the tolerance written in the code (0.80-0.97 and
# 0.80-1.25).
#
# One draw of each design can be lucky or unlucky, so the cell below repeats the whole comparison for five design seeds
# (the data stay the same). For each seed CES sets the budget and the other two designs get the same number of runs.

# %%
t0 = time.time()
REP = {k: [] for k in DES}
budgets = []
for seed in range(1, 6):
    Uc_, _ = ces_design(y, seed=seed)
    nb = len(Uc_); budgets.append(nb)
    designs = {"prior-wide": lhs(np.random.default_rng(50 + seed), nb), "CES": Uc_}
    for name, U_ in designs.items():
        em = Emulator(U_, run(U_))
        REP[name].append(grade(GridPost(lambda X: em.loglik(X, y, True))))
    Ua_, Ya_ = active_design(y, nb, seed=70 + seed)
    em = Emulator(Ua_, Ya_)
    REP["active learning"].append(grade(GridPost(lambda X: em.loglik(X, y, True))))
RUNS["n"] = 0
print(f"five design seeds, budgets {budgets} runs ({time.time() - t0:.0f} s)")
print(f"{'design':16s} | {'shift (sd)':>18s} | {'sd ratio':>18s} | {'share inside 90 %':>18s}   (median [min, max])")
for name, gs in REP.items():
    cols = []
    for key in ("shift", "sd_ratio", "in90"):
        v = np.array([g[key] for g in gs])
        cols.append(f"{np.median(v):6.2f} [{v.min():.2f}, {v.max():.2f}]")
    print(f"{name:16s} | " + " | ".join(f"{c:>18s}" for c in cols))
print(f"{'exact (benchmark)':16s} | {G_EXACT['shift']:6.2f}{'':12s} | {G_EXACT['sd_ratio']:6.2f}{'':12s} | {G_EXACT['in90']:6.2f}")

# %% [markdown]
# **How to read it.** The last row is a benchmark, not a limit: the exact posterior graded against the same finite set of
# NUTS draws. Its scores differ from the ideal (0 shift, sd ratio 1, 0.90 inside) only because the reference sample is
# finite. An emulated posterior can score slightly better than this row by chance (a slightly wider region holds more
# draws, for example), so differences smaller than the row's own departure from the ideal are noise.
#
# **The image to keep.** At the same number of runs, putting them where the posterior is makes the emulator error at
# the posterior far smaller (printed above) and the emulated posterior indistinguishable from the exact one (compare
# with the benchmark row). CES gets there by letting a cheap ensemble method do the searching; active learning gets there
# by letting the emulator ask for runs where it is unsure and the posterior might be. Prior-wide designs spend most runs
# on regions the data rule out.
#
# **Common misreadings.**
# * "The CES training set is biased, because EnKI is not an exact sampler." CES uses the EnKI members only as
#   *inputs* for training runs; the posterior is then sampled from the emulated likelihood. The EnKI ensemble only has to
#   cover the posterior region, not represent it exactly.
# * "Active learning needs no prior-wide runs." It needs a first emulator good enough to point roughly at the
#   posterior. Here it started from a third of the budget spread over the box; we did not test smaller starts.

# %% [markdown]
# ## 5. Coverage over repeated data sets
#
# **In plain words.** One data set can mislead. Make 16 data sets from the same true values with fresh noise, run each
# emulator-based inversion, and count how often the 90 % region contains the true values. A calibrated posterior does
# so about 90 % of the time; with 16 data sets, the count itself scatters (the binomial range is printed: where a count
# of yes/no outcomes with probability 0.9 falls by chance in 16 tries).
#
# **Set-up.** Prior-wide designs of 30, 60 and 120 runs (a new Latin hypercube for each data set), and CES with J = 10
# (its own budget for each data set), each with and without the $s^2$ term, independent GPs throughout. As a yardstick,
# the *linearised* exact posterior (Laplace at the true values, 2.08), whose coverage is 90 % by construction up to the
# small nonlinearity. Width is reported as the sd ratio against the linearised posterior.
#
# A side remark on cost. A prior-wide emulator does not depend on the data, so in practice it could be built once and
# reused for all 16 data sets (*amortised*); a CES design is calibrated on each data set and needs new runs every time.
# Here every data set gets a new design, so the coverage also averages over design luck.

# %%
t0 = time.time()
J_true = np.asarray(jax.jacrev(fwd_u)(jnp.asarray(u_true)))
C_lin = np.linalg.inv(J_true.T @ J_true / SIG ** 2); sd_lin = np.sqrt(np.diag(C_lin)); Ci_lin = np.linalg.inv(C_lin)
q90 = CHI2.ppf(0.9, 2)
R = 16
NS = (30, 60, 120)
COV = {}
for r in range(R):
    yr = clean + SIG * np.random.default_rng(1000 + r).standard_normal(clean.size)
    m_lin = u_true + C_lin @ J_true.T @ (yr - clean) / SIG ** 2
    COV.setdefault("linearised exact", []).append(((u_true - m_lin) @ Ci_lin @ (u_true - m_lin) <= q90, 1.0, 0))
    ems = {f"prior-wide {N}": Emulator(U_, run(U_)) for N in NS for U_ in [lhs(np.random.default_rng(2000 + 10 * r + N), N)]}
    Uc_, _ = ces_design(yr, seed=3000 + r)
    ems["CES"] = Emulator(Uc_, run(Uc_))
    for name, em in ems.items():
        for use_var in (False, True):
            p = GridPost(lambda X: em.loglik(X, yr, use_var))
            COV.setdefault((name, use_var), []).append((bool(p.inside90(u_true)[0]),
                                                        float(np.exp(np.mean(np.log(p.sd / sd_lin)))), len(em.U)))
RUNS["n"] = 0
lo_b, hi_b = binom.ppf(0.05, R, 0.9) / R, binom.ppf(0.95, R, 0.9) / R
print(f"{R} data sets ({time.time() - t0:.0f} s). A calibrated 90 % region covers the true values in "
      f"{lo_b:.2f}-{hi_b:.2f} of {R} data sets (binomial 5-95 %)")
print(f"{'':28s} {'coverage':>9s} {'sd ratio (median)':>18s} {'runs':>6s}")
for key, v in COV.items():
    name = key if isinstance(key, str) else f"{key[0]}, {'with s²' if key[1] else 'no s²'}"
    v = np.array(v, dtype=float)
    print(f"{name:28s} {v[:, 0].mean():9.2f} {np.median(v[:, 1]):18.2f} {np.median(v[:, 2]):6.0f}")

# %%
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.1))
for use_var, ls, mk in ((False, "--", "o"), (True, "-", "s")):
    lab = "with s²" if use_var else "no s² (mean only)"
    cov = [np.mean([v[0] for v in COV[(f"prior-wide {N}", use_var)]]) for N in NS]
    wid = [np.median([v[1] for v in COV[(f"prior-wide {N}", use_var)]]) for N in NS]
    axs[0].plot(NS, cov, ls=ls, marker=mk, color=C_PW, mfc=C_PW if use_var else S.SURFACE, label=f"prior-wide, {lab}")
    axs[1].plot(NS, wid, ls=ls, marker=mk, color=C_PW, mfc=C_PW if use_var else S.SURFACE, label=f"prior-wide, {lab}")
    v = np.array(COV[("CES", use_var)], dtype=float)
    xc = np.median(v[:, 2]) * (1.04 if use_var else 0.96)        # small offset so the two CES markers do not overlap
    axs[0].plot(xc, v[:, 0].mean(), marker=mk, ls="", ms=8, color=C_CES,
                mfc=C_CES if use_var else S.SURFACE, label=f"CES, {lab}")
    axs[1].plot(xc, np.median(v[:, 1]), marker=mk, ls="", ms=8, color=C_CES,
                mfc=C_CES if use_var else S.SURFACE, label=f"CES, {lab}")
cl = np.mean([v[0] for v in COV["linearised exact"]])
axs[0].axhspan(lo_b, hi_b, color=S.GRID, zorder=0, label=f"binomial 5-95 % range for {R} data sets")
axs[0].axhline(0.9, color=S.INK, lw=0.8)
axs[0].axhline(cl, color=S.INK, lw=0.8, ls=":", label="linearised exact posterior")
axs[0].set_ylim(-0.03, 1.05); axs[0].set_xscale("log"); plain_ticks(axs[0], x=list(NS) + [80])
axs[0].set_xlabel("runs in the design"); axs[0].set_ylabel("share of data sets whose 90 % region holds the truth")
axs[0].set_title(f"Coverage over {R} data sets"); axs[0].legend(fontsize=7, loc="lower right")
axs[1].axhline(1, color=S.INK, lw=0.8)
axs[1].set_xscale("log"); axs[1].set_yscale("log"); plain_ticks(axs[1], x=list(NS) + [80], y=[1, 2, 4, 8])
axs[1].set_xlabel("runs in the design"); axs[1].set_ylabel("posterior sd / linearised exact sd (median)")
axs[1].set_title("Width of the emulated posterior")
fig.tight_layout()
bk.savefig(fig, "3-03_coverage")
plt.show()

# %% [markdown]
# **How to read it.** Left: open markers and dashed lines are the emulated posteriors without $s^2$, filled markers and
# solid lines with it; the grey band is where a calibrated posterior's coverage falls by chance. The two CES markers
# are drawn slightly left and right of their common budget so that both stay visible. Right: how wide the
# same posteriors are, relative to the exact one. Coverage alone can be bought with width, so read the two panels
# together.
#
# **The image to keep.** Without $s^2$, a coarse emulator (30 runs) gives posteriors of the right width in the wrong
# place: coverage falls below the band. With $s^2$ the coverage is restored by widening, and here the widening remains
# when the emulator no longer needs it (60 and 120 runs), because the GP overstates its error (section 3). More runs
# bring the posterior without $s^2$ to the right coverage *and* width; in these two dimensions 120 prior-wide runs do,
# and CES does with its own, smaller budget (printed).

# %% [markdown]
# ## 6. Emulating the log-likelihood instead
#
# **In plain words.** Instead of emulating the 40 predicted numbers, emulate the one number we actually need: the
# log-likelihood $\ell(\boldsymbol\theta) = -\tfrac12\chi^2(\boldsymbol\theta)$. This is a common shortcut when the
# unknowns are a few hyperparameters and each evaluation of their (marginal) log-likelihood is a whole inversion: the
# GP target is then a single number instead of a curve.
#
# **Precisely.** Here we fit one GP to $\ell(\boldsymbol\theta_i)$ at the same runs as section 4 and set
# $p(\boldsymbol\theta\mid\mathbf d) \propto \exp[\mu_\ell(\boldsymbol\theta)]$ inside the box. Three things differ from
# the forward emulator.
# 1. The target spans a huge range: χ² of a bad parameter is enormous next to that of a good one, while near the peak an
#    error of 1 in $\ell$ already changes the posterior density by a factor $e$.
# 2. Far from the runs a GP reverts toward the average of its training values, or overshoots between them. For a forward
#    emulator, an "average curve" fits the data badly, so the posterior stays away. For an $\ell$ emulator, an "average
#    $\ell$" can be far *higher* than the true $\ell$ there, which invents posterior mass.
# 3. The $\ell$ emulator belongs to one data set and one noise model; the forward emulator can be reused for new data.
#
# A remedy for point 2 is a **trust region**: accept only inputs within a set distance of well-fitting training runs.
# We use this rule: an input is allowed if its distance to the nearest well-fitting training run is below the 99th
# percentile of the distances between neighbouring well-fitting runs, where well-fitting means $\ell$ within 200 of the
# best training run.

# %%
def loglik_true(Yout, yobs):
    return -0.5 * np.sum(((yobs - Yout) / SIG) ** 2, 1)


LL = {}
for name in ("prior-wide", "CES"):
    U_, Y_, c = DES[name]
    ll_ = loglik_true(Y_, y)
    gp_l = GP(U_, ll_)
    good = U_[ll_ > ll_.max() - 200]
    tree = cKDTree(good) if len(good) >= 2 else None
    r99 = float(np.percentile(tree.query(good, k=2)[0][:, 1], 99)) if tree is not None else np.nan
    LL[name] = dict(gp=gp_l, ll=ll_, tree=tree, r=r99, n_good=len(good))
    print(f"{name:10s}: {len(U_)} runs, training log-likelihood from {ll_.min():.3g} to {ll_.max():.3g}; "
          f"{len(good)} run(s) within 200 of the best; "
          + (f"trust radius {r99:.4f} (unit box)" if tree is not None else "too few to define a trust region"))
ll_check = loglik_true(Ycheck, y)
LPOST = {}
for name in ("prior-wide", "CES"):
    L = LL[name]
    err = L["gp"].predict(Ucheck, var=False)[:, 0] - ll_check
    print(f"{name:10s}: log-likelihood emulator error at 40 posterior draws, rms {np.sqrt(np.mean(err ** 2)):.3g}")
    LPOST[(name, False)] = GridPost(lambda X, L=L: L["gp"].predict(X, var=False)[:, 0])
    if L["tree"] is not None:
        LPOST[(name, True)] = GridPost(lambda X, L=L: np.where(L["tree"].query(X)[0] <= L["r"],
                                                               L["gp"].predict(X, var=False)[:, 0], -np.inf))
for (name, trust), p in LPOST.items():
    print(f"   {name:10s} {'with trust region' if trust else 'no trust region':18s}: {fmt(grade(p))}; "
          f"posterior mean at a_sigma = {phys(p.mean)[0]:.3f} MPa, v0 = {phys(p.mean)[1]:.2e} m/yr")
gbox = (np.arange(100) + 0.5) / 100
Gbox = np.stack(np.meshgrid(gbox, gbox, indexing="ij"), -1).reshape(-1, 2)
mu_box = LL["CES"]["gp"].predict(Gbox, var=False)[:, 0]
print(f"CES log-likelihood GP over the box: largest mean value {mu_box.max():.3g} vs best training run "
      f"{LL['CES']['ll'].max():.3g}; share of the box where the GP mean exceeds the best training run "
      f"{np.mean(mu_box > LL['CES']['ll'].max()):.3f}")

# %%
fig, axs = plt.subplots(1, 3, figsize=(16, 4.4), gridspec_kw=dict(width_ratios=[1.05, 1.1, 1]))
ax = axs[0]
ll_sl = loglik_true(Ysl, y)
for name, c in (("prior-wide", C_PW), ("CES", C_CES)):
    ax.plot(asl, LL[name]["gp"].predict(Usl, var=False)[:, 0], color=c, lw=1.4, ls="--", label=f"GP of ℓ, {name} runs")
ax.plot(asl, ll_sl, color=S.INK, lw=1.4, label="true ℓ")
ax.set_xscale("log"); ax.set_yscale("symlog", linthresh=10); plain_ticks(ax, x=[0.2, 0.5, 1, 2])
ax.axvline(TRUE[0], color=C_REF, lw=0.9, ls=":")
ax.set_xlabel("aσ (MPa), v₀ at its posterior mean"); ax.set_ylabel("log-likelihood ℓ (symlog)")
ax.set_title("ℓ along a slice: a huge range"); ax.legend(fontsize=7.5, loc="lower left")
ax = axs[1]
PB = phys(Gbox)
dmu = mu_box - LL["CES"]["ll"].max()
lim = max(abs(dmu.min()), abs(dmu.max()))
im = ax.pcolormesh(PB[:, 0].reshape(100, 100), PB[:, 1].reshape(100, 100), dmu.reshape(100, 100), cmap=DIV,
                   norm=SymLogNorm(linthresh=100, vmin=-lim, vmax=lim), shading="nearest")
fig.colorbar(im, ax=ax, label="GP mean of ℓ minus best training ℓ", fraction=0.05)
pc = phys(DES["CES"][0])
ax.scatter(pc[:, 0], pc[:, 1], s=8, color=S.SURFACE, ec=S.INK, lw=0.5, zorder=3, label="CES runs")
ax.plot(*TRUE, "*", color=C_REF, ms=11, mec=S.SURFACE, zorder=4, label="posterior")
box_axes(ax); ax.set_title("Away from the runs the GP of ℓ overshoots")
ax.legend(fontsize=7.5, loc="upper right", facecolor=S.SURFACE, framealpha=0.9, frameon=True, edgecolor=S.AXIS)
ax = axs[2]
pr = phys(ref)
ax.scatter(pr[::3, 0], pr[::3, 1], s=3, color=C_REF, alpha=0.5, zorder=3)
DPOST["CES"][0].contour(ax, S.AXIS, "-", 5.0, label="forward emulator, CES runs (section 4)")
LPOST[("CES", True)].contour(ax, C_CES, "-", 1.5, label="GP of ℓ, CES runs, trust region")
ax.plot(*TRUE, "*", color=S.INK, ms=10, mec=S.SURFACE, zorder=5)
ax.plot([], [], "o", color=C_REF, ms=3, label="reference (NUTS)")
set_view(ax, [LPOST[("CES", True)], DPOST["CES"][0]])
ax.set_xlabel("aσ (MPa)"); ax.set_ylabel("v₀ (m/yr)"); ax.set_title("With a trust region: 90 % regions")
ax.legend(fontsize=7, loc="upper left")
fig.tight_layout()
bk.savefig(fig, "3-03_loglik_emulator")
plt.show()

# %% [markdown]
# **How to read it.** Left: along a slice the true ℓ falls by orders of magnitude away from the posterior (symlog axis).
# The GP fitted to prior-wide runs follows it closely far from the peak and is wrong by tens of units *at* the peak,
# the only place that matters; the GP fitted to CES runs lies on the true curve there.
# Middle: the CES-trained GP of ℓ over the whole box, as the difference from the best training run. Red areas are
# where the GP claims a better fit than any run ever achieved; far from the runs this is pure extrapolation, and without
# a trust region the emulated posterior goes there (the "no trust region" lines printed above). Right: with the trust
# region, the CES ℓ emulator's 90 % region (violet) lies on the forward emulator's (wide grey line) and on the
# reference draws.
#
# **The image to keep.** A log-likelihood emulator is accurate only where it has runs and can be wildly optimistic
# elsewhere; confine it with a trust region. A forward emulator fails more gracefully, because an uncertain prediction
# fits the data badly.
#
# **Common misreadings.**
# * "A GP error of a few percent of the ℓ range is small." The relevant scale is absolute: near the peak, an error of
#   1 in ℓ is a factor $e$ in posterior density. The prior-wide ℓ emulator's error printed above is a tiny fraction of
#   the range and still useless.
# * "Hyperparameter surfaces behave like this one." Not necessarily: a log-evidence over smoothing hyperparameters is
#   often smooth and of modest range. This toy only shows what to check (absolute error near the peak, overshoot away
#   from the runs).

# %% [markdown]
# ## 7. Kernels and cost (all three parameters, cached runs)
#
# **In plain words.** Two practical knobs remain: which kernel, and how many runs a GP can take. Kernels encode how
# smooth we expect the outputs to be (2.12): Matérn-3/2 (rough), Matérn-5/2, squared exponential (infinitely smooth).
# Cost: fitting a GP needs the Cholesky factor (2.06) of an $N \times N$ matrix: time grows like $N^3$ and memory like
# $N^2$ (written $O(N^3)$ and $O(N^2)$).
#
# **Set-up.** The three-parameter model with the cached runs of `afterslip.training_set_single`: training runs from the
# 4,000-run Latin hypercube, held-out test runs from the separate 4,000-run uniform set (the first 400). PCA on the
# first 1,000 training runs, K = 8 (truncation printed). Error: rms over the 40 epochs of each test run, median over
# test runs, in units of σ. Hyperparameters are fitted on at most 400 runs. One shared Matérn-5/2 kernel is included as
# a fourth line.

# %%
t0 = time.time()
th_tr, u_tr = A.training_set_single(4000, seed=0, design="lhs")
th_te, u_te = A.training_set_single(4000, seed=0)
B3 = A.SP_BOUNDS
X_tr = (th_tr - B3[:, 0]) / (B3[:, 1] - B3[:, 0]); X_te = (th_te[:400] - B3[:, 0]) / (B3[:, 1] - B3[:, 0])
W_tr, W_te = u_tr / SIG, u_te[:400] / SIG
wm = W_tr[:1000].mean(0)
_, _, Vt3 = np.linalg.svd(W_tr[:1000] - wm, full_matrices=False)
V3 = Vt3[:8].T
print("three parameters: PCA truncation (rms, sigma units) on the test runs, K = 6, 8, 10:",
      [round(float(np.sqrt(np.mean(((W_te - wm) - (W_te - wm) @ Vt3[:k].T @ Vt3[:k]) ** 2))), 4) for k in (6, 8, 10)])
NS3 = (50, 100, 200, 400)
LC = {}
for label, kind, shared in (("Matérn-3/2", "m32", False), ("Matérn-5/2", "m52", False),
                            ("squared exponential", "se", False), ("Matérn-5/2, one shared kernel", "m52", True)):
    for n in NS3:
        Ctr = (W_tr[:n] - wm) @ V3
        gps = ([GP(X_tr[:n], Ctr, kind, starts=(0.2, 1.0))] if shared else
               [GP(X_tr[:n], Ctr[:, k], kind, starts=(0.2, 1.0)) for k in range(8)])
        mu = np.hstack([g.predict(X_te, var=False) for g in gps])
        per_run = np.sqrt(np.mean(((mu @ V3.T + wm) - W_te) ** 2, 1))        # sigma units
        LC[(label, n)] = (np.median(per_run), np.percentile(per_run, 90))
    print(f"{label:30s} median test error (sigma units) for N = {NS3}: "
          f"{np.array2string(np.array([LC[(label, n)][0] for n in NS3]), precision=3)}; 90th percentile at N = 400: "
          f"{LC[(label, 400)][1]:.3f}")
CH = {}
for n in (250, 500, 1000, 2000, 4000):
    Kx = kern(X_tr[:n], X_tr[:n], np.array([0.3, 0.3, 0.3]), 1.0) + 1e-6 * np.eye(n)
    t1 = time.time()
    for _ in range(3):
        cho_factor(Kx, lower=True)
    CH[n] = (time.time() - t1) / 3
print("Cholesky factorisation time (s):", {n: float(f"{t:.2g}") for n, t in CH.items()},
      f"; memory of the N = 4000 matrix: {4000 ** 2 * 8 / 1e6:.0f} MB")
print(f"({time.time() - t0:.0f} s)")

# %%
fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.1))
ax = axs[0]
for (label, c, ls) in (("Matérn-3/2", S.BLUE, "-"), ("Matérn-5/2", S.INK, "-"), ("squared exponential", S.AQUA, "-"),
                       ("Matérn-5/2, one shared kernel", S.INK2, ":")):
    ax.plot(NS3, [LC[(label, n)][0] for n in NS3], ls=ls, marker="o", color=c, label=label)
ax.axhline(1, color=S.MUTED, lw=0.8, ls="--")
ax.text(NS3[0], 1.05, "1 σ", color=S.MUTED, fontsize=7.5, ha="left", va="bottom")
ax.set_xscale("log"); ax.set_yscale("log"); plain_ticks(ax, x=list(NS3))
ax.set_xlabel("training runs N (prior-wide, 3 parameters)"); ax.set_ylabel("median test error (σ units)")
ax.set_title("Learning curves by kernel"); ax.legend(fontsize=7.5)
ax = axs[1]
ns = np.array(list(CH)); ts = np.array(list(CH.values()))
ax.loglog(ns, ts, "o-", color=S.INK)
ax.loglog(ns, ts[-1] * (ns / ns[-1]) ** 3, color=S.MUTED, ls="--", lw=0.9, label="∝ N³")
plain_ticks(ax, x=list(CH)); ax.set_xlabel("training runs N"); ax.set_ylabel("one Cholesky factorisation (s)")
ax.set_title("Cost of conditioning a GP"); ax.legend()
fig.tight_layout()
bk.savefig(fig, "3-03_kernels_cost")
plt.show()

# %% [markdown]
# **How to read it.** Left: test error against the number of runs for each kernel, on log axes; lower is better and
# 1 σ is marked. The three independent-GP lines bunch together, the shared-kernel line sits above them. Right: the
# time of one Cholesky factorisation grows close to $N^3$ (dashed guide), so 10× more runs cost about 1,000× more time;
# fitting the hyperparameters repeats it at every optimiser step.
#
# **The image to keep.** On a smooth forward model the kernel family matters less than independent fits per output and,
# above all, the number and placement of runs. GPs are the right tool for hundreds to a few thousand runs; beyond that,
# use sparse GPs (which summarise the runs by a few hundred "inducing" points) or local GPs (separate GPs for
# neighbourhoods), or switch to the neural surrogates of 3.02, 3.04 and 3.05.
#
# **Common misreadings.**
# * "The squared exponential is the default, so it must be safe." It assumes infinite smoothness and gives the most
#   ill-conditioned kernel matrices; check the fitted length scales (a length scale stuck at its lower bound means the
#   fit failed).
# * "More runs always fix a GP." Not if they are in the wrong place (sections 4-5) or if the cost of $N^3$ stops you first.

# %% [markdown]
# ## 8. Emulator error is not model discrepancy
#
# **In plain words.** Everything above treats the *emulator's* error, which shrinks with more runs. A different error
# does not: the physics in the black box may itself be wrong (a viscoelastic contribution the afterslip model lacks, a
# wrong elastic structure). No number of runs fixes that.
#
# **Precisely (concept only).** Kennedy & O'Hagan (2001; cited for the idea only) write
# $\mathbf d = \boldsymbol\eta(\boldsymbol\theta) + \boldsymbol\delta + \boldsymbol\varepsilon$: emulator of the
# simulator $\boldsymbol\eta$, a *discrepancy* $\boldsymbol\delta$ given its own GP prior, and noise. Two warnings come
# with it. The discrepancy and the parameters can trade off (a flexible $\boldsymbol\delta$ can absorb part of what
# $\boldsymbol\theta$ should explain), so the prior on $\boldsymbol\delta$ must say what kind of misfit is plausible.
# And the emulator's $s^2$ is *not* a discrepancy term: a perfect emulator of a wrong model still gives a confidently
# wrong posterior. In geodetic practice the structured version of $\boldsymbol\delta$ is the prediction covariance
# $\mathbf C_p$ of 2.13 and 2.31 (Duputel et al. 2014).

# %% [markdown]
# ## 9. Summary
#
# | choice | what we found here | what to check in a real application |
# |---|---|---|
# | what to emulate | forward map (PCA coefficients) fails gracefully; log-likelihood is fine near its runs and overshoots away from them | absolute ℓ error near the peak; overshoot away from the runs; trust region |
# | how to build the GP | independent GPs per coefficient: smaller errors than one shared kernel, and at 120 runs error bars much closer to right; both still overstate the error on the leading coefficients | per-coefficient $\lvert\text{error}\rvert/s$ on true runs near the posterior |
# | emulator variance $1+s^2$ | needed when the emulator is coarse; with a miscalibrated $s$ it widens and shifts the posterior | coverage over repeated synthetic data sets |
# | design | CES and active learning beat prior-wide at the same budget | where the training runs sit relative to the posterior |
# | kernel, size | kernel family matters less; cost grows as $N^3$ | fitted length scales; N in the hundreds to thousands |
#
# **In practice.**
# * Multi-output GP software often uses one shared kernel with each output standardised (section 3). In this toy that
#   choice overstated the error bars of the leading coefficients and widened the posterior far beyond the reference.
#   The check is cheap: run the true model at a few hundred points drawn from the emulated posterior and compare the
#   actual coefficient errors with $s$, coefficient by coefficient. A check of the emulated χ² against the true χ² at
#   those points tests the mean, not $s$. Training only on runs that fit reasonably well may reduce the effect; we did
#   not test that here.
# * When the GP emulates a log-likelihood or log-evidence over a few hyperparameters (section 6), judge its error in
#   absolute log units near the peak, not as a share of the range, and check whether the GP mean anywhere exceeds the
#   best training run, which calls for a trust region.
# * An emulator of an approximate target (for example a proxy for the evidence) inherits whatever is wrong with the
#   target; emulation does not fix it.
#
# Next: **3.04 · Physics-informed neural networks**.

# %%
print(f"notebook run time: {time.time() - T0:.0f} s")
