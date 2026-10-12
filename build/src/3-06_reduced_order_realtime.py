# %% [markdown]
# # 3.06 · Reduced-order models and real-time inference
#
# *Topic 3 is under development: content and numbering may still change.*
#
# **Before this:** 2.06 (SVD), 2.08 (Bayesian linear inversion and the gain form), 3.01 (PCA of outputs), 3.05 (the 1-D
# fault model) · **You will be able to:** compress solutions with POD and read the energy curve; build a data-driven and
# a Galerkin reduced model and judge their accuracy and speed; precompute an exact linear Bayesian inversion so that
# each new data set costs one matrix-vector product · **Optional sections:** 4 (outside the training range).
#
# 3.05 replaced the fault solver with a network that knows nothing about the equations. This notebook takes the other
# road: keep the physics, but find out how few numbers really describe its solutions, and solve for those few. That is a
# reduced-order model (ROM). The second half asks what "real time" can mean for an inversion: when the forward model is
# linear, everything except one matrix-vector product can be done before the earthquake, and the answer is still exact.
#
# **Words this notebook defines:** snapshot, proper orthogonal decomposition (POD), mode, energy, projection error,
# full-order model (FOM), data-driven (non-intrusive) ROM, radial basis function (RBF) interpolation, Galerkin
# projection (intrusive ROM), hyper-reduction, discrete empirical interpolation (DEIM), offline and online phases,
# quantity of interest (QoI), goal-oriented inference.

# %% [markdown]
# **Problems in this notebook**
#
# | problem | unknowns | data or output | forward model | linear? |
# |---|---|---|---|---|
# | POD and ROMs (sections 1-4) | $\ln a\sigma(z)$ on 40 patches | afterslip history of the 31 patches that slip, 40 epochs (day 1 to year 1) | 1-D antiplane fault, $v = 2v_0\sinh(\tau/a\sigma)$, $\dot\tau = \mathbf K v$ (`bayeskit.afterslip`) | nonlinear |
# | real-time inference (sections 5-6) | afterslip $s(z, t)$ on 40 patches × 40 epochs = 1,600 numbers, no friction law (kinematic), Gaussian prior | 30 GPS stations × 40 epochs = 1,200 numbers, σ = 2 mm | $\mathbf d_k = \mathbf G\,\mathbf s_k$ at every epoch | linear |
#
# Training and test runs: the cached `training_set_fault(1500, seed=0)`; runs 1-1,200 build the reduced models, runs
# 1,201-1,500 test them. Errors are surface displacements in units of the GPS noise σ = 2 mm, as in 3.05.

# %%
import sys, pathlib, time, warnings
sys.path.insert(0, str(pathlib.Path.cwd().parent))
import numpy as np
import matplotlib.pyplot as plt
import bayeskit as bk
from bayeskit import problems as P, samplers as SM, diagnostics as D, plotting as BP, style as S
from bayeskit import afterslip as A
import jax, jax.numpy as jnp
import diffrax as dfx
from scipy.interpolate import RBFInterpolator
from scipy.linalg import cho_factor, cho_solve

T0 = time.time()
SIG = 2.0                                                    # GPS noise (mm)
C_FULL, C_GAL, C_DEIM, C_RBF = S.INK, S.BLUE, S.VIOLET, S.ORANGE

la_all, slip_all, surf_all, setup = A.training_set_fault(1500, seed=0)
zc, free, K, G40 = setup["zc"], setup["free"], setup["K"], setup["G"]
Kff, dtf, Gf = K[np.ix_(free, free)], setup["dtau"][free], G40[:, free]
sf = slip_all[:, :, free]                                    # (run, epoch, free patch), metres
T = A.T_OBS
TR, TE = np.arange(1200), np.arange(1200, 1500)
nf = int(free.sum())
print(f"{nf} free patches, {len(T)} epochs; snapshot matrix from the training runs: {len(TR) * len(T):,} profiles of {nf} numbers")


def surf_rms(s_pred, s_true):
    """rms surface error (mm) for free-patch slip arrays (..., nf)."""
    return np.sqrt(np.mean((1000.0 * (s_pred - s_true) @ Gf.T) ** 2))


# %% [markdown]
# ## 1. Snapshots and POD: how many shapes does afterslip need?
#
# **In plain words.** Take every slip profile the solver produced (every run, every epoch) and stack them. If afterslip
# always looks like some mixture of a few basic shapes, a few numbers per profile are enough: the weights of those
# shapes. The singular value decomposition (SVD, 2.06) finds the best shapes, ordered from most to least important.
#
# **Precisely.** Collect the snapshots as rows of $\mathbf S\in\mathbb R^{M\times N_f}$ ($M$ = runs × epochs) and take
# the SVD $\mathbf S = \mathbf U\boldsymbol\Sigma\mathbf V^\top$. The first $r$ columns of $\mathbf V$ form the POD
# basis $\boldsymbol\Psi_r$ (the *modes*). Any profile is approximated by its projection
# $\mathbf s \approx \boldsymbol\Psi_r\boldsymbol\Psi_r^\top\mathbf s$, and over the snapshots the squared error equals
# the discarded "energy" $\sum_{k>r}\sigma_k^2$. We do not subtract a mean, so the zero profile at $t = 0$ is
# represented exactly. POD is the same algebra as principal component analysis (3.01) applied to solution snapshots.
#
# "Best" has a precise meaning here: the projection $\boldsymbol\Psi_r\boldsymbol\Psi_r^\top\mathbf s$ is the r-mode
# profile closest to $\mathbf s$ in slip (the state), summed over patches. An inversion sees the surface instead,
# $\mathbf G_f\mathbf s$. The r-mode profile that best reproduces the *surface* displacement uses different
# coefficients, $\mathbf b^\star = \arg\min_{\mathbf b}\lVert\mathbf G_f\boldsymbol\Psi_r\mathbf b - \mathbf G_f\mathbf s\rVert$
# (a least-squares problem, 2.07), and its surface error can only be smaller. Both are computed below.

# %%
Smat = sf[TR].reshape(-1, nf)
_, sv, Vt = np.linalg.svd(Smat, full_matrices=False)
Vt *= np.sign(Vt.sum(axis=1))[:, None]                       # a mode's sign is arbitrary; make each sum positive
energy_left = 1 - np.cumsum(sv ** 2) / np.sum(sv ** 2)
RS = np.arange(1, 17)
proj_err = [surf_rms(sf[TE] @ Vt[:r].T @ Vt[:r], sf[TE]) / SIG for r in RS]   # state (slip) projection


def surface_best(r):
    """The r-mode slip whose surface displacement is closest to the true one: Psi b*, b* = pinv(G Psi) G s."""
    Psi = Vt[:r].T
    M = Psi @ np.linalg.pinv(Gf @ Psi, rcond=1e-10) @ Gf       # (nf, nf): slip -> best r-mode slip for the surface
    # rcond drops directions whose surface signature is below 1e-10 of the largest; without it round-off in a nearly
    # singular G Psi makes the error grow again with r, although the best fit over nested sets of modes cannot get worse
    return sf[TE] @ M.T


surf_best_err = [surf_rms(surface_best(r), sf[TE]) / SIG for r in RS]
for r in [1, 2, 4, 8, 12]:
    print(f"r = {r:2d}: energy left {energy_left[r - 1]:.1e}; held-out rms surface error of the state projection "
          f"{proj_err[r - 1]:.3f} σ, of the best surface fit with the same modes {surf_best_err[r - 1]:.1e} σ")

# %%
fig, axs = plt.subplots(1, 3, figsize=(14.5, 3.9))
axs[0].semilogy(RS, energy_left[:len(RS)], "o-", color=S.INK, ms=4)
axs[0].set_xlabel("number of modes r"); axs[0].set_ylabel("share of energy not captured")
axs[0].set_title("Singular values: energy left out")
for k, c in enumerate([S.BLUE, S.ORANGE, S.AQUA, S.VIOLET]):
    full = np.full(len(zc), np.nan)
    full[free] = Vt[k]                                        # NaN on pinned patches: the line breaks across 2-11 km
    axs[1].plot(full, zc, color=c, lw=1.6, marker="o", ms=2.5, label=f"mode {k + 1}")
axs[1].axhspan(zc[~free].min() - 0.5, zc[~free].max() + 0.5, color=S.GRID, alpha=0.6, lw=0)
axs[1].axvline(0, color=S.AXIS, lw=0.8)
axs[1].invert_yaxis(); axs[1].set_xlabel("mode value"); axs[1].set_ylabel("depth (km)")
axs[1].set_title("The first four POD modes"); axs[1].legend(loc="lower right")
axs[2].semilogy(RS, proj_err, "o-", color=C_GAL, ms=4, label="POD projection of the slip")
axs[2].semilogy(RS, surf_best_err, "s--", color=S.ORANGE, ms=3.5, label="best surface fit, same r modes")
axs[2].axhline(1.0, color=S.INK2, lw=0.8, ls=":")
axs[2].text(RS[-1], 0.9, "error = GPS noise", ha="right", va="top", fontsize=8, color=S.INK2)
axs[2].set_xlabel("number of modes r"); axs[2].set_ylabel("rms surface error / σ (held-out runs)")
axs[2].set_ylim(1e-4, None)                                   # below this: smaller than the solver's own accuracy
axs[2].set_title("Surface error of r-mode slip"); axs[2].legend(fontsize=8, loc="upper right")
fig.tight_layout()
bk.savefig(fig, "3-06_pod")
plt.show()

# %% [markdown]
# **How to read it.** Left: each added mode removes a share of the energy; the curve falls by orders of magnitude within
# a handful of modes. Middle: the modes live on the free patches only (the grey band is pinned). Mode 1 is close to
# the average afterslip shape: slip on the two shallowest patches and a bulge just below the rupture. The next modes are
# corrections that trade slip between the shallow patches and the deep bulge and move the bulge up or down the fault.
# Right: the same thing in the units of an inversion, the surface error of r-mode approximations of held-out runs. The
# blue curve projects the slip onto the modes, the best r-mode slip in the slip norm; it is a lower bound for that norm,
# not for the surface error. The orange curve picks the coefficients to fit the surface instead and is lower (values
# printed above): measured at the surface, an r-mode model can beat the POD projection by a wide margin, because 30
# stations see only a few combinations of the slip. The axis stops at $10^{-4}$ σ: the two forms of the full model in
# section 3 differ from the cached runs by less than 0.001 mm (printed there), a few $10^{-4}$ σ, so smaller differences are below
# the solver's own accuracy. The blue curve is still the natural reference for a reduced model that solves for slip,
# such as the Galerkin model of section 3.
#
# **Worked example.** One held-out run at year 1: its POD coefficients and the error of its r-mode reconstruction.

# %%
i0 = TE[0]
s1 = sf[i0, -1]
for r in [2, 4, 8]:
    b = Vt[:r] @ s1
    print(f"r = {r}: coefficients {np.round(b, 3)}; reconstruction rms slip error {1000 * np.sqrt(np.mean((Vt[:r].T @ b - s1) ** 2)):.1f} mm "
          f"(slip rms {1000 * np.sqrt(np.mean(s1 ** 2)):.0f} mm)")

# %% [markdown]
# ## 2. A data-driven ROM: interpolate the POD coefficients
#
# **In plain words.** If every solution is a few POD coefficients, a ROM can skip the equations altogether: learn the
# map from the input parameters to those coefficients from the training runs, and interpolate for new inputs. This is
# what Magen et al. (2026) did for slow-slip cycles in Cascadia, with RBF interpolation. It never touches the solver
# code, which is why it is called *non-intrusive*.
#
# **Precisely.** Each run's whole history (40 epochs × 31 patches) is one vector; POD over runs (mean removed) gives
# $r = 16$ coefficients per run. The input, a 40-value profile, is reduced to its leading $k_x$ principal components.
# A radial basis function (RBF) interpolant predicts at a new input by a weighted sum of bumps centred on the training
# inputs, each depending only on the distance to its centre, with weights chosen so that the training runs are
# reproduced exactly. Here the bump is the thin-plate spline $r^2\log r$ (`scipy.interpolate.RBFInterpolator`, plus a
# linear polynomial), and it maps the $k_x$ input numbers to the 16 coefficients. We vary $k_x$, because an interpolant has to fill a $k_x$-dimensional space
# with training runs.

# %%
Yflat = sf.reshape(len(sf), -1)
xm = la_all[TR].mean(0)
_, sx, Vx = np.linalg.svd(la_all[TR] - xm, full_matrices=False)
R_OUT = 16
KXS = [2, 4, 8, 12, 16, 24]


def rbf_rom(idx_tr, kx, r=R_OUT):
    ym = Yflat[idx_tr].mean(0)
    _, _, Vy = np.linalg.svd(Yflat[idx_tr] - ym, full_matrices=False)
    Zx = (la_all - xm) @ Vx[:kx].T
    rb = RBFInterpolator(Zx[idx_tr], (Yflat[idx_tr] - ym) @ Vy[:r].T, kernel="thin_plate_spline", degree=1)
    return lambda la_: (ym + rb(np.atleast_2d((la_ - xm) @ Vx[:kx].T)) @ Vy[:r]).reshape(-1, len(T), nf)


rbf_err = {}
for n in [300, 1200]:
    rbf_err[n] = [surf_rms(rbf_rom(TR[:n], kx)(la_all[TE]), sf[TE]) / SIG for kx in KXS]
    print(f"{n:5d} training runs: rms surface error / σ for k_x = {KXS}: {np.round(rbf_err[n], 2)}")
left = 1 - np.cumsum(sx ** 2) / np.sum(sx ** 2)
print("share of input variance left out by k_x components: " + ", ".join(f"k_x = {k}: {left[k - 1]:.3f}" for k in KXS))
rom_rbf = rbf_rom(TR, 16)
t1 = time.time()
for i in TE[:50]:
    rom_rbf(la_all[i])
t_rbf = (time.time() - t1) / 50
print(f"one RBF-ROM evaluation (k_x = 16): {1000 * t_rbf:.2f} ms")

# %% [markdown]
# Magen et al. (2026, p. 1) report: "Achieving a speedup of ∼ 360 000× compared to volumetric simulations, the ROMs
# enable comprehensive parameter exploration and Bayesian Markov chain Monte Carlo (MCMC) inversion." Their inputs are
# few (p. 4): "In this work we consider a 2-D parameter space". Our input is a whole profile.

# %%
fig, axs = plt.subplots(1, 2, figsize=(12.0, 3.9))
for n, c, mk in [(300, S.MUTED, "s"), (1200, C_RBF, "o")]:
    axs[0].semilogy(KXS, rbf_err[n], marker=mk, color=c, lw=1.6, label=f"{n} training runs")
axs[0].axhline(1.0, color=S.INK2, lw=0.8, ls=":")
axs[0].set_xlabel("input dimensions k_x (principal components of the profile)")
axs[0].set_ylabel("rms surface error / σ (held-out runs)"); axs[0].set_title("Data-driven ROM: error against input dimension")
axs[0].legend()
j_st = 8
s_rb = rom_rbf(la_all[i0])[0]
axs[1].plot(T * 365.25, 1000 * sf[i0] @ Gf[j_st], color=C_FULL, lw=2.6, alpha=0.35, label="solver")
axs[1].plot(T * 365.25, 1000 * s_rb @ Gf[j_st], color=C_RBF, lw=1.4, ls="--", label="RBF ROM, k_x = 16")
axs[1].set_xscale("log"); axs[1].set_xlabel("days after the earthquake")
axs[1].set_ylabel(f"displacement at {setup['x_obs'][j_st]:.1f} km (mm)"); axs[1].set_title("One held-out run at one station")
axs[1].legend()
fig.tight_layout()
bk.savefig(fig, "3-06_data_driven_rom")
plt.show()

# %% [markdown]
# **How to read it.** Left: with few input dimensions the interpolant cannot tell profiles apart (the left points),
# because the discarded components of the friction profile still change the afterslip. Adding dimensions helps until
# the curves flatten, above the dotted line: the extra dimensions carry information, but the runs are too sparse in a
# space of that many dimensions to use it, and four times as many runs lowers the plateau only part of the way (values
# printed above). Right: one held-out run at one station, where the ROM is close except in the first days.
#
# **The image to keep.** Interpolating outputs works when the inputs are few. A 2-D parameter space can be covered
# densely by simulations; a 31-dimensional friction field cannot. For field inputs the options are a network built for
# fields (3.05) or a ROM that keeps the equations (next section).

# %% [markdown]
# ## 3. An intrusive ROM: project the equations
#
# **In plain words.** Instead of learning outputs, keep solving the friction equations, but only for the weights of the
# r POD shapes. The ODE for 31 slips becomes an ODE for r numbers. Because the physics is still solved, the ROM works for
# any friction profile; the only approximation is that slip must be a mixture of the r shapes.
#
# **Precisely.** On the free patches, pinned patches do not slip, so stress is a linear function of slip,
# $\boldsymbol\tau_f = \Delta\boldsymbol\tau_f + \mathbf K_{ff}\,\mathbf s_f$, and the full-order model (FOM) is
# $\dot{\mathbf s}_f = 2v_0\sinh\big((\Delta\boldsymbol\tau_f + \mathbf K_{ff}\mathbf s_f)/a\sigma\big)$, $\mathbf s_f(0) = 0$.
# Galerkin projection substitutes $\mathbf s_f = \boldsymbol\Psi_r\mathbf b$ and keeps only the part of the equation
# that lies along the modes (multiplying it by $\boldsymbol\Psi_r^\top$), which gives
# $$\dot{\mathbf b} = \boldsymbol\Psi_r^\top\,2v_0\sinh\!\big((\Delta\boldsymbol\tau_f + \mathbf K_{ff}\boldsymbol\Psi_r\mathbf b)/a\sigma\big),\qquad \mathbf b(0) = \mathbf 0 .$$
# The right side still evaluates $\sinh$ on all 31 patches. *Hyper-reduction* removes that cost: DEIM (Chaturantabut &
# Sorensen 2010, cited for the idea only) builds a POD basis $\mathbf U_m$ of the slip-rate snapshots, picks $m$ patches
# greedily, and reconstructs the whole slip-rate vector from those $m$ values,
# $\mathbf v \approx \mathbf U_m(\mathbf P^\top\mathbf U_m)^{-1}\mathbf P^\top\mathbf v$. The ROM then evaluates
# $\sinh$ at $m$ patches only. Both ROMs use the same stiff solver as the FOM (Kvaerno5, an implicit solver suited to
# stiff equations, 3.04; relative tolerance $10^{-5}$).

# %%
v_snap = 2 * 0.03 * np.sinh((dtf[None, None, :] + sf[TR] @ Kff.T) / np.exp(la_all[TR][:, None, free]))
Uv = np.linalg.svd(v_snap.reshape(-1, nf), full_matrices=False)[2].T
print(f"slip-rate snapshots span {v_snap.min():.1e} to {v_snap.max():.1e} m/yr")


def deim_points(U):
    """Greedy DEIM selection: each new point is where the current interpolant of the next basis vector is worst."""
    p = [int(np.argmax(np.abs(U[:, 0])))]
    for j in range(1, U.shape[1]):
        c = np.linalg.solve(U[p, :j], U[p, j])
        p.append(int(np.argmax(np.abs(U[:, j] - U[:, :j] @ c))))
    return np.array(p)


def make_rom(r, m=None):
    Psi = jnp.asarray(Vt[:r].T)
    KP, d0 = jnp.asarray(Kff) @ Psi, jnp.asarray(dtf)
    if m is None:
        rhs_v = lambda b, a: Psi.T @ (2 * 0.03 * jnp.sinh((d0 + KP @ b) / a))
        sel = jnp.arange(nf)
    else:
        p = deim_points(Uv[:, :m])
        M = jnp.asarray(Vt[:r] @ Uv[:, :m] @ np.linalg.inv(Uv[p, :m]))
        KPp, d0p, sel = KP[p], d0[p], jnp.asarray(p)
        rhs_v = lambda b, a: M @ (2 * 0.03 * jnp.sinh((d0p + KPp @ b) / a))

    @jax.jit
    def solve(la_f):
        a = jnp.exp(la_f)[sel]
        sol = dfx.diffeqsolve(dfx.ODETerm(lambda t, b, args: rhs_v(b, a)), dfx.Kvaerno5(), t0=0.0, t1=T[-1], dt0=1e-8,
                              y0=jnp.zeros(r), saveat=dfx.SaveAt(ts=jnp.asarray(T)),
                              stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-7), max_steps=400_000)
        return sol.ys @ Psi.T
    return solve


def run_timed(f, inputs):
    jax.block_until_ready(f(inputs[0]))                      # compile first, then time
    t1 = time.time()
    out = np.stack([np.asarray(f(x)) for x in inputs])
    return out, (time.time() - t1) / len(inputs)


TEST = TE[:20]
fom = jax.jit(lambda l: A.fault_afterslip(l, setup)[:, free])          # bayeskit form: stress and slip on all 40 patches
Kj_, dj_ = jnp.asarray(Kff), jnp.asarray(dtf)


@jax.jit
def fom31(la_f):
    """The same physics as an ODE for the 31 free slips only (the FOM that the ROMs below reduce)."""
    a = jnp.exp(la_f)
    sol = dfx.diffeqsolve(dfx.ODETerm(lambda t, s_, args: 2 * 0.03 * jnp.sinh((dj_ + Kj_ @ s_) / a)), dfx.Kvaerno5(),
                          t0=0.0, t1=T[-1], dt0=1e-8, y0=jnp.zeros(nf), saveat=dfx.SaveAt(ts=jnp.asarray(T)),
                          stepsize_controller=dfx.PIDController(rtol=1e-5, atol=1e-7), max_steps=400_000)
    return sol.ys


s_fom, t_fom = run_timed(fom, [jnp.asarray(la_all[i]) for i in TEST])
s_f31, t_f31 = run_timed(fom31, [jnp.asarray(la_all[i, free]) for i in TEST])
print(f"FOM as in bayeskit (80 unknowns: stress and slip on every patch): {1000 * t_fom:.0f} ms per run; agrees with the "
      f"cache to {1000 * np.abs(s_fom - sf[TEST]).max():.4f} mm")
print(f"FOM as 31 free slips only: {1000 * t_f31:.0f} ms per run; agrees with the cache to "
      f"{1000 * np.abs(s_f31 - sf[TEST]).max():.4f} mm")
gal, deim = {}, {}
for r in [2, 4, 6, 8, 12]:
    out, tt = run_timed(make_rom(r), [jnp.asarray(la_all[i, free]) for i in TEST])
    gal[r] = (surf_rms(out, sf[TEST]) / SIG, tt, surf_rms(sf[TEST] @ Vt[:r].T @ Vt[:r], sf[TEST]) / SIG, out)
    print(f"Galerkin r = {r:2d}: rms surface error {gal[r][0]:.3f} σ (POD projection of the slip {gal[r][2]:.3f} σ), {1000 * tt:.0f} ms per run")
for r, m in [(8, 8), (8, 12), (8, 16)]:
    out, tt = run_timed(make_rom(r, m), [jnp.asarray(la_all[i, free]) for i in TEST])
    deim[(r, m)] = (surf_rms(out, sf[TEST]) / SIG, tt)
    print(f"Galerkin r = {r} + DEIM m = {m:2d}: rms surface error {deim[(r, m)][0]:.3f} σ, {1000 * tt:.0f} ms per run")

# %%
fig, axs = plt.subplots(1, 3, figsize=(15.8, 3.9), gridspec_kw=dict(width_ratios=[1.05, 1.25, 1.05]))
rr = sorted(gal)
axs[0].semilogy(rr, [gal[r][2] for r in rr], "o:", color=S.MUTED, ms=4, label="POD projection of the slip")
axs[0].semilogy(rr, [gal[r][0] for r in rr], "o-", color=C_GAL, ms=4, label="Galerkin ROM")
axs[0].semilogy([8, 8, 8], [deim[(8, m)][0] for m in (8, 12, 16)], "D", color=C_DEIM, ms=5,
                label="Galerkin + DEIM, m = 8, 12, 16")
axs[0].axhline(1.0, color=S.INK2, lw=0.8, ls=":")
axs[0].set_xlabel("number of modes r"); axs[0].set_ylabel("rms surface error / σ (20 held-out runs)")
axs[0].set_title("Intrusive ROM: error against modes"); axs[0].legend(fontsize=7.5)
labs = ["FOM, 80\nunknowns", "FOM, 31\nunknowns", "Galerkin\nr = 8", "Galerkin r = 8\nDEIM m = 16", "RBF ROM\nk_x = 16"]
tvals = [t_fom, t_f31, gal[8][1], deim[(8, 16)][1], t_rbf]
axs[1].bar(labs, [1000 * t for t in tvals], color=[S.MUTED, C_FULL, C_GAL, C_DEIM, C_RBF], width=0.6)
axs[1].set_yscale("log"); axs[1].set_ylabel("time per run (ms)"); axs[1].tick_params(axis="x", labelsize=7.5)
for k, t in enumerate(tvals):
    axs[1].text(k, 1000 * t, f"{1000 * t:.2g}", ha="center", va="bottom", fontsize=8)
axs[1].set_title("Cost of one forward run")
k0 = 0
axs[2].plot(T * 365.25, 1000 * s_fom[k0] @ Gf[j_st], color=C_FULL, lw=2.8, alpha=0.3, label="FOM")
for r, ls in [(2, ":"), (4, "--"), (8, "-")]:
    axs[2].plot(T * 365.25, 1000 * gal[r][3][k0] @ Gf[j_st], color=C_GAL, lw=1.2, ls=ls, label=f"Galerkin r = {r}")
axs[2].set_xscale("log"); axs[2].set_xlabel("days after the earthquake")
axs[2].set_ylabel(f"displacement at {setup['x_obs'][j_st]:.1f} km (mm)"); axs[2].set_title("Same held-out run, Galerkin ROMs")
axs[2].legend(fontsize=7.5)
fig.tight_layout()
bk.savefig(fig, "3-06_galerkin_rom")
plt.show()

# %% [markdown]
# **How to read it.** Left: the Galerkin ROM (blue) follows the POD projection error (grey) down as modes are added,
# several times above it, because the reduced ODE also accumulates error in time. DEIM (diamonds, r = 8) with few interpolation
# points adds error; with 16 of the 31 patches it is as accurate as the Galerkin ROM that evaluates all of them. Middle:
# time per forward run, all measured here (log scale). The first two bars are the same physics: writing the ODE for the
# 31 free slips only (stress follows from slip, pinned patches never move) is an exact reduction and already makes the
# solver several times faster. Against that 31-unknown FOM the Galerkin ROM is no faster here, and DEIM gains little
# (times printed above):
# with so few unknowns the cost of a stiff solve is set by the number of time steps and fixed overheads, not by the size
# of the linear algebra. The RBF ROM, which solves no ODE, is faster by orders of magnitude but, as section 2 showed,
# far less accurate. Right:
# one run at one station; two modes are wrong through most of the year, four come close, eight lie on the FOM.
#
# **Common misreadings.**
# * "The speed-up is the ratio of unknowns." The cost of a stiff solver depends on the number of time steps as much as
#   on the size of the system; at 31 unknowns projection buys accuracy control, not speed. It pays when the FOM has
#   thousands to millions of unknowns (Magen et al. quote 360,000× for a data-driven ROM of a volumetric model).
# * "Reduction means POD." The largest saving in the middle panel came from rewriting the equations exactly
#   (eliminating stress and the pinned patches). Check for exact reductions before building an approximate one.
# * "DEIM is optional." For a nonlinear term it is what makes the ROM's cost independent of the full size; without it
#   every step still touches every patch.

# %% [markdown]
# ## 4. Outside the training range: physics against interpolation
#
# **In plain words.** The intrusive ROM still solves the friction law; the data-driven ROM only interpolates. Give both
# the rough friction profiles of 3.05 (correlation length 3 km instead of 8 km) and compare their errors with those on
# ordinary held-out runs.

# %%
la_ood = A.sample_log_asig(np.random.default_rng(3), 20, zc, ell=3.0)
s_ood, _ = run_timed(fom, [jnp.asarray(l) for l in la_ood])          # truth from the bayeskit solver
rom8, rom12 = make_rom(8), make_rom(12)
ood = {
    "Galerkin r = 8": (gal[8][0], surf_rms(np.stack([np.asarray(rom8(jnp.asarray(l[free]))) for l in la_ood]), s_ood) / SIG),
    "Galerkin r = 12": (gal[12][0], surf_rms(np.stack([np.asarray(rom12(jnp.asarray(l[free]))) for l in la_ood]), s_ood) / SIG),
    "RBF ROM, k_x = 16": (surf_rms(rom_rbf(la_all[TEST]), sf[TEST]) / SIG, surf_rms(rom_rbf(la_ood), s_ood) / SIG),
}
for k, (a_, b_) in ood.items():
    print(f"{k:18s}: rms surface error / σ, ordinary held-out {a_:.3f}, rough profiles {b_:.3f} (ratio {b_ / a_:.1f})")

# %%
fig, ax = plt.subplots(figsize=(7.2, 3.6))
names = list(ood)
x = np.arange(len(names))
ax.bar(x - 0.18, [ood[k][0] for k in names], width=0.34, color=[C_GAL, C_GAL, C_RBF], label="ordinary held-out runs")
ax.bar(x + 0.18, [ood[k][1] for k in names], width=0.34, color=[C_GAL, C_GAL, C_RBF], alpha=0.45, hatch="//",
       edgecolor=S.SURFACE, label="rough profiles (ℓ = 3 km)")
ax.set_yscale("log"); ax.axhline(1.0, color=S.INK2, lw=0.8, ls=":")
ax.set_xticks(x); ax.set_xticklabels(names); ax.set_ylabel("rms surface error / σ")
ax.set_title("Errors inside and outside the training range"); ax.legend(fontsize=8)
fig.tight_layout()
bk.savefig(fig, "3-06_rough_inputs")
plt.show()

# %% [markdown]
# **How to read it.** Solid bars: ordinary held-out runs; hatched: rough profiles. Every method's error grows on the
# rough profiles (ratios printed above), so solving the physics does not make a ROM immune: rough friction produces
# slip shapes that the snapshots did not contain, and no basis built from those snapshots represents them exactly. What
# differs is the level. The Galerkin ROMs stay far below the GPS noise on both sets; the RBF ROM is above it on both.

# %% [markdown]
# ## 5. Real-time Bayesian inference: offline once, online in microseconds
#
# **In plain words.** For a linear forward model with a Gaussian prior and Gaussian noise, the posterior mean is a fixed
# matrix times the data (2.08). That matrix does not depend on the data, so it can be computed before the earthquake.
# When data arrive, inference is one matrix-vector product, and it is exact, not approximate. If only a few quantities
# matter (what a warning needs), precompute the matrix that maps data straight to those quantities: that is
# *goal-oriented* inference.
#
# **Precisely.** With $\mathbf d = \mathbf G\mathbf m + \mathbf e$, $\mathbf m\sim\mathcal N(\mathbf m_0, \mathbf C_m)$,
# $\mathbf e\sim\mathcal N(\mathbf 0, \mathbf C_d)$ and quantities of interest $\mathbf q = \mathbf Q\mathbf m$,
# $$\bar{\mathbf q} = \mathbf Q\mathbf m_0 + \underbrace{\mathbf Q\mathbf C_m\mathbf G^\top(\mathbf G\mathbf C_m\mathbf G^\top + \mathbf C_d)^{-1}}_{\mathbf A_q\ \text{(offline)}}(\mathbf d - \mathbf G\mathbf m_0),\qquad
#   \tilde{\mathbf C}_q = \mathbf Q\mathbf C_m\mathbf Q^\top - \mathbf A_q\mathbf G\mathbf C_m\mathbf Q^\top .$$
# Both $\mathbf A_q$ and $\tilde{\mathbf C}_q$ are data-independent (offline); the online step is $\mathbf A_q\mathbf d$.
# Forming $\mathbf G\mathbf C_m\mathbf G^\top$ needs $\mathbf G^\top$ applied to one unit vector per datum: in a PDE
# model (a forward model given by a partial differential equation, such as the wave equation) each is one adjoint solve
# (2.22), which is why Henneking et al.'s offline phase costs one adjoint wave
# propagation per sensor (their time-invariance lets one solve serve every time sample of a sensor).
#
# **The toy.** Kinematic afterslip: the unknowns are the slip of all 40 patches at all 40 epochs (1,600 numbers; the
# inversion does not know the friction law or which patches are pinned). Prior: mean zero, standard deviation 0.6 m,
# Matérn correlation in depth (ℓ = 8 km, ν = 1.5, 2.12) times a Gaussian correlation in log-time (scale 1 in ln t).
# Coverage, used below, is the share of held-out runs whose true value falls inside the stated interval (2.01, 2.32). Data: 30
# stations × 40 epochs, σ = 2 mm. The truth is a run of the nonlinear friction model, so the prior is not the process
# that made it. Quantities of interest: surface displacement after one year at 60 points between 0.3 and 150 km (most
# of them between stations), and the potency (slip × depth extent, summed over the fault) at every epoch.

# %%
nt, npch = len(T), len(zc)
dz = np.diff(setup["z_edges"])
Cz = A.matern_cov(zc, 8.0, 1.5, 1.0)
lt = np.log(T)
Ct = 0.6 ** 2 * np.exp(-0.5 * (lt[:, None] - lt[None, :]) ** 2)
Cm = np.kron(Ct, Cz)                                          # (1600, 1600), epoch-major ordering
Gbig = np.kron(np.eye(nt), 1000.0 * G40)                      # mm per m, (1200, 1600)
Cd = SIG ** 2 * np.eye(Gbig.shape[0])
x_q = np.geomspace(0.3, 150.0, 60)
Gq = 1000.0 * A.surface_greens(x_q, setup["z_edges"])
Q = np.zeros((60 + nt, nt * npch))
Q[:60, (nt - 1) * npch:] = Gq                                 # displacement at year 1 (mm)
for k in range(nt):
    Q[60 + k, k * npch:(k + 1) * npch] = dz                    # potency at epoch k (m km)
print(f"unknowns {Cm.shape[0]}, data {Gbig.shape[0]}, quantities of interest {Q.shape[0]}")

t1 = time.time()
CmGt = Cm @ Gbig.T
Sfac = cho_factor(Gbig @ CmGt + Cd)
Aq = cho_solve(Sfac, (Q @ CmGt).T).T                          # (n_q, n_d)
Kgain = cho_solve(Sfac, CmGt.T).T                             # full gain (n_m, n_d), for the slip itself
Cq = Q @ Cm @ Q.T - Aq @ (Q @ CmGt).T
t_off = time.time() - t1
sd_q = np.sqrt(np.diag(Cq))
print(f"offline phase (once): {t_off:.2f} s")


def direct_posterior_q(d):
    """What one would do without the offline phase: build and solve the whole linear system for this data set."""
    CmGt_ = Cm @ Gbig.T
    fac = cho_factor(Gbig @ CmGt_ + Cd)
    return Q @ (CmGt_ @ cho_solve(fac, d))


ir = TE[5]
s_true = slip_all[ir]                                         # (40 epochs, 40 patches), from the friction model
rng = np.random.default_rng(0)
d_new = Gbig @ s_true.ravel() + SIG * rng.standard_normal(Gbig.shape[0])
reps = 2000
t1 = time.perf_counter()
for _ in range(reps):
    q_on = Aq @ d_new
t_on = (time.perf_counter() - t1) / reps
t1 = time.perf_counter()
for _ in range(200):
    m_on = Kgain @ d_new
t_on_m = (time.perf_counter() - t1) / 200
t1 = time.perf_counter()
for _ in range(3):
    q_dir = direct_posterior_q(d_new)
t_dir = (time.perf_counter() - t1) / 3
print(f"online, quantities of interest: {1e6 * t_on:.1f} µs per data set; online, all 1,600 slip values: {1e6 * t_on_m:.0f} µs")
print(f"direct solve for one data set: {1000 * t_dir:.0f} ms  ->  ratio direct / online (QoI) {t_dir / t_on:,.0f}")
print(f"online vs direct: largest difference {np.abs(q_on - q_dir).max():.2e} (largest |q| {np.abs(q_dir).max():.1f}); "
      f"QoI from the full slip mean {np.abs(Q @ m_on - q_on).max():.2e}")

q_true = Q @ s_true.ravel()
inside = []
for i in TE[:100]:
    d_i = Gbig @ slip_all[i].ravel() + SIG * np.random.default_rng(1000 + i).standard_normal(Gbig.shape[0])
    inside.append(np.abs(Aq @ d_i - Q @ slip_all[i].ravel()) <= 1.645 * sd_q)
inside = np.array(inside)
print(f"100 held-out runs: true QoI inside the 90 % interval, displacement {inside[:, :60].mean():.2f}, potency {inside[:, 60:].mean():.2f}")

# %%
fig, axs = plt.subplots(1, 3, figsize=(14.8, 3.9), gridspec_kw=dict(width_ratios=[1, 1.15, 1.15]))
labs = ["offline\n(once)", "direct solve\n(per data set)", "online, all slip\n(per data set)", "online, QoI\n(per data set)"]
tv = [t_off, t_dir, t_on_m, t_on]
axs[0].bar(labs, tv, color=[S.MUTED, S.INK2, S.BLUE, S.BLUE], width=0.62)
axs[0].set_yscale("log"); axs[0].set_ylabel("seconds"); axs[0].tick_params(axis="x", labelsize=7.5)
for k, t in enumerate(tv):
    axs[0].text(k, t, f"{t:.1e} s", ha="center", va="bottom", fontsize=7.5)
axs[0].set_title("Where the time goes")
dq = q_on[:60] - q_true[:60]                                 # the displacement itself spans hundreds of mm, the band a few
axs[1].fill_between(x_q, dq - 1.645 * sd_q[:60], dq + 1.645 * sd_q[:60], color=S.BLUE, alpha=0.2, lw=0, label="posterior 90 %")
axs[1].plot(x_q, dq, color=S.BLUE, lw=1.5, label="posterior mean (online) − truth")
axs[1].axhline(0, color=S.INK, lw=1.0, ls="--", label="truth (friction model)")
lo_ = (dq - 1.645 * sd_q[:60]).min()
axs[1].plot(setup["x_obs"], np.full(30, lo_ * 1.15), "|", color=S.INK2, ms=10, label="GPS stations")
axs[1].set_xscale("log"); axs[1].set_xlabel("distance from the fault (km)"); axs[1].set_ylabel("displacement after 1 yr, minus truth (mm)")
axs[1].set_title("Goal 1: displacement between and beyond stations"); axs[1].legend(fontsize=7.5, loc="upper right")
qp = q_on[60:]
axs[2].fill_between(T * 365.25, qp - 1.645 * sd_q[60:], qp + 1.645 * sd_q[60:], color=S.BLUE, alpha=0.2, lw=0)
axs[2].plot(T * 365.25, qp, color=S.BLUE, lw=1.5, label="posterior mean ± 90 %")
axs[2].plot(T * 365.25, q_true[60:], color=S.INK, lw=1.2, ls="--", label="truth")
axs[2].set_xscale("log"); axs[2].set_xlabel("days after the earthquake"); axs[2].set_ylabel("potency (m km)")
axs[2].set_title("Goal 2: afterslip potency through time"); axs[2].legend(fontsize=7.5)
fig.tight_layout()
bk.savefig(fig, "3-06_realtime")
plt.show()

# %% [markdown]
# **How to read it.** Left: the one-time offline cost, the cost of solving each new data set from scratch, and the
# online cost (log scale; values and ratio printed above). Middle and right: the online answer for one held-out run,
# with intervals from the precomputed covariance. The displacement after one year (middle) is so well constrained that
# the figure shows the online answer minus the truth; the potency (right) is shown as it is. The printed check shows that the online numbers equal the direct
# solve to rounding error: the speed comes from reorganising exact algebra, not from approximating it. The coverage over
# 100 held-out runs (printed) tells whether this Gaussian prior, which is not the friction model, still gives honest
# intervals for these two goals.
#
# **Common misreadings.**
# * "Real time needs a surrogate." Not for a linear problem with fixed hyperparameters: the exact posterior is already a
#   matrix-vector product. Surrogates and ROMs are needed when the forward model is nonlinear or the hyperparameters
#   are sampled, because then the gain changes with every candidate.
# * "Offline cost is unimportant." It is paid once per network and prior, before the event, and it is the expensive
#   part (printed). Henneking et al. (2025, p. 1) report that "The offline component requires just one adjoint wave
#   propagation per sensor", run on 43,520 GPUs.

# %% [markdown]
# ## 6. Where the field is
#
# Quotes are verbatim, with page numbers.
#
# * **Exact, real-time, goal-oriented.** Henneking, Venkat & Ghattas (2026, p. 1) describe the split used in section 5:
#   they exploit "the autonomous structure of the forward model to decompose the solution of the inverse problem into a
#   one-time-only offline phase in which the PDE model is solved a limited number of times (equal to the number of
#   sensors), and an online phase that maps well onto GPUs and computes the parameter inference and prediction of
#   quantities of interest in real time, given observational data." At Cascadia scale, Henneking et al. (2025, p. 1):
#   "given real-time data, the online component exactly solves the Bayesian inverse and forecasting problems in 0.2
#   seconds on a modest GPU system, a ten-billion-fold speedup."
# * **ROMs for slow slip.** Magen, May & Gabriel (2026, p. 1), quoted in section 2: POD plus RBF interpolation over a
#   2-D parameter space, used inside MCMC.
# * **ROMs for Green's functions.** Rekoske, May & Gabriel (2024, p. 1): "The ROM constructed for this region enables
#   rapid computation (0.001 CPU hours) of complete, high-resolution (500 m spacing), 0.5 Hz surface velocity
#   wavefields that are accurate for a shortest wavelength of 1.0 km for a single moment tensor source."
# * **Galerkin projection and DEIM** are classical model-reduction tools (e.g. Chaturantabut & Sorensen 2010), cited for
#   the idea only.
#
# **In practice.**
# * An intrusive ROM still solves the friction law, so it needs no new training when the friction field changes, only a
#   basis that contains the slip shapes; a data-driven ROM or a network has to see similar inputs in training.
# * The offline-online split of section 5 is exact only for a linear forward model with fixed hyperparameters and a
#   Gaussian prior. If noise scales or smoothing weights are sampled, or the prior is bounded (for example slip that
#   must be positive), the gain changes with every draw, and the split holds only conditionally on them.
# * For rapid tsunami or slip estimates from a fixed sensor network, section 5 is the whole recipe at toy scale: a linear
#   map, a prior, a gain computed before the event, and a goal-oriented product when data arrive.
#
# Next: **3.07 · Inference with surrogates: putting it together**.

# %%
print(f"notebook run time: {time.time() - T0:.0f} s")
