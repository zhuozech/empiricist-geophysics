# %% [markdown]
# # 3.05 · Neural operators: learning the map from a friction profile to afterslip
#
# *Topic 3 is under development: content and numbering may still change.*
#
# **Before this:** 2.12 (Gaussian-process priors on fields), 2.13 (model error), 3.01 (surrogates, $\mathbf C_p$),
# 3.02 (neural networks, Adam) · **You will be able to:** explain what an operator and a Fourier layer are; train an
# FNO and judge it by learning curves; test resolution transfer and out-of-distribution inputs; use an FNO inside an
# inversion with and without its error covariance · **Optional sections:** 6 (frequency bias), 9 (the field).
#
# Notebooks 3.01-3.03 fitted surrogates to a forward model with three input numbers. The fault model of Topic 3 takes a
# whole function, the friction parameter $a\sigma(z)$ at every depth, and returns another function, the afterslip
# $s(z, t)$. A *neural operator* is a network built to learn such function-to-function maps. This notebook writes a
# Fourier neural operator (FNO) from scratch in JAX, trains it on the cached fault runs, and then asks the questions that
# decide whether it can replace the solver inside an inversion: how many runs it needs, whether it works on a finer grid
# than it was trained on, which details it gets wrong, what happens outside its training range, and what its errors do
# to a posterior.
#
# **Words this notebook defines:** operator, function space, discretisation, neural operator, Fourier neural operator
# (FNO), lifting and projection, spectral convolution, Fourier modes $k_{\max}$, padding, resolution (discretisation)
# invariance, zero-shot super-resolution, frequency (spectral) bias, out-of-distribution input, surrogate-error
# covariance $\mathbf C_p$, Jacobian accuracy.

# %% [markdown]
# **Problems in this notebook**
#
# | problem | unknowns | data | forward model | linear? |
# |---|---|---|---|---|
# | operator learning (sections 1-7) | input function $\ln a\sigma(z)$ on 40 patches, 0-40 km (31 of them slip after the earthquake) | output function: afterslip $s(z, t)$ on the 40 patches at 8 epochs between day 1.8 and year 1 | 1-D antiplane fault with stress-driven afterslip, $v = 2v_0\sinh(\tau/a\sigma)$, $\dot\tau = \mathbf K v$ (`bayeskit.afterslip`) | nonlinear |
# | inversion (section 8) | $\ln a\sigma(z)$, 40 values, Gaussian-process prior | 30 GPS stations × 8 epochs = 240 numbers, σ = 2 mm | the same solver, or the FNO in its place | nonlinear |
#
# Training data: the 1,500 cached runs of `training_set_fault(1500, seed=0)`, whose inputs are drawn from a Matérn
# prior (a Gaussian-process prior with a Matérn kernel, 2.12; mean $\ln 0.8$ MPa, standard deviation 0.5, correlation
# length ℓ = 8 km, smoothness ν = 1.5). Runs 1-1,200 are for training, runs 1,201-1,500
# are never seen in training. Every error is judged in the units that matter for an inversion: the surface displacement
# it causes, divided by the GPS noise σ = 2 mm.

# %%
import os
os.environ["EQX_ON_ERROR"] = "nan"      # section 8: let rejected solver steps carry NaN instead of raising while differentiating
import sys, pathlib, time, warnings
sys.path.insert(0, str(pathlib.Path.cwd().parent))
import numpy as np
import matplotlib.pyplot as plt
import bayeskit as bk
from bayeskit import problems as P, samplers as SM, diagnostics as D, plotting as BP, style as S
from bayeskit import afterslip as A
import jax, jax.numpy as jnp
import diffrax as dfx
import numpyro, numpyro.distributions as dist
from numpyro.infer import MCMC, NUTS
from scipy.fft import dct
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel

T0 = time.time()
SIG = 2.0                                                    # GPS noise (mm)
C_TRUE, C_FNO, C_MLP, C_GP, C_CP = S.INK, S.BLUE, S.ORANGE, S.AQUA, S.VIOLET

la_all, slip_all, surf_all, setup = A.training_set_fault(1500, seed=0)
EP = np.arange(4, 40, 5)                                     # 8 of the 40 cached epochs
T8 = A.T_OBS[EP]
Y_all = slip_all[:, EP, :]                                   # afterslip (run, epoch, patch), metres
zc, free, G40 = setup["zc"], setup["free"], setup["G"]
TR, TE = np.arange(1200), np.arange(1200, 1500)
print(f"{len(la_all)} cached runs; epochs (days): {np.round(T8 * 365.25, 1)}")
print(f"{free.sum()} of {len(zc)} patches slip after the earthquake (stress went up); free patches from "
      f"{zc[free].min():.1f} to {zc[free].max():.1f} km, pinned: {np.round(zc[~free], 1)}")
print(f"afterslip at 1 yr, held-out runs: largest {1000 * Y_all[TE, -1].max():.0f} mm; "
      f"surface displacement at 1 yr: {1000 * (Y_all[TE, -1] @ G40.T).min():.0f} to {1000 * (Y_all[TE, -1] @ G40.T).max():.0f} mm")

# %% [markdown]
# ## 1. An operator maps a function to a function
#
# **In plain words.** A forward model that takes three numbers and returns forty is a function. The fault model takes a
# *curve* (how strong the friction is at every depth) and returns other curves (how much the fault slips at every depth
# and time). A map from curves to curves is called an operator. On a computer every curve is stored as a list of
# numbers, one per patch, but the list is only a sampling of the curve: the same physics sampled on 80 patches is the
# same operator.
#
# **Precisely.** The solution operator of the afterslip model is
# $\mathcal G: \ln a\sigma(\cdot) \mapsto s(\cdot, t)$, a map between function spaces (sets whose members are whole
# functions, here all possible friction profiles and all possible slip histories). A discretisation on $N$ patches
# turns it into a map $\mathbb R^N \to \mathbb R^{N\times n_t}$. A plain network (3.02) learns the discretised map, so
# its input layer has exactly $N$ entries. A neural operator is built so that its weights belong to the continuous map:
# the same trained weights can be applied on any grid.

# %%
fig, axs = plt.subplots(1, 3, figsize=(14.5, 3.9))
show = TE[:5]
cols5 = [S.BLUE, S.ORANGE, S.AQUA, S.VIOLET, S.INK2]
for c, i in zip(cols5, show):
    axs[0].plot(np.exp(la_all[i]), zc, color=c, lw=1.5)
    axs[1].plot(1000 * Y_all[i, -1], zc, color=c, lw=1.5)
axs[1].plot(1000 * setup["s_co"], zc, color=S.MUTED, lw=1.2, ls="--", label="coseismic slip")
for ax in axs[:2]:
    ax.axhspan(zc[~free].min() - 0.5, zc[~free].max() + 0.5, color=S.GRID, alpha=0.6, lw=0)
    ax.invert_yaxis(); ax.set_ylabel("depth (km)")
axs[0].set_xscale("log"); axs[0].set_xlabel("aσ (MPa)")
axs[0].set_xticks([0.3, 0.5, 1.0, 2.0]); axs[0].set_xticklabels(["0.3", "0.5", "1", "2"]); axs[0].minorticks_off()
axs[0].set_title("Input functions: five friction profiles")
axs[1].set_xlabel("slip (mm)"); axs[1].set_title("Output functions: afterslip after 1 yr")
axs[1].text(0.97, 0.03, "grey band: patches pinned\n(their stress dropped)", transform=axs[1].transAxes, ha="right",
            va="bottom", fontsize=8, color=S.INK2)
axs[1].legend(loc="lower right", bbox_to_anchor=(1.0, 0.13))
i0 = show[0]
for k, (j, ls) in enumerate(zip([2, 10, 20], ["-", "--", ":"])):
    axs[2].plot(T8 * 365.25, 1000 * Y_all[i0] @ G40[j], color=cols5[0], ls=ls, marker="o", ms=3,
                label=f"station at {setup['x_obs'][j]:.1f} km")
axs[2].set_xscale("log"); axs[2].set_xlabel("days after the earthquake"); axs[2].set_ylabel("surface displacement (mm)")
axs[2].set_title("What GPS sees (blue profile)"); axs[2].legend()
fig.tight_layout()
bk.savefig(fig, "3-05_operator")
plt.show()

# %% [markdown]
# **How to read it.** Each colour is one run: a friction profile (left) and the afterslip it produces after one year
# (middle). Afterslip concentrates just below the rupture and on the two shallowest patches, where the coseismic slip
# tapers toward the surface and the earthquake raised the stress; the patches between 2 and 11 km (grey band) do not
# slip at all, because their stress dropped. How much
# each depth slips depends on the whole friction profile, not only on aσ at that depth. The right panel is the part an inversion
# sees: the surface displacement of the blue run at three stations, growing roughly with the logarithm of time.
#
# **The image to keep.** One input curve in, a family of output curves out. The surrogate has to learn how the *shape*
# of the input curve moves slip up and down the fault, not a fixed list of 40 numbers.

# %% [markdown]
# ## 2. Anatomy of a Fourier layer
#
# **In plain words.** A Fourier layer splits a curve into waves of different lengths, keeps only the longest few,
# multiplies each by a learned number, and adds the waves back together. A second, simpler path changes each depth on
# its own. Long waves carry information across the whole fault (slip at 20 km depends on friction at 15 km); the local
# path handles what happens at each depth. Because a wave of a given length is the same wave on any grid, the learned
# numbers do not depend on how finely the curve is sampled.
#
# **Precisely.** With $v_\ell(z) \in \mathbb R^{W}$ the $W$ channels at layer $\ell$, an FNO computes
#
# $$v_0(z) = \mathbf P\,[\,a(z),\ z/40\,],\qquad
#   v_{\ell+1}(z) = \phi\Big(\mathbf W_\ell\, v_\ell(z) + \mathcal F^{-1}\big[\mathbf R_\ell(k)\,\mathcal F[v_\ell](k)\big](z)\Big),\qquad
#   s(z) = \mathbf Q\,v_L(z),$$
#
# where $\mathcal F$ is the Fourier transform over depth (it rewrites a curve as a sum of waves; wavenumber $k$ counts
# how many waves fit along the fault), $\mathbf R_\ell(k)\in\mathbb C^{W\times W}$ is learned for the
# lowest $k_{\max}$ wavenumbers and zero above, $\mathbf W_\ell$ is a pointwise linear map, $\phi$ an activation (GELU,
# a smooth version of the ReLU of 3.02),
# $\mathbf P$ the *lifting* from the input channels (here $\ln a\sigma$ and the depth coordinate) to $W$ channels and
# $\mathbf Q$ the *projection* to the 8 output epochs. The FFT assumes a periodic curve; padding the depth axis with
# zeros (here by a quarter of its length) keeps the bottom of the fault from leaking into the top. The FFT (fast Fourier
# transform) is the standard algorithm that computes $\mathcal F$ on a grid. The architecture is Li et al. (2021),
# cited for the idea, not quoted.
#
# The figure applies only the spectral half of one layer to one input profile: transform, keep $k_{\max}$ modes,
# transform back.

# %%
a0 = la_all[TE[0]] - la_all[TE[0]].mean()
N, pad = len(a0), len(a0) // 4
ap = np.r_[a0, np.zeros(pad)]
ah = np.fft.rfft(ap, norm="forward")
z_pad = np.arange(N + pad) + 0.5
fig, axs = plt.subplots(1, 3, figsize=(14.5, 3.6))
axs[0].plot(z_pad, ap, color=S.INK, lw=1.6, label="input (padded with zeros below 40 km)")
axs[0].axvspan(40, 50, color=S.GRID, lw=0)
axs[0].set_xlabel("depth (km)"); axs[0].set_ylabel("ln aσ minus its mean"); axs[0].legend(loc="lower right")
axs[0].set_title("One input curve")
axs[1].bar(np.arange(len(ah)), np.abs(ah), color=S.MUTED, width=0.8)
for km, c in [(4, S.ORANGE), (12, S.BLUE)]:
    axs[1].axvline(km - 0.5, color=c, lw=1.4, ls="--", label=f"k_max = {km}")
axs[1].set_xlabel("wavenumber k (cycles per 50 km)"); axs[1].set_ylabel("|Fourier coefficient|")
axs[1].set_title("Its Fourier coefficients"); axs[1].legend()
axs[2].plot(z_pad, ap, color=S.INK, lw=1.0, alpha=0.6, label="input")
for km, c in [(4, S.ORANGE), (12, S.BLUE)]:
    keep = np.where(np.arange(len(ah)) < km, ah, 0)
    rec = np.fft.irfft(keep, n=N + pad, norm="forward")
    axs[2].plot(z_pad, rec, color=c, lw=1.6, label=f"keep k < {km}")
    print(f"k_max = {km:2d}: rms difference to the input over 0-40 km {np.sqrt(np.mean((rec[:N] - a0) ** 2)):.3f} "
          f"(input rms {np.sqrt(np.mean(a0 ** 2)):.3f})")
axs[2].axvspan(40, 50, color=S.GRID, lw=0)
axs[2].set_xlabel("depth (km)"); axs[2].set_title("Back to depth: only long waves survive"); axs[2].legend(loc="lower right")
fig.tight_layout()
bk.savefig(fig, "3-05_fourier_layer")
plt.show()

# %% [markdown]
# **How to read it.** The input curve (left) has most of its variation in the lowest wavenumbers (middle), because the
# prior has an 8 km correlation length. Keeping the lowest 12 modes reproduces the curve closely; keeping 4 gives only
# its broad shape (right; misfits printed above). In the network, the truncated spectral path is added to the pointwise
# path $\mathbf W_\ell v_\ell$ and passed through the nonlinearity, so fine detail can still appear in the output; the
# spectral path supplies the long-range coupling.
#
# The network below is the whole FNO: about thirty lines. Width $W = 32$, $k_{\max} = 12$, four Fourier layers, and the
# output is multiplied by the known pinned/free mask, so the pinned patches give exactly zero (physics we know need not
# be learned).

# %%
W_CH, KMAX, NLAY, NOUT = 32, 12, 4, len(EP)
mu_in, sd_in = la_all[TR].mean(), la_all[TR].std()
ysd = Y_all[TR].std(axis=(0, 2))                             # one scale per epoch, so early and late epochs weigh alike


def features(la, z):
    """(n, N) profiles -> (n, N, 2) input channels: standardised ln a_sigma and depth / 40 km."""
    a = (jnp.asarray(la) - mu_in) / sd_in
    return jnp.stack([a, jnp.broadcast_to(jnp.asarray(z) / 40.0, a.shape)], -1)


def fno_init(key, kmax=KMAX, width=W_CH):
    ks = jax.random.split(key, 3 + 3 * NLAY)
    p = {"P": jax.random.normal(ks[0], (2, width)) / np.sqrt(2), "Pb": jnp.zeros(width),
         "Q1": jax.random.normal(ks[1], (width, 64)) / np.sqrt(width), "Q1b": jnp.zeros(64),
         "Q2": jax.random.normal(ks[2], (64, NOUT)) / 8.0, "Q2b": jnp.zeros(NOUT), "L": []}
    for l in range(NLAY):
        s = 1.0 / width ** 2
        p["L"].append({"Rr": s * jax.random.uniform(ks[3 + 3 * l], (kmax, width, width)),
                       "Ri": s * jax.random.uniform(ks[4 + 3 * l], (kmax, width, width)),
                       "W": jax.random.normal(ks[5 + 3 * l], (width, width)) / np.sqrt(width), "b": jnp.zeros(width)})
    return p


def fno(p, x, mask):
    """One input (N, 2) -> afterslip (N, 8) in units of ysd. Works for any number of patches N."""
    N = x.shape[0]
    pad = N // 4
    v = x @ p["P"] + p["Pb"]                                  # lift: 2 channels -> W channels, at every depth
    v = jnp.concatenate([v, jnp.zeros((pad, v.shape[1]))], 0)
    for l, lay in enumerate(p["L"]):
        km = lay["Rr"].shape[0]
        vh = jnp.fft.rfft(v, axis=0, norm="forward")[:km]     # lowest k_max modes of every channel
        oh = jnp.einsum("ki,kio->ko", vh, lay["Rr"] + 1j * lay["Ri"])   # learned complex W x W matrix per mode
        oh = jnp.concatenate([oh, jnp.zeros((v.shape[0] // 2 + 1 - km, v.shape[1]))], 0)
        v = jnp.fft.irfft(oh, n=v.shape[0], axis=0, norm="forward") + v @ lay["W"] + lay["b"]
        if l < NLAY - 1:
            v = jax.nn.gelu(v)
    v = v[:N]                                                 # drop the padding
    return (jax.nn.gelu(v @ p["Q1"] + p["Q1b"]) @ p["Q2"] + p["Q2b"]) * mask[:, None]


fno_batch = jax.vmap(fno, in_axes=(None, 0, None))
MASK40 = jnp.asarray(free, dtype=float)
X_all = features(la_all, zc)
Yn_all = jnp.asarray(Y_all / ysd[None, :, None]).transpose(0, 2, 1)       # (run, patch, epoch), scaled
n_par = sum(x.size for x in jax.tree_util.tree_leaves(fno_init(jax.random.PRNGKey(0))))
print(f"FNO parameters: {n_par:,} (real numbers; each complex weight counts twice)")
print("input", X_all.shape, "-> output", Yn_all.shape)


def to_slip(pred_n):
    """Network output (n, N, 8) in scaled units -> slip (n, 8, N) in metres."""
    return np.asarray(pred_n).transpose(0, 2, 1) * ysd[None, :, None]


def surf_err(slip_pred, slip_true, G=G40):
    """Surface-displacement error (mm) for slip arrays (n, 8, N)."""
    return 1000.0 * (slip_pred - slip_true) @ G.T


# %% [markdown]
# ## 3. Training: the loss and Adam, as in 3.02
#
# **In plain words.** Training adjusts the weights so that the network's output curves match the solver's on the
# training runs. The loss is the mean squared difference, each epoch scaled to the same size; Adam (written by hand in
# 3.02, repeated here in ten lines) takes the steps. A second set of runs the network never sees tells us whether it
# learned the physics or memorised the examples.
#
# **Precisely.** Loss $\mathcal L(\boldsymbol\theta) = \frac{1}{|B|}\sum_{i\in B}\lVert \mathrm{FNO}_{\boldsymbol\theta}(a_i) - s_i\rVert^2$
# over mini-batches $B$ of 100 runs, Adam with a learning rate that decays from $3\times10^{-3}$ to zero along a cosine.
# The test error is reported as the surface displacement it would cause, in units of σ = 2 mm.

# %%
def adam_train(loss_fn, params, X, Y, steps, bs=100, lr0=3e-3, seed=0, every=100, callback=None):
    """Plain Adam (Kingma & Ba 2015) with a cosine learning-rate decay."""
    m = jax.tree_util.tree_map(jnp.zeros_like, params)
    v = jax.tree_util.tree_map(jnp.zeros_like, params)

    @jax.jit
    def step(p, m, v, t, xb, yb, lr):
        l, g = jax.value_and_grad(loss_fn)(p, xb, yb)
        m = jax.tree_util.tree_map(lambda a, b: 0.9 * a + 0.1 * b, m, g)
        v = jax.tree_util.tree_map(lambda a, b: 0.999 * a + 0.001 * b * b, v, g)
        p = jax.tree_util.tree_map(lambda a, mm, vv: a - lr * (mm / (1 - 0.9 ** t)) / (jnp.sqrt(vv / (1 - 0.999 ** t)) + 1e-8),
                                   p, m, v)
        return p, m, v, l

    rng = np.random.default_rng(seed)
    hist = []
    n = X.shape[0]
    for t in range(1, steps + 1):
        idx = rng.choice(n, size=min(bs, n), replace=False)
        lr = lr0 * 0.5 * (1 + np.cos(np.pi * t / steps))
        params, m, v, l = step(params, m, v, t, X[idx], Y[idx], lr)
        if t % every == 0 or t == 1:
            hist.append((t, float(l)) + ((callback(params, t),) if callback else ()))
    return params, hist


fno_loss = lambda p, xb, yb: jnp.mean((fno_batch(p, xb, MASK40) - yb) ** 2)
fno_test = jax.jit(lambda p: fno_batch(p, X_all[TE], MASK40))
CHECK = [100, 300, 1000, 3000]
snap = {}


def watch(p, t):
    pred = fno_test(p)
    if t in CHECK:
        snap[t] = to_slip(pred)
    return float(jnp.mean((pred - Yn_all[TE]) ** 2))


t1 = time.time()
fno_p, fno_hist = adam_train(fno_loss, fno_init(jax.random.PRNGKey(0)), X_all[TR], Yn_all[TR], steps=3000, callback=watch)
t_fno = time.time() - t1
slip_fno = to_slip(fno_test(fno_p))
e_fno = surf_err(slip_fno, Y_all[TE])
print(f"training on 1,200 runs, 3,000 Adam steps: {t_fno:.0f} s")
print(f"held-out runs: afterslip rms error {1000 * np.sqrt(np.mean((slip_fno - Y_all[TE]) ** 2)):.2f} mm "
      f"(afterslip rms {1000 * np.sqrt(np.mean(Y_all[TE] ** 2)):.0f} mm)")
print(f"held-out runs: surface rms error {np.sqrt(np.mean(e_fno ** 2)):.2f} mm = {np.sqrt(np.mean(e_fno ** 2)) / SIG:.2f} σ; "
      f"largest single error {np.abs(e_fno).max():.1f} mm = {np.abs(e_fno).max() / SIG:.1f} σ; "
      f"share of errors above 1 σ: {np.mean(np.abs(e_fno) > SIG):.3f}")
fno_one = jax.jit(fno)
jax.block_until_ready(fno_one(fno_p, X_all[TE[0]], MASK40))      # compile once, then time
t_one = time.time()
for _ in range(200):
    jax.block_until_ready(fno_one(fno_p, X_all[TE[0]], MASK40))
print(f"one FNO evaluation: {1000 * (time.time() - t_one) / 200:.2f} ms")

# %%
fig, axs = plt.subplots(1, 3, figsize=(14.5, 3.9))
h = np.array(fno_hist)
axs[0].semilogy(h[:, 0], h[:, 1], color=C_FNO, lw=1.2, alpha=0.6, label="training mini-batch")
axs[0].semilogy(h[:, 0], h[:, 2], color=C_TRUE, lw=1.6, label="held-out runs")
axs[0].set_xlabel("Adam step"); axs[0].set_ylabel("mean squared error (scaled units)")
axs[0].set_title("Loss during training"); axs[0].legend()
for c, i in zip([S.BLUE, S.ORANGE, S.VIOLET], [0, 1, 2]):
    for k, ls in [(2, "--"), (7, "-")]:
        axs[1].plot(1000 * Y_all[TE[i], k], zc, color=S.INK, lw=2.6, alpha=0.25)
        axs[1].plot(1000 * slip_fno[i, k], zc, color=c, lw=1.3, ls=ls)
axs[1].invert_yaxis(); axs[1].set_xlabel("slip (mm)"); axs[1].set_ylabel("depth (km)")
axs[1].set_title("Held-out runs: FNO (colour) on the solver (grey)")
axs[1].text(0.97, 0.03, f"dashed: day {T8[2] * 365.25:.0f}\nsolid: year 1", transform=axs[1].transAxes, ha="right",
            va="bottom", fontsize=8, color=S.INK2)
lim = np.ceil(np.abs(e_fno).max() / SIG) + 0.5
axs[2].hist(e_fno.ravel() / SIG, bins=np.arange(-lim, lim + 0.01, 0.25), color=C_FNO, alpha=0.8)
axs[2].set_yscale("log")
axs[2].axvline(-1, color=S.MUTED, lw=0.8, ls=":"); axs[2].axvline(1, color=S.MUTED, lw=0.8, ls=":")
axs[2].set_xlabel("surface error / σ   (300 runs × 8 epochs × 30 stations)"); axs[2].set_ylabel("count")
axs[2].set_title("Surface error in units of the GPS noise")
fig.tight_layout()
bk.savefig(fig, "3-05_training")
plt.show()

# %% [markdown]
# **How to read it.** The held-out loss (left, black) follows the training loss down, so the network is not memorising.
# On held-out runs (middle) the FNO profiles lie on the solver's at both epochs. Right: almost all surface errors are a
# small fraction of σ, but the histogram has tails; the printed share above 1 σ and the largest error say how long.
#
# **Common misreadings.**
# * "Low average error means safe to use." An inversion is sensitive to errors that are systematic across many data;
#   section 8 measures that directly.
# * "The FNO is fast, so it is free." Its cost moved into the 1,200 solver runs that trained it.

# %% [markdown]
# ## 4. How many runs does it need? FNO against a plain network and a GP
#
# **In plain words.** A surrogate is only cheap if it needs few expensive runs. We train three surrogates on 100, 300
# and 1,200 runs and test them on the same 300 held-out runs: the FNO; a plain fully connected network (3.02) with 40
# inputs and 320 outputs; and the 3.03 recipe, a Gaussian process on the leading principal components of input and
# output.
#
# **Precisely.** The plain network has two hidden layers of 128 GELU units and the same mask, loss, Adam schedule and
# number of steps as the FNO at each size. The GP uses the 12 leading input components (scaled to unit variance), an
# anisotropic squared-exponential kernel with a noise term fitted on up to 300 runs, and predicts the 16 leading output
# components. Errors are rms surface errors in units of σ on the held-out runs.

# %%
def mlp_init(key, width=128):
    k = jax.random.split(key, 3)
    return {"W1": jax.random.normal(k[0], (40, width)) / np.sqrt(40), "b1": jnp.zeros(width),
            "W2": jax.random.normal(k[1], (width, width)) / np.sqrt(width), "b2": jnp.zeros(width),
            "W3": jax.random.normal(k[2], (width, 40 * NOUT)) / np.sqrt(width) / 8, "b3": jnp.zeros(40 * NOUT)}


def mlp(p, a):
    h = jax.nn.gelu(a @ p["W1"] + p["b1"])
    h = jax.nn.gelu(h @ p["W2"] + p["b2"])
    return (h @ p["W3"] + p["b3"]).reshape(40, NOUT) * MASK40[:, None]


mlp_batch = jax.vmap(mlp, in_axes=(None, 0))
A_all = (jnp.asarray(la_all) - mu_in) / sd_in
mlp_loss = lambda p, ab, yb: jnp.mean((mlp_batch(p, ab) - yb) ** 2)

xm = la_all[TR].mean(0)
_, sx, Vx = np.linalg.svd(la_all[TR] - xm, full_matrices=False)


def gp_fit_predict(idx_tr, idx_te, kx=12, r=16):
    Zx = (la_all - xm) @ Vx[:kx].T / sx[:kx] * np.sqrt(len(TR))
    Yf = Y_all.reshape(len(Y_all), -1)
    ym = Yf[idx_tr].mean(0)
    _, _, Vy = np.linalg.svd(Yf[idx_tr] - ym, full_matrices=False)
    C = (Yf[idx_tr] - ym) @ Vy[:r].T
    cs = C.std(0)
    kern = ConstantKernel(1.0, (1e-2, 1e3)) * RBF(np.full(kx, 3.0), (0.1, 1e3)) + WhiteKernel(1e-4, (1e-8, 1e-1))
    sub = idx_tr[:300]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        k_fit = GaussianProcessRegressor(kern, n_restarts_optimizer=0).fit(Zx[sub], C[:len(sub)] / cs).kernel_
        gp = GaussianProcessRegressor(k_fit, optimizer=None).fit(Zx[idx_tr], C / cs)
    return (ym + (gp.predict(Zx[idx_te]) * cs) @ Vy[:r]).reshape(len(idx_te), NOUT, 40)


SIZES = [100, 300, 1200]
lc = {"FNO": {}, "plain network": {}, "PCA + GP": {}}
fno_models = {1200: fno_p}
for n in SIZES:
    idx = TR[:n]
    steps = 3000 if n == 1200 else 1500
    t1 = time.time()
    if n != 1200:
        fno_models[n], _ = adam_train(fno_loss, fno_init(jax.random.PRNGKey(0)), X_all[idx], Yn_all[idx], steps=steps)
    t_f = time.time() - t1 if n != 1200 else t_fno
    lc["FNO"][n] = np.sqrt(np.mean(surf_err(to_slip(fno_test(fno_models[n])), Y_all[TE]) ** 2)) / SIG
    t1 = time.time()
    mp, _ = adam_train(mlp_loss, mlp_init(jax.random.PRNGKey(0)), A_all[idx], Yn_all[idx], steps=steps)
    t_m = time.time() - t1
    lc["plain network"][n] = np.sqrt(np.mean(surf_err(to_slip(mlp_batch(mp, A_all[TE])), Y_all[TE]) ** 2)) / SIG
    t1 = time.time()
    lc["PCA + GP"][n] = np.sqrt(np.mean(surf_err(gp_fit_predict(idx, TE), Y_all[TE]) ** 2)) / SIG
    t_g = time.time() - t1
    print(f"{n:5d} training runs: rms surface error / σ   FNO {lc['FNO'][n]:.3f} ({t_f:.0f} s)   "
          f"plain network {lc['plain network'][n]:.3f} ({t_m:.0f} s)   PCA + GP {lc['PCA + GP'][n]:.3f} ({t_g:.0f} s)")
n_mlp = sum(x.size for x in jax.tree_util.tree_leaves(mlp_init(jax.random.PRNGKey(0))))
print(f"parameters: FNO {n_par:,}, plain network {n_mlp:,}")

# %%
fig, ax = plt.subplots(figsize=(6.2, 4.0))
for (name, c, mk) in [("FNO", C_FNO, "o"), ("plain network", C_MLP, "s"), ("PCA + GP", C_GP, "D")]:
    ax.loglog(SIZES, [lc[name][n] for n in SIZES], marker=mk, color=c, lw=1.6, label=name)
ax.axhline(1.0, color=S.INK2, lw=0.8, ls=":")
ax.text(500, 0.95, "error = GPS noise", fontsize=8, color=S.INK2, va="top", ha="center")
ax.set_xlabel("number of solver runs used for training"); ax.set_ylabel("rms surface error / σ (held-out runs)")
ax.set_xticks(SIZES); ax.set_xticklabels([str(n) for n in SIZES])
ax.set_title("Learning curves: error against training runs"); ax.legend()
fig.tight_layout()
bk.savefig(fig, "3-05_learning_curves")
plt.show()

# %% [markdown]
# **How to read it.** Each line is one surrogate family; lower is better, and the dotted line marks an error equal to
# the GPS noise. Read the gap between the lines at a fixed number of runs, and the slope: a steeper line means each new
# solver run buys more accuracy. The printed numbers give the values and training times. The plain network treats the
# 40 depths as 40 unrelated inputs, and the GP sees only 12 summary numbers of each profile; the FNO's layers are built
# for inputs that are curves, an assumption about the problem that happens to be true here.
#
# **Common misreadings.**
# * "The FNO always wins." This is one problem, one set of settings and one seed per point; with three input numbers
#   (3.01) a GP needs far fewer runs. The architecture pays off when the input is a field.

# %% [markdown]
# ## 5. A finer grid without retraining
#
# **In plain words.** Because the FNO's weights refer to waves, not to grid points, the network trained on 40 patches can
# be run on 80 patches as it is. This is what papers call zero-shot super-resolution. The test: draw 30 new friction
# profiles on an 80-patch grid, run the solver at 80 patches, and compare with the FNO evaluated directly on the
# 80-patch input. A careful comparison also asks how different the 40-patch *solver* is from the 80-patch solver,
# because the network learned the 40-patch solver.
#
# **Precisely.** Three numbers, all rms surface errors over the same 30 profiles: (i) FNO at 80 patches against the
# solver at 80; (ii) the solver at 40 patches (profiles averaged over pairs of patches) against the solver at 80, the
# discretisation difference; (iii) the FNO at 40 patches against the solver at 40, its ordinary error. The mask at 80
# patches comes from the 80-patch stress change.

# %%
s80 = A.fault_setup(n_patch=80)
la80 = A.sample_log_asig(np.random.default_rng(80), 30, s80["zc"])
la80_to40 = 0.5 * (la80[:, 0::2] + la80[:, 1::2])                        # average pairs of patches
solve80 = jax.jit(lambda l: A.fault_afterslip(l, s80, t_obs=T8))
solve40 = jax.jit(lambda l: A.fault_afterslip(l, setup, t_obs=T8))
t1 = time.time()
Y80 = np.stack([np.asarray(solve80(jnp.asarray(l))) for l in la80])
t80 = (time.time() - t1) / len(la80)
t1 = time.time()
Y40c = np.stack([np.asarray(solve40(jnp.asarray(l))) for l in la80_to40])
t40 = (time.time() - t1) / len(la80)
pred80 = to_slip(fno_batch(fno_p, features(la80, s80["zc"]), jnp.asarray(s80["free"], dtype=float)))
pred40c = to_slip(fno_batch(fno_p, features(la80_to40, zc), MASK40))
G80 = s80["G"]
r_fno80 = np.sqrt(np.mean(surf_err(pred80, Y80, G80) ** 2))
r_disc = np.sqrt(np.mean((1000 * (Y40c @ G40.T - Y80 @ G80.T)) ** 2))
r_fno40 = np.sqrt(np.mean(surf_err(pred40c, Y40c) ** 2))
print(f"solver cost per run: 80 patches {t80:.2f} s, 40 patches {t40:.2f} s")
print(f"(i)   FNO on 80 patches vs solver on 80:  rms surface error {r_fno80:.2f} mm = {r_fno80 / SIG:.2f} σ")
print(f"(ii)  solver on 40 vs solver on 80:       rms surface difference {r_disc:.2f} mm = {r_disc / SIG:.2f} σ")
print(f"(iii) FNO on 40 patches vs solver on 40:  rms surface error {r_fno40:.2f} mm = {r_fno40 / SIG:.2f} σ")
print(f"free patches at 80: {s80['free'].sum()} of 80")

# %%
fig, axs = plt.subplots(1, 3, figsize=(14.5, 3.9), gridspec_kw=dict(width_ratios=[1, 1, 1.1]))
j = 0
axs[0].plot(np.exp(la80[j]), s80["zc"], color=S.INK, lw=1.2, label="80 patches")
axs[0].step(np.exp(la80_to40[j]), zc, where="mid", color=S.MUTED, lw=1.2, label="averaged to 40")
axs[0].set_xscale("log"); axs[0].invert_yaxis(); axs[0].set_xlabel("aσ (MPa)"); axs[0].set_ylabel("depth (km)")
axs[0].set_xticks([0.5, 1.0, 2.0]); axs[0].set_xticklabels(["0.5", "1", "2"]); axs[0].minorticks_off()
axs[0].set_title("A new input on the finer grid"); axs[0].legend(loc="lower left")
axs[1].plot(1000 * Y80[j, -1], s80["zc"], color=S.INK, lw=2.8, alpha=0.3, label="solver, 80 patches")
axs[1].plot(1000 * pred80[j, -1], s80["zc"], color=C_FNO, lw=1.4, label="FNO trained on 40, run on 80")
axs[1].plot(1000 * Y40c[j, -1], zc, color=S.ORANGE, lw=1.0, ls="--", label="solver, 40 patches")
axs[1].invert_yaxis(); axs[1].set_xlabel("slip after 1 yr (mm)"); axs[1].set_title("Output on the finer grid")
axs[1].legend(loc="lower right")
labs = ["(iii) FNO vs solver,\nboth 40 patches", "(i) FNO vs solver,\nboth 80 patches", "(ii) solver 40\nvs solver 80"]
vals = [r_fno40 / SIG, r_fno80 / SIG, r_disc / SIG]
axs[2].bar(labs, vals, color=[C_FNO, C_FNO, S.ORANGE], width=0.6)
for k, v_ in enumerate(vals):
    axs[2].text(k, v_, f"{v_:.2f}", ha="center", va="bottom", fontsize=8.5)
axs[2].set_ylabel("rms surface error / σ (30 profiles)"); axs[2].tick_params(axis="x", labelsize=8)
axs[2].set_title("Errors on the two grids")
fig.tight_layout()
bk.savefig(fig, "3-05_resolution")
plt.show()

# %% [markdown]
# **How to read it.** The network was never shown an 80-patch input, yet it runs on one (middle, blue) and its output
# can be compared point by point with the 80-patch solver (grey). The bars (right) compare its error on the fine grid
# (i) with its ordinary error on the coarse grid (iii), and with the difference between the 40- and 80-patch solvers
# themselves (ii). The printed values show that the fine-grid error is many times the coarse-grid error, and of the
# same order as the difference between the two solvers. The middle panel shows where: at the bottom of the rupture and
# at the surface, where the coseismic stress change is concentrated over a short distance, so a finer grid changes the
# solution itself. The network learned the 40-patch solver, including that solver's discretisation error. Here the FNO
# on 80 patches happens to land closer to the 80-patch solver than the 40-patch solver does; that is a property of this
# problem, not something the architecture guarantees.
#
# The same bar says something about the solver: the 40-patch model, which every other section of this notebook treats
# as the true model, differs from the 80-patch model by more than the GPS noise. Here the 40-patch solver defines the
# truth by construction; in a real inversion that difference is one more model error, in the forward model itself.
#
# **Common misreadings.**
# * "Super-resolution adds physics." It evaluates the learned operator on more points. Detail finer than anything in the
#   training runs is not in the operator. Yang et al. (2021, p. 7) say the same of their test on a twice finer velocity
#   grid: "Note that the velocity models with different meshes have the same roughness as the training data set."
# * The plain network of section 4 cannot even be run on the 80-patch input: its first layer has exactly 40 inputs.

# %% [markdown]
# ## 6. Frequency bias: the short waves stay worst
#
# **In plain words.** Neural networks fit smooth, long-wavelength features more easily than sharp, short ones. For an
# FNO this is reinforced by cutting the spectrum at $k_{\max}$. The test: split the error of the 1-yr slip profile into
# waves of different lengths, and watch how each part shrinks during training.
#
# **Precisely.** For each held-out run we take the discrete cosine transform (DCT-II, which does not assume a periodic
# profile) of the 1-yr slip error over the 40 patches and of the 1-yr slip itself, and average the squared coefficients
# over runs. Coefficient $k$ corresponds to a wavelength of $80/k$ km. We compare the error spectrum at Adam steps 100,
# 300, 1,000 and 3,000 with the signal spectrum, coefficient by coefficient and summed over five wavelength bands
# (dividing single coefficients would blow up wherever the signal spectrum happens to be near zero).

# %%
def dct_power(x):
    return np.mean(dct(x, type=2, norm="ortho", axis=-1) ** 2, axis=0)


kk = np.arange(1, 40)
wl = 80.0 / kk
sig_pow = dct_power(1000 * Y_all[TE, -1])[1:]
BANDS = [(1, 2), (3, 4), (5, 8), (9, 16), (17, 39)]                   # DCT index ranges, long to short waves
band_wl = [f"{80 / b:.0f}-{80 / a:.0f}" for a, b in BANDS]
band_x = np.array([80.0 / np.sqrt(a * b) for a, b in BANDS])


def band_ratio(err_pow):
    """Error power summed over each wavelength band / signal power summed over the same band."""
    return np.array([err_pow[a - 1:b].sum() / sig_pow[a - 1:b].sum() for a, b in BANDS])


fig, axs = plt.subplots(1, 2, figsize=(12.5, 3.9))
shade = {100: S.AXIS, 300: S.MUTED, 1000: S.INK2, 3000: C_FNO}
axs[0].loglog(wl, sig_pow, color=S.INK, lw=2.2, label="signal: afterslip at 1 yr")
ratios = {}
for t in CHECK:
    ep = dct_power(1000 * (snap[t][:, -1] - Y_all[TE, -1]))[1:]
    ratios[t] = band_ratio(ep)
    axs[0].loglog(wl, ep, color=shade[t], lw=1.5, label=f"error after {t:,} steps")
    axs[1].loglog(band_x, ratios[t], "o-", color=shade[t], lw=1.5, ms=4, label=f"after {t:,} steps")
for ax in axs:
    ax.invert_xaxis(); ax.set_xlabel("wavelength (km)")
    ax.set_xticks([40, 20, 10, 5, 2]); ax.set_xticklabels(["40", "20", "10", "5", "2"]); ax.minorticks_off()
axs[0].set_ylim(top=1e9); axs[0].set_ylabel("power (mm²)")
axs[0].set_title("Error spectrum during training"); axs[0].legend(fontsize=7.5, ncol=2, loc="upper right")
axs[1].set_ylabel("error power / signal power, per band")
axs[1].set_title("Relative error by wavelength band"); axs[1].legend(fontsize=7.5)
fig.tight_layout()
bk.savefig(fig, "3-05_frequency_bias")
plt.show()
print("relative error power per band (wavelengths in km: " + ", ".join(band_wl) + ")")
for t in CHECK:
    print(f"after {t:5d} steps: " + "  ".join(f"{r:.1e}" for r in ratios[t]))

# %% [markdown]
# **How to read it.** Left: the signal (black) has most of its power at long wavelengths and falls by orders of
# magnitude toward short ones, while the error (grey to blue, after more and more training) is spread almost evenly
# over all wavelengths. Right: the error divided by the signal, summed over five wavelength bands. The shortest band
# (2-5 km) has the largest relative error at every checkpoint, more than an order of magnitude above the best band; the
# bands in between are not in strict order. Reading the printed table, the shortest band reaches at the end of training
# the relative accuracy that the longest band had after a few hundred steps. All bands improve at broadly similar
# rates, so in this small problem the bias appears as a persistent gap: short waves are learned last because they are
# always the worst, not because they start late.
#
# Kong et al. (2025, p. 12) describe the effect in 3-D wavefield simulations: "Figure 3 shows the frequency bias from the FNO
# model, where the model tends to learn how to reproduce the lower frequencies more quickly than the higher
# frequencies." Their fix is multi-stage training, one network per frequency band.
#
# **In practice.** If the information you invert for sits in short wavelengths (a sharp downdip edge of afterslip, for
# example), the surrogate is least accurate exactly there; check its error in that band, not only on average.

# %% [markdown]
# ## 7. Outside the training distribution
#
# **In plain words.** A surrogate interpolates between the runs it has seen. Give it a profile unlike any of them and it
# can be wrong without warning. The training profiles have an 8 km correlation length; here we give the FNO 50 rougher
# profiles (correlation length 3 km, same mean and spread) and compare its errors with those on ordinary held-out runs.

# %%
la_ood = A.sample_log_asig(np.random.default_rng(3), 50, zc, ell=3.0)
t1 = time.time()
Y_ood = np.stack([np.asarray(solve40(jnp.asarray(l))) for l in la_ood])
pred_ood = to_slip(fno_batch(fno_p, features(la_ood, zc), MASK40))
e_ood = surf_err(pred_ood, Y_ood)
rms_in = np.sqrt(np.mean(e_fno ** 2, axis=(1, 2))) / SIG
rms_out = np.sqrt(np.mean(e_ood ** 2, axis=(1, 2))) / SIG
print(f"50 solver runs in {time.time() - t1:.0f} s")
print(f"rms surface error per run / σ: ordinary held-out median {np.median(rms_in):.2f} (largest {rms_in.max():.2f}); "
      f"rough profiles median {np.median(rms_out):.2f} (largest {rms_out.max():.2f})")
print(f"share of single errors above 1 σ: ordinary {np.mean(np.abs(e_fno) > SIG):.3f}, rough {np.mean(np.abs(e_ood) > SIG):.3f}")

# %%
fig, axs = plt.subplots(1, 3, figsize=(14.5, 3.9))
jo = int(np.argsort(rms_out)[len(rms_out) // 2])                          # a typical rough profile
axs[0].plot(np.exp(la_all[TE[1]]), zc, color=S.MUTED, lw=1.4, label="training-like (ℓ = 8 km)")
axs[0].plot(np.exp(la_ood[jo]), zc, color=S.INK, lw=1.4, label="rough (ℓ = 3 km)")
axs[0].set_xscale("log"); axs[0].invert_yaxis(); axs[0].set_xlabel("aσ (MPa)"); axs[0].set_ylabel("depth (km)")
axs[0].set_xticks([0.2, 0.5, 1.0, 2.0, 4.0]); axs[0].set_xticklabels(["0.2", "0.5", "1", "2", "4"]); axs[0].minorticks_off()
axs[0].set_title("Inputs"); axs[0].legend(loc="upper right")
bins = np.geomspace(min(rms_in.min(), rms_out.min()) * 0.8, max(rms_in.max(), rms_out.max()) * 1.2, 30)
axs[1].hist(rms_in, bins=bins, color=C_FNO, alpha=0.7, label="ordinary held-out (300)")
axs[1].hist(rms_out, bins=bins, color=S.INK2, alpha=0.7, label="rough profiles (50)")
axs[1].set_xscale("log"); axs[1].axvline(1, color=S.MUTED, lw=0.8, ls=":")
axs[1].set_xlabel("rms surface error per run / σ"); axs[1].set_ylabel("runs"); axs[1].legend()
axs[1].set_title("Errors inside and outside the training range")
axs[2].plot(1000 * Y_ood[jo, -1], zc, color=S.INK, lw=2.8, alpha=0.3, label="solver")
axs[2].plot(1000 * pred_ood[jo, -1], zc, color=C_FNO, lw=1.4, label="FNO")
axs[2].invert_yaxis(); axs[2].set_xlabel("slip after 1 yr (mm)"); axs[2].set_title("The typical rough case, output")
axs[2].legend(loc="lower right")
fig.tight_layout()
bk.savefig(fig, "3-05_out_of_distribution")
plt.show()

# %% [markdown]
# **How to read it.** The rough profiles (left, black) vary over much shorter distances than the training profiles. The
# histogram (middle) compares the error of each run in the two groups; the right panel shows the median rough case.
# Nothing in the FNO's output flags the second group as unreliable: a surrogate gives no warning when it extrapolates.
# Two practical guards: check that the posterior stays inside the region the training runs cover, and keep a few true
# solver runs at the end of an inversion to measure the surrogate error where the posterior sits.

# %% [markdown]
# ## 8. The FNO inside an inversion
#
# **In plain words.** Now use the trained network as the forward model of a Bayesian inversion for the friction profile,
# from 240 noisy GPS numbers. The network is fast and differentiable, so NUTS can run on it. But its errors act like
# model errors (2.13): left out of the likelihood they make the posterior too narrow and can pull it off the truth. The
# remedy is the same as in 2.13: add the surrogate's error covariance $\mathbf C_p$ to $\mathbf C_d$.
#
# **Precisely.** Parameters $\mathbf u\in\mathbb R^{40}$ with $\ln a\sigma = \boldsymbol\mu + \mathbf L\mathbf u$,
# $\mathbf u\sim\mathcal N(\mathbf 0, \mathbf I)$ ($\mathbf L$ the Cholesky factor of the Matérn prior covariance), data
# $\mathbf d = \mathbf f(\ln a\sigma) + \mathbf e$, $\mathbf e\sim\mathcal N(\mathbf 0, \sigma^2\mathbf I)$. Three
# posteriors:
# * **true model**: $\mathbf f$ is the solver; Gauss-Newton (2.10) to the MAP with the exact Jacobian (forward-mode
#   autodiff through diffrax, 40 tangent directions, 2.17), and the Laplace approximation around it (2.08), whose
#   covariance here is the Gauss-Newton one, $(\mathbf J^\top\mathbf C^{-1}\mathbf J + \mathbf I)^{-1}$ in the
#   whitened prior coordinates $\mathbf u$ (2.25 explains how it differs from the full Laplace covariance);
# * **FNO, $\mathbf C_d$ only**: $\mathbf f$ is the FNO, likelihood with $\sigma^2\mathbf I$;
# * **FNO, $\mathbf C_d + \mathbf C_p$**: $\mathbf C_p = \frac1n\sum_i \mathbf r_i\mathbf r_i^\top$, the second moment
#   (a covariance taken about zero instead of about the mean) of the FNO's surface errors $\mathbf r_i$ on 200 held-out runs (prior-wide, so it describes the surrogate error over
#   the whole prior, not only where the posterior sits).
#
# The FNO posteriors are sampled with NumPyro NUTS (2 chains × 600 draws after 400 warm-up steps) for one data set, and approximated by
# Gauss-Newton plus Laplace for the coverage test over many data sets (the NUTS and Laplace answers are compared below).
# The true-model posterior has no NUTS run: NUTS needs thousands of gradients, and one Jacobian through the stiff
# solver costs as much as several solver runs (both times printed below), so the true model gives the Laplace
# reference. Its Gauss-Newton iteration starts from the MAP of the 1,200-run FNO and then uses only the solver.
#
# A practical detail worth knowing. Differentiating through the stiff solver at first stopped for some profiles with
# "A linear solver received non-finite (NaN or inf) input", although the forward runs were fine. The adaptive solver
# tries steps and rejects the bad ones; a rejected trial step can overflow ($\sinh$ of a huge number) without harming
# the forward answer, but the error check inside the solver's derivative still sees it. With `EQX_ON_ERROR=nan` (first
# code cell) the check returns NaN instead of stopping. In forward mode, used here, the rejected step's values are then
# not selected, and the Jacobians came out finite; in reverse mode the same overflow can give NaN gradients (2.22,
# section 6, explains why and gives the cure). So the setting is not a fix in general: the Jacobian is checked against
# finite differences below, and the coverage loop counts any non-finite Jacobian.

# %%
MU_P = np.log(0.8)
Lc = np.linalg.cholesky(A.matern_cov(zc, 8.0, 1.5, 0.5) + 1e-10 * np.eye(40))
Lj, MUj, G40j = jnp.asarray(Lc), jnp.asarray(MU_P), jnp.asarray(G40)


def solver_surf(la_, adjoint=None):
    """True model: surface displacement (mm) at 8 epochs x 30 stations, flattened (240,)."""
    Kj, dt = jnp.asarray(setup["K"]), jnp.asarray(setup["dtau"])
    fr = jnp.asarray(free, dtype=float)
    a_s = jnp.exp(la_)

    def rhs(t, y, args):
        v = fr * 2.0 * 0.03 * jnp.sinh(y[:40] / a_s)
        return jnp.concatenate([Kj @ v, v])

    kw = {} if adjoint is None else dict(adjoint=adjoint)
    sol = dfx.diffeqsolve(dfx.ODETerm(rhs), dfx.Kvaerno5(), t0=0.0, t1=T8[-1], dt0=1e-8,
                          y0=jnp.concatenate([dt, jnp.zeros(40)]), saveat=dfx.SaveAt(ts=jnp.asarray(T8)),
                          stepsize_controller=dfx.PIDController(rtol=1e-6, atol=1e-8), max_steps=400_000, **kw)
    return 1000.0 * (sol.ys[:, 40:] @ G40j.T).ravel()


def fno_surf(p):
    return lambda la_: 1000.0 * ((fno(p, features(la_[None], zc)[0], MASK40) * jnp.asarray(ysd)[None, :]).T @ G40j.T).ravel()


f_true = jax.jit(solver_surf)
J_true = jax.jit(jax.jacfwd(lambda la_: solver_surf(la_, dfx.ForwardMode())))
x_ = jnp.asarray(la_all[TE[0]])
jax.block_until_ready(f_true(x_)); jax.block_until_ready(J_true(x_))         # compile both first
t1 = time.time(); jax.block_until_ready(f_true(x_)); t_f1 = time.time() - t1
t1 = time.time(); jax.block_until_ready(J_true(x_)); t_j1 = time.time() - t1
print(f"true model: one forward run {t_f1:.2f} s, one Jacobian (40 directions, forward mode) {t_j1:.2f} s")
x_ = jnp.asarray(la_all[TE[202]])                                       # a profile whose Jacobian raised before the fix
e_ = np.zeros(40); e_[25] = 1e-4
fd_ = (np.asarray(f_true(x_ + e_)) - np.asarray(f_true(x_ - e_))) / 2e-4
J_ = np.asarray(J_true(x_))
print(f"Jacobian check against central finite differences (run {TE[202] + 1:,}, ln aσ at {zc[25]:.1f} km): "
      f"largest relative difference {np.abs(fd_ - J_[:, 25]).max() / np.abs(J_[:, 25]).max():.1e}")
SURF_all = 1000.0 * (Y_all @ G40.T).reshape(len(Y_all), -1)            # cached solver output (mm), (1500, 240)
CP_IDX, TRUTH_IDX = TE[:200], TE[200:]


def cp_matrix(p):
    pred = np.asarray(jax.vmap(fno_surf(p))(jnp.asarray(la_all[CP_IDX])))
    r = pred - SURF_all[CP_IDX]
    return r.T @ r / len(r)


BAD_JAC = []                                                            # counts non-finite Jacobians (expected: none)


def gauss_newton(fwd, jac, d, Cinv, u0=None, iters=20, tol=1e-6, max_step=1.0):
    """MAP of 1/2 (d - f)^T Cinv (d - f) + 1/2 |u|^2 by Gauss-Newton with a step cap (in prior standard deviations) and
    backtracking; returns (u, Laplace posterior covariance of u, number of steps)."""
    def obj(u_):
        try:
            r_ = d - np.asarray(fwd(MU_P + Lc @ u_))
        except Exception:                                     # the stiff solver can fail for extreme trial profiles
            return np.inf
        val = 0.5 * r_ @ Cinv @ r_ + 0.5 * u_ @ u_
        return val if np.isfinite(val) else np.inf

    u = np.zeros(40) if u0 is None else u0.copy()
    f0 = obj(u)
    for it in range(iters):
        la_ = MU_P + Lc @ u
        r = d - np.asarray(fwd(la_))
        Ju = np.asarray(jac(la_)) @ Lc
        if not np.isfinite(Ju).all():
            BAD_JAC.append(1)
            break
        du = np.linalg.solve(Ju.T @ Cinv @ Ju + np.eye(40), Ju.T @ Cinv @ r - u)
        du *= min(1.0, max_step / np.abs(du).max())
        step = 1.0
        f1 = obj(u + du)
        while f1 > f0 and step > 1e-3:
            step /= 2
            f1 = obj(u + step * du)
        if f1 > f0:
            break
        u, f0 = u + step * du, f1
        if np.linalg.norm(step * du) < tol * (1 + np.linalg.norm(u)):
            break
    la_ = MU_P + Lc @ u
    Ju = np.asarray(jac(la_)) @ Lc
    return u, np.linalg.inv(Ju.T @ Cinv @ Ju + np.eye(40)), it + 1


inv_models = {}
for n in [300, 1200]:
    p = fno_models[n]
    f_ = jax.jit(fno_surf(p))
    inv_models[n] = dict(f=f_, J=jax.jit(jax.jacfwd(fno_surf(p))), Cp=cp_matrix(p))
    print(f"FNO trained on {n:4d} runs: C_p from 200 held-out runs, rms surrogate error "
          f"{np.sqrt(np.trace(inv_models[n]['Cp']) / 240):.2f} mm = {np.sqrt(np.trace(inv_models[n]['Cp']) / 240) / SIG:.2f} σ")

# %% [markdown]
# **Worked example: one data set.** Truth: held-out run 1,401 (never used for training or for $\mathbf C_p$). Data: the
# solver's 240 surface values plus noise with σ = 2 mm. We invert with the true model and with the FNO trained on 300
# runs, which section 4 showed is the less accurate of the two.

# %%
it0 = TRUTH_IDX[0]
rng = np.random.default_rng(1)
d_obs = SURF_all[it0] + SIG * rng.standard_normal(240)
Cd = SIG ** 2 * np.eye(240)
u_start, _, _ = gauss_newton(inv_models[1200]["f"], inv_models[1200]["J"], d_obs, np.linalg.inv(Cd))
t1 = time.time()
u_t, Cu_t, n_it = gauss_newton(f_true, J_true, d_obs, np.linalg.inv(Cd), u0=u_start)
t_gn = time.time() - t1
print(f"true model (started from the 1,200-run FNO's MAP): Gauss-Newton stopped after {n_it} steps, {t_gn:.0f} s; "
      f"chi2/N at the MAP {np.sum((d_obs - np.asarray(f_true(MU_P + Lc @ u_t))) ** 2) / SIG ** 2 / 240:.2f}")
m3 = inv_models[300]
r_truth = np.asarray(m3["f"](la_all[it0])) - SURF_all[it0]
print(f"FNO (300 runs) error at the true profile: rms {np.sqrt(np.mean(r_truth ** 2)):.2f} mm = "
      f"{np.sqrt(np.mean(r_truth ** 2)) / SIG:.2f} σ, largest {np.abs(r_truth).max() / SIG:.1f} σ")


def nuts_fno(f_, d, C, seed=0):
    diag = np.allclose(C, np.diag(np.diag(C)))
    Lchol = jnp.asarray(np.linalg.cholesky(C))

    def model():
        u = numpyro.sample("u", dist.Normal(0.0, 1.0).expand([40]).to_event(1))
        pred = f_(MUj + Lj @ u)
        if diag:
            numpyro.sample("d", dist.Normal(pred, jnp.sqrt(jnp.diag(jnp.asarray(C)))).to_event(1), obs=jnp.asarray(d))
        else:
            numpyro.sample("d", dist.MultivariateNormal(pred, scale_tril=Lchol), obs=jnp.asarray(d))

    mc = MCMC(NUTS(model), num_warmup=400, num_samples=600, num_chains=2, chain_method="vectorized", progress_bar=False)
    mc.run(jax.random.PRNGKey(seed), extra_fields=("diverging", "num_steps"))
    u = np.asarray(mc.get_samples(group_by_chain=True)["u"])
    la_s = MU_P + u @ Lc.T
    ex = mc.get_extra_fields()
    div = int(np.asarray(ex["diverging"]).sum())
    rh = max(D.rhat(la_s[:, :, j]) for j in np.where(free)[0])
    return la_s.reshape(-1, 40), rh, div, float(np.mean(ex["num_steps"]))


post = {}
t1 = time.time()
for key, C in [("Cd", Cd), ("Cp", Cd + m3["Cp"])]:
    post[key] = nuts_fno(m3["f"], d_obs, C)
    print(f"NUTS on the FNO, {'C_d only' if key == 'Cd' else 'C_d + C_p'}: largest split R-hat over free patches "
          f"{post[key][1]:.3f}, divergences {post[key][2]}, mean leapfrog steps per draw {post[key][3]:.0f}")
print(f"both NUTS runs: {time.time() - t1:.0f} s")
for key, C in [("Cd", Cd), ("Cp", Cd + m3["Cp"])]:
    u_l, Cu_l, _ = gauss_newton(m3["f"], m3["J"], d_obs, np.linalg.inv(C))
    sd_l = np.sqrt(np.diag(Lc @ Cu_l @ Lc.T))
    la_n = post[key][0]
    print(f"   Laplace vs NUTS ({key}), free patches: largest |mean difference| / NUTS sd "
          f"{np.max(np.abs(MU_P + Lc @ u_l - la_n.mean(0))[free] / la_n.std(0)[free]):.2f}; "
          f"sd ratio Laplace/NUTS {np.min(sd_l[free] / la_n.std(0)[free]):.2f} to {np.max(sd_l[free] / la_n.std(0)[free]):.2f}")

# %%
sd_t = np.sqrt(np.diag(Lc @ Cu_t @ Lc.T))
m_t = MU_P + Lc @ u_t
fig, axs = plt.subplots(1, 3, figsize=(14.8, 4.1), gridspec_kw=dict(width_ratios=[1.15, 1.15, 1]))
for ax, key, col, title in [(axs[0], "Cd", C_FNO, "FNO, C_d only"), (axs[1], "Cp", C_CP, "FNO, C_d + C_p")]:
    ax.fill_betweenx(zc, np.exp(m_t - 1.645 * sd_t), np.exp(m_t + 1.645 * sd_t), color=S.GRID, lw=0,
                     label="true model, 90 % (Laplace)")
    lo, hi = np.percentile(post[key][0], [5, 95], axis=0)
    ax.plot(np.exp(lo), zc, color=col, lw=1.2); ax.plot(np.exp(hi), zc, color=col, lw=1.2, label="FNO, 90 % (NUTS)")
    ax.plot(np.exp(la_all[it0]), zc, color=S.INK, lw=1.8, label="true profile")
    ax.set_xscale("log"); ax.invert_yaxis(); ax.set_xlabel("aσ (MPa)"); ax.set_ylabel("depth (km)")
    ax.set_xticks([0.3, 0.5, 1.0, 2.0, 4.0]); ax.set_xticklabels(["0.3", "0.5", "1", "2", "4"]); ax.minorticks_off()
    ax.set_title(f"Posterior of aσ(z): {title}"); ax.legend(loc="upper right", fontsize=7.5)
    inside = np.mean((la_all[it0] >= lo) & (la_all[it0] <= hi))
    print(f"{title}: true aσ inside the NUTS 90 % band at {inside:.2f} of all patches, "
          f"{np.mean(((la_all[it0] >= lo) & (la_all[it0] <= hi))[free]):.2f} of free patches")
print(f"true model (Laplace): true aσ inside its 90 % band at "
      f"{np.mean(np.abs(la_all[it0] - m_t)[free] <= 1.645 * sd_t[free]):.2f} of free patches")
axs[2].plot(r_truth.reshape(8, 30)[-1] / SIG, setup["x_obs"], "o-", color=C_FNO, ms=3.5, lw=1.0, label="year 1")
axs[2].plot(r_truth.reshape(8, 30)[2] / SIG, setup["x_obs"], "s--", color=S.MUTED, ms=3, lw=1.0,
            label=f"day {T8[2] * 365.25:.0f}")
for xv in (-1, 1):
    axs[2].axvline(xv, color=S.MUTED, lw=0.8, ls=":")
axs[2].axvline(0, color=S.AXIS, lw=0.8)
axs[2].set_yscale("log"); axs[2].set_ylabel("station distance (km)"); axs[2].set_xlabel("(FNO − solver) / σ at the true profile")
axs[2].set_title("Surrogate error the likelihood sees"); axs[2].legend(loc="upper left", fontsize=7.5)
fig.tight_layout()
bk.savefig(fig, "3-05_inversion")
plt.show()

# %% [markdown]
# **How to read it.** In both left panels the grey band is the reference, the true-model posterior, and the black line
# the profile that made the data. The coloured lines bound the FNO posterior. Where the FNO band is narrower than the
# grey one or sits beside it, the surrogate posterior is overconfident or shifted; the printed shares say how often the
# true profile falls inside each band. The right panel shows why: at the true profile the FNO's predictions miss the
# solver's by amounts comparable to σ at some stations, with the same sign at neighbouring stations, so the likelihood
# reads them as signal. Adding $\mathbf C_p$ (middle) tells the likelihood that differences of that size and pattern are
# expected.
#
# One data set can mislead in either direction. The test that counts repeats the inversion on many data sets.

# %% [markdown]
# **Coverage over 12 data sets.** For each of held-out runs 1,401-1,412 (never used for training or $\mathbf C_p$), new
# noise, then the Laplace posterior for the true model and for both FNOs (300 and 1,200 training runs), each with and
# without $\mathbf C_p$. A calibrated posterior contains the true value in its 90 % interval 90 % of the time; we count
# over the 31 patches that slip (the pinned ones are not seen by the data). We also compare the FNO's Jacobian with the
# solver's at the true profiles, because gradients steer both NUTS and Gauss-Newton.

# %%
NREP = 12
cov = {k: [] for k in ["true model", "FNO 300, C_d", "FNO 300, C_d + C_p", "FNO 1200, C_d", "FNO 1200, C_d + C_p"]}
zabs = {k: [] for k in cov}
jac_err = {300: [], 1200: []}
pred_err = {300: [], 1200: []}
t1 = time.time()
for rep in range(NREP):
    i = TRUTH_IDX[rep]
    d = SURF_all[i] + SIG * np.random.default_rng(100 + rep).standard_normal(240)
    runs = []
    Jt = np.asarray(J_true(jnp.asarray(la_all[i])))
    for n in [300, 1200]:
        m = inv_models[n]
        u_c, Cu_c, _ = gauss_newton(m["f"], m["J"], d, np.linalg.inv(Cd))
        u_p, Cu_p, _ = gauss_newton(m["f"], m["J"], d, np.linalg.inv(Cd + m["Cp"]))
        runs += [(f"FNO {n}, C_d", u_c, Cu_c), (f"FNO {n}, C_d + C_p", u_p, Cu_p)]
        jac_err[n].append(np.linalg.norm(np.asarray(m["J"](la_all[i])) - Jt) / np.linalg.norm(Jt))
        pred_err[n].append(np.linalg.norm(np.asarray(m["f"](la_all[i])) - SURF_all[i]) / np.linalg.norm(SURF_all[i]))
    u_t2, Cu_t2, _ = gauss_newton(f_true, J_true, d, np.linalg.inv(Cd), u0=runs[-1][1])
    runs.append(("true model", u_t2, Cu_t2))
    for name, u_, Cu_ in runs:
        z = (la_all[i] - (MU_P + Lc @ u_)) / np.sqrt(np.diag(Lc @ Cu_ @ Lc.T))
        cov[name].append(np.mean(np.abs(z[free]) <= 1.645))
        zabs[name].append(np.median(np.abs(z[free])))
print(f"{NREP} data sets, all inversions: {time.time() - t1:.0f} s; non-finite Jacobians met: {len(BAD_JAC)}")
for name in cov:
    print(f"{name:20s}: share of true values inside the 90 % interval {np.mean(cov[name]):.2f} "
          f"(range over data sets {np.min(cov[name]):.2f}-{np.max(cov[name]):.2f}); median |error| / sd {np.median(zabs[name]):.2f}")
for n in [300, 1200]:
    print(f"FNO ({n} runs) vs the solver at the true profiles: relative error of the Jacobian median {np.median(jac_err[n]):.3f} "
          f"(largest {np.max(jac_err[n]):.3f}); of the predicted data median {np.median(pred_err[n]):.4f}")

# %%
fig, axs = plt.subplots(1, 2, figsize=(12.8, 4.0), gridspec_kw=dict(width_ratios=[1.5, 1]))
names = list(cov)
cols = [C_TRUE, C_FNO, C_CP, C_FNO, C_CP]
hatch = [None, "//", "//", None, None]
vals = [np.mean(cov[k]) for k in names]
bars = axs[0].bar(range(len(names)), vals, color=cols, width=0.62)
for b, hh in zip(bars, hatch):
    if hh:
        b.set_hatch(hh); b.set_edgecolor(S.SURFACE)
for k, v_ in enumerate(vals):
    axs[0].text(k, v_ - 0.03, f"{v_:.2f}", ha="center", va="top", fontsize=8.5, color=S.SURFACE, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.2", fc=cols[k], ec="none"))
axs[0].axhline(0.9, color=S.INK, lw=0.8, ls="--")
axs[0].set_xticks(range(len(names))); axs[0].set_xticklabels([n.replace(", ", "\n") for n in names], fontsize=8)
axs[0].set_ylim(0, 1.08); axs[0].set_ylabel("share of true aσ inside the 90 % interval")
axs[0].set_title(f"Coverage over {NREP} data sets (31 free patches each)")
i_last = TRUTH_IDX[NREP - 1]
Jt = np.asarray(J_true(jnp.asarray(la_all[i_last])))
col_j = int(np.argmin(np.abs(zc - 15.5)))
axs[1].plot(Jt[:, col_j], color=S.INK, lw=2.4, alpha=0.35, label="solver")
for n, c, ls in [(300, S.ORANGE, "--"), (1200, C_FNO, "-")]:
    axs[1].plot(np.asarray(inv_models[n]["J"](la_all[i_last]))[:, col_j], color=c, lw=1.1, ls=ls, label=f"FNO, {n} runs")
axs[1].set_xlabel("datum (8 epochs × 30 stations)"); axs[1].set_ylabel(f"∂d / ∂ln aσ at {zc[col_j]:.1f} km (mm)")
axs[1].set_title("One column of the Jacobian"); axs[1].legend(fontsize=7.5)
fig.tight_layout()
bk.savefig(fig, "3-05_coverage")
plt.show()

# %% [markdown]
# **How to read it.** Left: the dashed line is the 90 % a calibrated posterior should reach. The true-model bar shows
# what the Laplace approximation achieves with no surrogate error (the benchmark for this test, itself subject to the
# scatter of only 12 data sets); each
# FNO pair shows the same surrogate without and with $\mathbf C_p$. Right: one column of the Jacobian, the change of all
# 240 data when friction at one depth changes; the FNO's gradients are learned only indirectly, from values, and the
# printed relative errors show them to be far less accurate than its predictions, with the largest misses at the
# stations nearest the fault (the spikes at the start of each 30-station block).
#
# Cao et al. (2025, p. 1) make the gradient point the centre of their method: "To achieve a substantial speedup, the
# surrogate must accurately approximate the PtO map and its Jacobian, which often demands a prohibitively large number
# of PtO map samples via conventional operator learning methods." (PtO: parameter-to-observable.) Their DINO
# (derivative-informed neural operator) surrogates are trained on values and on reduced Jacobians together; 3.07
# discusses what those Jacobians cost.
#
# **The image to keep.** A surrogate inside a likelihood is a forward model with a model error. Measure that error on
# runs it did not train on, in units of σ, and put it in the likelihood as $\mathbf C_p$; then check coverage against
# the true model on a handful of synthetic data sets.
#
# **Common misreadings.**
# * "C_p fixes everything." Here $\mathbf C_p$ was estimated over the whole prior. If the surrogate is much worse (or
#   better) where the posterior sits than on average, a prior-wide $\mathbf C_p$ is the wrong size there. Running the
#   solver at a few posterior samples measures that.
# * "Accurate predictions mean accurate gradients." Not guaranteed; check both.

# %% [markdown]
# ## 9. Where the field is
#
# Quotes below are verbatim with page numbers; Li et al. (2021, FNO) and Lu et al. (2021, DeepONet, a branch-and-trunk
# operator network and another widely used architecture) are named for the idea only.
#
# * **Speed and gradients in geophysics.** Yang et al. (2023, p. 3): "Once completely trained, the U-NO model can be
#   evaluated on a new input with very little computational cost (0.02 sec compared to the FDM runtime of 1.23 sec)."
#   And (p. 4): "Neural operators are differentiable by design, which enables gradient computation with reverse-mode
#   automatic differentiation." Kong, Gurnis & Ross (2026, p. 1) use the same two properties for mantle convection:
#   "These neural operators significantly accelerate forward and inverse convection modelling by transforming forward
#   physical processes into surrogate models with lower complexity while utilizing auto-differentiation to calculate
#   gradients."
# * **Approximate by construction.** Yang et al. (2021, p. 9): "One of the limitations of the approach is that the
#   solutions are approximate, as seen in several of the figures." Section 8 is how to carry that into the posterior.
# * **Gradients as training data.** Cao et al. (2025), quoted above: derivative-informed operators for MCMC.
# * **Posteriors over whole fields.** Shi, Ross & Yue (2026, p. 1) build function-space priors with flow matching (a way
#   of training a network to turn simple random draws into draws that resemble a training set):
#   their method "enables likelihood-guided posterior inference from sparse and noisy observations, supports variable
#   query discretizations, and avoids explicit prior-density evaluation." That is the resolution-invariance of section 5
#   applied to the prior as well as to the forward model.
#
# **In practice.** Ensemble methods run the forward model once per member per iteration, and gradient-based samplers once
# per leapfrog step; a trained operator makes each of those runs nearly free, which is why operators are attractive for
# fault-friction inversions with hundreds of unknowns. The price is the surrogate error, which propagates into an
# ill-posed inverse problem. Sections 7 and 8 are the two checks to run before trusting the result: errors outside the
# training range, and coverage with and without $\mathbf C_p$ against the true model on synthetic data. The training
# runs are forward runs, so their cost does not disappear; it moves to before the data arrive.
#
# Next: **3.06 · Reduced-order models and real-time inference**.

# %%
print(f"notebook run time: {time.time() - T0:.0f} s")
