# %% [markdown]
# # 3.02 · Neural networks and ResNets from scratch
#
# *Topic 3 is under development: content and numbering may still change.*
#
# **Before this:** 2.10 (loss functions, gradient descent, Adam), 2.17 (automatic differentiation), 3.01 (surrogate
# workflow, PCA of the outputs) · **You will be able to:** build and train a small network by hand; explain
# backpropagation as reverse-mode autodiff; diagnose bad initialisation, bad scaling and overfitting; judge whether
# an ensemble's spread is a usable error bar · **Optional sections:** 10 (a ResNet is an ODE solver).
#
# A neural network is a flexible function with many adjustable numbers, fitted to examples by gradient descent on a
# loss (2.10). 3.01 used one as a black-box surrogate with a small, fixed training budget, and it stopped improving
# early. This notebook opens the box: we build a network by hand in JAX, train it with Adam written out in ten lines,
# and see why initialisation, scaling, training length and depth matter. Residual networks (ResNets) turn out to be
# Euler steps of an ODE, and an ensemble of networks gives a rough, sometimes misleading, error bar.
#
# **Words this notebook defines:** neuron, weight, bias, activation function, layer, width, depth, multilayer
# perceptron (MLP), universal approximation, forward pass, backpropagation, initialisation, vanishing and exploding
# gradients, learning rate, gradient descent, momentum, Adam, training step, training / validation / test loss,
# overfitting, weight decay, early stopping, input and output scaling, residual block (ResNet), neural ODE, deep
# ensemble, in and out of distribution.

# %% [markdown]
# **Problems in this notebook**
#
# | problem | inputs | outputs | examples | sections |
# |---|---|---|---|---|
# | 1-D afterslip slice: displacement between day 30 and year 1 as a function of log v0 (aσ = 0.5 MPa, Δτ = 2.5 MPa) | 1 | 1 | 200 true-model runs on a grid of log v0 (some sections use subsets, one adds noise) | 1-6, 11 |
# | single-patch surrogate, as in 3.01 | 3 (log aσ, log v0, Δτ, scaled) | 10 PCA coefficients, rebuilt to 40 epochs (mm) | cached runs: first n training runs, 400 validation, 400 test | 7-9, 11 |
# | a 2-D ODE with random weights | 2 | 2 | none (a forward pass only) | 10 |
#
# All networks are written in plain JAX (no neural-network library). GPS noise σ = 2 mm wherever surrogate errors are
# judged; section 6 uses a larger noise on purpose and says so. Papers cited by author and year for an idea (Cybenko
# 1989, Hornik 1991, Rumelhart, Hinton & Williams 1986, Glorot & Bengio 2010, Kingma & Ba 2015, He et al. 2016,
# Chen et al. 2018, Lakshminarayanan et al. 2017, MacKay 1992, Neal 1996) are credited for the ideas only; no numbers
# or quotes are taken from them.
#
# As in 3.01, every surrogate here builds its output compression (PCA, 3.01 section 4) from its own training runs, so
# the number of training runs is its whole budget.

# %%
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))
import numpy as np
import matplotlib.pyplot as plt
import bayeskit as bk
from bayeskit import problems as P, samplers as SM, diagnostics as D, plotting as BP, style as S
import time, inspect
import jax
import jax.numpy as jnp
import diffrax as dfx
from scipy.optimize import minimize
from bayeskit import afterslip as A

T0 = time.time()
SIG = 2.0                                                     # GPS noise (mm)

# --- the 1-D slice: displacement between day 30 and year 1 against log v0
A_S, DTAU = 0.5, 2.5
T2 = jnp.array([30 / 365.25, 1.0])
slice_runs = jax.jit(jax.vmap(lambda lv: 1000.0 * A.single_patch_series(jnp.stack([jnp.log(A_S), lv, DTAU]), T2)))
lv_grid = np.linspace(np.log(1e-3), 0.0, 200)
u2 = np.asarray(slice_runs(jnp.asarray(lv_grid)))
y_grid = u2[:, 1] - u2[:, 0]                                   # mm
x_grid = 2 * (lv_grid - lv_grid[0]) / (lv_grid[-1] - lv_grid[0]) - 1      # log v0 scaled to [-1, 1]
Y_MU, Y_SD = y_grid.mean(), y_grid.std()                       # output scaling for the 1-D fits

# --- the single-patch surrogate data (as in 3.01)
LO, HI = A.SP_BOUNDS[:, 0], A.SP_BOUNDS[:, 1]
th_all, u_all = A.training_set_single(4000, seed=0)
U_all, X_all = 1000.0 * u_all, (th_all - LO) / (HI - LO)
TR, VA, TE = np.arange(0, 3200), np.arange(3200, 3600), np.arange(3600, 4000)
K = 10                                                         # PCA modes (3.01's rule)


def pca_basis(idx):
    """Mean curve, K leading modes and the spread of coefficient 1, from the runs idx only."""
    mu = U_all[idx].mean(0)
    V = np.linalg.svd(U_all[idx] - mu, full_matrices=False)[2][:K]
    return mu, V, ((U_all[idx] - mu) @ V[0]).std()


Z_SCALE = pca_basis(TR[:800])[2]                               # the output scale of section 7's 800 runs
print(f"1-D slice: {len(y_grid)} true-model runs; output from {y_grid.min():.1f} to {y_grid.max():.1f} mm")
print(f"single patch: training {len(TR)}, validation {len(VA)}, test {len(TE)} runs; {K} PCA coefficients;"
      f" output scale of the first 800 training runs (sd of coefficient 1) {Z_SCALE:.0f} mm")

# %% [markdown]
# ## 1. A neuron, a layer, a network
#
# **In plain words.** A neuron takes its inputs, multiplies each by a weight, adds them up with a bias, and passes the
# sum through a fixed bend (the activation function). With the bend, one neuron can make a step-like or ramp-like
# shape; a layer is many neurons side by side, each making its own shape at its own place and scale; the output adds
# the shapes together. A network with several layers repeats this, building shapes out of shapes.
#
# **Precisely.** A layer maps $\mathbf x\in\mathbb R^{n_\text{in}}$ to $\mathbf h = \varphi(\mathbf W\mathbf x + \mathbf b)$ with weights
# $\mathbf W\in\mathbb R^{n_\text{out}\times n_\text{in}}$, bias $\mathbf b$ and an elementwise activation $\varphi$, for example
# $\tanh$ or $\mathrm{ReLU}(a)=\max(0,a)$. A multilayer perceptron (MLP) composes layers and ends with a linear one:
#
# $$f(\mathbf x) = \mathbf W_{L+1}\,\varphi\big(\mathbf W_L \cdots \varphi(\mathbf W_1\mathbf x + \mathbf b_1)\cdots + \mathbf b_L\big) + \mathbf b_{L+1}.$$
#
# Width = neurons per layer, depth = number of hidden layers; all weights and biases together are the parameters $w$.
# Below: the code of the whole network (two functions), then a network with one hidden layer of 4 tanh neurons fitted to
# the 1-D slice, taken apart into the shapes its neurons contribute.

# %%
def init_mlp(key, sizes, scale=1.0):
    """Weights ~ N(0, scale^2 / n_in) (scale 1 keeps the signal size roughly constant through tanh layers), biases 0."""
    params = []
    for n_in, n_out in zip(sizes[:-1], sizes[1:]):
        key, k = jax.random.split(key)
        params.append((scale * jax.random.normal(k, (n_in, n_out)) / np.sqrt(n_in), jnp.zeros(n_out)))
    return params


def mlp(params, x, act=jnp.tanh):
    """Forward pass: every hidden layer is act(x W + b); the last layer is linear."""
    for W, b in params[:-1]:
        x = act(x @ W + b)
    W, b = params[-1]
    return x @ W + b


def adam(loss, params, n_steps, lr=3e-3, b1=0.9, b2=0.999, eps=1e-8, cosine=True, monitor=None):
    """Adam (Kingma & Ba 2015): each parameter moves by lr * (running mean of its gradient) / (running RMS of its
    gradient), with bias correction for the first steps. cosine=True lowers lr smoothly to 0 over the run.
    Returns the trained parameters and, per step, the loss (and monitor(params) if given)."""
    def step(carry, i):
        p, m, v = carry
        L, g = jax.value_and_grad(loss)(p)
        t = i + 1.0
        lr_t = lr * 0.5 * (1 + jnp.cos(jnp.pi * i / n_steps)) if cosine else lr
        m = jax.tree.map(lambda m_, g_: b1 * m_ + (1 - b1) * g_, m, g)          # running mean of the gradient
        v = jax.tree.map(lambda v_, g_: b2 * v_ + (1 - b2) * g_ ** 2, v, g)     # running mean of its square
        p = jax.tree.map(lambda p_, m_, v_: p_ - lr_t * (m_ / (1 - b1 ** t)) / (jnp.sqrt(v_ / (1 - b2 ** t)) + eps),
                         p, m, v)
        return (p, m, v), (L, monitor(p) if monitor is not None else L)
    zeros = jax.tree.map(jnp.zeros_like, params)
    (params, _, _), (hist, mon) = jax.lax.scan(step, (params, zeros, zeros), jnp.arange(n_steps, dtype=jnp.float64))
    return params, hist, mon


def fit_1d(width_list, x, y, n_steps=4000, lr=1e-2, seed=0):
    """Fit an MLP with hidden widths width_list to (x, y) in scaled units; returns params and the loss history."""
    xs, ys = jnp.asarray(x)[:, None], jnp.asarray((y - Y_MU) / Y_SD)[:, None]
    loss = lambda p: jnp.mean((mlp(p, xs) - ys) ** 2)
    p, hist, _ = jax.jit(lambda p0: adam(loss, p0, n_steps, lr))(init_mlp(jax.random.PRNGKey(seed), [1, *width_list, 1]))
    return p, np.asarray(hist)


def pred_1d(p, x):
    return np.asarray(mlp(p, jnp.asarray(x)[:, None]))[:, 0] * Y_SD + Y_MU


print(inspect.getsource(mlp))
p4, _ = fit_1d([4], x_grid, y_grid)
W1, b1 = (np.asarray(a) for a in p4[0]); W2, b2 = (np.asarray(a) for a in p4[1])
parts = np.tanh(x_grid[:, None] * W1 + b1) * W2[:, 0] * Y_SD           # each neuron's shape, in mm
print(f"4-neuron fit: RMS error {np.sqrt(((pred_1d(p4, x_grid) - y_grid) ** 2).mean()):.2f} mm over the 200 points")

# %%
fig, axs = plt.subplots(1, 3, figsize=(13.5, 3.9), gridspec_kw=dict(width_ratios=[0.8, 1.1, 1.1]))
ax = axs[0]
a = np.linspace(-3, 3, 200)
ax.plot(a, np.tanh(a), color=S.BLUE, label="tanh(a)")
ax.plot(a, np.maximum(a, 0), color=S.ORANGE, label="ReLU(a) = max(0, a)")
ax.axhline(0, color=S.GRID, lw=0.8); ax.axvline(0, color=S.GRID, lw=0.8)
ax.set_xlabel("a = w·x + b"); ax.set_ylabel("activation"); ax.set_ylim(-1.3, 3.0)
ax.set_title("Two activation functions"); ax.legend(fontsize=8, loc="upper left")
ax = axs[1]
cols = [S.BLUE, S.ORANGE, S.AQUA, S.VIOLET]
for j in range(4):
    ax.plot(lv_grid, parts[:, j], color=cols[j], lw=1.3, label=f"neuron {j + 1}")
ax.axhline(0, color=S.GRID, lw=0.8)
ax.set_xlabel("log v0"); ax.set_ylabel("contribution (mm)"); ax.set_title("What each of 4 neurons contributes")
lo_, hi_ = ax.get_ylim(); ax.set_ylim(lo_ - 0.35 * (hi_ - lo_), hi_)   # room below the curves for the legend
ax.legend(fontsize=7.8, ncol=2, loc="lower left")
ax = axs[2]
ax.plot(lv_grid, y_grid, "o", ms=2.5, color=S.MUTED, label="true model (200 runs)")
ax.plot(lv_grid, parts.sum(1) + b2[0] * Y_SD + Y_MU, color=S.INK, lw=1.8, label="sum of the 4 + constant")
ax.set_xlabel("log v0"); ax.set_ylabel("displacement, day 30 to year 1 (mm)"); ax.set_title("The network's output")
ax.legend(fontsize=8)
fig.tight_layout()
bk.savefig(fig, "3-02_neurons")
plt.show()

# %% [markdown]
# **How to read it.** Left: tanh bends a straight input into an S between −1 and 1; ReLU cuts off the negative half.
# Middle: each tanh neuron, after its output weight, is a smooth step placed and stretched by its own weight and bias.
# Right: their sum (plus one constant) is the network's prediction. Fitting the network means choosing where to put
# the steps and how tall to make them.
#
# **Worked example: one forward pass by hand.** For the fitted 4-neuron network and one input, compute
# $a_j = w_{1j}x + b_{1j}$, $h_j=\tanh a_j$, then $f = \sum_j w_{2j}h_j + b_2$ (in scaled units), then mm.

# %%
x0 = x_grid[120]
a_ = W1[0] * x0 + b1
h_ = np.tanh(a_)
f_ = h_ @ W2[:, 0] + b2[0]
print(f"input: log v0 = {lv_grid[120]:.3f}  ->  scaled x = {x0:.4f}")
for j in range(4):
    print(f"  neuron {j + 1}: a = {W1[0, j]:+.3f} * {x0:.4f} {b1[j]:+.3f} = {a_[j]:+.4f};  tanh(a) = {h_[j]:+.4f};"
          f"  times output weight {W2[j, 0]:+.3f} = {h_[j] * W2[j, 0]:+.4f}")
print(f"  sum + output bias {b2[0]:+.4f} = {f_:+.4f} (scaled)  ->  {f_ * Y_SD + Y_MU:.2f} mm;"
      f" network function gives {pred_1d(p4, [x0])[0]:.2f} mm; true model {y_grid[120]:.2f} mm")

# %% [markdown]
# **Common misreadings.**
# * "Neurons are like brain cells." Here a neuron is one line of algebra; the name is historical.
# * "The weights mean something physical." Different weight sets give the same function (swap two neurons, or flip the
#   sign of a tanh neuron's input and output weights); individual weights are not identifiable.

# %% [markdown]
# ## 2. Width: universal approximation
#
# **In plain words.** More neurons, more steps to build with, so finer shapes. With enough neurons in a single hidden
# layer, a network can match any smooth curve on a bounded range as closely as we like.
#
# **Precisely.** Universal approximation (Cybenko 1989; Hornik 1991): for a continuous $g$ on a compact set (a closed,
# bounded range of inputs) and any
# $\varepsilon > 0$ there is a one-hidden-layer network with a non-polynomial activation and *some* finite width with
# $\max|f - g| < \varepsilon$. The theorem says such weights exist; it does not say how wide, how many examples are
# needed, or that training will find them. Below: one hidden layer of width 1 to 64, fitted to all 200 points; the error
# here measures what the network can represent, not how well it generalises (section 6 does that).

# %%
WIDTHS = [1, 2, 4, 8, 16, 64]
fits_w = {w: fit_1d([w], x_grid, y_grid) for w in WIDTHS}
err_w = {w: np.sqrt(((pred_1d(fits_w[w][0], x_grid) - y_grid) ** 2).mean()) for w in WIDTHS}
for w in WIDTHS:
    n_par = 3 * w + 1
    print(f"width {w:3d} ({n_par:4d} parameters): RMS error {err_w[w]:9.4f} mm")

# %%
fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.9))
ax = axs[0]
ax.plot(lv_grid, y_grid, "o", ms=2.5, color=S.MUTED, label="true model")
for w, col in zip([1, 2, 4], [S.ORANGE, S.AQUA, S.BLUE]):
    ax.plot(lv_grid, pred_1d(fits_w[w][0], x_grid), color=col, lw=1.6, label=f"width {w}")
ax.set_xlabel("log v0"); ax.set_ylabel("displacement, day 30 to year 1 (mm)"); ax.set_title("Fits with 1, 2 and 4 neurons")
ax.legend(fontsize=8)
ax = axs[1]
ax.loglog(WIDTHS, [err_w[w] for w in WIDTHS], "o-", color=S.INK)
ax.axhline(SIG, color=S.MUTED, ls="--", lw=1); ax.text(WIDTHS[-1], SIG * 1.2, "σ = 2 mm", ha="right", fontsize=8.5)
ax.set_xlabel("neurons in the hidden layer"); ax.set_ylabel("RMS error on the 200 points (mm)")
ax.set_title("Error against width"); ax.set_xticks(WIDTHS); ax.set_xticklabels([str(w) for w in WIDTHS])
fig.tight_layout()
bk.savefig(fig, "3-02_width")
plt.show()

# %% [markdown]
# **How to read it.** One neuron makes one smooth step and cannot make a bump; two can (up, then down) and already follow
# the curve closely. Past a few neurons the error stops improving and even goes up and down from width to width (printed
# values): it is then limited by the optimiser and its budget of steps, not by the width.
#
# **Common misreadings.**
# * "Universal approximation means a network will learn anything." It guarantees that good weights exist, not that we
#   have enough runs to pin them down or that gradient descent finds them.

# %% [markdown]
# ## 3. Backpropagation is reverse-mode autodiff
#
# **In plain words.** Training needs the slope of the loss with respect to every weight. Backpropagation computes all
# of them in one backward sweep: run the network forward, keep the intermediate values, then apply the chain rule from
# the loss back to the inputs (the chain rule: the derivative of a function of a function is the product of their
# derivatives). Notebook 2.17 showed this is what JAX's `grad` does for any program; a network is just a program.
#
# **Precisely.** For one hidden layer, $\mathbf a = \mathbf x\mathbf W_1 + \mathbf b_1$, $\mathbf h = \tanh\mathbf a$,
# $\mathbf f = \mathbf h\mathbf W_2 + \mathbf b_2$, loss $L = \frac1N\sum(\mathbf f - \mathbf y)^2$. With
# $\mathbf r = \partial L/\partial\mathbf f = \tfrac2N(\mathbf f-\mathbf y)$:
#
# $$\frac{\partial L}{\partial\mathbf W_2} = \mathbf h^\top\mathbf r,\quad \frac{\partial L}{\partial\mathbf b_2} = \textstyle\sum\mathbf r,\quad
# \boldsymbol\delta = (\mathbf r\mathbf W_2^\top)\odot(1-\mathbf h^2),\quad \frac{\partial L}{\partial\mathbf W_1} = \mathbf x^\top\boldsymbol\delta,\quad
# \frac{\partial L}{\partial\mathbf b_1} = \textstyle\sum\boldsymbol\delta.$$
#
# Cost: one backward sweep costs a small multiple of one forward pass, whatever the number of parameters $P$
# (Rumelhart, Hinton & Williams 1986). Finite differences need $2P$ forward passes for central differences.
#
# **Worked example.** The formulas above, coded by hand, against `jax.grad`, on the 4-neuron network of section 1.

# %%
xs1, ys1 = x_grid[:, None], ((y_grid - Y_MU) / Y_SD)[:, None]
a1 = xs1 @ W1 + b1; h1 = np.tanh(a1); f1 = h1 @ W2 + b2
r = 2 * (f1 - ys1) / len(xs1)
hand = {"W1": xs1.T @ ((r @ W2.T) * (1 - h1 ** 2)), "b1": ((r @ W2.T) * (1 - h1 ** 2)).sum(0), "W2": h1.T @ r, "b2": r.sum(0)}
auto = jax.grad(lambda p: jnp.mean((mlp(p, jnp.asarray(xs1)) - jnp.asarray(ys1)) ** 2))(p4)
auto = {"W1": auto[0][0], "b1": auto[0][1], "W2": auto[1][0], "b2": auto[1][1]}
for k_ in hand:
    print(f"dL/d{k_}: by hand {np.round(np.ravel(hand[k_])[:3], 6)} ...; largest difference to jax.grad "
          f"{np.abs(hand[k_] - np.asarray(auto[k_])).max():.1e}")

# cost against the number of parameters: forward pass vs reverse-mode gradient (measured), finite differences (derived)
xb = jnp.asarray(np.random.default_rng(0).uniform(-1, 1, (1000, 3)))
yb = jnp.asarray(np.random.default_rng(1).standard_normal((1000, K)))
cost = []
for w in (8, 32, 128, 512):
    pw = init_mlp(jax.random.PRNGKey(1), [3, w, w, w, K])
    n_par = sum(W.size + b.size for W, b in pw)
    lf = jax.jit(lambda p: jnp.mean((mlp(p, xb) - yb) ** 2))
    gf = jax.jit(jax.grad(lambda p: jnp.mean((mlp(p, xb) - yb) ** 2)))
    lf(pw).block_until_ready(); jax.block_until_ready(gf(pw))
    reps = 20
    t0 = time.time()
    for _ in range(reps):
        lf(pw).block_until_ready()
    t_f = (time.time() - t0) / reps
    t0 = time.time()
    for _ in range(reps):
        jax.block_until_ready(gf(pw))
    t_g = (time.time() - t0) / reps
    cost.append((n_par, t_f, t_g))
    print(f"width {w:3d}: {n_par:7,d} parameters; forward {1e3 * t_f:7.3f} ms, reverse-mode gradient {1e3 * t_g:7.3f} ms"
          f" (ratio {t_g / t_f:.1f}); central finite differences would need {2 * n_par:,} forward passes"
          f" = {2 * n_par * t_f:.1f} s (derived)")

# %%
cost = np.array(cost)
fig, ax = plt.subplots(figsize=(6.4, 4.0))
ax.loglog(cost[:, 0], 1e3 * cost[:, 1], "o-", color=S.INK, label="forward pass (measured)")
ax.loglog(cost[:, 0], 1e3 * cost[:, 2], "s-", color=S.BLUE, label="all P derivatives, reverse mode (measured)")
ax.loglog(cost[:, 0], 1e3 * 2 * cost[:, 0] * cost[:, 1], "^--", color=S.ORANGE,
          label="all P derivatives, finite differences (2P × forward, derived)")
ax.set_xlabel("number of parameters P"); ax.set_ylabel("time (ms), batch of 1,000 inputs")
ax.set_title("The cost of a gradient"); ax.legend(fontsize=7.8)
fig.tight_layout()
bk.savefig(fig, "3-02_backprop")
plt.show()

# %% [markdown]
# **How to read it.** The hand-coded derivatives agree with `jax.grad` to rounding error (printed). In the timing plot,
# the reverse-mode gradient stays within a small factor of one forward pass at every size, while finite differences
# grow with the number of parameters. That gap is why networks with many parameters can be trained at all.
#
# **Common misreadings.**
# * "Backpropagation is a special neural-network algorithm." It is the chain rule applied in reverse order, the same
#   reverse-mode autodiff that gives NUTS its gradients in notebook 2.17 and that 2.22 calls the adjoint.

# %% [markdown]
# ## 4. Initialisation: keeping the signal alive through many layers
#
# **In plain words.** Before training, the weights are random. If they are too small, every layer shrinks the signal
# and after many layers nothing is left; the slopes that training needs shrink the same way on the way back
# (vanishing gradients). If they are too large, every tanh is pushed to ±1 and the network becomes a jumble of saturated
# switches (exploding or chaotic). The usual recipe picks the spread of the weights so that the signal keeps roughly
# its size from layer to layer.
#
# **Precisely.** With weights $W_{ij}\sim N(0, s^2/n_\text{in})$ the variance of a pre-activation (the weighted sum
# $\mathbf W\mathbf x + \mathbf b$, before the bend) is about $s^2$ times
# the mean square of the layer's input. For tanh, $s = 1$ roughly preserves the signal (Glorot & Bengio 2010 give the
# closely related $2/(n_\text{in}+n_\text{out})$); $s < 1$ shrinks it geometrically with depth, $s > 1$ saturates it.
# Below: a 30-layer tanh network of width 64 at initialisation, fed 1,000 random inputs, with $s$ = 0.5, 1 and 2: the
# spread of each layer's output (forward), and the size of the loss gradient with respect to each layer's weights
# (backward).

# %%
DEPTH_I = 30
xi = jnp.asarray(np.random.default_rng(2).uniform(-1, 1, (1000, 3)))
yi = jnp.asarray(np.random.default_rng(3).standard_normal((1000, 1)))
init_stats = {}
for s_ in (0.5, 1.0, 2.0):
    pi = init_mlp(jax.random.PRNGKey(4), [3] + [64] * DEPTH_I + [1], scale=s_)
    hs, h = [], xi
    for W, b in pi[:-1]:
        h = jnp.tanh(h @ W + b); hs.append(float(h.std()))
    g = jax.grad(lambda p: jnp.mean((mlp(p, xi) - yi) ** 2))(pi)
    gn = [float(jnp.linalg.norm(W)) for W, _ in g[:-1]]
    sat = float(jnp.mean(jnp.abs(h) > 0.99))
    init_stats[s_] = (hs, gn)
    print(f"s = {s_}: output spread of layer 1 {hs[0]:.3f}, layer {DEPTH_I} {hs[-1]:.2e};  gradient size at layer 1"
          f" {gn[0]:.2e}, layer {DEPTH_I} {gn[-1]:.2e};  share of saturated units (|h| > 0.99) in layer {DEPTH_I} {sat:.2f}")

# %%
fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.9))
for s_, col in zip((0.5, 1.0, 2.0), (S.ORANGE, S.INK, S.BLUE)):
    hs, gn = init_stats[s_]
    axs[0].semilogy(np.arange(1, DEPTH_I + 1), hs, "o-", ms=3, color=col, label=f"s = {s_}")
    axs[1].semilogy(np.arange(1, DEPTH_I + 1), gn, "o-", ms=3, color=col, label=f"s = {s_}")
axs[0].set_xlabel("layer"); axs[0].set_ylabel("spread (sd) of the layer's output")
axs[0].set_title("Forward: does the signal survive?")
axs[1].set_xlabel("layer"); axs[1].set_ylabel("size of dL/dW for the layer")
axs[1].set_title("Backward: does the gradient survive?")
axs[0].legend(fontsize=8.5)
fig.tight_layout()
bk.savefig(fig, "3-02_init")
plt.show()

# %% [markdown]
# **How to read it.** With s = 0.5 (orange) the signal shrinks by a constant factor per layer (a straight line on the
# log scale) and the gradients of all layers are tiny: there is almost nothing for training to work with. With s = 1
# (black) the signal keeps a modest size and the gradient changes by about a factor of ten over thirty layers (printed).
# With s = 2 (blue) the signal stays large and the gradient grows towards the first layers instead of shrinking, an
# exploding gradient; section 9 shows that this start also trains badly.
#
# **Common misreadings.**
# * "Initialisation only affects the starting point." For deep networks it decides whether gradient descent can move
#   the early layers at all.

# %% [markdown]
# ## 5. Training: gradient descent, momentum and Adam
#
# **In plain words.** Training walks downhill on the loss. Plain gradient descent takes a step against the slope,
# scaled by the learning rate (the step size: how far one step moves, per unit of slope). Momentum keeps part of the previous step, so it rolls through flat stretches and damps
# zig-zags across narrow valleys (2.10 showed both on a 2-D landscape). Adam goes further and gives every parameter its
# own step size: the running average of its gradient divided by the running size of that gradient, so parameters with
# small, consistent gradients move as fast as those with large ones.
#
# **Precisely.** With gradient $\mathbf g_t$: gradient descent $w \leftarrow w - \eta\,\mathbf g_t$; momentum
# $\mathbf v \leftarrow \beta\mathbf v + \mathbf g_t$, $w\leftarrow w - \eta\mathbf v$; Adam
# $\mathbf m \leftarrow \beta_1\mathbf m + (1-\beta_1)\mathbf g_t$, $\mathbf v \leftarrow \beta_2\mathbf v + (1-\beta_2)\mathbf g_t^2$,
# $w \leftarrow w - \eta\,\hat{\mathbf m}/(\sqrt{\hat{\mathbf v}}+\epsilon)$ with bias-corrected $\hat{\mathbf m} = \mathbf m/(1-\beta_1^t)$,
# $\hat{\mathbf v} = \mathbf v/(1-\beta_2^t)$ (Kingma & Ba 2015; usual $\beta_1 = 0.9$, $\beta_2 = 0.999$). The bias
# correction is needed because both running averages start at zero and would be too small for the first steps. The
# whole of Adam, as used in every Topic 3 notebook, is printed below. The comparison: the 1-D slice, two hidden layers
# of 16 tanh neurons, 3,000 full-batch steps (every step uses all training examples, not a mini-batch, 2.10) at a
# constant learning rate, each optimiser at each of six learning rates; same starting weights for all.

# %%
print(inspect.getsource(adam))


def gd_momentum(loss, params, n_steps, lr, beta):
    """beta = 0: plain gradient descent; beta > 0: heavy-ball momentum."""
    def step(carry, i):
        p, vel = carry
        L, g = jax.value_and_grad(loss)(p)
        vel = jax.tree.map(lambda v_, g_: beta * v_ + g_, vel, g)
        return (jax.tree.map(lambda p_, v_: p_ - lr * v_, p, vel), vel), L
    (params, _), hist = jax.lax.scan(step, (params, jax.tree.map(jnp.zeros_like, params)), jnp.arange(n_steps))
    return params, hist


xs_, ys_ = jnp.asarray(x_grid)[:, None], jnp.asarray((y_grid - Y_MU) / Y_SD)[:, None]
loss_1d = lambda p: jnp.mean((mlp(p, xs_) - ys_) ** 2)
p_start = init_mlp(jax.random.PRNGKey(5), [1, 16, 16, 1])
LRS = [1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1]
runners = {"gradient descent": jax.jit(lambda lr: gd_momentum(loss_1d, p_start, 3000, lr, 0.0)[1]),
           "momentum (β = 0.9)": jax.jit(lambda lr: gd_momentum(loss_1d, p_start, 3000, lr, 0.9)[1]),
           "Adam": jax.jit(lambda lr: adam(loss_1d, p_start, 3000, lr, cosine=False)[1])}
OCOL = {"gradient descent": S.MUTED, "momentum (β = 0.9)": S.AQUA, "Adam": S.BLUE}
opt_hist = {o: {lr: np.asarray(f(lr)) for lr in LRS} for o, f in runners.items()}
to_mm = lambda L: np.sqrt(L) * Y_SD                             # RMS error in mm from the scaled MSE
best = {}
for o in runners:
    fin = {lr: (to_mm(h[-1]) if np.isfinite(h[-1]) else np.inf) for lr, h in opt_hist[o].items()}
    best[o] = min(fin, key=fin.get)
    print(f"{o:20s} final RMS error (mm) by learning rate {LRS}: "
          + " ".join(f"{v:9.3f}" if np.isfinite(v) else "  diverged" for v in fin.values())
          + f";  best {best[o]:g}")

# %%
fig, axs = plt.subplots(1, 2, figsize=(11.5, 3.9))
for o in runners:
    axs[0].loglog(np.arange(1, 3001), to_mm(opt_hist[o][best[o]]), color=OCOL[o], label=f"{o}, lr = {best[o]:g}")
    fin = [to_mm(opt_hist[o][lr][-1]) for lr in LRS]
    fin = [v if np.isfinite(v) else np.nan for v in fin]
    axs[1].loglog(LRS, fin, "o-", color=OCOL[o], label=o)
axs[0].axhline(SIG, color=S.INK, ls="--", lw=1)
axs[0].set_xlabel("training step"); axs[0].set_ylabel("RMS error on the 200 points (mm)")
axs[0].set_title("Each optimiser at its best learning rate"); axs[0].legend(fontsize=8)
axs[1].set_xlabel("learning rate"); axs[1].set_ylabel("RMS error after 3,000 steps (mm)")
axs[1].set_title("Sensitivity to the learning rate\n(missing points: diverged)", fontsize=10)
fig.tight_layout()
bk.savefig(fig, "3-02_optimisers")
plt.show()

# %% [markdown]
# **How to read it.** Left: the loss histories at each optimiser's best learning rate; right: the final error against
# the learning rate (a missing point means the run blew up). Here Adam reached the lowest error in the same number of
# steps, and momentum and Adam both reached a small error over a wider range of learning rates than plain gradient
# descent (printed). The spikes in Adam's curve are overshoots at a constant, fairly large learning rate, after which
# it recovers; the cosine schedule used in the later sections lowers the learning rate and the spikes die out.
#
# **Common misreadings.**
# * "Adam always wins." It is a robust default, which is why the Topic 3 notebooks use it; how much it gains depends on the
#   problem and on tuning, so compare optimisers each at its own best learning rate, as here.
# * "The training loss going down means the surrogate is good." It says nothing about runs the network has not seen
#   (next section).

# %% [markdown]
# ## 6. Overfitting, weight decay and early stopping
#
# **In plain words.** With few, noisy examples and many parameters, a network can thread its curve through the noise:
# a tiny training loss and a bad prediction everywhere else. Two standard defences: penalise large weights (weight
# decay), which prefers smoother functions, or stop training when the error on held-out examples starts to rise (early
# stopping). Both are choices about what functions we believe in before seeing the data, which is exactly what a prior
# is.
#
# **Precisely.** With noise $\sigma_\text{toy}$ and a Gaussian prior $w\sim N(0, s_w^2\mathbf I)$ on all weights, the
# negative log posterior is
#
# $$\mathcal L(w) = \sum_i \frac{(y_i - f(x_i;w))^2}{2\sigma_\text{toy}^2} + \frac{\lVert w\rVert^2}{2 s_w^2} + \text{const},$$
#
# so minimising "data misfit + weight decay" is a MAP estimate (2.08) of a Bayesian neural network, a network whose
# weights are given a prior and, after training, a posterior (MacKay 1992; Neal 1996),
# with decay coefficient $\lambda = 1/s_w^2$. Early stopping acts like a prior too: gradient descent from small weights
# explores simple functions first (the Landweber picture of 2.10). **This section uses a noise of 10 mm**, larger than
# GPS noise, so that overfitting is visible with only 20 examples. Setup: 20 training and 20 validation points drawn
# from the 200-point grid with independent noise; two hidden layers of 64 tanh neurons; Adam at a constant learning
# rate for 6,000 steps. The prior width $s_w$ is chosen from four values by the validation loss; the true curve (the
# 200 noise-free runs) is used only to grade.

# %%
SIG_TOY = 10.0
r6 = np.random.default_rng(6)
idx = r6.permutation(200)
i_tr, i_va = np.sort(idx[:20]), np.sort(idx[20:40])
y_tr = y_grid[i_tr] + SIG_TOY * r6.standard_normal(20)
y_va = y_grid[i_va] + SIG_TOY * r6.standard_normal(20)
sc = lambda y: (y - Y_MU) / Y_SD
xt, yt = jnp.asarray(x_grid[i_tr])[:, None], jnp.asarray(sc(y_tr))[:, None]
xv, yv = jnp.asarray(x_grid[i_va])[:, None], jnp.asarray(sc(y_va))[:, None]
s_toy = SIG_TOY / Y_SD                                            # noise in scaled units


def neg_log_post(p, sw):
    misfit = jnp.sum((mlp(p, xt) - yt) ** 2) / (2 * s_toy ** 2)
    prior = 0.0 if sw is None else sum(jnp.sum(W ** 2) for W, _ in p) / (2 * sw ** 2)
    return misfit + prior


val_mm = lambda p: jnp.sqrt(jnp.mean((mlp(p, xv) - yv) ** 2)) * Y_SD
p0_6 = init_mlp(jax.random.PRNGKey(7), [1, 64, 64, 1])
N6 = 6000
runs6 = {}
for sw in (None, 0.3, 1.0, 3.0, 10.0):
    p_, hist_, vhist = jax.jit(lambda p: adam(lambda q: neg_log_post(q, sw), p, N6, 3e-3, cosine=False,
                                              monitor=val_mm))(p0_6)
    tr_mm = float(jnp.sqrt(jnp.mean((mlp(p_, xt) - yt) ** 2)) * Y_SD)
    runs6[sw] = (p_, np.asarray(vhist), tr_mm)
for sw, (p_, vh, tr_mm) in runs6.items():
    true_err = np.sqrt(((pred_1d(p_, x_grid) - y_grid) ** 2).mean())
    print(f"prior sd s_w = {str(sw):5s} (decay λ = {('none' if sw is None else f'{1 / sw ** 2:.3g}'):>6s}): training RMS"
          f" {tr_mm:6.2f} mm, validation RMS {vh[-1]:6.2f} mm, error against the true curve {true_err:6.2f} mm")
sw_best = min([0.3, 1.0, 3.0, 10.0], key=lambda s: runs6[s][1][-1])
vh_plain = runs6[None][1]
stop = int(np.argmin(vh_plain))
p_es, _, _ = jax.jit(lambda p: adam(lambda q: neg_log_post(q, None), p, stop, 3e-3, cosine=False))(p0_6) if stop > 0 \
    else (p0_6, None, None)
err_es = np.sqrt(((pred_1d(p_es, x_grid) - y_grid) ** 2).mean())
print(f"weight decay chosen by validation: s_w = {sw_best} (λ = {1 / sw_best ** 2:.3g})")
print(f"early stopping: validation error is smallest at step {stop} of {N6}; that network's error against the true curve"
      f" {err_es:.2f} mm (noise of the examples: {SIG_TOY} mm)")

# %%
fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.1))
ax = axs[0]
ax.plot(lv_grid, y_grid, color=S.INK, lw=2.2, label="true curve")
ax.plot(lv_grid[i_tr], y_tr, "o", color=S.INK, ms=5, mfc="none", label="20 noisy training points")
ax.plot(lv_grid, pred_1d(runs6[None][0], x_grid), color=S.BAD, lw=1.4, label="no prior, 6,000 steps")
ax.plot(lv_grid, pred_1d(runs6[sw_best][0], x_grid), color=S.BLUE, lw=1.6, label=f"weight decay (s_w = {sw_best})")
ax.plot(lv_grid, pred_1d(p_es, x_grid), color=S.AQUA, lw=1.6, ls="--", label=f"early stopping (step {stop})")
ax.set_xlabel("log v0"); ax.set_ylabel("displacement, day 30 to year 1 (mm)"); ax.set_title("Three networks, 20 examples")
ax.legend(fontsize=7.8)
ax = axs[1]
ax.semilogx(np.arange(1, N6 + 1), vh_plain, color=S.BAD, label="validation, no prior")
ax.semilogx(np.arange(1, N6 + 1), runs6[sw_best][1], color=S.BLUE, label=f"validation, weight decay (s_w = {sw_best})")
ax.axvline(stop, color=S.AQUA, ls="--", lw=1.2); ax.text(stop * 1.1, ax.get_ylim()[1] * 0.92, "early stop", color=S.AQUA,
                                                         fontsize=8.5)
ax.axhline(SIG_TOY, color=S.MUTED, ls=":", lw=1); ax.text(1.2, SIG_TOY * 0.93, "noise level (10 mm)", fontsize=8, color=S.INK2)
ax.set_xlabel("training step"); ax.set_ylabel("RMS error on the 20 validation points (mm)")
ax.set_title("Held-out error during training"); ax.legend(fontsize=8)
fig.tight_layout()
bk.savefig(fig, "3-02_overfitting")
plt.show()

# %% [markdown]
# **How to read it.** Left: the unpenalised network after 6,000 steps bends towards individual noisy points (its
# training error is well below the noise, printed); the weight-decay and early-stopped networks stay closer to the true
# curve (errors printed above). Right: the held-out
# error of the unpenalised network falls, reaches a minimum and then rises as it starts fitting noise; the dashed line
# is where early stopping would stop.
#
# **Worked example: weight decay as a prior.** The chosen prior standard deviation $s_w$ (printed) means: before seeing
# data, each weight is believed to be within about $\pm 2 s_w$ of zero. The decay coefficient is $\lambda = 1/s_w^2$,
# and each gradient step shrinks every weight by $\eta\,w/s_w^2$ on top of the data term.
#
# **Common misreadings.**
# * "Weight decay is a numerical trick." It is a Gaussian prior on the weights; choosing $s_w$ by validation is the
#   network version of choosing a smoothing weight by cross-validation (2.29).
# * "A network's MAP is its posterior." It is one point; ensembles (section 11) and Bayesian neural networks try to
#   describe the spread.

# %% [markdown]
# ## 7. Input and output scaling
#
# **In plain words.** The initialisation recipe of section 4 assumes inputs of size about one, and the last layer starts
# with outputs of size about one. Feed the network log v0 between −7 and 0 and ask for coefficients in the hundreds of
# millimetres, and it spends most of its training just learning the units. Scaling inputs to a unit range and outputs
# to a unit spread costs nothing and is undone after prediction.
#
# **Precisely.** Single-patch surrogate (as 3.01), 800 training runs, three hidden layers of 64 tanh neurons, Adam with a
# cosine schedule, 3,000 steps, four versions: (a) raw θ in, raw PCA coefficients (mm) out; (b) θ scaled to [0, 1],
# raw coefficients out; (c) θ scaled, coefficients divided by one common scale (the spread of coefficient 1, printed
# at the top); (d) raw θ in, scaled coefficients out. One common scale keeps the loss proportional to the squared error
# in millimetres, because the PCA modes are orthonormal (mutually perpendicular and of unit length, 2.06). The PCA basis
# comes from the same 800 runs. The monitored quantity is the RMS error over the 40 epochs on the 400 validation runs.

# %%
def surrogate_task(n, x_in, scaled, sizes, n_steps, lr=3e-3, seed=0, net=mlp, init_scale=1.0, idx=None):
    """Train net on the first n training runs (or the runs idx): x_in (4000, 3) inputs, outputs the PCA coefficients of
    those runs' own basis, divided by the spread of coefficient 1 if scaled. Monitor validation RMS (mm)."""
    idx = TR[:n] if idx is None else idx
    mu, V, zsd = pca_basis(idx)
    z_scale = zsd if scaled else 1.0
    Xt, Zt = jnp.asarray(x_in[idx]), jnp.asarray(((U_all[idx] - mu) @ V.T) / z_scale)
    Xv, Uv, Vj, MUj = jnp.asarray(x_in[VA]), jnp.asarray(U_all[VA]), jnp.asarray(V), jnp.asarray(mu)
    loss = lambda p: jnp.mean((net(p, Xt) - Zt) ** 2)
    mon = lambda p: jnp.sqrt(jnp.mean(((net(p, Xv) * z_scale) @ Vj + MUj - Uv) ** 2))
    p0 = init_mlp(jax.random.PRNGKey(seed), sizes, init_scale)
    t0 = time.time()
    p, hist, vh = jax.jit(lambda p: adam(loss, p, n_steps, lr, monitor=mon))(p0)
    vh = np.asarray(vh)
    return dict(params=p, val=vh, time=time.time() - t0,
                predict=lambda xq: (np.asarray(net(p, jnp.asarray(xq))) * z_scale) @ V + mu)


SIZES = [3, 64, 64, 64, K]
scal = {"(a) raw inputs, raw outputs": surrogate_task(800, th_all, False, SIZES, 3000),
        "(b) scaled inputs, raw outputs": surrogate_task(800, X_all, False, SIZES, 3000),
        "(c) scaled inputs and outputs": surrogate_task(800, X_all, True, SIZES, 3000),
        "(d) raw inputs, scaled outputs": surrogate_task(800, th_all, True, SIZES, 3000)}
for k_, r_ in scal.items():
    print(f"{k_:32s}: validation RMS after 3,000 steps {r_['val'][-1]:8.2f} mm  ({r_['time']:.0f} s)")
print(f"spread of the validation displacements about their mean: {np.sqrt(((U_all[VA] - U_all[VA].mean(0)) ** 2).mean()):.0f} mm")
print(f"Adam moves a weight by at most about the learning rate (0.003) per step. With raw outputs the last layer must"
      f" produce coefficients of size ~{Z_SCALE:.0f} mm from 64 units of size <= 1, so its weights must reach at least"
      f" ~{Z_SCALE / 64:.0f}: at least ~{Z_SCALE / 64 / 3e-3:,.0f} steps (derived lower bound), vs 3,000 available")

# %%
fig, ax = plt.subplots(figsize=(6.6, 4.0))
for (k_, r_), col, ls in zip(scal.items(), (S.BAD, S.ORANGE, S.BLUE, S.AQUA), ("-", "--", "-", "--")):
    ax.loglog(np.arange(1, 3001), r_["val"], color=col, ls=ls, label=k_)
ax.axhline(SIG, color=S.INK, ls="--", lw=1); ax.text(1.2, SIG * 1.15, "σ = 2 mm", fontsize=8.5)
ax.set_xlabel("training step"); ax.set_ylabel("validation RMS error (mm)")
ax.set_title("Same network, same steps, different units"); ax.legend(fontsize=8)
fig.tight_layout()
bk.savefig(fig, "3-02_scaling")
plt.show()

# %% [markdown]
# **How to read it.** Same architecture, same optimiser, same number of steps; only the units differ. Here the output
# units decide everything: with raw outputs (a, b) the error stays near the spread of the data, whatever the inputs,
# and with scaled outputs (c, d) it falls by orders of magnitude (printed). The reason is in Adam's update: each weight
# moves by at most about the learning rate per step, so last-layer weights that must grow to the size of the raw
# coefficients need more steps than the run has (printed lower bound). In this problem the input scaling matters much
# less; compare (c) and (d).
#
# **Common misreadings.**
# * "Adam makes scaling irrelevant." Adam rescales each parameter's step, but the network's starting function and the
#   shape of its tanh units still depend on the size of the inputs and targets.
# * "Standardise every output separately." For PCA coefficients that gives the tiny late modes the same weight in the
#   loss as the first; one common scale keeps the loss in millimetres.

# %% [markdown]
# ## 8. A better-trained network against the Gaussian process
#
# **In plain words.** In 3.01 the network got 4,000 steps and its error stopped falling after a few hundred runs, while
# its error on its own training runs stayed as large as its test error: it was under-trained, not short of data. Here
# the same network gets four times as many steps, and we compare it with the GP of 3.01 on the same runs.
#
# **Precisely.** Three hidden layers of 64 tanh neurons, scaled inputs and outputs (version c), Adam with a cosine
# schedule, 16,000 full-batch steps, on the first 400 and 800 training runs; the GP is 3.01's (Matérn-5/2, one length
# scale per input, shared by the coefficients, hyperparameters by marginal likelihood from two starts). Validation runs
# monitor training; the 400 test runs grade once at the end.

# %%
def k52(X1, X2, ell):
    a, b = X1 / ell, X2 / ell
    d2 = jnp.sum(a * a, 1)[:, None] + jnp.sum(b * b, 1)[None, :] - 2 * a @ b.T
    r_ = jnp.sqrt(5.0 * jnp.maximum(d2, 0.0) + 1e-24)
    return (1 + r_ + r_ * r_ / 3) * jnp.exp(-r_)


def fit_gp(idx, n_hyper=800):
    """The GP surrogate of 3.01 on the runs idx (on PCA coefficients of their own basis); returns x -> u (mm)."""
    mu, V, _ = pca_basis(idx)
    X, Z = X_all[idx], (U_all[idx] - mu) @ V.T
    zm, zs = Z.mean(0), Z.std(0)
    Zs = (Z - zm) / zs
    nh = min(len(X), n_hyper)

    def nlml(lp):
        ell, amp, nug = jnp.exp(lp[:3]), jnp.exp(2 * lp[3]), jnp.exp(2 * lp[4])
        Kx = amp * k52(jnp.asarray(X[:nh]), jnp.asarray(X[:nh]), ell) + (nug + 1e-10) * jnp.eye(nh)
        L = jnp.linalg.cholesky(Kx)
        return 0.5 * jnp.sum(Zs[:nh] * jax.scipy.linalg.cho_solve((L, True), Zs[:nh])) + Zs.shape[1] * jnp.sum(jnp.log(jnp.diag(L)))
    f = jax.jit(jax.value_and_grad(nlml))
    res = [minimize(lambda p: [np.asarray(v, float) for v in f(jnp.asarray(p))], np.log(st), jac=True, method="L-BFGS-B",
                    bounds=[(-5, 2)] * 3 + [(-3, 3), (-12, 0)])
           for st in ([0.3, 0.3, 0.3, 1.0, 1e-2], [0.1, 0.1, 0.3, 1.0, 1e-4])]
    lp = min(res, key=lambda r_: r_.fun).x
    ell, amp, nug = np.exp(lp[:3]), np.exp(2 * lp[3]), np.exp(2 * lp[4])
    Kx = amp * np.asarray(k52(jnp.asarray(X), jnp.asarray(X), ell)) + (nug + 1e-10) * np.eye(len(X))
    W = amp * np.linalg.solve(Kx, Zs) * zs @ V
    return lambda xq: np.asarray(k52(jnp.asarray(xq), jnp.asarray(X), ell)) @ W + zm @ V + mu


def grade(pred):
    e = np.sqrt(((pred - U_all[TE]) ** 2).mean(1))
    return np.sqrt((e ** 2).mean()), np.median(e), (e < SIG).mean()


N8 = 16000
long_nn, gps = {}, {}
for n in (400, 800):
    long_nn[n] = surrogate_task(n, X_all, True, SIZES, N8)
    gps[n] = fit_gp(TR[:n])
    gv = np.sqrt(((gps[n](X_all[VA]) - U_all[VA]) ** 2).mean())
    long_nn[n]["gp_val"] = gv
    for name, pr in (("network, 16,000 steps", long_nn[n]["predict"]), ("Gaussian process", gps[n])):
        rms, med, share = grade(pr(X_all[TE]))
        print(f"n = {n}: {name:22s} test RMS {rms:6.2f} mm, median {med:5.2f} mm, share within sigma {share:.2f}")
    print(f"        network validation RMS at step 4,000 of this run {long_nn[n]['val'][3999]:.2f} mm, at the end"
          f" {long_nn[n]['val'][-1]:.2f} mm; training time {long_nn[n]['time']:.0f} s")

# %%
fig, ax = plt.subplots(figsize=(6.8, 4.1))
steps8 = np.arange(1, N8 + 1)
for n, col in ((400, S.ORANGE), (800, S.BLUE)):
    vh = long_nn[n]["val"]
    med = np.array([np.median(vh[i - i // 20:i + i // 20 + 1]) for i in range(N8)])    # running median over +-5 % of the step
    ax.loglog(steps8, vh, color=col, lw=0.5, alpha=0.3)
    ax.loglog(steps8, med, color=col, lw=2, label=f"network, {n} runs (running median)")
    ax.axhline(long_nn[n]["gp_val"], color=col, ls=":", lw=1.6, label=f"GP, {n} runs")
ax.axhline(SIG, color=S.INK, ls="--", lw=1); ax.text(1.2, SIG * 1.1, "σ = 2 mm", fontsize=8.5)
ax.axvline(4000, color=S.MUTED, lw=0.8); ax.text(4200, 150, "3.01's\nbudget", fontsize=8, color=S.INK2)
ax.set_xlabel("training step"); ax.set_ylabel("validation RMS error (mm)")
ax.set_title("Training longer, against the GP")
fig.legend(*ax.get_legend_handles_labels(), loc="lower center", ncol=2, fontsize=8)
fig.tight_layout(rect=(0, 0.12, 1, 1))
bk.savefig(fig, "3-02_nn_vs_gp")
plt.show()

# %% [markdown]
# **How to read it.** Thick lines: the network's validation error during one long run (running median; the faint
# lines are every step, with Adam's overshoots, which die out as the cosine schedule lowers the learning rate). Dotted
# lines: the GP on the same runs. The vertical line marks 3.01's budget; a run with a 4,000-step schedule is not the same
# as the first 4,000 steps of a 16,000-step one (the learning rate decays differently), so 3.01's numbers are not on
# these curves. The two networks follow almost the same curve until late in training: for most of the run, more steps
# matter more than more runs. The printed test errors grade the final networks: trained four times longer, the network
# on 400 or 800 runs beats the GP on the same runs on every printed measure, and its RMS and share within σ are close
# to those 3.01's GP reached with all 3,200 runs (whose typical error is still smaller).
#
# **Common misreadings.**
# * "Method A beats method B" from one budget. Learning curves (3.01) and training curves (here) both have to be at
#   their ends before two methods are compared, and each method should get comparable care (the GP's hyperparameters
#   were optimised; the network's training length was not, in 3.01).

# %% [markdown]
# ## 9. Depth: plain networks and residual networks
#
# **In plain words.** Deeper networks can build more elaborate shapes from simpler ones, but section 4 showed that a
# deep plain network only passes signal and gradient through if its weights start at just the right size. A residual
# network (ResNet) changes each layer from "replace the signal" to "add a correction to the signal". The identity path
# carries the signal and the gradient through any number of layers, whatever the corrections do.
#
# **Precisely.** Plain: $\mathbf h_{l+1} = \tanh(\mathbf W_l\mathbf h_l + \mathbf b_l)$. Residual (He et al. 2016):
# $\mathbf h_{l+1} = \mathbf h_l + \alpha\,\tanh(\mathbf W_l\mathbf h_l + \mathbf b_l)$, with a first linear layer that lifts the
# input to the width and a last linear layer to the outputs. Here $\alpha = 1/\sqrt{L}$, a common choice that keeps the
# sum of $L$ corrections of order one. Experiment: 24 layers of width 32 on the single-patch task (400 runs, 2,000
# steps), plain (24 tanh layers) and residual (a linear lift and 23 residual blocks), each started with $s$ = 0.5, 1
# and 2. The left panel shows the 23 layers after the first in both.

# %%
def resnet(params, x, act=jnp.tanh):
    """Residual network: lift, then L blocks h <- h + alpha * act(h W + b), then a linear read-out."""
    W, b = params[0]
    h = x @ W + b
    alpha = 1.0 / np.sqrt(len(params) - 2)
    for W, b in params[1:-1]:
        h = h + alpha * act(h @ W + b)
    W, b = params[-1]
    return h @ W + b


DEPTH = 24
SIZES9 = [3] + [32] * DEPTH + [K]
mu4, V4, zsd4 = pca_basis(TR[:400])
Xg, Zg = jnp.asarray(X_all[TR[:400]]), jnp.asarray(((U_all[TR[:400]] - mu4) @ V4.T) / zsd4)
deep = {}
for name, net in (("plain", mlp), ("residual", resnet)):
    for s_ in (0.5, 1.0, 2.0):
        p0 = init_mlp(jax.random.PRNGKey(8), SIZES9, s_)
        g = jax.grad(lambda p: jnp.mean((net(p, Xg) - Zg) ** 2))(p0)
        gn = np.array([float(jnp.linalg.norm(W)) for W, _ in g[1:-1]])
        r_ = surrogate_task(400, X_all, True, SIZES9, 2000, net=net, init_scale=s_, seed=8)
        deep[(name, s_)] = (gn, r_["val"])
        print(f"{name:8s} s = {s_}: gradient size at init, first / middle / last hidden layer {gn[0]:.1e} / "
              f"{gn[len(gn) // 2]:.1e} / {gn[-1]:.1e};  validation RMS after 2,000 steps {r_['val'][-1]:8.2f} mm")

# %%
fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.1))
DCOL = {0.5: S.ORANGE, 1.0: S.INK, 2.0: S.BLUE}
for (name, s_), (gn, vh) in deep.items():
    ls = "-" if name == "plain" else "--"
    axs[0].semilogy(np.arange(1, len(gn) + 1), gn, ls, color=DCOL[s_], lw=1.5, label=f"{name}, s = {s_}")
    axs[1].loglog(np.arange(1, len(vh) + 1), vh, ls, color=DCOL[s_], lw=1.5, label=f"{name}, s = {s_}")
axs[0].set_xlabel("hidden layer"); axs[0].set_ylabel("size of dL/dW at initialisation")
axs[0].set_title(f"Gradients through {DEPTH} layers")
axs[1].axhline(SIG, color=S.MUTED, ls=":", lw=1)
axs[1].set_xlabel("training step"); axs[1].set_ylabel("validation RMS error (mm)")
axs[1].set_title("Training (solid: plain, dashed: residual)")
axs[0].legend(fontsize=7.5, ncol=2)
fig.tight_layout()
bk.savefig(fig, "3-02_depth")
plt.show()

# %% [markdown]
# **How to read it.** Left: at initialisation, the plain network with s = 0.5 has gradients smaller by about six orders
# of magnitude than the others (solid orange), while every residual version (dashed) has a gradient of the same size in
# every layer. Right: the training curves. In this run the best network is the plain one with s = 1; the plain network
# ends far worse with s = 0.5 or s = 2, while the residual network trains well with s = 0.5 or 1 and is slowed, less
# severely, by s = 2 (printed). Adam rescues part of the tiny-gradient case, because it divides each step by the
# gradient's own size, but not all of it. The residual connection does not make a network better; it makes it much less
# sensitive to how it was started.
#
# **Common misreadings.**
# * "Deeper is better." For a three-input surrogate, two or three hidden layers are plenty; depth matters for operators
#   and images (3.05). The point of residual connections is that depth stops being dangerous.

# %% [markdown]
# ## 10. A residual network is Euler's method for an ODE
#
# **In plain words.** "New state = old state + small correction that depends on the state" is exactly one step of
# Euler's method for a differential equation. A residual network with many thin layers is therefore a discretised ODE,
# and as the layers multiply and the corrections shrink, it becomes the ODE itself. (Euler's method advances the state
# by the step length times its current rate of change; it is the simplest ODE solver.) Networks defined that way are called
# neural ODEs; they are trained by differentiating through an ODE solver (2.22's adjoints).
#
# **Precisely.** With one shared block $\mathbf F(\mathbf h)=\tanh(\mathbf W\mathbf h+\mathbf b)$ and $\alpha = T/L$,
# $\mathbf h_{l+1} = \mathbf h_l + \tfrac{T}{L}\mathbf F(\mathbf h_l)$ is forward Euler for $d\mathbf h/dt = \mathbf F(\mathbf h)$ on
# $[0, T]$, with global error (the error at the end, accumulated over all steps) proportional to $1/L$ (Chen et al.
# 2018 build networks on this view). Below: a 2-D state, random $\mathbf W$ and $\mathbf b$, $T = 2$; the ResNet's
# output for $L$ = 2 to 512 against diffrax's adaptive solution (Tsit5, a fifth-order explicit Runge-Kutta solver that
# adjusts its step to meet a tolerance of 1e-10).

# %%
rk = np.random.default_rng(9)
W_ode = jnp.asarray(np.array([[0.3, -2.0], [2.0, 0.3]]) + 0.5 * rk.standard_normal((2, 2)))
b_ode = jnp.asarray(0.3 * rk.standard_normal(2))
F = lambda h: jnp.tanh(h @ W_ode.T + b_ode)
T_END = 2.0
h_starts = jnp.asarray([[1.0, 0.0], [0.0, -1.2], [-0.8, 0.8]])


def resnet_shared(h, L):
    path = [h]
    for _ in range(L):
        h = h + (T_END / L) * F(h)
        path.append(h)
    return jnp.stack(path)


ode = dfx.diffeqsolve(dfx.ODETerm(lambda t, y, a: F(y)), dfx.Tsit5(), 0.0, T_END, 1e-3, h_starts,
                      saveat=dfx.SaveAt(ts=jnp.linspace(0, T_END, 200)),
                      stepsize_controller=dfx.PIDController(rtol=1e-10, atol=1e-10))
h_end = ode.ys[-1]
LS = [2, 4, 8, 16, 32, 64, 128, 256, 512]
err_L = [float(jnp.abs(resnet_shared(h_starts, L)[-1] - h_end).max()) for L in LS]
slope = np.polyfit(np.log(LS[3:]), np.log(err_L[3:]), 1)[0]
for L, e in zip(LS, err_L):
    print(f"L = {L:3d} layers: largest difference to the ODE solution at t = T: {e:.2e}")
print(f"fitted slope of log error against log L (L >= 16): {slope:.2f}  (Euler's method: -1)")

# %%
fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.2))
ax = axs[0]
for i in range(3):
    ax.plot(ode.ys[:, i, 0], ode.ys[:, i, 1], color=S.INK, lw=1.8, label="ODE (diffrax)" if i == 0 else None)
    for L, col, mk in ((4, S.ORANGE, "o"), (16, S.BLUE, "s")):
        pth = np.asarray(resnet_shared(h_starts, L))
        ax.plot(pth[:, i, 0], pth[:, i, 1], mk + "-", color=col, ms=3.5, lw=1,
                label=f"ResNet, {L} layers" if i == 0 else None)
    ax.plot(*np.asarray(h_starts[i]), "k*", ms=10)
ax.set_xlabel("state component 1"); ax.set_ylabel("state component 2"); ax.set_aspect("equal")
ax.set_title("Layers as time steps (★: input)"); ax.legend(fontsize=8)
ax = axs[1]
ax.loglog(LS, err_L, "o-", color=S.INK, label="ResNet vs ODE")
ax.loglog(LS, err_L[3] * LS[3] / np.array(LS, float), "--", color=S.MUTED, label="slope -1")
ax.set_xlabel("number of layers L"); ax.set_ylabel("largest difference at t = T")
ax.set_title("More, thinner layers converge to the ODE"); ax.legend(fontsize=8.5)
fig.tight_layout()
bk.savefig(fig, "3-02_resnet_ode")
plt.show()

# %% [markdown]
# **How to read it.** Left: each residual layer moves the state one step along the flow of the ODE; with 4 layers the
# steps are coarse and drift off the curve, with 16 they follow it closely. Right: the difference at the end shrinks in
# proportion to 1/L (the printed slope), the first-order convergence of Euler's method.
#
# **Common misreadings.**
# * "A ResNet is an ODE solver." Only when the corrections are small and smooth from layer to layer; a trained ResNet
#   with independent layers is a sequence of steps of a time-varying vector field.

# %% [markdown]
# ## 11. Deep ensembles: an error bar from disagreement
#
# **In plain words.** Train several networks the same way but from different random starting weights. Where training
# runs constrain the function, they agree; away from them, they may disagree, and the disagreement hints that the
# prediction is uncertain. But they may also agree on the same wrong extrapolation. The spread of an ensemble is cheap
# and often useful, but nothing forces it to be as large as the actual error.
#
# **Precisely.** A deep ensemble (Lakshminarayanan et al. 2017) of $M$ networks gives a mean $\bar f$ and spread
# $s_f$ at each input. It is not a posterior: the members differ only through their initialisation (and here not
# through resampled data), so the spread reflects how many functions fit the training runs equally well, not a
# probability. "In distribution" means inputs like those the networks were trained on; "out of distribution" means
# inputs outside that range. We test it in two places: (a) the 1-D slice, trained only on log v0 in the left part of the range; (b) the
# single-patch surrogate, 5 networks trained on the first 400 training runs with Δτ ≤ 4 MPa, graded on test runs with
# Δτ ≤ 4 (in distribution) and Δτ > 4 (out of distribution) by the ratio of the actual error to the ensemble spread.

# %%
# (a) 1-D: eight networks, inputs x <= 0.2 only (noise-free runs)
inside1 = x_grid <= 0.2
xs_e, ys_e = jnp.asarray(x_grid[inside1])[:, None], jnp.asarray((y_grid[inside1] - Y_MU) / Y_SD)[:, None]
train_one = lambda key: adam(lambda p: jnp.mean((mlp(p, xs_e) - ys_e) ** 2), init_mlp(key, [1, 32, 32, 1]), 4000, 1e-2)[0]
ens1 = jax.jit(jax.vmap(train_one))(jax.random.split(jax.random.PRNGKey(10), 8))
pred_e = np.stack([np.asarray(mlp(jax.tree.map(lambda a: a[m], ens1), jnp.asarray(x_grid)[:, None]))[:, 0]
                   for m in range(8)]) * Y_SD + Y_MU
mean_e, sd_e = pred_e.mean(0), pred_e.std(0, ddof=1)
err_e = np.abs(mean_e - y_grid)
for lab, msk in (("inside the training range", inside1), ("outside", ~inside1)):
    print(f"1-D ensemble, {lab:25s}: largest error {err_e[msk].max():7.2f} mm, largest spread {sd_e[msk].max():7.2f} mm,"
          f" share of points with error <= 2 x spread {(err_e[msk] <= 2 * sd_e[msk]).mean():.2f}")

# (b) single patch: five networks, first 400 training runs with dtau <= 4 MPa
cut = 4.0
tr_in = TR[:400][th_all[TR[:400], 2] <= cut]
mu_e, V_e, zsd_e = pca_basis(tr_in)
Xe, Ze = jnp.asarray(X_all[tr_in]), jnp.asarray(((U_all[tr_in] - mu_e) @ V_e.T) / zsd_e)
train_sp = lambda key: adam(lambda p: jnp.mean((mlp(p, Xe) - Ze) ** 2), init_mlp(key, SIZES), 6000, 3e-3)[0]
t0 = time.time()
ens2 = jax.jit(jax.vmap(train_sp))(jax.random.split(jax.random.PRNGKey(11), 5))
P5 = np.stack([(np.asarray(mlp(jax.tree.map(lambda a: a[m], ens2), jnp.asarray(X_all[TE]))) * zsd_e) @ V_e + mu_e
               for m in range(5)])
e_te = np.sqrt(((P5.mean(0) - U_all[TE]) ** 2).mean(1))
s_te = np.sqrt((P5.std(0, ddof=1) ** 2).mean(1))
ratio = e_te / s_te
ind = th_all[TE, 2] <= cut
print(f"single-patch ensemble: {len(tr_in)} training runs with dtau <= {cut} MPa; 5 networks trained in {time.time() - t0:.0f} s")
for lab, msk in (("in distribution (dtau <= 4)", ind), ("out of distribution (dtau > 4)", ~ind)):
    print(f"  {lab:31s}: {msk.sum():3d} test runs; median error {np.median(e_te[msk]):6.2f} mm, median spread "
          f"{np.median(s_te[msk]):6.2f} mm, median error / spread {np.median(ratio[msk]):5.2f},"
          f" share with error <= 2 x spread {(ratio[msk] <= 2).mean():.2f}")

# %%
fig, axs = plt.subplots(1, 2, figsize=(12.0, 4.1))
ax = axs[0]
ax.axvspan(lv_grid[~inside1].min(), lv_grid[-1], color=S.GRID, alpha=0.6, lw=0)
for m in range(8):
    ax.plot(lv_grid, pred_e[m], color=S.BLUE, lw=0.7, alpha=0.6, label="8 networks" if m == 0 else None)
ax.fill_between(lv_grid, mean_e - 2 * sd_e, mean_e + 2 * sd_e, color=S.BLUE, alpha=0.15, lw=0, label="mean ± 2 spread")
ax.plot(lv_grid, y_grid, color=S.INK, lw=2, label="true model")
ax.set_ylim(y_grid.min() - 20, max(y_grid.max(), (mean_e + 2 * sd_e).max()) + 20)
ax.set_xlabel("log v0"); ax.set_ylabel("displacement, day 30 to year 1 (mm)")
ax.set_title("1-D: trained left of the shaded range"); ax.legend(fontsize=8, loc="upper left")
ax = axs[1]
bins = np.geomspace(max(ratio.min(), 1e-2), ratio.max(), 30)
ax.hist(ratio[ind], bins=bins, color=S.BLUE, alpha=0.7, label="in distribution (Δτ ≤ 4 MPa)")
ax.hist(ratio[~ind], bins=bins, color=S.ORANGE, alpha=0.7, label="out of distribution (Δτ > 4 MPa)")
ax.axvline(2, color=S.INK, ls="--", lw=1); ax.text(2.1, ax.get_ylim()[1] * 0.9, "error = 2 × spread", fontsize=8)
ax.set_xscale("log"); ax.set_xlabel("actual error / ensemble spread (per test run)"); ax.set_ylabel("test runs")
ax.set_xticks([0.3, 0.5, 1, 2, 5, 10]); ax.set_xticklabels(["0.3", "0.5", "1", "2", "5", "10"])
ax.set_title("Single patch: is the spread an error bar?"); ax.legend(fontsize=8, loc="upper left")
fig.tight_layout()
bk.savefig(fig, "3-02_ensembles")
plt.show()

# %% [markdown]
# **How to read it.** Left: inside the training range (white) the eight networks agree with each other and with the
# true model. In the shaded range all eight keep rising while the true curve turns down: they fan out only a little,
# agree on the wrong answer, and the band misses the true curve almost everywhere (printed share).
# Right: for each single-patch test run, the actual error divided by the ensemble's spread. If the spread were an honest
# error bar, most runs would sit left of the dashed line. In distribution a majority do; out of distribution most do
# not, and the typical error is a few times the spread (printed). The spread is most misleading exactly where the
# surrogate is most wrong.
#
# **Common misreadings.**
# * "Ensemble spread = uncertainty." It is a lower bound on how much the fitted function is undetermined by the
#   training runs, given this architecture; it says nothing about errors that all members share.
# * "So use it as $\mathbf C_p$." Possibly, after checking it on validation runs as 3.01 checked its $\mathbf C_p$; 3.03 does
#   the same check for the GP's own variance.

# %% [markdown]
# ## 12. Summary
#
# | idea | what to remember | where it returns |
# |---|---|---|
# | network = sum of bent shapes | width adds shapes, depth composes them | 3.04 (PINN), 3.05 (operators) |
# | backpropagation | reverse-mode autodiff; a gradient costs a few forward passes | 2.17, 2.22 (adjoints), 3.07 (NUTS on a network) |
# | initialisation and scaling | signal size about one at every layer; inputs and outputs in unit ranges | every trained surrogate |
# | Adam | robust default; still check the training curve has flattened | 3.03-3.07 |
# | weight decay, early stopping | priors on the function; choose them on validation runs | 2.09, 2.29 |
# | residual blocks | identity path keeps deep networks trainable; Euler step of an ODE | 3.05, neural ODEs |
# | ensembles | cheap spread, not a calibrated error bar | 3.03 (GP variance), 3.07 (inference with surrogates) |
#
# **In practice.**
# * A trained network is a JAX function, so its gradient with respect to θ costs a few forward passes (section 3), and
#   gradient-based samplers such as NUTS (2.18) can run on it directly. That is the main attraction of a network
#   surrogate for an expensive forward model that has no gradient of its own.
# * Before trusting a network surrogate inside an inversion, check what 3.01 checked for the GP: error per test run in
#   units of σ, where it is large, and the coverage of the resulting posterior with and without $\mathbf C_p$.
#
# Next: **3.03 · GP surrogates and Calibrate-Emulate-Sample**.

# %%
print(f"notebook run time: {time.time() - T0:.0f} s")
