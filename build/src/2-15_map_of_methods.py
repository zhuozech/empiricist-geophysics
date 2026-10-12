# %% [markdown]
# # 2.15 · The map: five methods, six posterior shapes
#
# **Before this:** 2.01 (what a posterior is), 2.02 (posterior shapes).
# **You will be able to:** name the posterior shapes that make sampling hard; say which method family suits which
# shape, what each one needs (log-density, gradients, forward runs) and roughly what it costs; read a report card as a
# summary of particular runs, not a verdict on an algorithm.
# **Optional sections:** 6 (concept map).
#
# This notebook runs five methods on six small problems whose exact answer we know, and puts
# every result in one picture, then in one report card. Keep those two pictures in mind while reading the rest;
# each later notebook explains one row or one column of them.
#
# | question | where it is answered |
# |---|---|
# | What does it mean that a posterior is *Gaussian*, *flat*, *smooth*? | 2.02, then 2.12 and 2.33 |
# | What is Metropolis? What does "random walk" mean? | 2.16 |
# | What is gradient information? What do HMC and NUTS do with it? | 2.18 |
# | What is convergence, mixing, R-hat? | 2.19 |
# | What is tempering? What is SMC? | 2.20 |
# | What are EnKF, EKI and EnKI, and how do they differ? | 2.23, 2.24 |
# | What is a Gaussian process (GP)? | 2.12 |
# | What is JAX? What is "differentiable", what is "gradient-based"? | 2.17 |
# | All of it on one fault-slip inversion | 2.33 |
# | How nonlinear is a problem, and how do the EKF, EnKI and the samplers cope as it gets worse? | 2.25 |

# %%
import sys, pathlib, time
sys.path.insert(0, str(pathlib.Path.cwd().parent))

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
from scipy.ndimage import binary_dilation, label, maximum_filter
from IPython.display import Markdown, display

import bayeskit as bk
from bayeskit import problems as P, samplers as SM, diagnostics as D, plotting as BP, style as S

rng = np.random.default_rng(2026)

# %% [markdown]
# ## 1. The six problems: their size, their forward model and their exact answers
#
# Four two-parameter posteriors computed exactly on a fine grid (notebook 2.02 builds them one by one): a plain
# **Gaussian**; a **trade-off** (two slip patches seen from far away); a **banana** (one station's distance plus
# a regional prior); **two modes** (two stations on a line cannot tell north from south).
#
# Plus Neal's **funnel** in 10 dimensions: a log-variance $v \sim N(0, 3^2)$ and nine parameters
# $x_i \sim N(0, e^{v})$ whose spread it sets, like a hierarchical smoothing scale and the slip it scales. It
# appears twice: written directly (*centered*), and as the *same model* written as $x_i = e^{v/2} z_i$ with
# $z_i \sim N(0,1)$ (*non-centered*). We plot and score it in the $(v, x_1)$ plane.
#
# The **problem cards** below list, for each problem, the number of unknowns $d$, the number of data $M$, the
# forward model, the prior and the noise. They are generated from the problem objects themselves, so they cannot
# drift from what was actually run.

# %%
fun_c = P.funnel(dim=10, centered=True, scale_v=3.0)
fun_n = P.funnel(dim=10, centered=False, scale_v=3.0)
fun_pic = P.funnel(dim=2, centered=True, scale_v=3.0)          # the exact (v, x1) marginal of the 10-D funnel
fun_pic.extent = fun_c.extent = fun_n.extent = (-9, 9, -10, 10)

probs = {"Gaussian": P.gaussian(), "Trade-off": P.correlated(), "Banana": P.location(kind="banana"),
         "Two modes": P.location(kind="bimodal"),
         "Funnel (10-D), centered": fun_c, "Same funnel, non-centered": fun_n}
FUNNELS = ("Funnel (10-D), centered", "Same funnel, non-centered")
grids = {k: P.grid(fun_pic if k in FUNNELS else p, n=450) for k, p in probs.items()}

# station positions as defined in bayeskit/problems.py location(); checked against the forward model below
STATIONS = {"Banana": np.array([[0.0, -2.5]]), "Two modes": np.array([[-1.0, 0.0], [1.0, 0.0]])}
for k, st in STATIONS.items():
    m_test = np.array([0.37, -0.21])
    assert np.allclose(probs[k].forward(m_test), np.linalg.norm(m_test - st, axis=1)), k


def vec(v):
    return "(" + ", ".join(f"{x:g}" for x in np.atleast_1d(v)) + ")"


def prior_txt(pr):
    sd = np.sqrt(np.diag(pr.P))
    return f"N({vec(pr.m0)}, {sd[0]:g}² I)" if np.allclose(sd, sd[0]) else f"N({vec(pr.m0)}, diag{vec(sd ** 2)})"


rows = ["| problem | unknowns $d$ | data $M$ | forward model | data $\\mathbf d$ | noise σ | prior |", "|---|---|---|---|---|---|---|"]
for k, pr in probs.items():
    if k in FUNNELS:
        model = ("none (a hierarchical prior): $v \\sim N(0, 3^2)$, $x_i \\mid v \\sim N(0, e^{v})$, $i = 1..9$"
                 if k == FUNNELS[0] else
                 "none: $v \\sim N(0, 3^2)$, $z_i \\sim N(0, 1)$, $x_i = e^{v/2} z_i$ (same model, other coordinates)")
        rows.append(f"| {k} | {pr.dim} | 0 | {model} | none | none | the model itself |")
    elif hasattr(pr, "G"):
        G = "; ".join(" ".join(f"{g:g}" for g in r) for r in pr.G)
        rows.append(f"| {k} | {pr.dim} | {pr.y.size} | **linear**: $\\mathbf d = \\mathbf G \\mathbf m$, G = [{G}] | "
                    f"{vec(pr.y)} | {pr.sigma[0]:g} | {prior_txt(pr)} |")
    else:
        st = ", ".join(vec(s) for s in STATIONS[k])
        rows.append(f"| {k} | {pr.dim} | {pr.y.size} | **nonlinear**: $d_i = \\lVert \\mathbf m - \\mathbf s_i \\rVert$, "
                    f"distance to station $\\mathbf s_i$ = {st} | {vec(np.round(pr.y, 3))} | {pr.sigma[0]:g} | {prior_txt(pr)} |")
display(Markdown("\n".join(rows)))

# %% [markdown]
# **How nonlinear is each problem?** A single number helps: the **degree of nonlinearity** NL is the
# root-mean-square of what the forward model's tangent (its first-order Taylor expansion at the posterior mean)
# misses, over draws from the exact posterior, measured in units of the data noise σ. NL ≪ 1 means a straight-line
# model is as good as the data can tell; NL ≫ 1 means strongly nonlinear. The same number over draws from the prior
# (tangent at the prior mean) says how nonlinear the problem looks *before* the data shrink the region of interest,
# which is what a one-shot linearisation faces (a **linearisation** replaces the forward model by its tangent, a
# straight-line approximation around one point).
#
# The second number, **KL** (the Kullback-Leibler divergence), measures how different two distributions are: in plain
# words, how much information is lost when one is used in place of the other. It is 0 when they are identical and
# grows as they differ; it is measured in **nats** (information units based on the natural logarithm). Here it
# compares the exact posterior with the Gaussian of the same mean and covariance, so it is 0 for a Gaussian posterior.
# Notebook 2.28 defines KL properly; notebook 2.25 treats both measures and shows how each method degrades as
# nonlinearity grows.

# %%
def count_modes(g):
    """Local maxima of the gridded density above 1 % of the peak, merging maxima a few cells apart."""
    mx = (g.P == maximum_filter(g.P, size=9)) & (g.P > 0.01 * g.P.max())
    return int(label(binary_dilation(mx, iterations=3))[1])


rng_cards = np.random.default_rng(11)                            # separate stream: does not change the runs below
cards = {}
for k, pr in probs.items():
    g = grids[k]
    if k in FUNNELS:
        # exact KL of the (v, x1) marginal from its moment-matched Gaussian: s^2/4 with s = 3. In the non-centered
        # coordinates (v, z) the same model is exactly Gaussian, so KL = 0 there.
        cards[k] = dict(nl_post=np.nan, nl_prior=np.nan, kl=(3.0 ** 2 / 4 if k == FUNNELS[0] else 0.0), modes=count_modes(g))
        continue
    nl_post = D.nonlinearity_taylor(pr, g.sample(4000, rng_cards))["nl"]
    nl_prior = D.nonlinearity_taylor(pr, pr.sample_prior(rng_cards, 4000), m_ref=pr.m0)["nl"]
    cards[k] = dict(nl_post=nl_post, nl_prior=nl_prior, kl=D.gaussianity(g)["kl"], modes=count_modes(g))


def nl_word(x):
    return "linear" if x < 1e-8 else ("effectively linear" if x < 0.1 else ("mildly nonlinear" if x < 1 else "strongly nonlinear"))


rows = ["| problem | NL over the posterior | NL over the prior | in words | KL from its Gaussian (nats) | modes |", "|---|---|---|---|---|---|"]
for k, c in cards.items():
    if k in FUNNELS:
        coords = "(v, x₁) marginal" if k == FUNNELS[0] else "in (v, z)"
        rows.append(f"| {k} | no forward model | no forward model | prior structure | {c['kl']:.2f} {coords} | {c['modes']} |")
    else:
        rows.append(f"| {k} | {c['nl_post']:.3g} | {c['nl_prior']:.3g} | {nl_word(c['nl_post'])} | {c['kl']:.3g} | {c['modes']} |")
display(Markdown("\n".join(rows)))

truth = {}
for k, g in grids.items():
    if k in FUNNELS:
        truth[k] = dict(mean=np.zeros(1), sd=np.array([3.0]))          # exact: v ~ N(0, 3^2); score v only
    else:
        truth[k] = dict(mean=g.mean, sd=np.sqrt(np.diag(g.cov)))
north_true = grids["Two modes"].mass(lambda X, Y: Y > 0)
for k, t in truth.items():
    print(f"{k:26s} exact mean {np.round(t['mean'], 3)}  exact sd {np.round(t['sd'], 3)}")
print(f"Two modes: exact probability of the northern mode = {north_true:.3f}")

# %% [markdown]
# Reading the cards: the two linear problems have NL at round-off level and KL near zero, as they must. The banana
# and the two-mode problem are strongly nonlinear by this measure; the two-mode value over the posterior is large
# partly because the tangent is taken at the posterior mean, in the empty valley between the islands, where no
# straight line describes the data. The funnel has no forward model at all: its difficulty is the prior's geometry,
# and how non-Gaussian it looks depends on the coordinates (the same model is exactly Gaussian in $(v, z)$).

# %% [markdown]
# ## 2. The five methods, each with a typical budget
#
# Each method is explained in its own notebook; here is just enough to read the results.
#
# * **Metropolis** (random walk, 2.16): a **chain** of samples in which each new point is *proposed* near the current
#   one and *accepted* or *rejected* by comparing posterior values; such a sequence, where each step depends only on
#   the current point, is a **Markov chain**. Here: 4 chains, round proposals whose size is tuned during **warm-up**
#   (an initial stretch used only for tuning and then discarded) to about 30 % **acceptance** (the share of proposals
#   accepted), then 5,000 steps each. Needs only $\log p$, the logarithm of the (unnormalised) posterior density.
# * **NUTS** (gradient-based Hamiltonian Monte Carlo, 2.18): a chain that moves along smooth trajectories steered by
#   the **gradient** of $\log p$ (the vector of its slopes in every parameter direction, 2.17). 4 chains, 500 warm-up
#   + 1,000 draws each, identity **mass matrix** (no rescaling of the coordinates; 2.18). Needs $\log p$ and its
#   gradient (from JAX, 2.17).
# * **SMC** (sequential Monte Carlo with **tempering**, CATMIP style, 2.20): a population of 2,000 **particles**
#   (candidate parameter vectors) that starts from the prior while the likelihood is turned up from 0 to 1 in
#   adaptive steps (that gradual turning-up is the tempering); at each step the particles are reweighted, the best
#   are copied, and each takes 10 Metropolis moves. For the funnels, which have no data, it tempers from a broad
#   Gaussian reference to the target instead.
# * **EnKI** (tempered ensemble Kalman inversion, 2.24): an **ensemble** of 1,000 **members** (parameter vectors)
#   moved together by the Kalman update (2.08) while the data weight θ rises from 0 to 1; the update uses only the
#   ensemble's means and covariances, computed in square-root form. Needs only forward-model runs, no gradients and
#   no accept/reject.
# * **EKF (iterated)**, the extended Kalman filter (2.25): replace the forward model by its tangent at the current
#   estimate, apply the ordinary Kalman update (2.08), and repeat from the prior until the
#   estimate stops moving (Bell & Cathey 1993). The tangent is the **Jacobian**: the matrix of derivatives of every
#   predicted datum with respect to every parameter (here from JAX, 2.17). The iteration is undamped Gauss-Newton
#   (2.10), which, when it converges, stops at a local MAP (the peak of the posterior, 2.08). It returns one Gaussian:
#   that mean and the Gauss-Newton covariance, which differs from the full Laplace covariance (the curvature of the
#   log posterior at its peak, 2.08) when the forward model is curved (2.25). It is shown and scored here as 4,000 draws from that
#   Gaussian. The classic EKF, which linearises once at the prior mean, is also run and printed for comparison.
#
# EnKI and the EKF need data and a forward model, so they do not apply to the funnels. All chains start from random
# draws of the prior (or of $N(0, I)$ for the funnels), so no method is told where the answer is.
#
# **Timing.** Every run is timed with a wall clock. Before the clock starts, each method runs once on the same
# problem with a tiny budget and the same array shapes, so that JAX compilation is not counted. Wall-clock time
# depends on the implementation (here teaching code: Python loops for NUTS and Metropolis, vectorised arrays for SMC,
# EnKI and the EKF), while the evaluation counts do not.

# %%
class Tempered:
    """Any target written as reference N(0, s^2 I) times 'likelihood' = target / reference, so SMC can temper it."""

    def __init__(self, target, s_ref):
        self.t, self.s = target, np.asarray(s_ref, float)

    def sample_prior(self, rng, n):
        return rng.standard_normal((n, self.s.size)) * self.s

    def logprior_batch(self, X):
        return -0.5 * np.sum((np.atleast_2d(X) / self.s) ** 2, axis=1)

    def loglik_batch(self, X):
        return self.t.logp_batch(X) - self.logprior_batch(X)


def starts(k, prob, n, rng):
    return rng.standard_normal((n, prob.dim)) if k in FUNNELS else prob.sample_prior(rng, n)


def smc_target(k, prob):
    return Tempered(prob, np.full(prob.dim, 3.0)) if k in FUNNELS else prob


def run_rwm(k, prob, rng, chains=4, n=5000, n_warm=1000):
    prob.reset_counts()
    X, step = starts(k, prob, chains, rng), 0.5
    for _ in range(n_warm // 50):                                  # tune the step towards 30 % acceptance
        r = SM.rwm(prob, X, 50, step, rng=rng)
        step *= np.exp(2.0 * (r["accept_rate"] - 0.3))
        X = r["samples"][:, -1]
    r = SM.rwm(prob, X, n, step, rng=rng)
    return dict(samples=r["samples"], evals=prob.n_logp, unit="log p")


def run_nuts(k, prob, rng, chains=4, n=1000, n_warm=500):
    prob.reset_counts()
    out, div = [], 0
    for x0 in starts(k, prob, chains, rng):
        r = SM.nuts(prob, x0, n, n_warmup=n_warm, rng=rng)
        out.append(r["chain"])
        div += int(r["divergent"][n_warm:].sum())
    return dict(samples=np.array(out), evals=prob.n_grad, unit="gradients", divergences=div)


def run_smc(k, prob, rng, n=2000):
    r = SM.smc(smc_target(k, prob), n=n, rng=rng)
    return dict(samples=r["samples"][None], evals=r["n_logp"], unit="log p")


def run_enki(k, prob, rng, J=1000):
    if k in FUNNELS:
        return None
    r = SM.enki(prob, J=J, rng=rng)
    return dict(samples=r["samples"][None], evals=r["n_forward"], unit="forward runs",
                theta=float(r["theta"][-1]), updates=len(r["history"]) - 1, wrms=float(r["wrms"][-1]))


def run_ekf(k, prob, rng, mode="iterated", n_draws=4000):
    if k in FUNNELS:
        return None
    e = SM.ekf(prob, mode=mode)
    draws = rng.multivariate_normal(e["mean"], e["cov"], n_draws)
    return dict(samples=draws[None], evals=e["n_forward"] + e["n_jacobian"], unit="forward + Jacobian",
                ekf_mean=e["mean"], ekf_cov=e["cov"], iters=len(e["history"]))


def warm_up(mname, k, prob):
    """Run the method once with a tiny budget and the same array shapes, so JAX compiles before the clock starts."""
    wr = np.random.default_rng(0)                                  # separate stream: the timed runs are unchanged
    if mname == "Metropolis":
        SM.rwm(prob, starts(k, prob, 4, wr), 3, 0.5, rng=wr)
    elif mname == "NUTS":
        SM.nuts(prob, starts(k, prob, 1, wr)[0], 2, n_warmup=2, rng=wr)
    elif mname == "SMC":
        SM.smc(smc_target(k, prob), n=2000, rng=wr, n_mcmc=1, betas=[1.0])
    elif mname == "EnKI" and k not in FUNNELS:
        SM.enki(prob, J=1000, rng=wr, maxit=1)
    elif mname == "EKF (iterated)" and k not in FUNNELS:
        SM.ekf(prob, mode="classic")


def fmt_evals(n):
    return f"{n / 1e3:.0f}k" if n >= 1000 else f"{n:d}"


def fmt_time(dt):
    return f"{dt * 1e3:.2g} ms" if dt < 0.1 else f"{dt:.2g} s"


methods = {"Metropolis": run_rwm, "NUTS": run_nuts, "SMC": run_smc, "EnKI": run_enki, "EKF (iterated)": run_ekf}
colors = {"Metropolis": S.COLORS["rwm"], "NUTS": S.COLORS["nuts"], "SMC": S.COLORS["smc"], "EnKI": S.COLORS["enki"],
          "EKF (iterated)": S.INK2}
results = {}
for mname, run in methods.items():
    for pname, prob in probs.items():
        warm_up(mname, pname, prob)
        t0 = time.perf_counter()
        r = run(pname, prob, rng)
        dt = time.perf_counter() - t0
        if r is not None:
            r["time"] = dt
            r["cost"] = f"{fmt_evals(r['evals'])} {r['unit']} · {fmt_time(dt)}"
        results[mname, pname] = r
# the classic EKF (one linearisation at the prior mean), for comparison only (not plotted)
classic = {p: run_ekf(p, probs[p], rng, mode="classic") for p in probs if p not in FUNNELS}
print("done")

# %% [markdown]
# ## 3. Scoring every cell against the exact answer
#
# Matching the mean and the standard deviation is not enough: an ellipse can have the same mean and spread as
# a banana and still put its samples where the banana has no probability. So each cell must pass every check
# below (thresholds are a choice, stated so you can change them):
#
# | check | pass if | catches |
# |---|---|---|
# | mean | within 0.2 standard deviations of the exact mean | a shifted answer |
# | spread | each sd within 0.8 to 1.25 of the exact sd | over- or under-confidence |
# | shape | 85 to 95 % of samples inside the exact posterior's 90 % region | right moments, wrong shape |
# | modes | probability of the northern mode within 0.1 of the exact value (two-mode problem) | missing or misweighted islands |
# | chains agree | rank-normalised split R-hat < 1.01 (methods with 4 chains) | chains exploring different places |
#
# **R-hat** compares how much the chains differ from one another with how much each varies on its own; it is near 1
# when the chains agree and larger when they sit in different places (2.19 defines the rank-normalised, split version
# used here).
#
# For the funnels only $v$ is scored (mean, spread, R-hat), because the spread of $x$ is too heavy-tailed to
# estimate reliably from a few thousand draws even with a perfect sampler. SMC, EnKI and the EKF produce one
# population each, so R-hat does not apply to them.

# %%
def to_picture(k, X):
    """Map samples in the sampler's own coordinates to the plotted plane (v, x1) for the funnels."""
    if k == "Same funnel, non-centered":
        return np.stack([X[..., 0], np.exp(X[..., 0] / 2) * X[..., 1]], -1)
    return X[..., :2]


def score(k, samples):
    X = to_picture(k, samples)
    flat = X.reshape(-1, 2)
    C = X.shape[0]
    out = {}
    if k in FUNNELS:
        v = flat[:, 0]
        out["mean"] = abs(v.mean()) / 3.0
        out["sd"] = np.array([v.std() / 3.0])
        out["rhat"] = D.rhat(X[:, :, 0]) if C > 1 else np.nan
        out["in90"] = np.nan
    else:
        t = truth[k]
        out["mean"] = float(np.max(np.abs(flat.mean(0) - t["mean"]) / t["sd"]))
        out["sd"] = flat.std(0) / t["sd"]
        out["rhat"] = max(D.rhat(X[:, :, i]) for i in range(2)) if C > 1 else np.nan
        out["in90"] = grids[k].inside(flat, 0.9)
    out["north"] = float(np.mean(flat[:, 1] > 0)) if k == "Two modes" else np.nan
    fails = []
    if not np.isnan(out["north"]) and abs(out["north"] - north_true) >= 0.1:
        fails.append(f"north mode {out['north']:.2f} (exact {north_true:.2f})")
    if not np.isnan(out["rhat"]) and out["rhat"] >= 1.01:
        fails.append(f"R-hat {out['rhat']:.2f}")
    if not np.isnan(out["in90"]) and not (0.85 <= out["in90"] <= 0.95):
        fails.append(f"{100 * out['in90']:.0f} % inside exact 90 % region")
    if not np.all((out["sd"] > 0.8) & (out["sd"] < 1.25)):
        fails.append(("v " if k in FUNNELS else "") + "sd × " + "/".join(f"{r:.2f}" for r in out["sd"]))
    if out["mean"] >= 0.2:
        fails.append(("v " if k in FUNNELS else "") + f"mean off by {out['mean']:.2f} sd")
    out["fails"], out["ok"] = fails, not fails
    return out


print(f"{'method':15s} {'problem':27s} {'verdict':8s} {'mean':>6s} {'sd ratio':>14s} {'in90':>6s} {'R-hat':>6s} "
      f"{'north':>6s} {'time (s)':>9s}  evaluations")
for (mname, pname), r in results.items():
    if r is None:
        print(f"{mname:15s} {pname:27s} n/a")
        continue
    r.update(score(pname, r["samples"]))
    print(f"{mname:15s} {pname:27s} {('pass' if r['ok'] else 'FAIL'):8s} {r['mean']:6.2f} "
          f"{np.array2string(r['sd'], precision=2):>14s} {r['in90']:6.2f} {r['rhat']:6.3f} {r['north']:6.2f} "
          f"{r['time']:9.3f}  {r['evals']:,} {r['unit']}")
for pname, r in classic.items():
    r.update(score(pname, r["samples"]))
    print(f"{'EKF (classic)':15s} {pname:27s} {('pass' if r['ok'] else 'FAIL'):8s} {r['mean']:6.2f} "
          f"{np.array2string(r['sd'], precision=2):>14s} {r['in90']:6.2f} {'':>6s} {r['north']:6.2f} {'':>9s}  "
          f"{r['evals']:,} {r['unit']}  (not plotted)")
print("NUTS divergences after warm-up:", {p: results["NUTS", p]["divergences"] for p in probs})
for p in probs:
    r = results["EnKI", p]
    if r is not None:
        print(f"EnKI on {p:10s}: theta reached {r['theta']:.3f} after {r['updates']} updates (limit 20), final WRMS {r['wrms']:.2f}")
for p in probs:
    r = results["EKF (iterated)", p]
    if r is not None:
        print(f"EKF (iterated) on {p:10s}: {r['iters']} linearisations, mean {np.round(r['ekf_mean'], 3)}, "
              f"sd {np.round(np.sqrt(np.diag(r['ekf_cov'])), 3)}  (exact mean {np.round(truth[p]['mean'], 3)}, "
              f"exact sd {np.round(truth[p]['sd'], 3)})")
for p, r in classic.items():
    print(f"EKF (classic)  on {p:10s}: mean {np.round(r['ekf_mean'], 3)}, sd {np.round(np.sqrt(np.diag(r['ekf_cov'])), 3)}")
for m in ("Metropolis", "NUTS"):
    occ = (to_picture("Two modes", results[m, "Two modes"]["samples"])[:, :, 1] > 0).mean(1)
    print(f"{m}, two modes: fraction of each chain's samples in the northern mode = {np.round(occ, 2)}")
# how deep into the funnel's neck does each method get?  (exact 5th percentile of v ~ N(0, 3^2) is -4.93)
for k in FUNNELS:
    q5 = {m: np.percentile(to_picture(k, results[m, k]["samples"])[..., 0], 5) for m in ("Metropolis", "NUTS", "SMC")}
    print(f"{k}: 5th percentile of v  " + "  ".join(f"{m} {v:.2f}" for m, v in q5.items()) + f"  (exact {-3 * 1.6449:.2f})")

# %%
fig, axs = plt.subplots(5, 6, figsize=(18.5, 15.6))
for i, mname in enumerate(methods):
    for j, pname in enumerate(probs):
        ax = axs[i, j]
        g, r = grids[pname], results[mname, pname]
        BP.density(ax, g)
        BP.frame(ax, fun_pic if pname in FUNNELS else probs[pname], labels=False)
        ax.set_xticks([]); ax.set_yticks([])
        if r is None:
            ax.text(0.5, 0.5, f"not applicable:\n{mname.split(' ')[0]} needs data\nand a forward model",
                    transform=ax.transAxes, ha="center", va="center", fontsize=9, color=S.INK2,
                    bbox=dict(boxstyle="round,pad=0.4", fc=S.SURFACE, ec=S.AXIS, lw=0.8))
        else:
            BP.samples(ax, to_picture(pname, r["samples"]).reshape(-1, 2), colors[mname], n_max=2500, s=2.5,
                       alpha=0.45)
            text = "matches the exact answer" if r["ok"] else r["fails"][0]
            if mname == "NUTS" and r["divergences"]:
                text += f"; {r['divergences']} divergence" + ("s" if r["divergences"] > 1 else "")
            S.badge(ax, text, ok=bool(r["ok"]), loc="lower right", fs=7.8)
            ax.text(0.03, 0.97, r["cost"], transform=ax.transAxes, ha="left", va="top", fontsize=7.5, color=S.INK2)
        if i == 0:
            ax.set_title(pname, fontsize=11.5)
        if j == 0:
            ax.set_ylabel(mname, fontsize=12, fontweight="bold", color=S.INK)
            ax.add_patch(plt.Rectangle((-0.16, 0.0), 0.035, 1.0, transform=ax.transAxes, color=colors[mname],
                                       clip_on=False))
        if pname in FUNNELS and i == len(methods) - 1:
            ax.set_xlabel(r"$v$ (log variance) $\rightarrow$", fontsize=9)
fig.suptitle("Five methods × six posterior shapes   (grey = exact posterior: 50/90/99 % regions, dots = the method's samples; "
             "EKF dots = draws from its Gaussian)", fontsize=12.5, y=0.997)
fig.tight_layout()
bk.savefig(fig, "2-15_map")
plt.show()

# %% [markdown]
# ### How to read the map
#
# Each badge is computed by the rules of section 3, and the table above prints the numbers behind it. A badge
# summarises one run of one implementation on one example with one budget. It says that these checks passed or
# failed here; it is not a general guarantee, or a general verdict, about the algorithm.
#
# * **Columns are properties of the problem.** The Gaussian column is easy for every method. The others are
#   the shapes that break particular methods: a long thin ridge (trade-off), curvature (banana), separated
#   islands (two modes), and a scale that changes across the space (funnel).
# * **Metropolis (orange)** draws the right picture on the 2-D problems, but at this budget its four chains do
#   not yet agree to R-hat < 1.01 on the trade-off and the banana: in a narrow region the random-walk steps must
#   be short, so it needs many more evaluations (2.16). On the two-mode problem each chain stays on the island
#   where it started (the per-chain fractions printed above), so its share of the northern mode reflects how many
#   chains started there, not the exact weight of that mode. In 10 dimensions its chains have not converged even for the easy,
#   non-centered funnel: dimension alone slows a random walk (2.16).
# * **NUTS (blue)** uses the gradient to glide along the ridge and around the banana, and passes both. On the
#   two-mode problem every chain settled on the northern island during warm-up and none crossed the empty valley.
#   Its R-hat is fine, because all four chains agree, yet the southern island is missing: R-hat checks agreement,
#   not correctness (2.19). On the centered funnel it spends several times more gradients than on the non-centered
#   one and still fails, because no single step size suits both the neck and the mouth; written non-centered,
#   the same model is easy (2.18). (A NUTS **divergence**, counted in the badges, is a trajectory whose numerical
#   energy error blew up: a sign of a region the integrator cannot follow at its step size; 2.18.)
# * **SMC (aqua)** starts from the prior and turns the data up gradually, so particles occupy both islands
#   before the valley between them deepens, and resampling gives the islands the right weights: in these runs it is
#   the only method that passes the two-mode problem. It still fails the centered funnel, because its moves, scaled
#   to the whole cloud, are far too large for the narrow neck (see the 5th percentiles of $v$ printed above). Here
#   tempering handled the separated islands but not the changing scale; the two difficulties call for different
#   remedies (tempering; moves or coordinates adapted to the geometry), which can be combined (2.20).
# * **EnKI (violet)** matches the exact answer on the two linear-Gaussian problems (up to the sampling error of
#   1,000 members), with a few thousand forward runs and no gradients. On the banana (one datum), each update moves
#   every member along one common direction, by an amount set by that member's own predicted datum. That bends the
#   cloud partway toward the arc, but members spill out on both sides of it (badge). On two modes it produces one
#   stretched cloud across both islands and the empty valley between them, and θ never reaches 1 before the
#   20-update limit (printed above; WRMS, also printed, is the weighted root-mean-square misfit, the RMS of the
#   residuals divided by their noise σ, about 1 for a fit at the noise level). A stalled θ is an alarm with several possible causes (strong nonlinearity, too
#   few members, model error, a bug); here the cause is a two-island posterior that an update built from ensemble
#   means and covariances cannot represent (2.24).
# * **EKF (grey)** is the Kalman update with the *local* slope (the Jacobian) instead of EnKI's ensemble slope, so
#   it is gradient-based and returns one Gaussian; iterated, its mean here converges to a local MAP and its
#   covariance is the Gauss-Newton one, not the full Laplace covariance (2.25). It is exact on the two
#   linear problems with a handful of forward runs and Jacobians, the cheapest method in the table by far. On the
#   banana its Gaussian is centred at the peak with the curvature of the peak, so it is too narrow across the arc,
#   misses the tips and puts its centre where the posterior mean is not (badge and printed means). On two modes it
#   converges to the northern island and reports it with full confidence: one Gaussian, one island. The classic
#   EKF (one linearisation at the prior mean, printed above) does worse there: its Gaussian sits beside the
#   northern island's centre rather than on it, so only part of its draws fall inside the exact 90 % region, because
#   the tangent at the prior mean describes the data poorly. Iterating moves the tangent point onto the island.
# * **Cost and time** are printed in each cell: the evaluations in the units each method pays (log-density
#   evaluations, gradient evaluations, forward runs, Jacobians), then the wall-clock time. With reverse-mode
#   autodiff a gradient costs a small constant multiple of one log-density evaluation, whatever the number of
#   parameters (notebook 2.17 measures it). The seconds follow the implementation as much as the method: our NUTS
#   pays Python overhead at every leapfrog step (NumPyro compiles that loop away); SMC evaluates all 2,000 particles
#   in one vectorised call, so its many evaluations take little time; EnKI's time here goes almost entirely into
#   the 1,000 × 1,000 matrix algebra (an eigen-decomposition, 2.06 section 4) of each square-root update, because these toy forward models
#   cost almost nothing.
#   With a real forward model (an ODE solve or a PyLith run per member), the forward runs dominate and the
#   evaluation counts become the better guide to cost.
#
# None of these verdicts is a ranking, and none is a general property of the method: other budgets, tuning or
# implementations can move a cell either way. Which method is right depends on the shape you expect, on whether your
# forward model is differentiable, and on how expensive one forward run is.

# %% [markdown]
# ## 4. The report card: what each method can and cannot do
#
# The same runs, summarised per method. A check means the cell passed every rule of section 3 in this run, a cross
# means it failed at least one (the first failure is named). Evaluations and wall-clock time are medians over the
# problems the method ran on. The card summarises these six examples, these implementations and these budgets.

# %%
COLS = ["linear-Gaussian\n(Gaussian + trade-off)", "curved ridge\n(banana)", "two peaks with\nthe right weights",
        "funnel as written\n(centered)", "funnel\nnon-centered", "needs\ngradients", "evaluations\n(median)",
        "wall-clock\n(median)"]
NEEDS_GRAD = {"Metropolis": "no", "NUTS": "yes", "SMC": "no", "EnKI": "no", "EKF (iterated)": "yes (Jacobian)"}


def verdict(mname, pnames):
    rs = [results[mname, p] for p in pnames]
    if any(r is None for r in rs):
        return None, "not applicable:\nneeds data and\na forward model"
    bad = [(p, r) for p, r in zip(pnames, rs) if not r["ok"]]
    if not bad:
        detail = f"north share {rs[0]['north']:.2f}" if pnames == ["Two modes"] else ""
        if pnames == ["Banana"]:
            detail = f"{100 * rs[0]['in90']:.0f} % inside exact 90 %"
        return True, detail
    p, r = bad[0]
    detail = r["fails"][0]
    if len(pnames) > 1:
        detail = f"{p.lower()}: {detail}"
    if pnames == ["Two modes"] and not detail.startswith("north"):
        detail = f"north share {r['north']:.2f}; {detail}"
    return False, detail


card = {}
for mname in methods:
    runs = [results[mname, p] for p in probs if results[mname, p] is not None]
    ev = np.median([r["evals"] for r in runs])
    tt = np.median([r["time"] for r in runs])
    card[mname] = [verdict(mname, ["Gaussian", "Trade-off"]), verdict(mname, ["Banana"]), verdict(mname, ["Two modes"]),
                   verdict(mname, [FUNNELS[0]]), verdict(mname, [FUNNELS[1]]), (None, NEEDS_GRAD[mname]),
                   (None, f"{fmt_evals(int(ev))} {runs[0]['unit']}"), (None, fmt_time(tt))]
    print(f"{mname:15s} " + " | ".join(("pass" if ok else "FAIL" if ok is False else "-") + (f" ({d.replace(chr(10), ' ')})" if d else "")
                                        for ok, d in card[mname]))

fig, ax = plt.subplots(figsize=(17.5, 5.2))
ax.set_xlim(0, 10.6); ax.set_ylim(0, len(methods) + 1.25); ax.axis("off")
x0, colw = 1.75, [1.2, 1.1, 1.25, 1.15, 1.0, 0.85, 1.25, 1.0]
xs = x0 + np.concatenate([[0], np.cumsum(colw)[:-1]])
ytop = len(methods) + 0.55
for xc, w, title in zip(xs, colw, COLS):
    ax.text(xc + w / 2, ytop + 0.25, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color=S.INK)
ax.plot([0.05, xs[-1] + colw[-1]], [ytop - 0.22] * 2, color=S.AXIS, lw=0.8)
for i, mname in enumerate(methods):
    y = ytop - 0.75 - i
    if i % 2 == 0:
        ax.add_patch(plt.Rectangle((0.05, y - 0.47), xs[-1] + colw[-1] - 0.05, 0.94, color="#f3f2ee", lw=0, zorder=0))
    ax.add_patch(plt.Rectangle((0.08, y - 0.3), 0.07, 0.6, color=colors[mname], lw=0))
    ax.text(0.25, y, mname, ha="left", va="center", fontsize=11, fontweight="bold", color=S.INK)
    for (ok, detail), xc, w in zip(card[mname], xs, colw):
        if ok is None:
            ax.text(xc + w / 2, y, detail, ha="center", va="center", fontsize=9, color=S.INK2 if detail.startswith("not applicable") else S.INK)
            continue
        ax.text(xc + w / 2, y + 0.14, ("✓ pass" if ok else "✗ fail"), ha="center", va="center", fontsize=10.5,
                fontweight="bold", color=S.GOOD if ok else S.BAD)
        if detail:
            ax.text(xc + w / 2, y - 0.2, detail, ha="center", va="center", fontsize=7.6, color=S.INK2, wrap=True)
fig.tight_layout()
bk.savefig(fig, "2-15_report_card")
plt.show()

# %% [markdown]
# ### Reading the report card
#
# * **In these runs only SMC found both peaks with the right weights.** Every other method either stayed on the
#   island it reached first (Metropolis, NUTS, the EKF) or stretched one cloud over both (EnKI).
# * **The Kalman methods (EnKI, EKF) matched the exact answer cheaply on the linear-Gaussian problems**, and failed
#   the curved and two-peak problems for related reasons: the EKF returns one Gaussian by construction, and EnKI
#   moves its members with an update built only from ensemble means and covariances. EnKI needs no gradients; the
#   EKF needs the Jacobian and was the cheapest of all here.
# * **None of the five passed the funnel as written at these budgets.** Rewriting it non-centered fixed it for NUTS
#   and SMC; at this budget Metropolis's chains did not yet agree even then.
# * **The evaluation counts differ by orders of magnitude**, and so do the times. Counted in evaluations, the cheap
#   methods are cheap because they assume more (one Gaussian), and the methods that get the right shape pay for it.
#   In seconds the order changes, because time also reflects how each method is implemented (timing note above).
#
# The rows answer "can it?", not "should I?": an expensive forward model can make even a few thousand runs too
# many, and a non-differentiable one rules out NUTS and the EKF whatever the shape.

# %% [markdown]
# ## 5. Five mental images
#
# | method | the picture | needs | fails when |
# |---|---|---|---|
# | **Metropolis** | a walker who tries a random step and keeps it only if it is not too much downhill (a rejected step means staying put and counting the old point again) | $\log p$ | the region is long, thin, curved or high-dimensional |
# | **HMC / NUTS** | a frictionless puck on the surface $-\log p$, kicked in a random direction; the slope bends its path along valleys; NUTS stops each glide when the path starts to come back (U-turn) | $\log p$ and $\nabla\log p$ (differentiable model; JAX) | islands separated by deep valleys; scales that change across the space (funnel), unless reparameterised |
# | **SMC with tempering** | a crowd that starts spread over the prior; the data are turned up step by step; each step the crowd is reweighted, the good members are cloned, the bad ones dropped, and everyone takes a few random steps | $\log p$ split into prior and likelihood; many evaluations | too few particles or too-large steps in β (weights degenerate); moves that cannot reach part of the space |
# | **EnKI / EKI / EnKF** | a crowd that is never cloned or dropped; at each step every member is moved by the same Kalman rule built from the crowd's own spread | forward runs only, no gradients | strongly non-Gaussian posteriors; ensemble smaller than the number of unknowns (subspace property) |
# | **EKF** | replace the forward model by its tangent line at the current estimate and apply the Kalman formula; iterate to move the tangent point to the peak | forward runs and the Jacobian (differentiable model) | curvature over the region the posterior covers; several peaks (it reports one, confidently) |

# %% [markdown]
# ## 6. How the pieces connect

# %%
fig, ax = plt.subplots(figsize=(13, 5.4))
ax.set_xlim(0, 13); ax.set_ylim(0, 5.4); ax.axis("off")


def box(x, y, w, h, title, body, edge=S.AXIS, fill=S.SURFACE, title_color=S.INK):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fill, ec=edge, lw=1.4))
    ax.text(x + 0.15, y + h - 0.16, title, ha="left", va="top", fontsize=10.5, fontweight="bold", color=title_color)
    ax.text(x + 0.15, y + h - 0.5, body, ha="left", va="top", fontsize=8.6, color=S.INK2, linespacing=1.35)


def arrow(x0, y0, x1, y1):
    ax.annotate("", xy=(x1, y1), xytext=(x0, y0), arrowprops=dict(arrowstyle="-|>", color=S.MUTED, lw=1.1))


box(0.1, 3.2, 3.3, 2.05, "The problem", "prior p(m): Gaussian, flat, smooth (GP) [2.11, 2.12]\n"
    "forward model G(m): linear or nonlinear,\n   differentiable or a black box [2.17, 2.25]\n"
    "data d and noise σ\n"
    "Bayes: p(m | d) ∝ p(m) · p(d | m)")
box(0.1, 0.35, 3.3, 2.45, "The posterior's shape", "Gaussian  (linear + Gaussian)\n"
    "trade-off ridge  (data see combinations)\n"
    "banana  (nonlinear forward model)\n"
    "two modes  (ambiguity)\n"
    "funnel  (sampled hyperparameters)\n"
    "flat / improper  (flat prior + blind direction)")
arrow(1.75, 3.2, 1.75, 2.8)
methods_boxes = [
    (4.1, 4.35, "Closed form", "linear + Gaussian: Kalman formula, no sampling [2.08, 2.23]", S.INK),
    (4.1, 3.3, "Metropolis", "random walk; needs log p only [2.16]", S.COLORS["rwm"]),
    (4.1, 2.25, "HMC / NUTS", "follows the gradient; needs a differentiable log p (JAX) [2.17, 2.18]", S.COLORS["nuts"]),
    (4.1, 1.2, "SMC with tempering", "crowd: reweight, resample, move; can weigh several modes [2.20]", S.COLORS["smc"]),
    (4.1, 0.15, "EKF / EnKF / EKI / EnKI", "Kalman rule with the local slope (EKF) or the crowd's\n"
     "average slope (ensembles); means and covariances [2.24, 2.25]", S.COLORS["enki"]),
]
for (x, y, t_, b_, c) in methods_boxes:
    box(x, y, 4.6, 0.85, t_, b_, edge=c, title_color=S.INK)
    arrow(3.4, 3.0, x, y + 0.42)
box(9.3, 2.45, 3.6, 2.8, "Did it work? [2.19]", "trace plots: hairy caterpillar or not\n"
    "R-hat < 1.01: chains agree\nESS: how many independent draws\n"
    "divergences, tree depth (NUTS)\n"
    "θ and WRMS per iteration (EnKI)\n"
    "a second seed; per-chain energy\ncompare with a known answer")
box(9.3, 0.35, 3.6, 1.7, "Then report", "samples or credible regions,\nnot only a mean (banana) [2.02, 2.32]\n"
    "fit to data: WRMS, χ² per data set\nwhat the prior taught vs the data [2.33]")
for (x, y, t_, b_, c) in methods_boxes:
    arrow(x + 4.6, y + 0.42, 9.3, 3.85)
arrow(11.1, 2.45, 11.1, 2.05)
fig.tight_layout()
bk.savefig(fig, "2-15_concept_map")
plt.show()

# %% [markdown]
# **The one-sentence version.** The posterior's shape is decided by the prior, the forward model and the data;
# the method has to be chosen to match that shape (and your forward model's cost and differentiability); and the
# diagnostics tell you whether it did, without knowing the answer.
#
# **In practice.** A slip inversion that is linear in precomputed Green's functions gives exact gradients cheaply
# through automatic differentiation, which suits NUTS; hierarchical scales are then usually written non-centered to
# avoid the funnel (2.18). A forward model that is a nonlinear ODE or PDE solved by an external code offers forward
# runs only, which points to ensemble Kalman methods (2.24) or SMC; CATMIP-style SMC (Minson et al. 2013) is a common
# choice in finite-fault inversion. Network-inversion filters for transient slip, such as the extended filter of
# Qiu et al. (2018), are EKF-type filters stepping through a time series (2.23, 2.25).
#
# Next: **2.16 · Metropolis: the random walk**.
