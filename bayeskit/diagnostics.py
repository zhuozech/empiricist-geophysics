"""Convergence and mixing diagnostics, written out so you can see what each number measures.

R-hat (Gelman & Rubin 1992; split + rank-normalised version of Vehtari et al. 2021):
    compares the spread BETWEEN chains with the spread WITHIN chains. If every chain explores the same
    distribution, the two agree and R-hat = 1. Vehtari et al. recommend R-hat < 1.01.
ESS (effective sample size): how many independent draws the correlated chain is worth.
"""
from __future__ import annotations

import numpy as np
from scipy.special import ndtri
from scipy.stats import rankdata


def _as_chains(x):
    x = np.asarray(x, float)
    if x.ndim == 1:
        x = x[None]
    return x                                                  # (C, n)


def rhat_classic(x):
    """Original Gelman-Rubin potential scale reduction factor on (C, n) draws of one scalar."""
    x = _as_chains(x)
    C, n = x.shape
    means = x.mean(1)
    B = n * means.var(ddof=1)                                 # between-chain variance (times n)
    W = x.var(1, ddof=1).mean()                               # mean within-chain variance
    var_plus = (n - 1) / n * W + B / n
    return float(np.sqrt(var_plus / W))


def split(x):
    x = _as_chains(x)
    n = x.shape[1] // 2
    return np.vstack([x[:, :n], x[:, -n:]])


def _rank_normalise(x):
    r = rankdata(x.ravel()).reshape(x.shape)
    return ndtri((r - 3 / 8) / (x.size + 1 / 4))


def rhat(x):
    """Rank-normalised split R-hat (Vehtari et al. 2021): max of the bulk and tail versions."""
    xs = split(x)
    bulk = rhat_classic(_rank_normalise(xs))
    folded = np.abs(xs - np.median(xs))
    tail = rhat_classic(_rank_normalise(folded))
    return max(bulk, tail)


def autocorr(x, max_lag=None):
    """Autocorrelation of one chain (FFT based)."""
    x = np.asarray(x, float) - np.mean(x)
    n = x.size
    f = np.fft.rfft(x, 2 * n)
    ac = np.fft.irfft(f * np.conj(f))[:n]
    ac /= ac[0]
    return ac[: (max_lag or n)]


def ess(x):
    """Effective sample size over all chains (Geyer initial monotone sequence, as in Stan/NumPyro)."""
    x = _as_chains(x)
    C, n = x.shape
    if n < 4:
        return float("nan")
    acov = np.array([autocorr(c) * c.var() for c in x])     # per-chain autocovariance
    W = x.var(1, ddof=1).mean()                               # within-chain variance
    var_plus = W * (n - 1) / n + (x.mean(1).var(ddof=1) if C > 1 else 0.0)
    rho = 1.0 - (W - acov.mean(0)) / var_plus                 # combined autocorrelation at each lag
    rho[0] = 1.0
    # Geyer: add pairs (rho_2k + rho_2k+1) while they are positive, never letting a pair exceed the previous one
    pair_sums, prev = [], np.inf
    for k in range(0, n - 1, 2):
        ps = rho[k] + rho[k + 1]
        if ps <= 0:
            break
        prev = min(ps, prev)
        pair_sums.append(prev)
    tau = -1.0 + 2.0 * np.sum(pair_sums)                      # integrated autocorrelation time
    tau = max(tau, 1.0 / np.log10(C * n))                     # Stan's cap: ESS <= S log10(S) for anti-correlated chains
    return float(C * n / tau)


def ess_numpyro(x):
    """Same quantity computed by NumPyro, for cross-checking."""
    from numpyro.diagnostics import effective_sample_size
    return float(effective_sample_size(_as_chains(x)[..., None])[0])


def summary(samples, names=None):
    """samples: (C, n, d). Returns a small table (list of dicts) with mean, sd, R-hat, ESS per parameter."""
    S = np.asarray(samples, float)
    if S.ndim == 2:
        S = S[None]
    C, n, d = S.shape
    names = names or [f"m{i + 1}" for i in range(d)]
    rows = []
    for i in range(d):
        x = S[:, :, i]
        rows.append(dict(param=names[i], mean=x.mean(), sd=x.std(), rhat=rhat(x) if C > 1 else float("nan"),
                         ess=ess(x)))
    return rows


def print_summary(samples, names=None, title=None):
    rows = summary(samples, names)
    if title:
        print(title)
    print(f"{'param':>14s} {'mean':>9s} {'sd':>8s} {'R-hat':>7s} {'ESS':>8s}")
    for r in rows:
        print(f"{r['param']:>14s} {r['mean']:9.3f} {r['sd']:8.3f} {r['rhat']:7.3f} {r['ess']:8.0f}")
    return rows


# ============================================================================ how nonlinear is the problem?
def nonlinearity_taylor(problem, X, m_ref=None):
    """Degree of nonlinearity seen by a LOCAL linearisation (EKF, Laplace, Gauss-Newton).

    Over the parameter values X (n, d) (e.g. posterior draws, or prior draws), compare the forward model with its
    tangent at m_ref (default: mean of X):  r = (G(x) - G(m_ref) - J(m_ref)(x - m_ref)) / sigma.
    Returns
      nl          RMS of r: the part of the predictions the tangent misses, in units of the data noise.
                  << 1: linear for all practical purposes; ~1: noticeable; >> 1: strongly nonlinear.
      unexplained the same as a fraction of the (noise-weighted) spread of the predictions over X.
    """
    X = np.atleast_2d(X)
    m_ref = X.mean(0) if m_ref is None else np.asarray(m_ref, float)
    J, g0 = problem.jacobian(m_ref), problem.forward(m_ref)
    GX = problem.forward_batch(X)
    s = problem.sigma
    r = (GX - g0 - (X - m_ref) @ J.T) / s
    spread = (GX - GX.mean(0)) / s
    return dict(nl=float(np.sqrt(np.mean(r ** 2))), unexplained=float(np.mean(r ** 2) / max(np.mean(spread ** 2), 1e-300)))


def nonlinearity_ensemble(problem, X, folds: int = 5, seed: int = 0):
    """Degree of nonlinearity seen by an ENSEMBLE Kalman update (EnKF/EKI/EnKI).

    Fit the best straight-line relation between parameters and predicted data across the ensemble X (n, d):
    G(x) ~ a + B (x - mean), and measure what it misses on members it was NOT fitted to (k-fold cross-validation).
    Returns nl (held-out RMS residual in noise units), unexplained (its share of the prediction spread), nl_train (the
    in-sample residual, for reference only) and underdetermined.

    Why held out: with n members and d parameters the straight line has d + 1 coefficients. When n <= d + 1 it passes
    through every member exactly, so the in-sample residual is zero for ANY forward model, however nonlinear. If the
    training folds have fewer than d + 2 members the check cannot be made; nl is then NaN and underdetermined is True.
    """
    import warnings
    X = np.atleast_2d(X)
    n, d = X.shape
    GX = np.asarray(problem.forward_batch(X))
    s = problem.sigma
    spread = (GX - GX.mean(0)) / s
    A = np.column_stack([np.ones(n), X - X.mean(0)])
    coef, *_ = np.linalg.lstsq(A, GX, rcond=None)
    nl_train = float(np.sqrt(np.mean(((GX - A @ coef) / s) ** 2)))
    parts = np.array_split(np.random.default_rng(seed).permutation(n), folds)
    if n - max(len(q) for q in parts) < d + 2:
        warnings.warn(f"nonlinearity_ensemble: {n} members for {d} parameters; a straight line fitted to the training "
                      "folds interpolates them, so nonlinearity cannot be measured (returning NaN)")
        return dict(nl=float("nan"), unexplained=float("nan"), nl_train=nl_train, underdetermined=True)
    r = np.empty_like(GX, dtype=float)
    for q in parts:
        tr = np.setdiff1d(np.arange(n), q)
        mu = X[tr].mean(0)
        c, *_ = np.linalg.lstsq(np.column_stack([np.ones(len(tr)), X[tr] - mu]), GX[tr], rcond=None)
        r[q] = (GX[q] - np.column_stack([np.ones(len(q)), X[q] - mu]) @ c) / s
    return dict(nl=float(np.sqrt(np.mean(r ** 2))), unexplained=float(np.mean(r ** 2) / max(np.mean(spread ** 2), 1e-300)),
                nl_train=nl_train, underdetermined=False)


def gaussianity(g):
    """How far a gridded 2-D posterior is from the Gaussian with the same mean and covariance.

    Returns kl  = KL(p || q) in nats (0 for a Gaussian),
            in90 = probability p puts inside q's 90 % ellipse (0.90 for a Gaussian; smaller when the mass curls away).
    The grid must contain the whole posterior: a window that cuts off its tails biases kl low.
    """
    m, C = g.mean, g.cov
    Ci = np.linalg.inv(C)
    dX, dY = g.X - m[0], g.Y - m[1]
    md2 = Ci[0, 0] * dX ** 2 + 2 * Ci[0, 1] * dX * dY + Ci[1, 1] * dY ** 2
    q = np.exp(-0.5 * md2)
    q /= q.sum() * g.dx * g.dy
    p = g.P
    ok = p > 0
    kl = float(np.sum(p[ok] * np.log(p[ok] / np.maximum(q[ok], 1e-300))) * g.dx * g.dy)
    in90 = float((p * (md2 < -2 * np.log(0.1))).sum() * g.dx * g.dy)
    return dict(kl=kl, in90=in90)
