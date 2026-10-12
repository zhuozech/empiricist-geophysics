"""Samplers, written to be read rather than to be fast.

Every function returns a plain dict, so a notebook can plot everything the sampler did (proposals, leapfrog paths,
tempering stages, ensemble iterations). Production versions of the same ideas: NumPyro NUTS (gradient sampler),
AlTar/CATMIP (SMC), research ensemble-Kalman codes (EnKI).
"""
from __future__ import annotations

import numpy as np


# ============================================================================ random-walk Metropolis
def rwm(target, x0, n, step, cov=None, rng=None, record=False):
    """Random-walk Metropolis for one or several chains at once.

    x0: (d,) or (C, d) start points.  step: proposal scale.  cov: proposal shape (default identity).
    One iteration:  propose y = x + step * L z,  accept with probability min(1, p(y)/p(x)),  otherwise stay.
    Returns samples (C, n, d), accepted flags (C, n), and (if record) the proposals.
    """
    rng = rng or np.random.default_rng()
    X = np.atleast_2d(np.asarray(x0, float)).copy()
    C, d = X.shape
    Lc = np.linalg.cholesky(np.asarray(cov, float)) if cov is not None else np.eye(d)
    lp = target.logp_batch(X)
    S = np.empty((C, n, d))
    A = np.zeros((C, n), bool)
    LP = np.empty((C, n))
    Pr = np.empty((C, n, d)) if record else None
    for i in range(n):
        Y = X + step * rng.standard_normal((C, d)) @ Lc.T
        lpy = target.logp_batch(Y)
        acc = np.log(rng.uniform(size=C)) < lpy - lp          # the whole algorithm is this line
        X[acc], lp[acc] = Y[acc], lpy[acc]
        S[:, i], A[:, i], LP[:, i] = X, acc, lp
        if record:
            Pr[:, i] = Y
    return dict(samples=S, chain=S[0], accepted=A, logp=LP, proposals=Pr, accept_rate=A.mean(),
                n_logp=C * n, start=np.atleast_2d(np.asarray(x0, float)))


# ============================================================================ HMC
def _mass(M, d):
    if M is None:
        return np.eye(d), np.eye(d)
    M = np.asarray(M, float)
    if M.ndim == 1:
        M = np.diag(M)
    return np.linalg.inv(M), np.linalg.cholesky(M)


def leapfrog(target, x, p, g, eps, L, Minv, record=False):
    """L leapfrog steps of size eps. Returns end point, momentum, log p, gradient, and the path if record."""
    path = [x.copy()] if record else None
    p = p + 0.5 * eps * g                       # half kick: momentum feels the slope
    lp = None
    for l in range(L):
        x = x + eps * (Minv @ p)                # drift: position follows momentum
        lp, g = target.logp_and_grad(x)
        if not np.isfinite(lp):
            break
        if l < L - 1:
            p = p + eps * g                     # full kick
        if record:
            path.append(x.copy())
    p = p + 0.5 * eps * g                       # final half kick
    return x, p, lp, g, (np.array(path) if record else None)


def hmc(target, x0, n, eps, L, M=None, rng=None, record=False, jitter_L=False):
    """Hamiltonian Monte Carlo with a fixed number L of leapfrog steps.

    Picture: put a frictionless puck on the surface U(m) = -log p(m), give it a random kick, let it slide for
    L steps, then accept or reject the end point by how well total energy was conserved.
    """
    rng = rng or np.random.default_rng()
    x = np.asarray(x0, float).copy()
    d = x.size
    Minv, Mchol = _mass(M, d)
    lp, g = target.logp_and_grad(x)
    S = np.empty((n, d))
    acc = np.zeros(n, bool)
    dH = np.empty(n)
    paths, momenta = [], []
    for i in range(n):
        p0 = Mchol @ rng.standard_normal(d)
        H0 = -lp + 0.5 * p0 @ Minv @ p0
        Li = rng.integers(1, 2 * L) if jitter_L else L
        x1, p1, lp1, g1, path = leapfrog(target, x, p0, g, eps, Li, Minv, record)
        H1 = -lp1 + 0.5 * p1 @ Minv @ p1 if (lp1 is not None and np.isfinite(lp1)) else np.inf
        dH[i] = H1 - H0
        if np.log(rng.uniform()) < H0 - H1:
            x, lp, g = x1, lp1, g1
            acc[i] = True
        S[i] = x
        if record:
            paths.append(path)
            momenta.append(p0)
    return dict(samples=S[None], chain=S, accepted=acc[None], accept_rate=acc.mean(), energy_error=dH,
                paths=paths, momenta=momenta, n_grad=n * L)


# ============================================================================ NUTS
def find_reasonable_eps(target, x, lp, g, Minv, Mchol, rng):
    """Hoffman & Gelman (2014) Algorithm 4: double/halve eps until one leapfrog step has acceptance ~0.5."""
    eps = 0.5
    p = Mchol @ rng.standard_normal(x.size)

    def logratio(e):
        x1, p1, lp1, _, _ = leapfrog(target, x, p, g, e, 1, Minv)
        if lp1 is None or not np.isfinite(lp1):
            return -np.inf
        return (lp1 - 0.5 * p1 @ Minv @ p1) - (lp - 0.5 * p @ Minv @ p)

    r = logratio(eps)
    a = 1.0 if r > np.log(0.5) else -1.0
    for _ in range(60):
        if not (a * r > -a * np.log(2.0)):
            break
        eps *= 2.0 ** a
        r = logratio(eps)
    return eps


def nuts(target, x0, n, n_warmup=500, eps=None, target_accept=0.8, max_depth=10, M=None, rng=None,
         record=(), delta_max=1000.0):
    """No-U-Turn Sampler: Hoffman & Gelman (2014), Algorithm 6 (slice NUTS + dual-averaging step size).

    NUTS is HMC that chooses the trajectory length itself: it doubles the trajectory (forward or backward in
    time, chosen at random) until the two ends start to come back towards each other (a U-turn), then draws the
    next sample from the points it visited.

    record: iteration numbers (counted from 0 including warm-up) whose full trees are stored for plotting.
    Returns post-warm-up samples plus per-iteration tree depth, leapfrog count, step size, divergences.
    """
    rng = rng or np.random.default_rng()
    x = np.asarray(x0, float).copy()
    d = x.size
    Minv, Mchol = _mass(M, d)
    lp, g = target.logp_and_grad(x)
    if eps is None:
        eps = find_reasonable_eps(target, x, lp, g, Minv, Mchol, rng)
    record = set(record)
    # dual averaging (Nesterov 2009 as used by Hoffman & Gelman)
    mu, log_eps_bar, H_bar, gamma, t0, kappa = np.log(10 * eps), 0.0, 0.0, 0.05, 10.0, 0.75
    N = n_warmup + n
    S = np.empty((N, d))
    depth = np.zeros(N, int)
    nleap = np.zeros(N, int)
    eps_hist = np.empty(N)
    div = np.zeros(N, bool)
    acc_stat = np.empty(N)
    energy = np.empty(N)
    trees = {}
    state = {}

    def one_step(xx, pp, gg, e):
        p_half = pp + 0.5 * e * gg
        x1 = xx + e * (Minv @ p_half)
        lp1, g1 = target.logp_and_grad(x1)
        p1 = p_half + 0.5 * e * g1
        return x1, p1, lp1, g1

    def build(xx, pp, gg, log_u, v, j, e, joint0):
        """Returns (x-, p-, g-, x+, p+, g+, x', lp', g', n', s', alpha, n_alpha)."""
        if j == 0:
            x1, p1, lp1, g1 = one_step(xx, pp, gg, v * e)
            joint = lp1 - 0.5 * p1 @ Minv @ p1 if np.isfinite(lp1) else -np.inf
            n1 = int(log_u <= joint)
            s1 = bool(log_u < joint + delta_max)
            if not s1:
                state["div"] = True
            state["nleap"] += 1
            if state["rec"] is not None:
                state["rec"].append((v, x1.copy(), n1 == 1))
            a = float(np.exp(min(0.0, joint - joint0))) if np.isfinite(joint) else 0.0   # = min(1, e^x), no overflow
            return x1, p1, g1, x1, p1, g1, x1, lp1, g1, n1, s1, a, 1
        xm, pm, gm, xp, pp_, gp, x1, lp1, g1, n1, s1, a1, na1 = build(xx, pp, gg, log_u, v, j - 1, e, joint0)
        if s1:
            if v == -1:
                xm, pm, gm, _, _, _, x2, lp2, g2, n2, s2, a2, na2 = build(xm, pm, gm, log_u, v, j - 1, e, joint0)
            else:
                _, _, _, xp, pp_, gp, x2, lp2, g2, n2, s2, a2, na2 = build(xp, pp_, gp, log_u, v, j - 1, e, joint0)
            if n1 + n2 > 0 and rng.uniform() < n2 / (n1 + n2):
                x1, lp1, g1 = x2, lp2, g2
            a1 += a2
            na1 += na2
            dx = xp - xm
            s1 = bool(s2 and dx @ (Minv @ pm) >= 0 and dx @ (Minv @ pp_) >= 0)
            n1 += n2
        return xm, pm, gm, xp, pp_, gp, x1, lp1, g1, n1, s1, a1, na1

    for it in range(N):
        p0 = Mchol @ rng.standard_normal(d)
        joint0 = lp - 0.5 * p0 @ Minv @ p0
        log_u = joint0 + np.log(rng.uniform())             # slice level
        xm = xp = x
        pm = pp = p0
        gm = gp = g
        xn, lpn, gn = x, lp, g
        j, nval, s = 0, 1, True
        state.update(div=False, nleap=0, rec=[] if it in record else None)
        doublings = []
        a_sum, na_sum = 0.0, 0
        while s and j < max_depth:
            v = int(rng.choice([-1, 1]))
            start_len = len(state["rec"]) if state["rec"] is not None else 0
            if v == -1:
                xm, pm, gm, _, _, _, x1, lp1, g1, n1, s1, a, na = build(xm, pm, gm, log_u, v, j, eps, joint0)
            else:
                _, _, _, xp, pp, gp, x1, lp1, g1, n1, s1, a, na = build(xp, pp, gp, log_u, v, j, eps, joint0)
            if state["rec"] is not None:
                doublings.append((v, j, start_len, len(state["rec"])))
            if s1 and rng.uniform() < min(1.0, n1 / nval):
                xn, lpn, gn = x1, lp1, g1
            nval += n1
            a_sum, na_sum = a, na
            dx = xp - xm
            s = bool(s1 and dx @ (Minv @ pm) >= 0 and dx @ (Minv @ pp) >= 0)
            j += 1
        if state["rec"] is not None:
            trees[it] = _assemble_tree(x, state["rec"], doublings, xn, j, s, eps)
        x, lp, g = xn, lpn, gn
        alpha = a_sum / max(na_sum, 1)
        S[it], depth[it], nleap[it], eps_hist[it], div[it], acc_stat[it] = x, j, state["nleap"], eps, state["div"], alpha
        energy[it] = -lp
        if it < n_warmup:                                   # adapt the step size towards target_accept
            m = it + 1
            H_bar = (1 - 1 / (m + t0)) * H_bar + (target_accept - alpha) / (m + t0)
            log_eps = mu - np.sqrt(m) / gamma * H_bar
            eta = m ** (-kappa)
            log_eps_bar = eta * log_eps + (1 - eta) * log_eps_bar
            eps = float(np.exp(log_eps))
            if it == n_warmup - 1:
                eps = float(np.exp(log_eps_bar))
    post = slice(n_warmup, N)
    return dict(samples=S[post][None], chain=S[post], warmup=S[:n_warmup], all=S, depth=depth, n_leapfrog=nleap,
                eps=eps_hist, divergent=div, accept_stat=acc_stat, potential=energy, trees=trees,
                step_size=eps, n_grad=int(nleap.sum()), n_warmup=n_warmup)


def _assemble_tree(x0, rec, doublings, chosen, depth, finished_ok, eps):
    """Turn the recorded leapfrog states of one NUTS iteration into a time-ordered trajectory."""
    t_plus, t_minus = 0, 0
    times = np.empty(len(rec), int)
    stage = np.empty(len(rec), int)
    for (v, j, a, b) in doublings:
        for k in range(a, b):
            if v == 1:
                t_plus += 1
                times[k] = t_plus
            else:
                t_minus -= 1
                times[k] = t_minus
            stage[k] = j
    xs = np.array([r[1] for r in rec]) if rec else np.empty((0, x0.size))
    valid = np.array([r[2] for r in rec], bool)
    order = np.argsort(np.r_[0, times])
    path = np.vstack([x0[None], xs])[order]
    return dict(x0=x0, states=xs, times=times, stage=stage, valid=valid, path=path, doublings=doublings,
                chosen=chosen, depth=depth, uturn=not finished_ok, eps=eps)


# ============================================================================ SMC with tempering (CATMIP style)
def systematic_resample(w, rng):
    n = w.size
    u = (rng.uniform() + np.arange(n)) / n
    return np.minimum(np.searchsorted(np.cumsum(w), u), n - 1)


def smc(problem, n=2000, rng=None, n_mcmc=10, cov_target=1.0, betas=None, scale0=0.5, record=True):
    """Sequential Monte Carlo with likelihood tempering, in the style of CATMIP (Minson et al. 2013).

    Targets p_beta(m) ~ p(m) L(m)^beta for beta going 0 -> 1. Each stage:
      1. choose the next beta so that the importance weights w = L^dbeta have coefficient of variation
         cov_target (=1 means effective sample size N/2), unless a fixed schedule `betas` is given;
      2. weight and resample the particles;
      3. move every particle with n_mcmc Metropolis steps whose proposal covariance is the particle
         covariance times c^2, with c = 1/9 + 8/9 * (last acceptance rate)  (the a + bR rule).
    Returns the stages (particles, beta, ESS, acceptance) and the log evidence estimate.
    """
    rng = rng or np.random.default_rng()
    X = problem.sample_prior(rng, n)
    d = X.shape[1]
    ll = problem.loglik_batch(X)
    lpri = problem.logprior_batch(X)
    beta, logZ, c = 0.0, 0.0, scale0
    stages = [dict(beta=0.0, X=X.copy(), ess=float(n), accept=np.nan, dbeta=0.0)]
    sched = list(betas) if betas is not None else None
    n_eval = n
    while beta < 1.0 - 1e-12:
        llc = ll - ll.max()

        def cov_w(db):
            w = np.exp(db * llc)
            return w.std() / w.mean()
        if sched is not None:
            nb = sched.pop(0) if sched else 1.0
            db = max(min(nb, 1.0) - beta, 0.0)
        elif cov_w(1.0 - beta) <= cov_target:
            db = 1.0 - beta
        else:
            lo, hi = 0.0, 1.0 - beta
            for _ in range(60):                               # bisection on the coefficient of variation
                mid = 0.5 * (lo + hi)
                lo, hi = (mid, hi) if cov_w(mid) < cov_target else (lo, mid)
            db = lo if lo > 0 else hi
        w = np.exp(db * llc)
        logZ += np.log(w.mean()) + db * ll.max()
        wn = w / w.sum()
        ess = 1.0 / np.sum(wn ** 2)
        mean = wn @ X
        Cw = (X - mean).T @ ((X - mean) * wn[:, None])        # weighted particle covariance
        idx = systematic_resample(wn, rng)
        X, ll, lpri = X[idx], ll[idx], lpri[idx]
        beta = min(beta + db, 1.0)
        Lc = np.linalg.cholesky(Cw + 1e-12 * np.eye(d))
        acc_sum = 0.0
        for _ in range(n_mcmc):
            Y = X + c * rng.standard_normal((n, d)) @ Lc.T
            llY = problem.loglik_batch(Y)
            lpY = problem.logprior_batch(Y)
            ok = np.log(rng.uniform(size=n)) < (lpY + beta * llY) - (lpri + beta * ll)
            X[ok], ll[ok], lpri[ok] = Y[ok], llY[ok], lpY[ok]
            acc_sum += ok.mean()
        n_eval += n * n_mcmc
        acc = acc_sum / n_mcmc
        c = 1.0 / 9.0 + 8.0 / 9.0 * acc
        if record:
            stages.append(dict(beta=beta, X=X.copy(), ess=float(ess), accept=float(acc), dbeta=float(db),
                               X_before_move=None))
    return dict(samples=X, stages=stages, betas=np.array([s["beta"] for s in stages]), logZ=float(logZ),
                n_logp=n_eval)


# ============================================================================ ensemble Kalman family
def etkf_update(X, Y, y, sigma, alpha=1.0):
    """One ensemble square-root (ETKF) Kalman update with the noise covariance inflated by alpha.

    X: (d, J) parameters,  Y: (M, J) predicted data for each member,  y: (M,) data,  sigma: (M,) noise std.
    The ensemble transform Kalman filter (ETKF) square-root form, written in the M-dimensional data space.
    """
    J = X.shape[1]
    Rinv = 1.0 / np.asarray(sigma) ** 2
    xm, ym = X.mean(1), Y.mean(1)
    Xf = (X - xm[:, None]) / np.sqrt(J - 1)
    Yf = (Y - ym[:, None]) / np.sqrt(J - 1)
    YtR = (Yf * Rinv[:, None]).T / alpha                      # (J, M)
    Tinv = np.eye(J) + YtR @ Yf
    ev, U = np.linalg.eigh(Tinv)
    T = (U / ev) @ U.T
    Ts = (U / np.sqrt(ev)) @ U.T
    xm = xm + Xf @ (T @ (YtR @ (y - ym)))                    # mean moves by the Kalman gain times the misfit
    return xm[:, None] + np.sqrt(J - 1) * (Xf @ Ts)         # anomalies shrink by T^(1/2)


def _misfit(problem, Y):
    r = (problem.y[:, None] - Y) / problem.sigma[:, None]
    chi = np.sum(r ** 2, axis=0)                              # chi^2 per member
    return chi


def enki(problem, J=500, rng=None, X0=None, max_dtheta=1.0, maxit=20, record=True):
    """Tempered ensemble Kalman inversion (EnKI) with the adaptive step of Iglesias & Yang (2021).

    theta = fraction of the likelihood assimilated so far (0 = prior, 1 = posterior).
    Step rule (Iglesias & Yang 2021):  dtheta = min( max(WRMS^-2, sqrt(M / (2 Var Phi))), max_dtheta, 1 - theta )
    then one ETKF update with the noise covariance inflated by alpha = 1/dtheta.
    For a linear forward model and Gaussian prior the final ensemble samples the exact posterior (J large).
    """
    rng = rng or np.random.default_rng()
    X = (problem.sample_prior(rng, J) if X0 is None else np.asarray(X0, float)).T.copy()   # (d, J)
    M = problem.y.size
    theta, it = 0.0, 0
    hist = []
    while True:
        Y = problem.forward_batch(X.T).T                       # (M, J): one forward run per member
        chi = _misfit(problem, Y)
        Phi = 0.5 * chi
        wrms = np.sqrt(chi.sum() / (J * M))
        ym = Y.mean(1)
        wrms_mean = np.sqrt(np.sum(((problem.y - ym) / problem.sigma) ** 2) / M)
        rec = dict(iter=it, theta=theta, wrms=float(wrms), wrms_of_mean=float(wrms_mean))
        if record:
            rec["X"] = X.T.copy()
        hist.append(rec)
        if theta >= 1.0 - 1e-12 or it >= maxit:
            break
        ainv = min(max(wrms ** -2, np.sqrt(M / (2 * np.var(Phi)))), max_dtheta, 1.0 - theta)
        X = etkf_update(X, Y, problem.y, problem.sigma, alpha=1.0 / ainv)
        theta += ainv
        it += 1
        hist[-1]["dtheta_next"] = ainv
    return dict(samples=X.T, history=hist, theta=np.array([h["theta"] for h in hist]),
                wrms=np.array([h["wrms"] for h in hist]), n_forward=J * len(hist))


def eki_classic(problem, J=500, n_iter=10, rng=None, X0=None, record=True):
    """The original iterated EKI used as an optimiser: assimilate the FULL data at every iteration (alpha = 1).

    Same ETKF update as enki() above, so the only difference is the schedule: theta = 1, 2, 3, ...
    After iteration 1 a linear-Gaussian ensemble equals the posterior; every later iteration counts the data
    again, so the ensemble keeps shrinking (collapse) towards the least-squares point.
    """
    rng = rng or np.random.default_rng()
    X = (problem.sample_prior(rng, J) if X0 is None else np.asarray(X0, float)).T.copy()
    hist = []
    for it in range(n_iter + 1):
        Y = problem.forward_batch(X.T).T
        chi = _misfit(problem, Y)
        hist.append(dict(iter=it, theta=float(it), wrms=float(np.sqrt(chi.mean() / problem.y.size)),
                         **({"X": X.T.copy()} if record else {})))
        if it == n_iter:
            break
        X = etkf_update(X, Y, problem.y, problem.sigma, alpha=1.0)
    return dict(samples=X.T, history=hist)


# ============================================================================ extended Kalman filter (EKF)
def _kalman_from(m_from, P_from, H, innovation, R):
    """Kalman update of N(m_from, P_from) with a linear(ised) observation operator H. Joseph-form covariance."""
    S = H @ P_from @ H.T + R
    K = np.linalg.solve(S, H @ P_from).T                     # P H^T S^-1
    m = m_from + K @ innovation
    IKH = np.eye(len(m_from)) - K @ H
    P = IKH @ P_from @ IKH.T + K @ R @ K.T
    return m, 0.5 * (P + P.T)


def ekf(problem, mode="classic", n_iter=20, n_steps=10, tol=1e-10, record=True):
    """Extended Kalman filter for a static inverse problem with a Gaussian prior N(m0, P0).

    The EKF replaces the forward model by its tangent line, G(m) ~ G(m_lin) + H (m - m_lin) with H = dG/dm at
    m_lin (the Jacobian; here from JAX), and then applies the ordinary Kalman update. It returns ONE Gaussian.
      mode="classic"  : linearise once, at the prior mean (what a filter does with one data batch).
      mode="iterated" : iterated EKF (Bell & Cathey 1993) = undamped Gauss-Newton: re-linearise at the latest
                        estimate, always updating from the prior, until it stops moving. When it converges it stops at
                        a stationary point of the posterior (a local MAP); convergence is not guaranteed.
    The returned covariance is always the Gauss-Newton one, (P0^-1 + H^T R^-1 H)^-1. It equals the Laplace covariance
    (inverse Hessian of -log posterior) only when the term sum_i (R^-1 r)_i d2G_i/dm2, with residual r = G(m) - y, is
    negligible: small residuals or nearly linear G. Otherwise the two differ.
      mode="tempered" : n_steps equal steps of the data weight (alpha = n_steps, like EnKI's theta schedule),
                        re-linearising at the current mean before each step.
    Compare: EnKI uses the ensemble's average slope instead of the local Jacobian.
    """
    m0, P0 = np.asarray(problem.m0, float), np.asarray(problem.P, float)
    y, R = problem.y, np.diag(problem.sigma ** 2)
    hist, n_fwd, n_jac = [], 0, 0
    if mode in ("classic", "iterated"):
        m_lin = m0.copy()
        iters = 1 if mode == "classic" else n_iter
        for i in range(iters):
            H, g = problem.jacobian(m_lin), problem.forward(m_lin)
            n_fwd += 1; n_jac += 1
            m, P = _kalman_from(m0, P0, H, y - g - H @ (m0 - m_lin), R)   # Bell & Cathey form
            if record:
                hist.append(dict(iter=i + 1, mean=m.copy(), cov=P.copy(), lin_point=m_lin.copy()))
            done = np.linalg.norm(m - m_lin) < tol * (1 + np.linalg.norm(m_lin))
            m_lin = m
            if done:
                break
    elif mode == "tempered":
        m, P = m0.copy(), P0.copy()
        for k in range(n_steps):
            H, g = problem.jacobian(m), problem.forward(m)
            n_fwd += 1; n_jac += 1
            m, P = _kalman_from(m, P, H, y - g, R * n_steps)    # data counted 1/n_steps at a time
            if record:
                hist.append(dict(iter=k + 1, theta=(k + 1) / n_steps, mean=m.copy(), cov=P.copy()))
    else:
        raise ValueError(mode)
    return dict(mean=m, cov=P, history=hist, n_forward=n_fwd, n_jacobian=n_jac, mode=mode)


def ekf_step(m, P, y, sigma, h, H, Q=None):
    """One step of a sequential EKF for a parameter state with random-walk dynamics (Network-Inversion-Filter style).

    predict: m stays, P <- P + Q;  update with the new data y: linearise the measurement function h at m
    (H = dh/dm at m) and apply the Kalman update. Returns the updated (m, P).
    """
    P = P + (Q if Q is not None else 0.0)
    Hm = H(m)
    return _kalman_from(m, P, Hm, y - h(m), np.diag(np.broadcast_to(sigma, (len(y),)) ** 2))
