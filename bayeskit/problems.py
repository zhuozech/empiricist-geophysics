"""Small Bayesian problems whose true posterior we can compute on a grid.

Every problem is written as prior + forward model + data, the same way a geodetic inversion is written:

    log p(m | d) = log p(m) - 1/2 * sum(((d - G(m)) / sigma)^2) + const

so that every sampler (Metropolis, HMC/NUTS, SMC, EnKI) can run on every problem. The log densities are JAX
functions, so the gradient comes from automatic differentiation (jax.grad), never from a hand-written formula.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)


# ============================================================================ base classes
class Target:
    """A log density on R^d plus its JAX gradient. Counts how often it is evaluated."""

    def __init__(self, logp, dim, name="", extent=None, labels=None, note=""):
        self.name, self.dim, self.note = name, dim, note
        self.extent = extent                                    # (xmin, xmax, ymin, ymax) for 2-D pictures
        self.labels = labels or tuple(f"$m_{i + 1}$" for i in range(dim))
        self._logp_fn = logp
        self._logp = jax.jit(logp)
        self._vg = jax.jit(jax.value_and_grad(logp))
        self._grad = jax.jit(jax.grad(logp))
        self._batch = jax.jit(jax.vmap(logp))
        self.reset_counts()

    # ---- counters (so samplers can be compared at equal cost)
    def reset_counts(self):
        self.n_logp = 0
        self.n_grad = 0

    # ---- numpy-friendly calls
    def logp(self, x):
        self.n_logp += 1
        return float(self._logp(jnp.asarray(x, dtype=jnp.float64)))

    def grad(self, x):
        self.n_grad += 1
        return np.array(self._grad(jnp.asarray(x, dtype=jnp.float64)))

    def logp_and_grad(self, x):
        self.n_grad += 1
        v, g = self._vg(jnp.asarray(x, dtype=jnp.float64))
        return float(v), np.array(g)

    def logp_batch(self, X):
        X = np.atleast_2d(X)
        self.n_logp += X.shape[0]
        return np.array(self._batch(jnp.asarray(X, dtype=jnp.float64)))


class InverseProblem(Target):
    """prior + forward model + data. Gaussian prior N(m0, P) unless logprior/prior_sampler are given."""

    def __init__(self, forward, y, sigma, m0=None, P=None, name="", extent=None, labels=None, note="",
                 logprior=None, prior_sampler=None, truth=None):
        y = np.asarray(y, float)
        sigma = np.broadcast_to(np.asarray(sigma, float), y.shape).copy()
        self.y, self.sigma, self.truth = y, sigma, truth
        self.forward_fn = forward
        yj, sj = jnp.asarray(y), jnp.asarray(sigma)

        if logprior is None:                                   # Gaussian prior
            m0 = np.asarray(m0, float)
            P = np.atleast_2d(np.asarray(P, float))
            self.m0, self.P = m0, P
            Pinv = jnp.asarray(np.linalg.inv(P))
            m0j = jnp.asarray(m0)
            logprior = lambda m: -0.5 * (m - m0j) @ Pinv @ (m - m0j)
            Lc = np.linalg.cholesky(P)
            prior_sampler = lambda rng, n: m0 + rng.standard_normal((n, m0.size)) @ Lc.T
        else:
            self.m0, self.P = m0, P
        self.prior_sampler = prior_sampler

        def loglik(m):
            return -0.5 * jnp.sum(((yj - forward(m)) / sj) ** 2)

        self._loglik = jax.jit(loglik)
        self._loglik_batch = jax.jit(jax.vmap(loglik))
        self._logprior_batch = jax.jit(jax.vmap(logprior))
        self._forward_batch = jax.jit(jax.vmap(forward))
        self.logprior_fn, self.loglik_fn = logprior, loglik
        dim = int(np.size(m0)) if m0 is not None else None
        super().__init__(lambda m: logprior(m) + loglik(m), dim, name, extent, labels, note)

    # ---- pieces used by SMC (tempering) and EnKI (forward runs)
    def sample_prior(self, rng, n):
        return np.asarray(self.prior_sampler(rng, n))

    def loglik_batch(self, X):
        X = np.atleast_2d(X)
        self.n_logp += X.shape[0]
        return np.array(self._loglik_batch(jnp.asarray(X)))

    def logprior_batch(self, X):
        return np.array(self._logprior_batch(jnp.asarray(np.atleast_2d(X))))

    def forward_batch(self, X):
        """Run the forward model for every row of X (an ensemble). Counts as len(X) forward runs."""
        X = np.atleast_2d(X)
        self.n_logp += X.shape[0]
        return np.array(self._forward_batch(jnp.asarray(X)))

    def forward(self, m):
        """One forward run G(m)."""
        return self.forward_batch(np.asarray(m, float)[None])[0]

    def jacobian(self, m):
        """Jacobian dG/dm (M x d) at m, from JAX automatic differentiation. Counted separately in n_jac."""
        if not hasattr(self, "_jac"):
            self._jac = jax.jit(jax.jacfwd(self.forward_fn))
            self.n_jac = 0
        self.n_jac += 1
        x = jnp.asarray(m, dtype=jnp.float64)
        try:
            return np.array(self._jac(x))
        except TypeError:          # forward mode is unavailable through custom_vjp code (diffrax solvers): use reverse mode
            self._jac = jax.jit(jax.jacrev(self.forward_fn))
            return np.array(self._jac(x))


# ============================================================================ grid "truth" for 2-D problems
@dataclass
class Grid:
    X: np.ndarray
    Y: np.ndarray
    P: np.ndarray            # normalised so that sum(P) * dx * dy = 1
    dx: float
    dy: float
    levels: dict = field(default_factory=dict)

    @property
    def mean(self):
        w = self.P * self.dx * self.dy
        return np.array([(w * self.X).sum(), (w * self.Y).sum()])

    @property
    def cov(self):
        w = self.P * self.dx * self.dy
        mx, my = self.mean
        cxx = (w * (self.X - mx) ** 2).sum()
        cyy = (w * (self.Y - my) ** 2).sum()
        cxy = (w * (self.X - mx) * (self.Y - my)).sum()
        return np.array([[cxx, cxy], [cxy, cyy]])

    def marginal(self, axis):
        """axis=0 -> density of the first parameter (x); axis=1 -> second (y)."""
        if axis == 0:
            return self.X[0], self.P.sum(0) * self.dy
        return self.Y[:, 0], self.P.sum(1) * self.dx

    def mass(self, mask_fn):
        return float((self.P * mask_fn(self.X, self.Y)).sum() * self.dx * self.dy)

    def sample(self, n, rng):
        """Exact draws from the gridded posterior (pick cells with probability P, jitter inside the cell)."""
        w = self.P.ravel() / self.P.sum()
        idx = rng.choice(w.size, n, p=w)
        return np.column_stack([self.X.ravel()[idx] + (rng.uniform(size=n) - 0.5) * self.dx,
                                self.Y.ravel()[idx] + (rng.uniform(size=n) - 0.5) * self.dy])

    def density_at(self, X2):
        """Gridded density at points X2 (n, 2), bilinear interpolation, 0 outside the grid."""
        from scipy.interpolate import RegularGridInterpolator
        f = RegularGridInterpolator((self.Y[:, 0], self.X[0]), self.P, bounds_error=False, fill_value=0.0)
        X2 = np.atleast_2d(X2)
        return f(np.column_stack([X2[:, 1], X2[:, 0]]))

    def inside(self, X2, q=0.9):
        """Share of the points X2 that fall inside the true highest-density region holding probability q.
        A correct sampler gives about q; samples piled in the wrong place give less."""
        lev = self.levels.get(q) or hdr_levels(self.P, (q,))[q]
        return float(np.mean(self.density_at(X2) >= lev))


def hdr_levels(P, probs=(0.5, 0.9, 0.99)):
    """Density thresholds whose super-level sets contain the given probability (highest-density regions)."""
    p = np.sort(P.ravel())[::-1]
    c = np.cumsum(p)
    c /= c[-1]
    return {q: p[min(np.searchsorted(c, q), p.size - 1)] for q in probs}


def grid(target: Target, extent=None, n=300) -> Grid:
    ext = extent or target.extent
    x = np.linspace(ext[0], ext[1], n)
    y = np.linspace(ext[2], ext[3], n)
    X, Y = np.meshgrid(x, y)
    pts = np.column_stack([X.ravel(), Y.ravel()])
    L = np.asarray(target._batch(jnp.asarray(pts))).reshape(X.shape)   # not counted: this is the answer key
    L = np.where(np.isfinite(L), L, -np.inf)
    P = np.exp(L - L.max())
    dx, dy = x[1] - x[0], y[1] - y[0]
    P /= P.sum() * dx * dy
    return Grid(X, Y, P, dx, dy, hdr_levels(P))


# ============================================================================ linear-Gaussian closed form
def linear_gaussian_posterior(G, y, sigma, m0, P):
    """Exact posterior N(mean, C) for d = G m + e, e ~ N(0, diag(sigma^2)), m ~ N(m0, P)."""
    G = np.atleast_2d(G)
    R = np.diag(np.broadcast_to(sigma, (G.shape[0],)) ** 2)
    S = G @ P @ G.T + R
    K = np.linalg.solve(S, G @ P).T                  # Kalman gain P G^T S^-1 (S symmetric)
    mean = m0 + K @ (y - G @ m0)
    IKG = np.eye(P.shape[0]) - K @ G
    C = IKG @ P @ IKG.T + K @ R @ K.T                # Joseph form: stays positive definite for very broad priors
    return mean, 0.5 * (C + C.T)


def linear(G, y, sigma, m0, P, **kw) -> InverseProblem:
    Gj = jnp.asarray(np.atleast_2d(G), dtype=jnp.float64)
    prob = InverseProblem(lambda m: Gj @ m, y, sigma, m0, P, **kw)
    prob.G = np.atleast_2d(G)
    prob.exact = linear_gaussian_posterior(prob.G, prob.y, prob.sigma, np.asarray(m0, float), np.atleast_2d(P))
    return prob


# ============================================================================ the zoo
def gaussian() -> InverseProblem:
    """Two data, each mostly sensitive to one parameter: a round, exactly Gaussian posterior."""
    G = np.array([[1.0, 0.2], [0.1, 1.0]])
    return linear(G, y=[1.0, -0.4], sigma=0.35, m0=[0.0, 0.0], P=np.eye(2) * 1.5 ** 2,
                  name="Gaussian", extent=(-2.5, 2.5, -2.5, 2.5),
                  note="linear forward model + Gaussian prior and noise")


def correlated(rho_target="strong") -> InverseProblem:
    """Two stations that see two neighbouring slip patches almost equally: a tilted, narrow ellipse (trade-off)."""
    G = np.array([[1.0, 0.85], [0.85, 1.0]])
    return linear(G, y=[0.8, 0.9], sigma=0.12, m0=[0.0, 0.0], P=np.eye(2) * 1.5 ** 2,
                  name="Correlated (trade-off)", extent=(-3.0, 3.5, -3.0, 3.5),
                  note="data constrain m1+m2 well and m1-m2 poorly")


def flat_ridge(prior="weak") -> InverseProblem:
    """One datum that only sees m1 + m2. With a flat prior the posterior is an infinite ridge."""
    G = np.array([[1.0, 1.0]])
    if prior == "flat":
        prob = InverseProblem(lambda m: jnp.asarray(G) @ m, y=[1.0], sigma=0.15, m0=np.zeros(2), P=None,
                              logprior=lambda m: 0.0 * jnp.sum(m), prior_sampler=None,
                              name="Flat ridge (flat prior)", extent=(-6, 7, -6, 7),
                              note="improper: nothing stops the walk along m1 - m2")
        prob.dim = 2
        return prob
    return linear(G, y=[1.0], sigma=0.15, m0=[0.0, 0.0], P=np.eye(2) * 2.0 ** 2,
                  name="Flat direction (weak prior)", extent=(-6, 7, -6, 7),
                  note="only the prior limits m1 - m2")


def _station_forward(stations):
    st = jnp.asarray(np.atleast_2d(stations), dtype=jnp.float64)

    def fwd(m):
        return jnp.sqrt(jnp.sum((m[None, :] - st) ** 2, axis=1) + 1e-12)   # epicentral distance to each station
    return fwd


def location(n_stations=1, kind=None, seed=0, r=2.5, prior_mean=(0.5, 0.3)) -> InverseProblem:
    """Locate an epicentre from noisy distances to 1, 2 or 3 stations (same physics, different data).

    kind: "ring" (1 station, broad prior), "banana" (1 station, regional prior), "bimodal" (2 stations),
          "gaussian" (3 stations), "knob" (1 station at distance r: the nonlinearity knob, prior mean prior_mean).
    """
    kind = kind or {1: "ring", 2: "bimodal", 3: "gaussian"}[n_stations]
    rng = np.random.default_rng(seed)
    if kind == "banana":
        st = np.array([[0.0, -2.5]])
        src = np.array([0.6, -0.07])                                           # a point on the arc
        d = np.array([2.5])                                                    # noise-free radius: a symmetric arc
        return InverseProblem(_station_forward(st), d, 0.1, m0=[0.0, 0.0], P=np.eye(2) * 1.0 ** 2,
                              name="Banana (1 station + regional prior)", extent=(-2.6, 2.6, -2.9, 1.1),
                              labels=("east (x 10 km)", "north (x 10 km)"), truth=src,
                              note="one distance = a circle; the prior keeps only an arc")
    if kind == "knob":
        # nonlinearity knob: the station sits r below the point (0, 0); the datum says "distance = r", so the
        # data circle passes through (0, 0). Far station (large r): the circle is nearly a straight line across the
        # prior and the problem is nearly linear. Close station: the circle bends inside the prior (banana, then ring).
        st = np.array([[0.0, -float(r)]])
        return InverseProblem(_station_forward(st), np.array([float(r)]), 0.1, m0=np.asarray(prior_mean, float),
                              P=np.eye(2) * 1.0 ** 2, name=f"Station at distance {r:g}", extent=(-3.0, 3.0, -3.0, 3.0),
                              labels=("east (x 10 km)", "north (x 10 km)"), note="nonlinearity knob")
    if kind == "ring":
        st = np.array([[-0.6, 0.0]])
        src = np.array([0.6, 0.9])
        d = np.linalg.norm(st - src, axis=1) + 0.05 * rng.standard_normal(1)
        return InverseProblem(_station_forward(st), d, 0.08, m0=[0.0, 0.0], P=np.eye(2) * 3.0 ** 2,
                              name="Ring (1 station, broad prior)", extent=(-3, 2, -2.5, 2.5),
                              labels=("east (x 10 km)", "north (x 10 km)"), truth=src)
    if kind == "bimodal":
        st = np.array([[-1.0, 0.0], [1.0, 0.0]])
        src = np.array([0.3, 0.8])
        d = np.linalg.norm(st - src, axis=1) + np.array([0.02, -0.03])
        return InverseProblem(_station_forward(st), d, 0.06, m0=[0.0, 0.5], P=np.eye(2) * 1.0 ** 2,
                              name="Two modes (2 stations)", extent=(-2.0, 2.4, -2.0, 2.2),
                              labels=("east (x 10 km)", "north (x 10 km)"), truth=src,
                              note="two stations on a line cannot tell north from south")
    if kind == "gaussian":
        st = np.array([[-1.0, 0.0], [1.0, 0.0], [0.2, 1.6]])
        src = np.array([0.3, 0.8])
        d = np.linalg.norm(st - src, axis=1) + np.array([0.02, -0.03, 0.01])
        return InverseProblem(_station_forward(st), d, 0.06, m0=[0.0, 0.5], P=np.eye(2) * 1.0 ** 2,
                              name="One blob (3 stations)", extent=(-2.0, 2.4, -2.0, 2.2),
                              labels=("east (x 10 km)", "north (x 10 km)"), truth=src)
    raise ValueError(kind)


def funnel(dim=10, centered=True, scale_v=3.0) -> Target:
    """Neal's funnel. v = log-variance of the 'slip' x (the hierarchical scale), x_i ~ N(0, exp(v)).

    centered=True  samples (v, x)         -> funnel geometry (the shape NUTS struggles with)
    centered=False samples (v, z), x = exp(v/2) z  -> independent Gaussians (easy)
    Coordinates: m[0] = v, m[1:] = x (or z).
    """
    if centered:
        def lp(m):
            v, x = m[0], m[1:]
            return -0.5 * v ** 2 / scale_v ** 2 - 0.5 * jnp.sum(x ** 2) * jnp.exp(-v) - 0.5 * (dim - 1) * v
        name = "Funnel (centered)"
    else:
        def lp(m):
            v, z = m[0], m[1:]
            return -0.5 * v ** 2 / scale_v ** 2 - 0.5 * jnp.sum(z ** 2)
        name = "Funnel (non-centered)"
    t = Target(lp, dim, name, extent=(-10, 10, -9, 9), labels=("$v$ (log variance)", "$x_1$"))
    t.centered = centered
    return t


def screw_dislocation(stations_km=(20, 30, 45, 65, 90), sigma=2.0, s_true=30.0, D_true=15.0, seed=5) -> InverseProblem:
    """Interseismic velocity across a strike-slip fault, v(x) = s/pi * atan(x/D) (Savage & Burford 1973).

    m = (D in km, s in mm/yr). Far-field stations see s and s*D in combination: a trade-off ridge.
    Prior: D ~ N(15, 15^2) truncated to D > 0, s ~ N(30, 20^2).
    """
    xs = np.asarray(stations_km, float)
    rng = np.random.default_rng(seed)
    d = s_true / np.pi * np.arctan(xs / D_true) + sigma * rng.standard_normal(xs.size)
    xj = jnp.asarray(xs)
    m0, sd = np.array([15.0, 30.0]), np.array([15.0, 20.0])

    def fwd(m):
        D = jnp.maximum(m[0], 1e-3)
        return m[1] / jnp.pi * jnp.arctan(xj / D)

    def logprior(m):
        g = -0.5 * jnp.sum(((m - m0) / sd) ** 2)
        return jnp.where(m[0] > 0, g, -jnp.inf)            # locking depth must be positive

    def sampler(rng, n):
        out = np.empty((0, 2))
        while len(out) < n:
            z = m0 + sd * rng.standard_normal((2 * n, 2))
            out = np.vstack([out, z[z[:, 0] > 0]])
        return out[:n]
    prob = InverseProblem(fwd, d, sigma, m0=m0, P=np.diag(sd ** 2), logprior=logprior, prior_sampler=sampler,
                          name="Locking depth vs slip rate", extent=(0.0, 60, 10, 60),
                          labels=("locking depth D (km)", "slip rate s (mm/yr)"), truth=np.array([D_true, s_true]))
    prob.stations_km = xs
    return prob


ZOO = {
    "gaussian": gaussian,
    "correlated": correlated,
    "flat": flat_ridge,
    "banana": lambda: location(kind="banana"),
    "bimodal": lambda: location(kind="bimodal"),
    "funnel": lambda: funnel(dim=2, centered=True),
}
