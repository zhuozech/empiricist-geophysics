"""Stress-driven afterslip with rate-strengthening friction: the shared forward model of notebooks 2.10, 2.22 and Topic 3.

The standard stress-driven afterslip model of postseismic geodesy, reduced to 1-D:
    slip rate     v = 2 v0 sinh(tau / a_sigma)          (rate-strengthening friction, a_sigma = (a-b) sigma > 0)
    stress        d tau / dt = K v                       (elastic interaction; K_ii < 0 is the stress drop per metre)
    initial state tau(0) = dtau (the coseismic stress change), slip s(0) = 0.

Two versions:
  * single patch (a spring-slider): K = -k. Three parameters theta = (log a_sigma, log v0, dtau); observable: the
    surface displacement time series u(t) = g s(t) at one GPS site. For dtau >> a_sigma the exact solution of the
    exponential approximation is s(t) = (a_sigma / k) log(1 + t / t_c), t_c = a_sigma / (k v+), v+ = v0 exp(dtau / a_sigma),
    the classic logarithmic afterslip curve (used to check the solver).
  * 1-D antiplane fault: N patches between 0 and z_max km on a vertical strike-slip fault, stress interaction from screw
    dislocations with a free-surface image, coseismic slip on shallow patches, afterslip on the patches whose stress
    increased (patches with a stress drop are pinned, a common simplification), a_sigma varying with depth.
    Observables: afterslip on every patch and surface displacement at stations, at chosen times.

Units: time in years, slip in metres, stress in MPa, lengths in km (converted internally), v0 in m/yr.
Solver: diffrax, implicit Kvaerno5 with adaptive steps (the problem is stiff at early times), float64.
"""
from __future__ import annotations

import hashlib
import json
import pathlib

import diffrax as dfx
import jax
import jax.numpy as jnp
import numpy as np

jax.config.update("jax_enable_x64", True)

MU = 30_000.0                       # shear modulus, MPa (30 GPa)
DATA = pathlib.Path(__file__).resolve().parent.parent / "data"


# ============================================================================ elastic Green's functions (antiplane)
def surface_greens(x_obs_km, z_edges_km):
    """Surface displacement (m per m of slip) at x_obs from uniform slip on each patch between consecutive z_edges.
    u(x) = s / pi * [atan(z2 / x) - atan(z1 / x)]  (screw dislocation in a half-space; same formula as notebook 2.33)."""
    x = np.asarray(x_obs_km, float)[:, None]
    z1, z2 = np.asarray(z_edges_km[:-1], float)[None, :], np.asarray(z_edges_km[1:], float)[None, :]
    return (np.arctan(z2 / x) - np.arctan(z1 / x)) / np.pi


def stress_kernel(z_edges_km, mu=MU):
    """Shear stress change (MPa per m of slip) at each patch centre from uniform slip on each patch.

    From the displacement field of slip below depth xi with its free-surface image,
    u(x, z) = s / (2 pi) [atan(x / (xi - z)) + atan(x / (xi + z))], the fault-plane stress mu du/dx at x = 0 is
    (mu s / 2 pi) [1 / (xi - z) + 1 / (xi + z)]. A patch [a, b] is slip below a minus slip below b:
    K_ij = mu / (2 pi) [1/(a_j - z_i) + 1/(a_j + z_i) - 1/(b_j - z_i) - 1/(b_j + z_i)]  (lengths in metres).
    K_ii < 0: slip on a patch lowers its own stress; slip just below a patch raises it (stress concentration).
    """
    ze = np.asarray(z_edges_km, float) * 1000.0
    zc = 0.5 * (ze[1:] + ze[:-1])[:, None]
    a, b = ze[:-1][None, :], ze[1:][None, :]
    return mu / (2 * np.pi) * (1 / (a - zc) + 1 / (a + zc) - 1 / (b - zc) - 1 / (b + zc))


# ============================================================================ single patch (spring-slider)
SP_DEFAULTS = dict(k=3.0, g=0.6)    # stiffness MPa/m (about mu / patch size); surface displacement per metre of slip


def single_patch_series(theta, t_obs, k=SP_DEFAULTS["k"], g=SP_DEFAULTS["g"], rtol=1e-8, atol=1e-10):
    """Surface displacement u(t_obs) = g s(t_obs) (metres) for theta = (log a_sigma [MPa], log v0 [m/yr], dtau [MPa]).
    JAX-traceable: jit, vmap and grad work through the solver (discretise-then-differentiate)."""
    log_as, log_v0, dtau = theta[0], theta[1], theta[2]
    a_s, v0 = jnp.exp(log_as), jnp.exp(log_v0)

    def rhs(t, y, args):
        v = 2.0 * v0 * jnp.sinh(y[0] / a_s)
        return jnp.stack([-k * v, v])

    ts = jnp.asarray(t_obs)
    sol = dfx.diffeqsolve(dfx.ODETerm(rhs), dfx.Kvaerno5(), t0=0.0, t1=ts[-1], dt0=1e-7,
                          y0=jnp.stack([dtau, jnp.zeros_like(dtau)]), saveat=dfx.SaveAt(ts=ts),
                          stepsize_controller=dfx.PIDController(rtol=rtol, atol=atol), max_steps=200_000)
    return g * sol.ys[:, 1]


def single_patch_log_formula(theta, t_obs, k=SP_DEFAULTS["k"], g=SP_DEFAULTS["g"]):
    """Closed form of the exponential approximation (valid for dtau >> a_sigma): u = g a_s / k log(1 + t / t_c)."""
    a_s, v0, dtau = np.exp(theta[0]), np.exp(theta[1]), theta[2]
    v_plus = v0 * np.exp(dtau / a_s)
    t_c = a_s / (k * v_plus)
    return g * a_s / k * np.log1p(np.asarray(t_obs) / t_c)


T_OBS = np.geomspace(1 / 365.25, 1.0, 40)       # 40 GPS epochs between day 1 and year 1 (log spaced)
# prior box for theta = (log a_sigma, log v0, dtau): a_sigma 0.2-2 MPa, v0 1e-3-1 m/yr, dtau 1-5 MPa
SP_BOUNDS = np.array([[np.log(0.2), np.log(2.0)], [np.log(1e-3), np.log(1.0)], [1.0, 5.0]])
SP_NAMES = ("log a_sigma", "log v0", "dtau")


def sample_theta(rng, n, bounds=SP_BOUNDS):
    lo, hi = bounds[:, 0], bounds[:, 1]
    return lo + (hi - lo) * rng.uniform(size=(n, len(lo)))


# ============================================================================ 1-D antiplane fault
def fault_setup(n_patch=40, z_max=40.0, x_obs=None, rupture_bottom=12.0, slip_peak=3.0):
    """Geometry, kernels and a coseismic slip model (a smooth bump from 0 to rupture_bottom km)."""
    z_edges = np.linspace(0.0, z_max, n_patch + 1)
    zc = 0.5 * (z_edges[1:] + z_edges[:-1])
    if x_obs is None:
        x_obs = np.geomspace(0.5, 120.0, 30)
    K = stress_kernel(z_edges)
    G = surface_greens(x_obs, z_edges)
    s_co = np.where(zc < rupture_bottom, slip_peak * np.clip(np.sin(np.pi * zc / rupture_bottom), 0, None) ** 0.7, 0.0)
    dtau = K @ s_co                                   # coseismic stress change on every patch
    free = dtau > 0                                   # afterslip only where the stress went up (others pinned)
    return dict(z_edges=z_edges, zc=zc, x_obs=np.asarray(x_obs, float), K=K, G=G, s_co=s_co, dtau=dtau, free=free)


def fault_afterslip(log_asig, setup, v0=0.03, t_obs=T_OBS, rtol=1e-5, atol=1e-7):
    """Afterslip s(z, t_obs) (n_t, n_patch) for a depth profile log a_sigma(z) (n_patch,). JAX-traceable."""
    K, dtau = jnp.asarray(setup["K"]), jnp.asarray(setup["dtau"])
    free = jnp.asarray(setup["free"], dtype=jnp.float64)
    a_s = jnp.exp(log_asig)
    n = dtau.shape[0]

    def rhs(t, y, args):
        tau = y[:n]
        v = free * 2.0 * v0 * jnp.sinh(tau / a_s)
        return jnp.concatenate([K @ v, v])

    ts = jnp.asarray(t_obs)
    sol = dfx.diffeqsolve(dfx.ODETerm(rhs), dfx.Kvaerno5(), t0=0.0, t1=ts[-1], dt0=1e-8,
                          y0=jnp.concatenate([dtau, jnp.zeros(n)]), saveat=dfx.SaveAt(ts=ts),
                          stepsize_controller=dfx.PIDController(rtol=rtol, atol=atol), max_steps=400_000)
    return sol.ys[:, n:]


def fault_surface(log_asig, setup, v0=0.03, t_obs=T_OBS):
    """Postseismic surface displacement (n_t, n_station), metres."""
    return fault_afterslip(log_asig, setup, v0, t_obs) @ jnp.asarray(setup["G"]).T


def matern_cov(z, ell, nu=1.5, sd=1.0):
    """Matérn covariance (nu = 0.5, 1.5 or 2.5) on 1-D coordinates z (km)."""
    r = np.abs(np.asarray(z)[:, None] - np.asarray(z)[None, :]) / ell
    if nu == 0.5:
        c = np.exp(-r)
    elif nu == 1.5:
        c = (1 + np.sqrt(3) * r) * np.exp(-np.sqrt(3) * r)
    elif nu == 2.5:
        c = (1 + np.sqrt(5) * r + 5 * r ** 2 / 3) * np.exp(-np.sqrt(5) * r)
    else:
        raise ValueError("nu must be 0.5, 1.5 or 2.5")
    return sd ** 2 * c


def sample_log_asig(rng, n, zc, mean=np.log(0.8), sd=0.5, ell=8.0, nu=1.5):
    """Random depth profiles of log a_sigma from a Gaussian-process (Matérn) prior."""
    C = matern_cov(zc, ell, nu, sd) + 1e-10 * np.eye(len(zc))
    return mean + rng.standard_normal((n, len(zc))) @ np.linalg.cholesky(C).T


# ============================================================================ cached training sets
def _cache_path(tag, settings):
    h = hashlib.sha1(json.dumps(settings, sort_keys=True).encode()).hexdigest()[:10]
    return DATA / f"{tag}_{h}.npz"


def training_set_single(n, seed=0, design="uniform", t_obs=T_OBS):
    """n forward runs of the single-patch model on a design over SP_BOUNDS. Cached in data/ (deterministic)."""
    settings = dict(n=int(n), seed=int(seed), design=design, t=list(np.round(t_obs, 12)), b=SP_BOUNDS.round(12).tolist(),
                    k=SP_DEFAULTS["k"], g=SP_DEFAULTS["g"])
    path = _cache_path("single", settings)
    if path.exists():
        z = np.load(path)
        return z["theta"], z["u"]
    rng = np.random.default_rng(seed)
    if design == "uniform":
        theta = sample_theta(rng, n)
    elif design == "lhs":                                     # Latin hypercube: one sample in each of n strata per axis
        d = SP_BOUNDS.shape[0]
        u01 = (np.argsort(rng.uniform(size=(n, d)), axis=0) + rng.uniform(size=(n, d))) / n
        theta = SP_BOUNDS[:, 0] + (SP_BOUNDS[:, 1] - SP_BOUNDS[:, 0]) * u01
    else:
        raise ValueError(design)
    f = jax.jit(jax.vmap(lambda th: single_patch_series(th, t_obs)))
    u = np.concatenate([np.asarray(f(jnp.asarray(theta[i:i + 256]))) for i in range(0, n, 256)])
    DATA.mkdir(exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, theta=theta, u=u)
    tmp.replace(path)
    return theta, u


def training_set_fault(n, seed=0, setup_kw=None, prior_kw=None, v0=0.03, t_obs=T_OBS):
    """n forward runs of the 1-D fault model with log a_sigma(z) drawn from the GP prior. Cached in data/."""
    setup_kw, prior_kw = setup_kw or {}, prior_kw or {}
    settings = dict(n=int(n), seed=int(seed), setup=setup_kw, prior=prior_kw, v0=v0, t=list(np.round(t_obs, 12)))
    path = _cache_path("fault", settings)
    setup = fault_setup(**setup_kw)
    if path.exists():
        z = np.load(path)
        return z["log_asig"], z["slip"], z["surf"], setup
    rng = np.random.default_rng(seed)
    la = sample_log_asig(rng, n, setup["zc"], **prior_kw)
    f = jax.jit(jax.vmap(lambda l: fault_afterslip(l, setup, v0, t_obs)))
    slip = np.concatenate([np.asarray(f(jnp.asarray(la[i:i + 64]))) for i in range(0, n, 64)])
    surf = slip @ setup["G"].T
    DATA.mkdir(exist_ok=True)
    tmp = path.with_suffix(".tmp.npz")
    np.savez(tmp, log_asig=la, slip=slip, surf=surf)
    tmp.replace(path)
    return la, slip, surf, setup
