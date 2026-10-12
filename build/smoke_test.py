"""Quick check that the environment works (about a minute): versions, one exact posterior, one NUTS run, one ODE solve.

usage: python build/smoke_test.py
"""
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent.parent))
t0 = time.time()
import jax  # noqa: E402
import numpy as np  # noqa: E402
import numpyro  # noqa: E402
import diffrax  # noqa: E402
import matplotlib  # noqa: E402

matplotlib.use("Agg")
import bayeskit as bk  # noqa: E402
from bayeskit import afterslip as A, problems as P  # noqa: E402

print(f"python {sys.version.split()[0]}, jax {jax.__version__} ({jax.default_backend()}), numpyro {numpyro.__version__}, "
      f"diffrax {diffrax.__version__}, numpy {np.__version__}")

prob = P.correlated()                                   # 2-parameter linear-Gaussian problem with a closed-form answer
m_ex, C_ex = prob.exact
from numpyro.infer import MCMC, NUTS  # noqa: E402
import numpyro.distributions as dist  # noqa: E402
G, m0, P0, y, sig = (np.asarray(v, float) for v in (prob.G, prob.m0, prob.P, prob.y, prob.sigma))


def model():
    m = numpyro.sample("m", dist.MultivariateNormal(m0, P0))
    numpyro.sample("y", dist.Normal(G @ m, sig), obs=y)


mc = MCMC(NUTS(model), num_warmup=500, num_samples=2000, progress_bar=False)
mc.run(jax.random.PRNGKey(0))
from numpyro.diagnostics import effective_sample_size  # noqa: E402
draws = np.asarray(mc.get_samples(group_by_chain=True)["m"])
mean = draws.reshape(-1, 2).mean(0)
ess = effective_sample_size(draws)
err = np.abs(mean - m_ex) / np.sqrt(np.diag(C_ex))                 # in posterior sd
mcse = 1 / np.sqrt(ess)                                             # Monte Carlo standard error of the mean, same units
print(f"NUTS mean vs exact posterior mean: {np.round(err, 3)} posterior sd; Monte Carlo error {np.round(mcse, 3)} "
      f"(ESS {np.round(ess)}); pass if within 4 Monte Carlo errors")

u = np.asarray(A.single_patch_series(np.array([np.log(0.5), np.log(0.03), 3.0]), A.T_OBS))
print(f"afterslip single patch: {u.size} epochs, displacement at 1 yr {1000 * u[-1]:.1f} mm")
ok = bool(np.all(err < 4 * mcse)) and np.isfinite(u).all()
print(f"smoke test {'passed' if ok else 'FAILED'} in {time.time() - t0:.0f} s")
sys.exit(0 if ok else 1)
