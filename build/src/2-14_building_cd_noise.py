# %% [markdown]
# # 2.14 · Building $\mathbf C_d$: noise in GNSS, InSAR and tsunami data
#
# 2.07-2.13 wrote the measurement covariance as $\mathbf C_d = \sigma^2\mathbf I$: every datum has the same error and no two
# errors are related. Real geodetic errors are not like that. A GNSS position today is wrong in nearly the same way as
# yesterday, two InSAR pixels a kilometre apart see the same patch of wet air, and a filtered tsunami record has
# neighbouring samples that share their noise. This notebook builds $\mathbf C_d$ for the three data types, and measures
# what it costs to pretend the errors are independent: error bars that are too small, and weights between data sets
# that are wrong.
#
# **Words this notebook defines:** white, flicker and random-walk noise, power-law noise and spectral index κ,
# periodogram, maximum-likelihood noise estimation, empirical covariogram, sill, range, nugget, exponential covariance,
# decimation (uniform, quadtree), averaging operator, information share of a data set, effective number of independent
# data $N_\text{eff}$, noise scale $\sigma_k$ vs full covariance.
#
# **Before this:** 2.05 (coloured noise), 2.07, 2.13. · **You will be able to:** recognise white, flicker and
# random-walk noise in a GNSS series and say what each does to a velocity's error bar; estimate a spatial covariance
# from an interferogram (an InSAR image of the ground displacement between two radar acquisitions) and carry it through
# decimation; carry a time covariance through filtering; compute the
# effective number of independent data; explain why one noise scale per data set cannot replace a full $\mathbf C_d$. ·
# **Optional sections:** 4 (tsunami records) if you work only with GNSS and InSAR. Topic 4 (data and data processing,
# in preparation) will cover each data type in depth.

# %% [markdown]
# **Problems in this notebook**
#
# | problem | unknowns | data | forward model | linear? |
# |---|---|---|---|---|
# | GNSS daily positions (section 2) | offset and velocity (plus three noise amplitudes in 2.4) | one component, daily, 4 years | straight line in time | linear in offset and velocity |
# | inflating source seen by InSAR and GNSS (sections 3, 5, 6) | volume change of a shallow (3 km) and a deep (9 km) point source | 64 × 64 line-of-sight pixels over 48 km × 48 km, decimated; 8 GNSS stations × 3 components | Mogi point sources in an elastic half-space (Poisson ratio 0.25) | linear in the two volume changes |
# | tsunami record at one bottom-pressure gauge (section 4) | one amplitude | 3 hours sampled every 15 s | one fixed waveform times the amplitude | linear |
#
# A *Mogi source* is a small pressurised sphere buried in an elastic half-space, the standard simple model of an
# inflating magma chamber; the *Poisson ratio* describes how much rock bulges sideways when squeezed. A
# *bottom-pressure gauge* sits on the sea floor and records sea level through the weight of the water above it.
#
# All noise amplitudes below are **inputs chosen for illustration**, not measured values for any network:
# GNSS white 1.5 mm, flicker 3 mm/yr$^{1/4}$, random walk 0.7 mm/yr$^{1/2}$; InSAR atmosphere sill 6 mm, range 5 km,
# pixel noise 2 mm; static GNSS offsets 2 mm per component; tsunami gauge 1 cm of slowly varying noise with an 8-minute
# correlation time plus 0.4 cm of white noise. Every data set is synthetic, so the true answer and the true $\mathbf C_d$
# are known and each estimate can be graded.

# %%
import sys, pathlib
sys.path.insert(0, str(pathlib.Path.cwd().parent))
import numpy as np
import matplotlib.pyplot as plt
import bayeskit as bk
from bayeskit import problems as P, samplers as SM, diagnostics as D, plotting as BP, style as S
import time
from scipy.linalg import toeplitz, cho_factor, cho_solve
from scipy.optimize import minimize, curve_fit
from scipy.signal import butter, filtfilt
from scipy.stats import chi2
from matplotlib.patches import Rectangle

T0 = time.time()
C_DIAG, C_SK, C_FULL = S.ORANGE, S.AQUA, S.BLUE        # same meaning as in 2.13: C_d diagonal, sigma_k scaled, full C_d
C_W, C_F, C_R = S.BLUE, S.ORANGE, S.AQUA                # white, flicker, random walk
DT = 1 / 365.25                                         # one day in years


def powerlaw_T(n, kappa):
    """Lower-triangular Toeplitz matrix that turns white noise into power-law noise with spectral index kappa
    (Hosking's fractional differencing as used for GNSS by Williams 2003): h_0 = 1, h_i = h_{i-1} (i - 1 - kappa/2) / i."""
    h = np.empty(n)
    h[0] = 1.0
    for i in range(1, n):
        h[i] = h[i - 1] * (i - 1 - kappa / 2) / i
    return np.tril(toeplitz(h))


def gnss_components(n, dt=DT):
    """Unit-amplitude covariance matrices of flicker (kappa = -1) and random-walk (kappa = -2) noise, sampling dt (yr)."""
    Tf, Tr = powerlaw_T(n, -1.0), powerlaw_T(n, -2.0)
    return dt ** 0.5 * Tf @ Tf.T, dt * Tr @ Tr.T


def gls(A, d, C):
    """Generalised least squares: estimate and its covariance (A^T C^-1 A)^-1. d may hold many columns."""
    cf = cho_factor(C, lower=True)
    CiA = cho_solve(cf, A)
    cov = np.linalg.inv(A.T @ CiA)
    return cov @ (CiA.T @ d), cov


# %% [markdown]
# ## 1. Three data types, three shapes of error
#
# **In plain words.** The error of a measurement is the part we cannot predict. For GNSS it includes slow wobbles of the
# monument and of the processing, so a daily position is wrong in nearly the same way for weeks. For InSAR it is mostly
# the atmosphere: water vapour delays the radar signal in blobs several kilometres wide, so nearby pixels are wrong
# together. For a tsunami gauge it is everything left after removing the tide, which drifts on time scales of minutes.
# In all three cases a datum's error tells you something about its neighbours' errors, and that is what the
# off-diagonal part of $\mathbf C_d$ records.
#
# **Precisely.** The errors $\mathbf e$ in $\mathbf d = \mathbf G\mathbf m + \mathbf e$ are modelled as $\mathbf e\sim N(\mathbf 0,\mathbf C_d)$ with
# $(\mathbf C_d)_{ij} = \mathrm{Cov}(e_i, e_j)$. The diagonal holds each datum's variance; the off-diagonal entries
# hold the covariances. A diagonal $\mathbf C_d$ claims that knowing $e_i$ says nothing about $e_j$. The misfit that every
# inversion minimises or samples is $(\mathbf d-\mathbf G\mathbf m)^\top\mathbf C_d^{-1}(\mathbf d-\mathbf G\mathbf m)$, so the off-diagonal entries
# change both the estimate and its uncertainty.
#
# The figure draws one realisation of each noise model used below, and how strongly a datum's error is correlated with
# that of a datum some time or distance away.

# %%
rng = np.random.default_rng(1)
N_G = int(4 * 365.25)                                   # 4 years of daily GNSS positions
t_g = np.arange(N_G) * DT
SW, SF, SR = 1.5, 3.0, 0.7                              # white (mm), flicker (mm/yr^0.25), random walk (mm/yr^0.5)
Cf8, Cr8 = gnss_components(int(8 * 365.25))             # 8-year unit matrices (sliced for shorter spans: T is triangular)
Cf_g, Cr_g = Cf8[:N_G, :N_G], Cr8[:N_G, :N_G]
C_g = SW ** 2 * np.eye(N_G) + SF ** 2 * Cf_g + SR ** 2 * Cr_g
L_g = np.linalg.cholesky(C_g)

NG, DXP = 64, 0.75                                       # InSAR grid: 64 x 64 pixels, 0.75 km spacing (48 km square)
xg = (np.arange(NG) - NG / 2 + 0.5) * DXP
XX, YY = np.meshgrid(xg, xg)
PIX = np.column_stack([XX.ravel(), YY.ravel()])
S_ATM, ELL, NUG = 6.0, 5.0, 2.0                          # sill (mm), range (km), pixel noise (mm)
DIST = np.sqrt(((PIX[:, None, :] - PIX[None, :, :]) ** 2).sum(-1))
C_i = S_ATM ** 2 * np.exp(-DIST / ELL) + NUG ** 2 * np.eye(NG * NG)
L_i = np.linalg.cholesky(C_i)

DT_T = 15.0 / 60.0                                       # tsunami gauge: 15 s sampling, in minutes
t_t = np.arange(0, 180, DT_T)
S_RED, TAU, S_WT = 1.0, 8.0, 0.4                         # cm, minutes, cm
C_t = S_RED ** 2 * np.exp(-np.abs(t_t[:, None] - t_t[None, :]) / TAU) + S_WT ** 2 * np.eye(t_t.size)
L_t = np.linalg.cholesky(C_t)

e_g = L_g @ rng.standard_normal(N_G)
e_i = (L_i @ rng.standard_normal(NG * NG)).reshape(NG, NG)
e_t = L_t @ rng.standard_normal(t_t.size)

fig, axs = plt.subplots(2, 3, figsize=(15, 7.2), gridspec_kw=dict(height_ratios=[1.15, 1]))
ax = axs[0, 0]
ax.plot(t_g, e_g, color=S.INK2, lw=0.5)
ax.set_xlabel("time (years)"); ax.set_ylabel("position error (mm)")
ax.set_title("(a) GNSS: 4 years of daily position errors")
ax = axs[0, 1]
im = ax.imshow(e_i, origin="lower", extent=[xg[0], xg[-1], xg[0], xg[-1]], cmap="RdBu_r",
               vmin=-3 * S_ATM, vmax=3 * S_ATM)
fig.colorbar(im, ax=ax, label="line-of-sight error (mm)", shrink=0.85)
ax.set_xlabel("east (km)"); ax.set_ylabel("north (km)")
ax.set_title("(b) InSAR: one interferogram's error")
ax = axs[0, 2]
ax.plot(t_t, e_t, color=S.INK2, lw=0.7)
ax.set_xlabel("time (minutes)"); ax.set_ylabel("sea-level error (cm)")
ax.set_title("(c) Tsunami gauge: 3 hours of residual noise")

ax = axs[1, 0]
ref = N_G // 2
lags = np.arange(0, N_G - ref)
for (C, col, lab) in [(SW ** 2 * np.eye(N_G), C_W, "white only"), (SF ** 2 * Cf_g, C_F, "flicker only"),
                      (SR ** 2 * Cr_g, C_R, "random walk only"), (C_g, S.INK, "all three (used here)")]:
    sd = np.sqrt(np.diag(C))
    ax.plot(lags, C[ref, ref:] / (sd[ref] * sd[ref:]), color=col, lw=1.5 if col == S.INK else 1.1, label=lab)
ax.set_xlabel("days after a reference day (mid-series)"); ax.set_ylabel("correlation with the reference day")
ax.set_ylim(-0.05, 1.02); ax.legend(fontsize=8)
ax.set_title("(d) GNSS errors stay correlated for years")
ax = axs[1, 1]
rr = np.linspace(0, 30, 301)
ax.plot(rr, np.where(rr == 0, 1.0, S_ATM ** 2 * np.exp(-rr / ELL) / (S_ATM ** 2 + NUG ** 2)), color=S.INK)
ax.axvline(ELL, color=S.MUTED, lw=0.8, ls=(0, (3, 2)))
ax.text(ELL + 0.4, 0.85, f"range ℓ = {ELL:.0f} km", fontsize=8.5, color=S.INK2)
ax.set_xlabel("distance between pixels (km)"); ax.set_ylabel("correlation")
ax.set_ylim(-0.05, 1.02)
ax.set_title("(e) InSAR: correlation falls over kilometres")
ax = axs[1, 2]
lt = t_t[t_t <= 60]
ax.plot(lt, C_t[0, :lt.size] / C_t[0, 0], color=S.INK)
ax.axvline(TAU, color=S.MUTED, lw=0.8, ls=(0, (3, 2)))
ax.text(TAU + 1, 0.85, f"correlation time τ = {TAU:.0f} min", fontsize=8.5, color=S.INK2)
ax.set_xlabel("time between samples (minutes)"); ax.set_ylabel("correlation")
ax.set_ylim(-0.05, 1.02)
ax.set_title("(f) Tsunami gauge: correlation over minutes")
fig.tight_layout()
bk.savefig(fig, "2-14_three_noises")
plt.show()
print(f"GNSS: {N_G} daily epochs; InSAR: {NG * NG} pixels; tsunami gauge: {t_t.size} samples")
for name, C, far in [("GNSS", C_g, 365), ("InSAR", C_i, 8), ("tsunami", C_t, 120)]:
    sd = np.sqrt(np.diag(C))
    i0 = C.shape[0] // 2 if name != "InSAR" else (NG // 2) * NG + NG // 2      # a datum in the middle
    unit, gap = ("cm", f"{far * DT_T:.0f} min") if name == "tsunami" else ("mm", "1 yr" if name == "GNSS" else f"{far * DXP:.0f} km")
    print(f"{name:8s}: error sd from {sd.min():.2f} to {sd.max():.2f} {unit}; correlation with the next datum "
          f"{C[i0, i0 + 1] / (sd[i0] * sd[i0 + 1]):.3f}, with a datum {gap} away {C[i0, i0 + far] / (sd[i0] * sd[i0 + far]):.3f}")

# %% [markdown]
# **How to read it.** Top row: one draw of each error. The GNSS error wanders in excursions of months to years, the
# InSAR error is a field of blobs a few kilometres across, and the tide-gauge residual is smooth over minutes. Bottom
# row: the correlation between one datum's error and another's, as a function of their separation. White noise alone
# (blue in d) is uncorrelated after zero lag; flicker noise stays correlated over the whole series, and a random walk's
# correlation with the reference day never falls far. Neighbouring data are correlated in all three cases (printed):
# strongly for InSAR pixels and gauge samples, moderately for GNSS days, where white noise dominates the day-to-day
# jitter. But the GNSS correlation then hardly decays: a position error a year away is still correlated with today's
# (printed), while InSAR and gauge correlations have died out over a few ranges or correlation times. A diagonal
# $\mathbf C_d$ says all of these correlations are zero.
#
# **Common misreadings.**
# * "Correlation is a second-order detail." For neighbouring InSAR pixels and gauge samples it is close to one, so a
#   diagonal $\mathbf C_d$ treats near-duplicates as independent confirmations; for GNSS a modest correlation that lasts
#   for years does the same damage to a velocity (section 2).
# * "The noise level is the standard deviation." The standard deviation says how big one error is; the correlation
#   says how many errors are really independent. Both enter the uncertainty of anything you estimate.

# %% [markdown]
# ## 2. GNSS time series: white, flicker and random-walk noise
#
# ### 2.1 Power-law noise
#
# **In plain words.** Split a noise record into slow and fast wiggles. White noise has the same power at every period.
# Flicker noise has more power the longer the period (each tenfold longer period carries ten times the power), and a
# random walk has even more (a hundredfold). A daily GNSS position series is usually described as a sum of these:
# white at short periods, flicker and possibly a random walk at long periods (Mao et al. 1999; Williams 2003).
#
# **Precisely.** A power-law noise has a power spectrum $P(f) \propto f^{\kappa}$ with spectral index $\kappa = 0$ (white),
# $-1$ (flicker) or $-2$ (random walk). For data sampled every $\Delta T$ it can be written as
# $\mathbf e = \sigma_\kappa\,\Delta T^{-\kappa/4}\,\mathbf T_\kappa\,\mathbf z$ with $\mathbf z$ white and $\mathbf T_\kappa$ lower-triangular Toeplitz
# (constant along each diagonal) with
# $h_0 = 1$, $h_i = h_{i-1}\,(i - 1 - \kappa/2)/i$, so
#
# $$\mathbf C_d = \sigma_w^2\,\mathbf I + \sigma_f^2\,\Delta T^{1/2}\,\mathbf T_{-1}\mathbf T_{-1}^\top + \sigma_{rw}^2\,\Delta T\,\mathbf T_{-2}\mathbf T_{-2}^\top .$$
#
# For κ = −2 every $h_i = 1$ and $\mathbf T$ is a running sum, so the random walk's variance grows as $\sigma_{rw}^2 t$.
# Units: $\sigma_f$ in mm/yr$^{1/4}$ and $\sigma_{rw}$ in mm/yr$^{1/2}$ with $\Delta T$ in years.
#
# The spectrum is measured with a **periodogram**: write the series, after removing its straight-line trend
# (*detrending*), as a sum of sine waves of different periods (a Fourier decomposition) and square each wave's
# amplitude. Below, it is
# averaged over 200 simulated series of each component so its slope is visible.

# %%
NREP_SP = 200
Tf_g, Tr_g = powerlaw_T(N_G, -1.0), powerlaw_T(N_G, -2.0)
A_g = np.column_stack([np.ones(N_G), t_g])
Z = rng.standard_normal((N_G, NREP_SP))
comp = {"white": SW * Z, "flicker": SF * DT ** 0.25 * (Tf_g @ Z), "random walk": SR * DT ** 0.5 * (Tr_g @ Z)}
comp["all three"] = comp["white"] + comp["flicker"] + comp["random walk"]
freq = np.fft.rfftfreq(N_G, d=DT)[1:]                   # cycles per year
spec = {}
for k, X in comp.items():
    Xd = X - A_g @ np.linalg.lstsq(A_g, X, rcond=None)[0]   # remove offset and trend, as for a velocity fit
    spec[k] = (np.abs(np.fft.rfft(Xd, axis=0)) ** 2)[1:].mean(1) * 2 * DT / N_G
band = (freq >= 2) & (freq <= 100)
slopes = {k: np.polyfit(np.log10(freq[band]), np.log10(spec[k][band]), 1)[0] for k in spec}

fig, axs = plt.subplots(1, 2, figsize=(12.5, 4.0), gridspec_kw=dict(width_ratios=[1.2, 1]))
ax = axs[0]
for k, col in [("white", C_W), ("flicker", C_F), ("random walk", C_R), ("all three", S.INK)]:
    ax.loglog(freq, spec[k], color=col, lw=1.6 if k == "all three" else 1.0,
              label=f"{k}: slope {slopes[k]:+.2f}")
ax.axvspan(2, 100, color=S.GRID, alpha=0.5, lw=0)
ax.set_xlabel("frequency (cycles per year)"); ax.set_ylabel("power (mm² per cycle/yr)")
ax.set_title("(a) Average periodogram of 200 series; slopes fitted in the shaded band", fontsize=10)
ax.legend(fontsize=8.2, loc="lower left")
ax = axs[1]
for k, col in [("white", C_W), ("flicker", C_F), ("random walk", C_R)]:
    ax.plot(t_g, comp[k][:, 0], color=col, lw=0.6, label=k)
ax.set_xlabel("time (years)"); ax.set_ylabel("error (mm)")
ax.set_title("(b) One series of each component")
ax.legend(fontsize=8.2)
fig.tight_layout()
bk.savefig(fig, "2-14_gnss_spectrum")
plt.show()
print("fitted log-log slopes between 2 and 100 cycles/yr:", {k: round(v, 2) for k, v in slopes.items()})
cross = freq[np.argmin(np.abs(np.log(spec["white"]) - np.log(spec["flicker"])))]
print(f"white and flicker power are equal near {cross:.0f} cycles/yr (period {365.25 / cross:.0f} days)")

# %% [markdown]
# **How to read it.** (a) On log-log axes each component is a straight line whose slope is its spectral index
# (printed: close to 0, −1 and −2). The sum (black) is flat at high frequencies, where white noise dominates, and
# rises at low frequencies, where flicker and random walk dominate. The crossover between white and flicker power
# is printed. (b) The three components drawn alone: the random walk drifts, flicker wanders, white noise only jitters.
#
# **Common misreadings.**
# * "The daily scatter tells you the noise." The daily scatter is set by the flat, high-frequency part. A velocity is
#   a very low-frequency quantity, so its error is set by the low-frequency end of the spectrum, which the daily scatter
#   does not see.

# %% [markdown]
# ### 2.2 What a white-noise assumption does to a velocity
#
# **In plain words.** Fit a straight line to four years of daily positions and ask how well the slope is known. If the
# errors were independent, 1461 days would pin the slope down very precisely. With wandering errors, a slow excursion
# can tilt the whole line, so the slope is much less certain than the white-noise formula says.
#
# **Precisely.** For $\mathbf d = \mathbf A[a, v]^\top + \mathbf e$ with $\mathbf A = [\mathbf 1, \mathbf t]$, the least-squares rate under the
# white assumption has standard error $\hat\sigma/\sqrt{\sum (t_i-\bar t)^2}$, with $\hat\sigma$ the residual scatter. The
# generalised least-squares (GLS) estimate with the true $\mathbf C_d$ has covariance $(\mathbf A^\top\mathbf C_d^{-1}\mathbf A)^{-1}$. We
# check both against the actual scatter of rates over 400 simulated series (true velocity 12 mm/yr).

# %%
NREP_V = 400
V_TRUE = 12.0
Zg = L_g @ rng.standard_normal((N_G, NREP_V))
Dg = (3.0 + V_TRUE * t_g)[:, None] + Zg
coef_ols = np.linalg.lstsq(A_g, Dg, rcond=None)[0]
res = Dg - A_g @ coef_ols
sig_hat = res.std(axis=0, ddof=2)
se_white = sig_hat / np.sqrt(((t_g - t_g.mean()) ** 2).sum())
coef_gls, cov_gls = gls(A_g, Dg, C_g)
rate = dict(white_formal=se_white.mean(), ols_actual=coef_ols[1].std(), gls_formal=np.sqrt(cov_gls[1, 1]),
            gls_actual=coef_gls[1].std())
print(f"rate uncertainty over {N_G} days ({NREP_V} simulated series, true rate {V_TRUE} mm/yr):")
print(f"  least squares, white-noise formal error (mean over series): {rate['white_formal']:.3f} mm/yr")
print(f"  least squares, actual scatter of the rates:                 {rate['ols_actual']:.3f} mm/yr")
print(f"  GLS with the true C_d, formal error:                        {rate['gls_formal']:.3f} mm/yr")
print(f"  GLS with the true C_d, actual scatter:                      {rate['gls_actual']:.3f} mm/yr")
print(f"  white-noise error bar is too small by a factor {rate['ols_actual'] / rate['white_formal']:.1f}")
print(f"  mean rate: least squares {coef_ols[1].mean():.3f}, GLS {coef_gls[1].mean():.3f} mm/yr (both unbiased)")

# %% [markdown]
# The white-noise formula understates the rate error by the factor printed above. GLS with the true covariance gives a
# formal error that matches its actual scatter, and its rates scatter slightly less than plain least squares
# (both printed). Both estimators are unbiased: the damage is in the error bar, not in the mean.
#
# ### 2.3 How the rate error shrinks with the length of the series

# %%
spans = np.array([0.5, 1, 2, 3, 4, 6, 8])
I8 = np.eye(Cf8.shape[0])
rows = {"white only": [], "flicker only": [], "random walk only": [], "all three": [], "all three, white assumption": []}
for yrs in spans:
    n = int(yrs * 365.25)
    tt = np.arange(n) * DT
    A = np.column_stack([np.ones(n), tt])
    for k, C in [("white only", SW ** 2 * I8[:n, :n]), ("flicker only", SF ** 2 * Cf8[:n, :n]),
                 ("random walk only", SR ** 2 * Cr8[:n, :n])]:
        rows[k].append(np.sqrt(gls(A, np.zeros(n), C)[1][1, 1]))
    Call = SW ** 2 * I8[:n, :n] + SF ** 2 * Cf8[:n, :n] + SR ** 2 * Cr8[:n, :n]
    rows["all three"].append(np.sqrt(gls(A, np.zeros(n), Call)[1][1, 1]))
    rows["all three, white assumption"].append(np.sqrt(np.diag(Call).mean() / ((tt - tt.mean()) ** 2).sum()))
rows = {k: np.array(v) for k, v in rows.items()}
fit = spans >= 1
span_slope = {k: np.polyfit(np.log(spans[fit]), np.log(v[fit]), 1)[0] for k, v in rows.items()}

fig, ax = plt.subplots(figsize=(7.2, 4.3))
for k, col, ls in [("white only", C_W, "-"), ("flicker only", C_F, "-"), ("random walk only", C_R, "-"),
                   ("all three", S.INK, "-"), ("all three, white assumption", S.INK, (0, (4, 2)))]:
    ax.loglog(spans, rows[k], "o", ls=ls, color=col, ms=4, lw=1.2, label=f"{k}: slope {span_slope[k]:+.2f}")
ax.set_xticks(spans); ax.set_xticklabels([f"{s_:g}" for s_ in spans]); ax.minorticks_off()
ax.set_xlabel("length of the daily series (years)"); ax.set_ylabel("formal rate uncertainty (mm/yr)")
ax.set_title("Rate uncertainty vs length of the series")
ax.legend(fontsize=8)
fig.tight_layout()
bk.savefig(fig, "2-14_rate_vs_span")
plt.show()
print("log-log slopes for spans >= 1 yr:", {k: round(v, 2) for k, v in span_slope.items()})
print(f"8-year series: true (all three) {rows['all three'][-1]:.3f} mm/yr vs white assumption "
      f"{rows['all three, white assumption'][-1]:.4f} mm/yr (ratio {rows['all three'][-1] / rows['all three, white assumption'][-1]:.0f})")

# %% [markdown]
# **How to read it.** Each line is the formal GLS rate error for one noise model (computed, not simulated). White noise
# alone falls steeply with the series length (slope near −1.5), flicker noise more slowly (near −1), and a random walk
# slowest (near −0.5): a longer series cannot average a random walk away nearly as fast. The black lines are the noise
# used here: the solid one is the true error, the dashed one is what a white-noise formula with the same daily scatter
# reports. The gap between them widens with time (8-year ratio printed).
#
# ### 2.4 Estimating the noise model: maximum likelihood
#
# **In plain words.** In practice the amplitudes are unknown and are estimated from the series itself. The likelihood of
# the series under a candidate noise model is computed exactly with the full covariance, and the amplitudes that make
# the observed wobbles most probable are kept.
#
# **Precisely.** With the offset and rate *profiled out* (re-estimated by GLS for each trial set of noise amplitudes),
#
# $$-\ln p(\mathbf d\mid\sigma_w,\sigma_f,\sigma_{rw}) = \tfrac12\,\mathbf r^\top\mathbf C_d^{-1}\mathbf r + \tfrac12\ln\det\mathbf C_d + \text{const},$$
#
# minimised over the three amplitudes (on a log scale, lower bound $e^{-5}$). We repeat this on 12 independent
# 4-year series with the same true amplitudes, to see how well each amplitude is determined.

# %%
I_g = np.eye(N_G)


def nll_noise(p, x):
    C = np.exp(2 * p[0]) * I_g + np.exp(2 * p[1]) * Cf_g + np.exp(2 * p[2]) * Cr_g
    cf = cho_factor(C, lower=True)
    CiA = cho_solve(cf, A_g)
    coef = np.linalg.solve(A_g.T @ CiA, CiA.T @ x)
    r = x - A_g @ coef
    return 0.5 * (r @ cho_solve(cf, r)) + np.log(np.diag(cf[0])).sum()


NREP_ML = 12
LB = -5.0
rng_ml = np.random.default_rng(21)
est = []
t1 = time.time()
for _ in range(NREP_ML):
    x = 3.0 + V_TRUE * t_g + L_g @ rng_ml.standard_normal(N_G)
    r_ = minimize(nll_noise, np.log([1.0, 1.0, 0.3]), args=(x,), method="L-BFGS-B", bounds=[(LB, 3.0)] * 3)
    est.append(np.exp(r_.x))
est = np.array(est)
print(f"{NREP_ML} maximum-likelihood fits in {time.time() - t1:.0f} s")
print(" series   white (mm)   flicker (mm/yr^0.25)   random walk (mm/yr^0.5)   rate error implied (mm/yr)")
rate_ml = []
for j, (w_, f_, r_) in enumerate(est):
    Cm_ = w_ ** 2 * I_g + f_ ** 2 * Cf_g + r_ ** 2 * Cr_g
    rate_ml.append(np.sqrt(gls(A_g, np.zeros(N_G), Cm_)[1][1, 1]))
    rw_txt = f"{r_:.2f}" if r_ >= 0.1 else f"{r_:.3f} (≈ 0)"
    print(f"  {j + 1:3d}      {w_:6.2f}         {f_:6.2f}              {rw_txt:>22s}       {rate_ml[-1]:.3f}")
rate_ml = np.array(rate_ml)
n_zero = int(np.sum(est[:, 2] < 0.1))
print(f"truth:     {SW:6.2f}         {SF:6.2f}              {SR:22.2f}       {rate['gls_formal']:.3f}")
print(f"random-walk amplitude estimated below 0.1 mm/yr^0.5 (effectively zero; true {SR}) in {n_zero} of {NREP_ML} series")

fig, axs = plt.subplots(1, 2, figsize=(12, 3.8), gridspec_kw=dict(width_ratios=[1.3, 1]))
ax = axs[0]
jit = np.linspace(-0.12, 0.12, NREP_ML)
for i, (lab, tru, col) in enumerate([("white\n(mm)", SW, C_W), ("flicker\n(mm/yr$^{1/4}$)", SF, C_F),
                                     ("random walk\n(mm/yr$^{1/2}$)", SR, C_R)]):
    ax.plot(i + jit, est[:, i], "o", color=col, ms=5, alpha=0.85)
    ax.plot([i - 0.3, i + 0.3], [tru, tru], color=S.INK, lw=1.6)
ax.set_xticks(range(3)); ax.set_xticklabels(["white (mm)", "flicker (mm/yr$^{1/4}$)", "random walk (mm/yr$^{1/2}$)"])
ax.set_ylabel("estimated amplitude")
ax.set_title(f"(a) Amplitudes from {NREP_ML} independent 4-year series (black: truth)")
ax = axs[1]
ax.plot(np.arange(1, NREP_ML + 1), rate_ml, "o", color=S.INK2, ms=5, label="from each fitted noise model")
ax.axhline(rate["gls_formal"], color=S.INK, lw=1.4, label="from the true noise model")
ax.axhline(rate["white_formal"], color=C_DIAG, lw=1.2, ls=(0, (4, 2)), label="white-noise formula")
ax.set_ylim(0, None)
ax.set_xlabel("series"); ax.set_ylabel("formal rate uncertainty (mm/yr)")
ax.set_title("(b) The rate error each fitted noise model implies")
ax.legend(fontsize=8, loc="lower right", bbox_to_anchor=(1.0, 0.1))
fig.tight_layout()
bk.savefig(fig, "2-14_gnss_mle")
plt.show()

# %% [markdown]
# **How to read it.** (a) The white amplitude is recovered tightly and the flicker amplitude reasonably well. The
# random-walk amplitude is not: in some series it is estimated above the truth, in others it collapses to nearly zero
# (count printed above), because over four years a small random walk is hard to tell apart from flicker noise. (b) Each
# fitted noise model implies a rate error; where the random walk was estimated as zero, the implied rate error is smaller
# than the true one. All of them are far above the white-noise formula.
#
# **Worked example.** Five yearly measurements (a campaign site, $\Delta T$ = 1 yr) with white noise 1.5 mm and flicker
# noise 3 mm/yr$^{1/4}$. The steps: (1) the coefficients $h_i$ for κ = −1; (2) $\mathbf T$ and
# $\mathbf C_d = \sigma_w^2\mathbf I + \sigma_f^2\,\Delta T^{1/2}\,\mathbf T\mathbf T^\top$; (3) the rate error from the white formula with the same average
# variance and from GLS with the full $\mathbf C_d$.

# %%
n5 = 5
T5 = powerlaw_T(n5, -1.0)
C5 = SW ** 2 * np.eye(n5) + SF ** 2 * 1.0 ** 0.5 * T5 @ T5.T
t5 = np.arange(n5, dtype=float)
A5 = np.column_stack([np.ones(n5), t5])
print("step 1  h_i = h_{i-1} (i - 1/2) / i:", np.round(powerlaw_T(n5, -1.0)[:, 0], 4))
print("step 2  T =\n", np.round(T5, 4))
print("        C_d (mm^2) =\n", np.round(C5, 2))
sd5 = np.sqrt(np.diag(C5))
print("        correlation between year 1 and years 2-5:", np.round(C5[0, 1:] / (sd5[0] * sd5[1:]), 2))
se_w5 = np.sqrt(np.diag(C5).mean() / ((t5 - t5.mean()) ** 2).sum())
w_ols = (t5 - t5.mean()) / ((t5 - t5.mean()) ** 2).sum()          # least-squares rate = w_ols . d
se_o5 = np.sqrt(w_ols @ C5 @ w_ols)
se_f5 = np.sqrt(np.linalg.inv(A5.T @ np.linalg.solve(C5, A5))[1, 1])
print(f"step 3  white formula: sqrt({np.diag(C5).mean():.2f} / {((t5 - t5.mean()) ** 2).sum():.0f}) = {se_w5:.3f} mm/yr")
print(f"        actual error of that least-squares rate, sqrt(w^T C_d w): {se_o5:.3f} mm/yr  (ratio to the formula {se_o5 / se_w5:.2f})")
print(f"        GLS with the full C_d: {se_f5:.3f} mm/yr")

# %% [markdown]
# With five yearly epochs the white formula is close to the actual error (step 3): one year apart, the errors are only
# weakly correlated (step 2), and the positive correlation that remains partly cancels in a difference of positions,
# which is what a rate is. The damage done by a white-noise assumption comes from many closely spaced samples, as with
# the 1461 daily epochs above; section 5 makes this general.
#
# **In geodesy.** Published GNSS velocities are usually given with power-law noise models estimated this way, often with
# the random-walk amplitude fixed or bounded because a single series cannot determine it (panel a). The covariance for
# an inversion of *velocities* is then the GLS covariance of the rates, one 2 × 2 block (offset, rate) per component
# and station.

# %% [markdown]
# ## 3. InSAR: spatially correlated atmosphere
#
# ### 3.1 The empirical covariogram
#
# **In plain words.** To learn how InSAR errors are correlated, take an interferogram area with no deformation (or the
# residuals after removing a model), and for every pair of pixels multiply their values. Group the pairs by distance and
# average: close pairs have a large average product (they err together), distant pairs an average near zero. That
# curve is the empirical covariogram; fit a simple function to it and you have a formula for every entry of $\mathbf C_d$.
#
# **Precisely.** For a zero-mean, stationary (the same statistics everywhere) and isotropic (the same in every
# direction) error field, $\mathrm{Cov}(e(\mathbf x), e(\mathbf x')) = c(\lVert\mathbf x-\mathbf x'\rVert)$.
# The estimate is $\hat c(r) = \text{mean}\{e_ie_j : \lVert\mathbf x_i-\mathbf x_j\rVert\in\text{bin}(r)\}$. A common model is the
# exponential $c(r) = s^2 e^{-r/\ell}$ for $r>0$ plus a **nugget** $n^2$ at $r = 0$ (uncorrelated pixel noise). The **sill**
# $s^2 + n^2$ is the total variance; the **range** $\ell$ is the distance over which correlation falls by $e$. (Jónsson
# et al. 2002 and Lohman & Simons 2005 describe this practice for InSAR.)
#
# Below, the covariogram is estimated from one simulated deformation-free interferogram on the 64 × 64 grid.

# %%
rng_i = np.random.default_rng(2)
e_free = L_i @ rng_i.standard_normal(NG * NG)
ef = e_free - e_free.mean()
iu = np.triu_indices(NG * NG, 1)
dist_pairs = DIST[iu]
prod = ef[iu[0]] * ef[iu[1]]
bins = np.arange(0, 30.01, 0.75)
idx = np.digitize(dist_pairs, bins) - 1
ok = (idx >= 0) & (idx < len(bins) - 1)
cnt = np.bincount(idx[ok], minlength=len(bins) - 1)
cov_emp = np.bincount(idx[ok], weights=prod[ok], minlength=len(bins) - 1) / np.maximum(cnt, 1)
rc = 0.5 * (bins[1:] + bins[:-1])
use = cnt > 0
fitr = use & (rc <= 15.0)                                 # fit the part where the covariogram is well determined
p_fit, _ = curve_fit(lambda r, s, l: s ** 2 * np.exp(-r / l), rc[fitr], cov_emp[fitr], p0=[5.0, 5.0])
s_fit, l_fit = abs(p_fit[0]), p_fit[1]
var0 = ef.var()
nug_fit = np.sqrt(max(var0 - s_fit ** 2, 0.0))
del iu, dist_pairs, prod, idx
print(f"pairs used: {cnt.sum():,}")
print(f"fitted sill part s = {s_fit:.2f} mm (true {S_ATM}), range = {l_fit:.2f} km (true {ELL}), "
      f"nugget = {nug_fit:.2f} mm (true {NUG}; from total variance {var0:.1f} mm² minus s²)")

# %% [markdown]
# ### 3.2 Decimation: uniform and quadtree
#
# **In plain words.** An interferogram has far more pixels than an inversion needs or can afford, so it is thinned.
# *Uniform* decimation keeps every k-th pixel. *Quadtree* decimation splits the image into squares, keeps splitting a
# square while the data inside it vary more than a threshold, and replaces each final square by the mean of its pixels.
# Quadtree keeps many points where the signal changes fast (near the source) and few where it is flat.
#
# **Precisely.** Both are linear operators on the pixel vector, $\mathbf d_\text{dec} = \mathbf A\mathbf d_\text{pix}$ (a selection matrix,
# or an averaging matrix with $1/n_\text{pix}$ in each leaf's row). The decimated data then have exactly
#
# $$\mathbf C_d^\text{dec} = \mathbf A\,\mathbf C_d^\text{pix}\,\mathbf A^\top,\qquad \mathbf G^\text{dec} = \mathbf A\,\mathbf G^\text{pix}.$$
#
# A leaf that averages many pixels has a smaller variance, but much less smaller than $1/n_\text{pix}$ when its pixels are
# correlated. Here the quadtree threshold is a data variance of 25 mm² inside a square and the smallest square is
# 2 × 2 pixels, computed on one noisy interferogram and then held fixed (as one would in practice).
#
# The signal: two point sources (Mogi) under the centre of the image, at 3 km and 9 km depth, with volume changes
# 1 and 4 (× 10⁶ m³), seen along a line of sight 38° from vertical. Eight GNSS stations measure the three displacement
# components with independent 2 mm errors.

# %%
NU_P = 0.25
inc, head = np.deg2rad(38.0), np.deg2rad(10.0)
LOS = np.array([-np.sin(inc) * np.cos(head), np.sin(inc) * np.sin(head), np.cos(inc)])   # east, north, up
DEPTHS = (3.0, 9.0)
M_TRUE = np.array([1.0, 4.0])


def mogi(depth, pts):
    """East, north, up displacement (mm) at pts (km) for a volume change of 1e6 m^3 at (0, 0, -depth)."""
    r = np.hypot(pts[:, 0], pts[:, 1]) * 1e3
    d = depth * 1e3
    R3 = (r ** 2 + d ** 2) ** 1.5
    c = (1 - NU_P) * 1e6 / np.pi
    uz, ur = c * d / R3 * 1e3, c * r / R3 * 1e3
    rr_ = np.maximum(r, 1e-9)
    return np.column_stack([ur * pts[:, 0] * 1e3 / rr_, ur * pts[:, 1] * 1e3 / rr_, uz])


G_pix = np.column_stack([mogi(z, PIX) @ LOS for z in DEPTHS])
az = np.deg2rad([20, 75, 130, 170, 215, 260, 300, 340])
rst = np.array([4, 6, 8, 11, 14, 17, 20, 24.0])
GPS = np.column_stack([rst * np.cos(az), rst * np.sin(az)])
G_gps = np.column_stack([mogi(z, GPS).ravel() for z in DEPTHS])
SG = 2.0
I_gps = G_gps.T @ G_gps / SG ** 2
sig_pix = G_pix @ M_TRUE
d_obs_pix = sig_pix + L_i @ rng_i.standard_normal(NG * NG)


def quadtree(img, thr, nmin=2):
    leaves = []

    def split(i0, j0, n):
        blk = img[i0:i0 + n, j0:j0 + n]
        if n > nmin and blk.var() > thr:
            h = n // 2
            for a in (0, h):
                for b in (0, h):
                    split(i0 + a, j0 + b, h)
        else:
            leaves.append((i0, j0, n))
    split(0, 0, img.shape[0])
    return leaves


def averaging_operator(leaves):
    A = np.zeros((len(leaves), NG * NG))
    for k, (i0, j0, n) in enumerate(leaves):
        ii, jj = np.meshgrid(np.arange(i0, i0 + n), np.arange(j0, j0 + n), indexing="ij")
        A[k, (ii * NG + jj).ravel()] = 1.0 / n ** 2
    return A


leaves = quadtree(d_obs_pix.reshape(NG, NG), thr=25.0)
A_q = averaging_operator(leaves)
keep = np.zeros((NG, NG), bool)
keep[1::4, 1::4] = True
A_u = np.eye(NG * NG)[keep.ravel()]
DEC = {"uniform": A_u, "quadtree": A_q}
sizes, counts = np.unique([l_[2] for l_ in leaves], return_counts=True)
print(f"uniform: every 4th pixel -> {A_u.shape[0]} points; quadtree -> {A_q.shape[0]} leaves "
      f"(sizes in pixels: {dict(zip(sizes.tolist(), counts.tolist()))})")
for name, A in DEC.items():
    Cdec = A @ C_i @ A.T
    sd = np.sqrt(np.diag(Cdec))
    print(f"{name:8s}: point error sd from {sd.min():.2f} to {sd.max():.2f} mm")
big = [k for k, l_ in enumerate(leaves) if l_[2] == sizes.max()][0]
npx = leaves[big][2] ** 2
print(f"largest quadtree leaf: {npx} pixels; its error sd {np.sqrt((A_q @ C_i @ A_q.T)[big, big]):.2f} mm, "
      f"vs {np.sqrt(S_ATM ** 2 + NUG ** 2) / np.sqrt(npx):.2f} mm if its pixels were independent")

fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.5), gridspec_kw=dict(width_ratios=[1.05, 1, 1]))
ax = axs[0]
ax.plot(rc[use], cov_emp[use], "o", color=S.INK2, ms=3.5, label="empirical (one interferogram)")
rr = np.linspace(0.01, 30, 300)
ax.plot(rr, S_ATM ** 2 * np.exp(-rr / ELL), color=S.INK, lw=1.4, label="true exponential")
ax.plot(rr, s_fit ** 2 * np.exp(-rr / l_fit), color=C_FULL, lw=1.3, ls=(0, (4, 2)), label="fitted exponential")
ax.plot([0], [var0], "s", color=S.INK2, ms=5, label="variance at r = 0 (sill + nugget)")
ax.axhline(0, color=S.AXIS, lw=0.7)
ax.set_xlabel("distance between pixels (km)"); ax.set_ylabel("covariance (mm²)")
ax.set_title("(a) Empirical covariogram and exponential fit")
ax.legend(fontsize=8)
ext = [xg[0] - DXP / 2, xg[-1] + DXP / 2] * 2
for ax, name in [(axs[1], "uniform"), (axs[2], "quadtree")]:
    im = ax.imshow(d_obs_pix.reshape(NG, NG), origin="lower", extent=ext, cmap="RdBu_r", vmin=-40, vmax=40)
    if name == "uniform":
        ax.plot(PIX[keep.ravel(), 0], PIX[keep.ravel(), 1], ".", color=S.INK, ms=2.5)
    else:
        for (i0, j0, n) in leaves:
            ax.add_patch(Rectangle((xg[0] - DXP / 2 + j0 * DXP, xg[0] - DXP / 2 + i0 * DXP), n * DXP, n * DXP,
                                   fill=False, lw=0.5, ec=S.INK))
    ax.plot(GPS[:, 0], GPS[:, 1], "^", color=S.VIOLET, ms=6.5, mec=S.SURFACE, mew=0.8, label="GNSS stations")
    ax.set_xlabel("east (km)"); ax.set_ylabel("north (km)")
    ax.set_title(f"({'b' if name == 'uniform' else 'c'}) {name.capitalize()} decimation: {DEC[name].shape[0]} points")
    ax.legend(fontsize=8, loc="lower left")
fig.colorbar(im, ax=axs[1:], label="observed line-of-sight displacement (mm)", shrink=0.85)
bk.savefig(fig, "2-14_insar_covariogram")
plt.show()

# %% [markdown]
# **How to read it.** (a) The empirical covariances (dots) follow the true exponential closely at short distances and
# dip below zero at long ones. That dip is a known artefact of estimating from one finite image after removing its mean:
# the deviations from the image's own mean must average to zero, so distant pairs are pushed negative. The fit
# therefore uses distances up to 15 km. The fitted values are printed above next to the truth; the nugget, obtained by
# subtracting the fitted sill part from the total variance, is the least well determined of the three. (b) Uniform decimation spreads points evenly, most of them far from the source.
# (c) Quadtree leaves are small where the deformation changes quickly and large where it is flat. The largest leaf's
# error is printed: much larger than the independent-pixel value, because its pixels share the same atmosphere.
#
# **Worked example.** Three pixels on a line at 0, 1 and 6 km, with the covariance above (s = 6 mm, ℓ = 5 km,
# n = 2 mm). The steps: (1) the 3 × 3 $\mathbf C_d$; (2) the correlations; (3) how many independent pixels the three are
# worth for estimating a common offset: $N_\text{eff} = (s^2+n^2)\,\mathbf 1^\top\mathbf C_d^{-1}\mathbf 1$ (section 5 explains the
# formula).

# %%
x3 = np.array([0.0, 1.0, 6.0])
C3 = S_ATM ** 2 * np.exp(-np.abs(x3[:, None] - x3[None, :]) / ELL) + NUG ** 2 * np.eye(3)
sd3 = np.sqrt(np.diag(C3))
print("step 1  C_d (mm^2) =\n", np.round(C3, 2))
print("step 2  correlations =\n", np.round(C3 / np.outer(sd3, sd3), 3))
neff3 = (S_ATM ** 2 + NUG ** 2) * np.ones(3) @ np.linalg.solve(C3, np.ones(3))
neff2 = (S_ATM ** 2 + NUG ** 2) * np.ones(2) @ np.linalg.solve(C3[:2, :2], np.ones(2))
print(f"step 3  N_eff of the three pixels = {neff3:.2f}; of the two pixels 1 km apart alone = {neff2:.2f}")

# %% [markdown]
# Two pixels 1 km apart are worth little more than one (step 3); the third pixel, 6 km away, adds most of the
# remaining information.

# %% [markdown]
# ### 3.3 What ignoring the correlation does to the weights
#
# **In plain words.** In a joint inversion, each data set pulls the answer toward what it sees, with a strength set by
# how much information it is believed to carry. If the InSAR covariance is taken as diagonal, every decimated point
# counts as an independent measurement, and InSAR is believed to carry far more information than it does. The
# GNSS stations, whose errors really are independent here, are drowned out.
#
# **Precisely.** The information (inverse covariance) about the two volume changes is the sum
# $\mathbf F = \mathbf G_\text{InSAR}^\top\mathbf C_\text{InSAR}^{-1}\mathbf G_\text{InSAR} + \mathbf G_\text{GNSS}^\top\mathbf C_\text{GNSS}^{-1}\mathbf G_\text{GNSS}$.
# The InSAR **information share** for parameter $j$ is its term's $jj$ entry divided by the total's. "Diagonal" keeps
# each point's correct variance but drops the off-diagonal entries; "full" uses $\mathbf A\mathbf C_d^\text{pix}\mathbf A^\top$.

# %%
share = {}
print(f"{'decimation':10s} {'C_d model':9s} {'InSAR-only sd shallow':>22s} {'deep':>7s}   {'InSAR share shallow':>20s} {'deep':>6s}")
for name, A in DEC.items():
    Cdec = A @ C_i @ A.T
    G = A @ G_pix
    F_diag = G.T @ (G / np.diag(Cdec)[:, None])
    F_full = G.T @ np.linalg.solve(Cdec, G)
    for lab, F in [("diagonal", F_diag), ("full", F_full)]:
        sh = np.diag(F) / (np.diag(F) + np.diag(I_gps))
        share[(name, lab)] = sh
        sdo = np.sqrt(np.diag(np.linalg.inv(F)))
        print(f"{name:10s} {lab:9s} {sdo[0]:22.3f} {sdo[1]:7.3f}   {sh[0]:20.2f} {sh[1]:6.2f}")
print("GNSS-only sd (shallow, deep):", np.round(np.sqrt(np.diag(np.linalg.inv(I_gps))), 3))

# %% [markdown]
# The printed table is the core result. With a diagonal $\mathbf C_d$, InSAR alone claims to determine the deep source
# much better than it does (compare the "deep" sd columns), and its share of the information about the deep source is
# several times what the full covariance gives. The reason is physical: the deep source makes a broad, smooth bowl, and
# a broad smooth pattern is exactly what correlated atmosphere can imitate. The shallow source's sharp, localised
# signal is harder for the atmosphere to imitate, so its share changes less.
#
# ### 3.4 Cost of ignoring the correlation: 400 synthetic interferograms
#
# The same joint GNSS + InSAR estimate (weighted least squares, flat prior), repeated for 400 independent noise draws,
# with the InSAR covariance taken as diagonal or full. A 90 % confidence ellipse from the formal covariance should
# contain the true volume changes in 90 % of the repeats.

# %%
NREP_I = 400
rng_mc = np.random.default_rng(11)
Z_i = L_i @ rng_mc.standard_normal((NG * NG, NREP_I))
Z_gps = SG * rng_mc.standard_normal((G_gps.shape[0], NREP_I))
d_gps = (G_gps @ M_TRUE)[:, None] + Z_gps
Q90 = chi2.ppf(0.9, 2)


def joint_estimates(A, scheme):
    """Joint GNSS + InSAR weighted least squares for all repeats. scheme: diagonal, sigma_k (iterated), full."""
    G = A @ G_pix
    Cdec = A @ C_i @ A.T
    dI = (A @ sig_pix)[:, None] + A @ Z_i
    vi = np.diag(Cdec)
    est, covs, ks = [], [], []
    if scheme == "full":
        Wi = np.linalg.inv(Cdec)
        F = G.T @ Wi @ G + I_gps
        cov = np.linalg.inv(F)
        E = (cov @ (G.T @ Wi @ dI + G_gps.T @ d_gps / SG ** 2)).T
        return E, np.repeat(cov[None], NREP_I, 0), None
    for r in range(NREP_I):
        ki = kg = 1.0
        for _ in range(30 if scheme == "sigma_k" else 1):
            F = G.T @ (G / (ki * vi)[:, None]) + G_gps.T @ G_gps / (SG ** 2 * kg)
            m = np.linalg.solve(F, G.T @ (dI[:, r] / (ki * vi)) + G_gps.T @ d_gps[:, r] / (SG ** 2 * kg))
            if scheme == "sigma_k":          # variance-component update: sigma_k^2 = chi^2 / N per data set
                ki = np.mean((dI[:, r] - G @ m) ** 2 / vi)
                kg = np.mean((d_gps[:, r] - G_gps @ m) ** 2 / SG ** 2)
        if scheme == "sigma_k":
            F = G.T @ (G / (ki * vi)[:, None]) + G_gps.T @ G_gps / (SG ** 2 * kg)
            m = np.linalg.solve(F, G.T @ (dI[:, r] / (ki * vi)) + G_gps.T @ d_gps[:, r] / (SG ** 2 * kg))
        est.append(m); covs.append(np.linalg.inv(F)); ks.append((ki, kg))
    return np.array(est), np.array(covs), np.array(ks)


MC = {}
for name, A in DEC.items():
    for scheme in ["diagonal", "sigma_k", "full"]:
        E, CV, ks = joint_estimates(A, scheme)
        e = E - M_TRUE
        md = np.einsum("ri,rij,rj->r", e, np.linalg.inv(CV), e)
        MC[(name, scheme)] = dict(E=E, CV=CV, ks=ks, cover=np.mean(md < Q90), actual=E.std(0),
                                  formal=np.sqrt(CV[:, [0, 1], [0, 1]].mean(0)), mean=E.mean(0))
print(f"{'decimation':10s} {'scheme':9s} {'mean (shallow, deep)':>22s} {'actual sd':>16s} {'formal sd':>16s} {'90% coverage':>13s}")
for (name, scheme), r_ in MC.items():
    print(f"{name:10s} {scheme:9s} {str(np.round(r_['mean'], 3)):>22s} {str(np.round(r_['actual'], 3)):>16s} "
          f"{str(np.round(r_['formal'], 3)):>16s} {r_['cover']:13.3f}")


def ellipse_xy(mean, cov, q):
    w, V = np.linalg.eigh(cov)
    a = np.linspace(0, 2 * np.pi, 200)
    return (mean[:, None] + V @ (np.sqrt(w * q)[:, None] * np.vstack([np.cos(a), np.sin(a)]))).T


fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.6), gridspec_kw=dict(width_ratios=[1, 1, 1.05]))
for ax, name, letter in [(axs[0], "uniform", "a"), (axs[1], "quadtree", "b")]:
    for scheme, col in [("diagonal", C_DIAG), ("full", C_FULL)]:
        r_ = MC[(name, scheme)]
        ax.plot(r_["E"][:, 0], r_["E"][:, 1], ".", color=col, ms=2.5, alpha=0.45)
        xy = ellipse_xy(M_TRUE, r_["CV"][0], Q90)
        ax.plot(xy[:, 0], xy[:, 1], color=col, lw=1.6, label=f"{scheme} $\\mathbf{{C}}_d$: formal 90 % ellipse")
    ax.plot(*M_TRUE, "*", color=S.INK, ms=12, mec=S.SURFACE, label="true value")
    ax.set_xlabel("shallow source ΔV (10⁶ m³)"); ax.set_ylabel("deep source ΔV (10⁶ m³)")
    ax.set_title(f"({letter}) Estimates from 400 repeats, {name} decimation", fontsize=10)
    ax.legend(fontsize=7.8, loc="upper right")
ax = axs[2]
labs = ["diagonal", "sigma_k", "full"]
cols = [C_DIAG, C_SK, C_FULL]
xx = np.arange(2)
for j, (scheme, col) in enumerate(zip(labs, cols)):
    vals = [MC[(n_, scheme)]["cover"] for n_ in ["uniform", "quadtree"]]
    ax.bar(xx + (j - 1) * 0.26, vals, width=0.24, color=col,
           label={"diagonal": "diagonal $\\mathbf{C}_d$", "sigma_k": "diagonal, scaled by $\\sigma_k^2$",
                  "full": "full $\\mathbf{C}_d$"}[scheme])
    for x_, v_ in zip(xx + (j - 1) * 0.26, vals):
        ax.text(x_, v_ - 0.02, f"{v_:.2f}", ha="center", va="top", fontsize=8, color=S.SURFACE, fontweight="bold")
ax.axhline(0.9, color=S.INK, lw=1.0, ls=(0, (4, 2)))
ax.text(0.5, 0.915, "nominal 0.90", ha="center", fontsize=8, color=S.INK2)
ax.set_xticks(xx); ax.set_xticklabels(["uniform", "quadtree"])
ax.set_ylim(0, 1.2); ax.set_ylabel("fraction of repeats inside the 90 % ellipse")
ax.set_title("(c) Coverage of the formal 90 % ellipse")
ax.legend(fontsize=7.8, loc="upper center", ncol=3, columnspacing=0.8, handlelength=1.2)
fig.tight_layout()
bk.savefig(fig, "2-14_insar_weights")
plt.show()

# %% [markdown]
# **How to read it.** (a, b) Each dot is the estimate from one noise draw. Orange: InSAR covariance taken as diagonal;
# blue: full. The orange cloud spills far outside its own 90 % ellipse, and it is wider than the blue cloud, so the
# diagonal weighting is both overconfident and less precise (actual vs formal sd in the table). The blue cloud fills its
# ellipse as it should. Both clouds are centred on the truth (means in the table): ignoring correlation here biases the
# error bars and the weights, not the average estimate. (c) Coverage: the full covariance gives close to the nominal
# 0.90, the diagonal one far less, and quadtree decimation makes the diagonal case worse because its points are
# concentrated where the atmosphere's correlated blobs are counted many times. The aqua bars (scaling each data set by
# a fitted $\sigma_k$) are discussed in section 6.
#
# **Common misreadings.**
# * "Decimation removes the correlation." Thinning to every 4th pixel (3 km apart) still leaves neighbours strongly
#   correlated when the range is 5 km; averaging into leaves does not remove it either.
# * "More InSAR points means more weight, so decimate harder to balance the data sets." That balances by accident.
#   With the full $\mathbf C_d$ the weight follows from the information, whatever the number of points.

# %% [markdown]
# ## 4. Tsunami records: temporal correlation, sampling and filtering
#
# **In plain words.** A bottom-pressure gauge records sea level many times a minute. After the tide is removed, the
# remaining error is not independent from sample to sample: it drifts over minutes. Two common processing steps change
# the apparent number of data without changing the information: low-pass filtering (which makes neighbouring samples
# even more alike) and resampling to a finer interval (which invents samples by interpolation). If the covariance is
# taken as diagonal, both make the error bars smaller for no reason.
#
# **Precisely.** The noise model is $c(\Delta t) = s^2 e^{-|\Delta t|/\tau} + n^2\,\delta_{\Delta t,0}$ (s = 1 cm, τ = 8 min,
# n = 0.4 cm). Any linear processing $\mathbf d' = \mathbf F\mathbf d$ (filter, interpolation, decimation) gives
# $\mathbf C_d' = \mathbf F\mathbf C_d\mathbf F^\top$ and must be applied to the predictions too, $\mathbf G' = \mathbf F\mathbf G$. The signal is
# one waveform (a wave packet with a 22-minute period, centred 70 minutes after the earthquake) times an unknown amplitude
# (true value 1). Five schemes, each graded by 1000 noise draws:
#
# | scheme | data | $\mathbf C_d$ assumed |
# |---|---|---|
# | A | 15-s samples | diagonal, correct variance |
# | B | 15-s samples | full (true) |
# | C | 15-s samples, low-pass filtered (fast wiggles removed with a standard Butterworth filter, order 4, cut-off period 4 min, run forward and backward) | diagonal, variance of the filtered noise |
# | D | interpolated to 5-s samples | diagonal |
# | E | decimated to 1-min samples | full |

# %%
NREP_T = 1000
rng_t = np.random.default_rng(4)
wave = np.exp(-((t_t - 70) / 18) ** 2) * np.sin(2 * np.pi * (t_t - 70) / 22) * 4.0    # cm per unit amplitude
D_t = 1.0 * wave[:, None] + L_t @ rng_t.standard_normal((t_t.size, NREP_T))


def scheme_fit(F, C_assumed):
    """Fit the amplitude to processed data F d with the assumed covariance; return actual sd and formal sd."""
    g, Dp = F @ wave, F @ D_t
    a_hat, cov = gls(g[:, None], Dp, C_assumed)
    return a_hat[0].std(), np.sqrt(cov[0, 0]), a_hat[0].mean()


n_t = t_t.size
b_f, a_f = butter(4, (1 / 4.0) / (0.5 / DT_T))           # cut-off 1/(4 min); Nyquist 1/(2 x 15 s)
F_filt = filtfilt(b_f, a_f, np.eye(n_t), axis=0)
t5 = np.arange(t_t[0], t_t[-1] + 1e-9, 5 / 60)
F_int = np.array([np.interp(t5, t_t, col) for col in np.eye(n_t)]).T
F_dec = np.eye(n_t)[::4]
schemes = {}
for key, F, full in [("A", np.eye(n_t), False), ("B", np.eye(n_t), True), ("C", F_filt, False),
                     ("D", F_int, False), ("E", F_dec, True)]:
    Cp_ = F @ C_t @ F.T
    C_ass = Cp_ if full else np.diag(np.diag(Cp_))
    schemes[key] = scheme_fit(F, C_ass) + (F.shape[0],)
names_t = {"A": "15 s, diagonal", "B": "15 s, full", "C": "filtered, diagonal", "D": "5 s interpolated, diagonal",
           "E": "1 min, full"}
print(f"{'scheme':30s} {'samples':>8s} {'actual sd':>10s} {'formal sd':>10s} {'actual/formal':>14s} {'mean':>7s}")
for k, (act, frm, mn, nn) in schemes.items():
    print(f"{k} {names_t[k]:28s} {nn:8d} {act:10.4f} {frm:10.4f} {act / frm:14.2f} {mn:7.3f}")

fig, axs = plt.subplots(1, 3, figsize=(15.5, 4.0), gridspec_kw=dict(width_ratios=[1.25, 1, 1.15]))
ax = axs[0]
ax.plot(t_t, D_t[:, 0], color=S.MUTED, lw=0.7, label="record (signal + noise)")
ax.plot(t_t, wave, color=S.INK, lw=1.4, label="signal (amplitude 1)")
ax.plot(t_t, (F_filt @ D_t[:, 0]), color=C_FULL, lw=1.0, label="low-pass filtered record")
ax.set_xlabel("time after the earthquake (minutes)"); ax.set_ylabel("sea level (cm)")
ax.set_title("(a) One synthetic tsunami record")
ax.legend(fontsize=8, loc="lower right")
ax = axs[1]
lagm = np.arange(0, 241) * DT_T
Cf_t = F_filt @ C_t @ F_filt.T
mid = n_t // 2
ax.plot(lagm, C_t[mid, mid:mid + lagm.size] / C_t[mid, mid], color=S.INK, lw=1.3, label="raw 15-s samples")
ax.plot(lagm, Cf_t[mid, mid:mid + lagm.size] / Cf_t[mid, mid], color=C_FULL, lw=1.3, ls=(0, (4, 2)),
        label="after low-pass filtering")
ax.set_xlim(0, 30); ax.set_ylim(-0.05, 1.02)
ax.set_xlabel("time between samples (minutes)"); ax.set_ylabel("noise correlation")
ax.set_title("(b) Filtering raises short-lag correlation")
ax.legend(fontsize=8)
ax = axs[2]
keys = list(schemes)
yy = np.arange(len(keys))[::-1]
ax.barh(yy + 0.19, [schemes[k][1] for k in keys], height=0.36, color=S.MUTED, label="formal sd")
ax.barh(yy - 0.19, [schemes[k][0] for k in keys], height=0.36, color=S.INK, label="actual sd (1000 draws)")
for y_, k in zip(yy, keys):
    ok_ = abs(schemes[k][0] / schemes[k][1] - 1) < 0.1
    ax.text(max(schemes[k][0], schemes[k][1]) + 0.004, y_, ("✓ " if ok_ else "✗ ") + f"×{schemes[k][0] / schemes[k][1]:.1f}",
            va="center", fontsize=8.5, color=S.GOOD if ok_ else S.BAD)
ax.set_yticks(yy); ax.set_yticklabels([f"{k}: {names_t[k]}" for k in keys], fontsize=8.5)
ax.set_xlim(0, 0.17); ax.set_ylim(-0.6, len(keys) + 0.2)
ax.set_xlabel("uncertainty of the amplitude")
ax.set_title("(c) Formal vs actual error, five schemes")
ax.legend(fontsize=8, loc="upper right", ncol=2)
fig.tight_layout()
bk.savefig(fig, "2-14_tsunami")
plt.show()

# %% [markdown]
# **How to read it.** (a) The wave packet and one noisy record; the filtered record (blue) has lost the fast jitter
# but keeps the slow drift, which is the part that resembles the wave. (b) The correlation between noise samples a given
# time apart: filtering pushes the short-lag correlation even closer to one. (c) The amplitude's actual scatter (black)
# against the error bar each scheme reports (grey), with the ratio actual/formal. With a diagonal $\mathbf C_d$ the error bar
# is too small (A), more so after filtering (C), and more still after interpolation (D), although the actual scatter
# hardly moves: no scheme added information. With the full covariance (B and E) the error bar is right, and keeping only
# one sample per minute (E) costs little precision compared with all 15-s samples (B).
#
# **Worked example.** For pure exponential (AR(1)) noise sampled every Δt, the lag-one correlation is
# $\rho = e^{-\Delta t/\tau}$ and the effective number of samples for estimating a mean is approximately
# $N_\text{eff}\approx N(1-\rho)/(1+\rho)$. The steps: (1) ρ for Δt = 15 s, τ = 8 min; (2) the approximation for N = 720;
# (3) the exact $N_\text{eff} = s^2\,\mathbf 1^\top\mathbf C_d^{-1}\mathbf 1$; (4) the same with the 0.4 cm white noise added, where the
# approximation uses the observed lag-one correlation.

# %%
rho = np.exp(-DT_T / TAU)
C_ar = S_RED ** 2 * np.exp(-np.abs(t_t[:, None] - t_t[None, :]) / TAU)
neff_ar = S_RED ** 2 * np.ones(n_t) @ np.linalg.solve(C_ar, np.ones(n_t))
print(f"step 1  rho = exp(-{DT_T:.2f} / {TAU:.0f}) = {rho:.4f}")
print(f"step 2  N (1 - rho) / (1 + rho) = {n_t} x {(1 - rho) / (1 + rho):.4f} = {n_t * (1 - rho) / (1 + rho):.1f}")
print(f"step 3  exact N_eff for pure AR(1) noise = {neff_ar:.1f}")
rho_w = C_t[0, 1] / C_t[0, 0]
neff_w = C_t[0, 0] * np.ones(n_t) @ np.linalg.solve(C_t, np.ones(n_t))
print(f"step 4  with white noise added: lag-one correlation {rho_w:.4f}, approximation {n_t * (1 - rho_w) / (1 + rho_w):.1f}, "
      f"exact {neff_w:.1f}")

# %% [markdown]
# For pure AR(1) noise the approximation and the exact value agree closely (steps 2 and 3). Add a little white noise and
# the lag-one correlation drops, so the approximation claims many more independent samples, while the exact count hardly
# changes (step 4): the slow drift still links samples minutes apart. The lag-one correlation alone does not determine
# $N_\text{eff}$; the whole covariance does.
#
# **Common misreadings.**
# * "Filtering removes noise, so the filtered data deserve a smaller σ." It removes the fast part of the noise, which
#   was never the part that limited the estimate. The slow part, which competes with the signal, is untouched.
# * "Resampling to 1 s gives more data." It gives more numbers, all interpolated from the same measurements.

# %% [markdown]
# ## 5. The effective number of independent data
#
# **In plain words.** The question to ask of any data set is not "how many points?" but "how many independent points
# would tell me as much?". For independent errors the two numbers are equal. For correlated errors, adding points
# closer together than the correlation length or time adds numbers but little information, and the effective number
# levels off.
#
# **Precisely.** For a quantity $\theta$ estimated linearly from the data, define
#
# $$N_\text{eff}(\theta) = N\;\frac{\mathrm{Var}_\text{indep}(\hat\theta)}{\mathrm{Var}_\text{true}(\hat\theta)},$$
#
# where $\mathrm{Var}_\text{indep}$ is the GLS variance computed as if the $N$ errors were independent with the same
# variances (diagonal of $\mathbf C_d$) and $\mathrm{Var}_\text{true}$ the GLS variance with the full $\mathbf C_d$. It is the
# number of independent data of the same quality that would determine $\theta$ as well. For a common offset with equal
# variances $s^2$ it reduces to $N_\text{eff} = s^2\,\mathbf 1^\top\mathbf C_d^{-1}\mathbf 1$, the formula used in the worked examples.
# Below, each data set is sampled more and more densely over the same span or area: the GNSS rate (4 years, one sample
# every 60 days down to every day), the deep-source volume change from InSAR alone (48 km square, pixel spacing 12 km down to 0.75 km),
# and the tsunami amplitude (3 hours, one sample every 8 minutes down to every 15 s).

# %%
def neff(A, C, j):
    v_ind = np.linalg.inv(A.T @ (A / np.diag(C)[:, None]))[j, j]
    v_true = np.linalg.inv(A.T @ np.linalg.solve(C, A))[j, j]
    return A.shape[0] * v_ind / v_true


ne = {"GNSS rate (4 yr)": [], "InSAR deep source (48 km)": [], "tsunami amplitude (3 h)": []}
for k in [60, 30, 14, 7, 4, 2, 1]:
    sl = slice(0, N_G, k)
    ne["GNSS rate (4 yr)"].append((A_g[sl].shape[0], neff(A_g[sl], C_g[sl, sl], 1)))
for k in [16, 12, 8, 6, 4, 3, 2, 1]:
    msk = np.zeros((NG, NG), bool)
    msk[k // 2::k, k // 2::k] = True
    ii = np.flatnonzero(msk.ravel())
    ne["InSAR deep source (48 km)"].append((ii.size, neff(G_pix[ii], C_i[np.ix_(ii, ii)], 1)))
for k in [32, 16, 8, 4, 2, 1]:
    sl = slice(0, n_t, k)
    ne["tsunami amplitude (3 h)"].append((wave[sl].size, neff(wave[sl, None], C_t[sl, sl], 0)))

fig, ax = plt.subplots(figsize=(7.4, 4.6))
allN = [n_ for v in ne.values() for n_, _ in v]
lim = np.array([min(allN) * 0.7, max(allN) * 1.4])
ax.loglog(lim, lim, color=S.AXIS, lw=1.0, ls=(0, (4, 2)), label="independent errors: $N_{eff} = N$")
for (k, v), col in zip(ne.items(), [C_W, C_F, C_R]):
    v = np.array(v)
    ax.loglog(v[:, 0], v[:, 1], "o-", color=col, ms=4.5, lw=1.4, label=k)
ax.set_xlabel("number of data N (denser sampling →)"); ax.set_ylabel("effective number of independent data $N_{eff}$")
ax.set_title("More samples, little more information")
ax.legend(fontsize=8.2, loc="upper left")
fig.tight_layout()
bk.savefig(fig, "2-14_neff")
plt.show()
for k, v in ne.items():
    print(f"{k:28s}: " + ", ".join(f"N={n_} -> {e_:.1f}" for n_, e_ in v))

# %% [markdown]
# **How to read it.** Each line follows one data set as its sampling gets denser (left to right). Once the spacing drops
# below the correlation length or time, $N_\text{eff}$ flattens out while $N$ keeps growing (numbers printed). The flat
# level depends on the span or area relative to the correlation scale, and on how much the signal of interest looks
# like the noise. The GNSS rate competes with flicker and random-walk noise that is correlated over the whole four
# years, so it sits below the dashed line even at the sparsest sampling and levels off lowest. The deep source's broad
# bowl competes with the kilometre-scale atmosphere. The tsunami amplitude at the sparsest sampling lies *above* the
# dashed line: the noise varies slowly and the wave oscillates faster, so GLS can use the correlation to predict and
# subtract part of the noise, and correlated samples are then worth more than independent samples of the same variance.
# Correlation hurts when the signal looks like the noise and can help when it does not.
#
# **Common misreadings.**
# * "$N_\text{eff}$ is a property of the data set." It depends on the quantity being estimated: a sharp, local or
#   oscillating signal can be told apart from smooth noise much better than a broad one.

# %% [markdown]
# ## 6. One noise scale per data set vs a full covariance
#
# **In plain words.** A common shortcut, which 2.30 studies in a hierarchical setting, is to keep each data set's
# $\mathbf C_d$ diagonal and
# multiply it by an unknown factor $\sigma_k^2$ estimated from the misfit. That scale fixes the overall size of a data
# set's errors. It cannot fix their correlation: a correlated error field has the right variance pixel by pixel, so the
# residuals do not ask for a larger $\sigma_k$.
#
# **Precisely.** In section 3.4, the aqua scheme iterates $\sigma_k^2 = \chi^2_k/N_k$ for InSAR and GNSS separately
# (a variance-component estimate; the hierarchical version samples $\sigma_k$ instead, 2.30). We compare the fitted
# $\sigma_k^2$ with the factor by which each parameter's diagonal-$\mathbf C_d$ error bar would have to be inflated to match
# its actual scatter.

# %%
for name in ["uniform", "quadtree"]:
    ks = MC[(name, "sigma_k")]["ks"]
    rd = MC[(name, "diagonal")]
    need = rd["actual"] / rd["formal"]
    print(f"{name:8s}: fitted sigma_k^2 (median over repeats) InSAR {np.median(ks[:, 0]):.2f}, GNSS {np.median(ks[:, 1]):.2f};"
          f"  inflation the diagonal error bars need: shallow x{need[0]:.2f}, deep x{need[1]:.2f}")
    print(f"          actual sd with diagonal C_d {np.round(rd['actual'], 3)}, with sigma_k {np.round(MC[(name, 'sigma_k')]['actual'], 3)}, "
          f"with full C_d {np.round(MC[(name, 'full')]['actual'], 3)}")

# %% [markdown]
# The fitted InSAR $\sigma_k^2$ stays near one or even below it (printed), so the scale does not inflate the error
# bars, and coverage does not recover (aqua bars in the section 3.4 figure). The shortfall also differs between the two
# parameters (printed inflation factors), so it is not a matter of one overall error size. And the estimates themselves stay less precise than with the full covariance (actual sd printed), because the
# *relative* weights of the InSAR points, not only their overall size, are wrong. An estimated weight measures misfit
# (2.13 showed this; 2.30, section 5, shows it again for hierarchical weights), and here the misfit looks normal, so
# the weight cannot react.
#
# **Common misreadings.**
# * "Sampling $\sigma_k$ makes the weighting objective, so the covariance shape does not matter." $\sigma_k$ is one number
#   per data set; the correlation is a matrix. The hierarchical scale is a correction for a wrong size, not for a wrong
#   shape.

# %% [markdown]
# ## 7. Summary
#
# | data | main error structure | how to build $\mathbf C_d$ | what a diagonal $\mathbf C_d$ does |
# |---|---|---|---|
# | GNSS time series | white + flicker (+ random walk) in time | power-law model, amplitudes by maximum likelihood (2.4); GLS covariance of the rates | rate errors too small by a large factor; random walk is hard to estimate |
# | InSAR | atmosphere correlated over kilometres | empirical covariogram → exponential (sill, range, nugget); $\mathbf A\mathbf C\mathbf A^\top$ after decimation | InSAR over-weighted vs GNSS, especially for broad signals; coverage collapses |
# | tsunami gauge | slow drift over minutes; filtering adds correlation | exponential in time; $\mathbf F\mathbf C\mathbf F^\top$ for any filter or resampling | error bars shrink with every filter or resampling step |
# | any | | $N_\text{eff}$ shows how much the points are worth | |
# | several data sets | | $\sigma_k$ fixes the size, the full $\mathbf C_d$ the shape | a scale cannot fix a shape |
#
# The image to keep: **counting points is not counting information.** A diagonal $\mathbf C_d$ counts points.
#
# **In practice.** Many joint inversions estimate one noise scale $\sigma_k$ per data set and keep each data set's
# covariance diagonal. Section 6 shows what such a scale can and cannot do: it adapts each data set's overall error
# size, but with a diagonal covariance the data set's weight relative to the others still follows its number of points
# and its sampling. A cheap partial remedy is to down-weight a densely sampled data set by its effective number of
# data, $N_\text{eff}/N$ (section 5), as one weight. The full remedy is sections 3 and 4: a covariogram per
# interferogram and $\mathbf A\mathbf C\mathbf A^\top$ through the decimation, and an exponential time covariance per gauge
# propagated through the same filter that is applied to the records and to the Green's functions.
#
# Next: **2.15 · The map: five methods, six posterior shapes**.

# %%
print(f"notebook run time: {time.time() - T0:.0f} s")
