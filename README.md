# Empiricist Geophysics

**Data, Models, and Bayesian Inference.** Geophysical inverse problems, by picture and by computation.

These are the notebooks I keep to refresh my own thinking and to build my own map of how data, models and inference
fit together in geophysics. I share them as an open course for readers who are new to inversion, to Bayesian
statistics, or to the forward models of geophysics (earthquake science). Every idea is shown on a small problem whose correct answer is
known, every method is actually run, and every figure comes from that run. The notebooks explain each concept twice: first in plain words, then precisely
(definition, assumptions, formula), followed by a worked example whose numbers the code prints.

The course has four topics. Topic 2 is complete, Topic 3 is under development, and Topics 1 and 4 are planned.

| topic | what it covers | status |
|---|---|---|
| **1 · Geophysical forward problems and numerical methods** | dislocations, FD/FE/BE methods, earthquake-cycle and postseismic processes, tsunamis, seismic waves | planned |
| **2 · Inverse problems and Bayesian inversion** | from least squares to hierarchical Bayes, samplers and ensemble methods | 33 notebooks |
| **3 · Forward surrogate models** | cheap stand-ins for expensive forward models, and inference with them | under development, 7 notebooks |
| **4 · Data and data processing** | GNSS, InSAR, seafloor and ocean data (DART, S-net, pressure gauges), altimetry, seismic data, and their errors | planned |

`figures/roadmap.png` shows the reading order and what builds on what.

## How to use it

| you want to | do this |
|---|---|
| read without installing anything | open `html/index.html` in a browser (or start at `html/2-01_basics.html` and follow the "Next" links) |
| play with samplers live | open `playground.html` in a browser: pick a posterior shape and a sampler, move the knobs |
| run and change the code | install (below), then `jupyter lab notebooks/` |
| see one figure | `figures/<id>_<name>.png` (300 dpi); animations are `.gif` |

### Install and check

```bash
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python build/smoke_test.py      # versions, one exact posterior against NUTS, one ODE solve: a few seconds
python build/make_data.py       # optional: pre-builds the cached training runs used by 2.10, 2.22 and Topic 3
```

Everything runs on a CPU; no GPU is needed. `requirements.txt` pins the versions the course was built with
(Python 3.11.7, JAX 0.10.0, NumPyro 0.21.0, diffrax 0.7.2). The notebooks are generated from the sources in
`build/src/` by `python build/build.py` (all) or `python build/build.py 2-07` (one). If you edit a notebook
directly instead, do not rebuild it, or your edit is overwritten. Run times are listed under
[Reproducibility](#reproducibility).

## Topic 1 · Geophysical forward problems and numerical methods (planned)

How the physics turns a model of the Earth into predicted data. Planned content:

- **Elastic dislocation models**: point sources and finite faults in an elastic half-space (Okada 1985, 1992), triangular dislocations and layered Earth models; how the Green's-function matrix G of a slip inversion is built, and how its errors become the prediction covariance C_p (2.13).
- **Numerical methods**: finite-difference, finite-element and boundary-element methods; meshing a curved fault from a slab model; checking accuracy by mesh refinement.
- **Earthquake cycle and postseismic processes**: interseismic locking (backslip), rate-and-state and rate-strengthening friction, stress-driven afterslip (the forward model used in Topic 3), viscoelastic relaxation in Maxwell and Burgers materials, power-law flow laws, and poroelastic rebound.
- **Tsunami modelling**: from seafloor deformation to the initial sea surface (including the contribution of horizontal motion on sloping seafloor), the shallow-water equations, dispersion (Boussinesq equations), adaptive mesh refinement, and tsunami Green's functions.
- **Seismic waves**: the elastic wave equation, body and surface waves, synthetic seismograms, kinematic rupture models, and back projection of array recordings.

Until then, the forward models used in the course are introduced where they first appear (a screw dislocation
in 2.33, stress-driven afterslip in 2.10 and Topic 3).

## Topic 2 · Inverse problems and Bayesian inversion

Read in order; sections marked *optional* can be skipped on a first pass. Each notebook starts with what it
assumes ("Before this") and what you will be able to do afterwards.

| id | notebook | the question it answers | pictures to keep |
|---|---|---|---|
| | **A · Probability for inverse problems** | | |
| 2.01 | Deterministic physics, uncertain answers | if the physics is deterministic, why probability? what are "the true value", "the posterior" and its "shape"? | `2-01_backwards_one_station.png`, `2-01_crowd_forming.gif`, `2-01_coverage.png` |
| 2.02 | Bayes in pictures | what do prior, likelihood and posterior look like? what are Gaussian, flat and smooth posteriors? | `2-02_bands_2d.png`, `2-02_location_zoo.png` |
| 2.03 | Probability distributions | pdf, cdf, quantiles, heavy tails, χ², covariance and ellipses | `2-03_gallery.png`, `2-03_chisquare.png`, `2-03_multivariate.png` |
| 2.04 | Conditional probability, Bayes' rule | joint, marginal and conditional; why the posterior is a slice | `2-04_joint_marginal_conditional.png`, `2-04_gaussian_conditioning.png` |
| 2.05 | Law of large numbers, central limit theorem | why averages settle, why errors look Gaussian, when √N misleads | `2-05_clt.png`, `2-05_coloured_noise.png` |
| 2.06 | Matrix and notation primer | the matrices of inversion, their sizes and units; solve, don't invert | `2-06_covariance_ellipse.png`, `2-06_cholesky_sampling.png`, `2-06_solve_vs_inverse.png` |
| | **B · Linear inversion, ill-posedness, priors** | | |
| 2.07 | Least squares | normal equations, weighting, χ², SVD, null space, resolution | `2-07_line_fit.png`, `2-07_illposed.png`, `2-07_resolution.png` |
| 2.08 | Bayesian linear inversion and the MAP | prior × likelihood, MAP, the Kalman-gain form, mean vs median vs MAP | `2-08_map_objective.png`, `2-08_laplace.png` |
| 2.09 | Regularisation and smoothing are priors | why ill-posed problems need smoothing; L2, Laplacian, Matérn, L1, TV | `2-09_geometry.png`, `2-09_fault_six.png`, `2-09_l1_map_vs_posterior.png` |
| 2.10 | Loss functions and optimisation | loss = negative log posterior; L2, L1, Huber, Student-t; optimisers; early stopping | `2-10_losses.png`, `2-10_optimiser_race.gif`, `2-10_laplace.png` |
| 2.11 | Choosing a prior | Gaussian, Laplace, Student-t, bounded, log-normal, Dirichlet priors; prior-predictive checks | `2-11_families_1d.png`, `2-11_flat_not_neutral.png`, `2-11_prior_predictive.png` |
| 2.12 | Gaussian-process priors on fields | what L and ν do to a field; GP regression | `2-12_kernels_roughness.png`, `2-12_fault_fields.png` |
| | **C · Errors and data weighting** | | |
| 2.13 | Model errors C_d and C_p | measurement noise vs model error; coverage with and without C_p | `2-13_two_errors.png`, `2-13_coverage.png` |
| 2.14 | Noise models, building C_d | flicker and random-walk noise; spatially correlated noise; effective number of data | `2-14_rate_vs_span.png`, `2-14_insar_weights.png`, `2-14_neff.png` |
| | **D · Computing a posterior: sampling** | | |
| 2.15 | Map of methods | which method works on which posterior shape, at what cost, in these test problems | `2-15_map.png`, `2-15_report_card.png` |
| 2.16 | Metropolis | random-walk sampling, acceptance, step size, ESS | `2-16_rwm_animation.gif`, `2-16_step_size.png` |
| 2.17 | JAX and differentiability | autodiff, "differentiable", gradient-based methods (CPU vs GPU: in preparation) | `2-17_what_differentiable_means.png`, `2-17_nuts_on_terraces.png` |
| 2.18 | Gradients, HMC and NUTS | what "following the gradient of the log posterior" means | `2-18_random_walk_vs_gradient.png`, `2-18_nuts_tree.gif` |
| 2.19 | Convergence, mixing and R-hat | convergence, mixing, R-hat, Monte Carlo error; when R-hat misleads | `2-19_trace_dictionary.png`, `2-19_rhat_one_but_wrong.png` |
| 2.20 | Tempering and SMC | tempering, sequential Monte Carlo, two separated modes | `2-20_tempering_2d.png`, `2-20_smc_bimodal.gif` |
| 2.21 | Variational inference *(optional)* | inference as optimisation; mean field, full rank, flows | `2-21_families.png`, `2-21_nonidentifiable.png` |
| 2.22 | Gradients through physics: adjoints *(optional)* | finite differences, autodiff, adjoints; the adjoint is Gᵀ | `2-22_adjoint_transpose.png`, `2-22_fault_gradients.png` |
| | **E · Time-dependent and ensemble inversion** | | |
| 2.23 | The Kalman filter | sequential Bayes; the gain as a trust ratio; process noise | `2-23_gain_trust_ratio.png`, `2-23_process_noise.png` |
| 2.24 | Ensemble Kalman methods: EnKF, EKI, EnKI | how the three ensemble methods differ | `2-24_eki_collapse.png`, `2-24_enki_linear.png`, `2-24_subspace.png` |
| 2.25 | Nonlinearity and the extended Kalman filter | how nonlinear is a problem, as a number? how do EKF, EnKI and samplers cope? | `2-25_knob.png`, `2-25_degradation.png` |
| 2.26 | Localized EnKI *(optional)* | small ensembles, many unknowns: tapering, inflation, the two-seed test | `2-26_sampling_error.png`, `2-26_seed_test.png` |
| | **F · Hyperparameters and weights** | | |
| 2.27 | Entropy and information | information in bits; what each parameter learns | `2-27_fault_information.png`, `2-27_station_design.png` |
| 2.28 | KL divergence | distance between distributions; mode seeking vs mass covering | `2-28_asymmetry.png`, `2-28_information_gain.png` |
| 2.29 | Choosing hyperparameters, comparing models | L-curve, GCV, ABIC, AIC, BIC; evidence and prior width | `2-29_lambda_axis.png`, `2-29_evidence_prior_width.png` |
| 2.30 | Hierarchical Bayes: who sets the weights? | ABIC vs fully Bayesian; positivity; data weights absorb model error | `2-30_abic_vs_full.png`, `2-30_weights_absorb_misfit.png` |
| 2.31 | To smooth or not | regularised vs unregularised Bayesian slip; objectivity | `2-31_methods_one_case.png`, `2-31_model_error.png` |
| | **G · Checking and synthesis** | | |
| 2.32 | Using and checking a posterior | derived quantities, joint bands, predictive checks, simulation-based calibration, LOO | `2-32_derived_quantities.png`, `2-32_marginal_joint.png`, `2-32_sbc.png` |
| 2.33 | Capstone: one fault-slip inversion, every choice | all of the above on one antiplane slip inversion | `2-33_priors.png`, `2-33_hierarchical.png` |

## Topic 3 · Forward surrogate models (under development)

What to do when the forward model is too expensive to run as often as a sampler needs. All notebooks use one shared
forward model, stress-driven afterslip ($v = 2v_0\sinh(\tau/a\sigma)$, $\dot\tau = Kv$; `bayeskit/afterslip.py`),
on a single fault patch (3 unknowns) and on a 1-D fault (31 unknowns), so the methods can be compared directly.
The rule throughout: surrogate error is model error, so each surrogate posterior is graded against the true model.
Planned additions: convolutional, recurrent and graph neural networks.

| id | notebook | the question it answers | pictures to keep |
|---|---|---|---|
| 3.01 | Surrogates 101 | design, output compression, which regressor; is surrogate error model error? | `3-01_learning_curves.png`, `3-01_coverage.png` |
| 3.02 | Neural networks and ResNets from scratch | neurons, activation functions, backpropagation, overfitting, ResNets, ensembles | `3-02_neurons.png`, `3-02_resnet_ode.png`, `3-02_ensembles.png` |
| 3.03 | GP surrogates and calibrate-emulate-sample | where to put the runs; are the GP's error bars right? | `3-03_designs.png`, `3-03_emulator_variance.png` |
| 3.04 | Physics-informed neural networks | PINN forward and inverse; time coordinate, stiffness, loss weights | `3-04_time_coordinate.png`, `3-04_inverse.png`, `3-04_loss_weights.png` |
| 3.05 | Neural operators (FNO) | learning a map between functions; resolution transfer; frequency bias | `3-05_learning_curves.png`, `3-05_resolution.png`, `3-05_coverage.png` |
| 3.06 | Reduced-order models and real-time inference | POD, Galerkin and data-driven ROMs; offline-online inference | `3-06_pod.png`, `3-06_galerkin_rom.png`, `3-06_realtime.png` |
| 3.07 | Inference with surrogates | surrogate posteriors, delayed acceptance, amortised inference; which route when | `3-07_coverage.png`, `3-07_delayed_acceptance.png`, `3-07_ledger.png` |

## Topic 4 · Data and data processing (planned)

The observations and their errors: what each data type measures, how it is processed, and how its noise becomes
the data covariance. Planned content:

- **GNSS**: daily positions and velocities, high-rate GNSS displacement waveforms, reference frames, common-mode error, offsets, coloured noise (white, flicker and random walk; 2.14), and time-series models of postseismic motion.
- **InSAR**: interferograms, phase unwrapping, atmospheric and orbital errors, decimation, and time-series methods; Sentinel-1, ALOS-2 and NISAR.
- **Seafloor and ocean observations**: DART buoys; cabled seafloor networks such as S-net along the Japan Trench; ocean-bottom pressure and differential pressure gauges (DPG); GNSS-acoustic seafloor positioning; coastal tide gauges; and removing tides, instrument drift and non-tectonic signals.
- **Satellite altimetry**: SWOT wide-swath sea-surface height and conventional nadir altimeters, as tsunami observations.
- **Seismic data**: teleseismic body and surface waves, strong-motion accelerograms, instrument response, filtering and alignment.
- **From data to C_d**: noise models, correlated errors and data weighting for each data type (2.14, 2.30).

Until then, 2.14 covers how noise in such data becomes C_d.

## The words, in one or two lines each

- **Posterior** = prior × likelihood, renormalised: every model, weighted by how well it fits the data and the prior
  (2.02). Picture it as a crowd of possible Earths, each a complete deterministic model that fits the data within the
  noise (2.01).
- **True value** (★) vs **reference posterior** (grey): the true value is the parameter that generated a synthetic
  test's data (unknown in real life); the reference posterior is the correct answer for those data under the stated
  model, computed by brute force on a grid, and is what the samplers are graded against. With noise the ★ is usually
  near the posterior but not at its centre, and it can lie outside a 90 % region.
- **Credible region**: a region holding a stated posterior probability, say 0.90, given the data and the model. It can
  miss the true value. Averaged over parameters drawn from the prior and new data, 90 % regions contain the true value
  90 % of the time; at one fixed true value the share can be very different (2.01, 2.32).
- **Gaussian posterior**: elliptical contours, fully described by a mean and a covariance. You get one exactly when the
  forward model is linear and prior and noise are Gaussian; then the Kalman formula gives it with no sampling (2.02,
  2.08).
- **Flat**: the density does not change along some direction. A *flat prior* has no preference; a *flat direction* is
  one the data cannot see. Both together make an improper posterior that no sampler can converge on (2.02, 2.19).
- **Smooth** (a solution): neighbouring parameters are correlated by the prior (Laplacian smoothing or a GP) (2.09,
  2.12). **Smooth** (a log posterior): differentiable, so gradients exist and NUTS can use them (2.17).
- **Metropolis**: propose a random jump, accept it with probability min(1, p(new)/p(old)) for a symmetric proposal,
  otherwise stay and count the old point again (2.16).
- **Gradient information**: at every point you know which way is uphill in log p and how steep. **HMC** turns that
  into motion: a frictionless puck kicked in a random direction slides along the valleys of −log p. **NUTS** is HMC
  that stops each slide when it starts to come back (a U-turn), so you do not tune the path length (2.18).
- **Convergence**: the chain has forgotten where it started and samples the target. **Mixing**: how fast it moves
  once there (autocorrelation; the **ESS** is the number of independent draws the chain is worth). **R-hat** compares
  spread between chains with spread within chains; below about 1.01 the chains agree, which is not the same as being
  right (2.19).
- **Tempering**: raise the likelihood to a power β from 0 (prior) to 1 (posterior). **SMC**: a crowd of particles
  carried through that sequence by reweighting, resampling and a few Metropolis moves per step (2.20).
- **EnKF**: ensemble Kalman *filter*, for a state that evolves in time. **EKI**: ensemble Kalman *inversion*, the same
  update applied repeatedly with the full data, which behaves like an optimiser. **EnKI**: the tempered version, which
  assimilates the data in fractions that add up to 1; for linear-Gaussian problems it reproduces the posterior in the
  limit of a large ensemble, and otherwise gives a moment-based approximation (2.24).
- **EKF** (extended Kalman filter): the Kalman update applied to the forward model's tangent (its Jacobian). Its
  covariance is the Gauss-Newton approximation, which differs from the full Laplace covariance when residuals and
  curvature are both large (2.25).
- **Degree of nonlinearity**: the RMS of what the forward model's tangent misses over the region you are uncertain
  about, in units of the data noise (2.25).
- **GP (Gaussian process)**: a Gaussian prior on a whole function; its kernel decides how rough the function is (2.12).
  A GP can also be fitted to forward-model runs as a surrogate (3.03).
- **JAX**: NumPy with function transformations: `grad` (automatic differentiation), `jit` (compilation), `vmap`
  (vectorisation) (2.17).
- **Differentiable / gradient-based**: a log posterior is differentiable when its slope exists everywhere.
  Gradient-based methods (NUTS, HMC, L-BFGS) need that; derivative-free ones (Metropolis, SMC, EnKI) need only
  function values (2.17).

## What the experiments showed

Each of these is a result of the experiments in the notebook named, with its numbers printed there. They are
examples of what can happen, not general laws about the methods.

- **A 90 % credible region is not a 90 % guarantee for one fixed truth.** With prior N(0, 1) and one datum of
  unit noise, the central 90 % interval contained a fixed true value 0.980 of the time at θ = 0 and 0.250 at θ = 3;
  averaged over true values drawn from the prior it was 0.900 (2.01).
- **How you compute matters as much as what you compute.** On one ill-conditioned least-squares problem the explicit
  inverse of GᵀC_d⁻¹G recovered the coefficients with relative error 0.34; whitening followed by an SVD solve gave
  1.1e-9 (2.07).
- **A vague prior can be physically absurd.** A prior of N(0, 10²) m on every slip patch gave a negative total slip in
  0.498 of prior draws; checking what the prior predicts before fitting exposed it (2.11).
- **White-noise error bars on GNSS rates can be an order of magnitude too small.** With flicker and random-walk noise
  in 4-year series, the white-noise formula gave a rate uncertainty 11.4 times smaller than the actual scatter (2.14).
- **A diagonal C_d over-weights spatially correlated data, and one scale factor cannot repair it.** For two source
  parameters, 90 % coverage was 0.517 with a diagonal C_d, 0.507 after fitting a σ scale, and 0.895 with the full C_d
  (uniform decimation; 0.340, 0.275 and 0.877 with quadtree decimation) (2.14).
- **A non-differentiable forward model defeats NUTS, not the posterior.** Rounding the predictions barely moved the
  posterior mean, but NUTS used 1,021 leapfrog steps per draw on average (the maximum is 1,023) and reached an ESS of
  about 50, while Metropolis on the same rounded model reached about 430 to 500 (2.17).
- **R-hat measures agreement between chains, not correctness.** Four chains that all stayed in one of two modes gave
  R-hat 1.002 and 1.003 and put probability 1.000 on the northern mode, whose exact probability is 0.686 (2.19).
- **Variational inference can be confidently wrong without warning.** Where the data left a ridge between two
  parameters, the variational families tested covered none of it, while NUTS covered 0.86 of it and flagged trouble
  with 139 divergent transitions (2.21).
- **A gradient can be NaN while the forward run is fine.** In the stiff afterslip solver, trial steps that the solver
  later rejects overflow, and reverse-mode differentiation turned all 40 gradient entries into NaN; clipping the
  argument fixed it without changing the forward run (2.22).
- **Kalman-type methods degrade as the forward model becomes nonlinear; the samplers did not.** On the one-station
  test, the share of EKF draws inside the exact posterior's 90 % region fell from 0.89 at nonlinearity index 0.08 to
  0.36 at 5.94, and EnKI's to 0.17, while NUTS, SMC and Metropolis stayed between 0.88 and 0.92 (2.25).
- **The Gauss-Newton covariance of the iterated EKF is not the Laplace covariance.** At the same peak their long-axis
  variances were 1.000 and 0.880 (2.25).
- **With fewer ensemble members than unknowns, run a second seed.** Over 98 pairs of runs, the gap between two seeds
  (computable without knowing the answer) correlated 0.90 with the actual error (2.26).
- **Centered or non-centered hierarchical parameters: test both.** With precise data (3 mm noise) the non-centered
  form had about 400 divergent transitions per run and a smoothing-scale posterior too narrow (sd 0.37 against an
  exact 0.47); the centered form had none. With 100 times noisier data both forms produced divergences and the
  centered one mixed worse (ESS 239 to 332 against 928 to 1,517) (2.33).
- **Per-patch error bars are not a band for the whole profile.** Only 0.239 of sampled slip profiles stayed inside
  the per-patch 90 % band; a band holding 90 % of whole profiles needed a half-width of 2.78 sd instead of 1.645 (2.32).
- **Overlapping error bars do not mean "no difference".** Two independent estimates, 0 and 3 with standard error 1,
  have overlapping 95 % intervals, yet their difference has p = 0.0339 (2.32).
- **Surrogate error is model error.** Posteriors from a GP surrogate trained on 200 runs contained the true value in
  0.033 of repeats at the 90 % level; adding the surrogate's validation error as C_p raised that to 0.787. With 3,200
  runs: 0.627 without and 0.907 with (3.01).
- **A surrogate's gradients are worse than its predictions.** A neural operator trained on 1,200 runs predicted the
  data with median relative error 0.0039 but its Jacobian with 0.128 (3.05).
- **Check a physics-informed network by re-solving the equation.** A PINN whose residual was written in other units
  fitted the data (χ²/N 0.81) at aσ = 5.95 MPa when the true value was 0.5; re-solving at those parameters gave
  χ²/N 1,084 (3.04).
- **Delayed acceptance gives the exact posterior for fewer expensive runs.** Screening Metropolis proposals with a
  surrogate trained on 400 runs raised the effective samples per 1,000 true-model runs from 103 to 339 counting only
  the sampling runs, or to 273 when the 1,200 runs spent building the surrogate are included (3.07).

## Reproducibility

All 40 notebooks were rebuilt in one sequential pass on an Apple M3 Pro (11 CPU cores, 18 GB memory, no GPU) with
the versions in `requirements.txt`: 2,416 s (40 minutes) in total. Most notebooks run in under a minute; the longer
ones are 2.18 (135 s), 2.22 (180 s), 2.30 (130 s), 2.31 (172 s), 3.01 (122 s), 3.02 (146 s), 3.03 (178 s),
3.04 (182 s) and 3.05 (267 s). Random seeds are fixed, so a rebuild reproduces the printed numbers; timings, and
results that depend on parallel reductions, can differ slightly on other machines and library versions.

To rebuild everything: `./build/rebuild_all.sh` (sequential, resumable, log in `build/rebuild_all.log`), then
`python build/link_pages.py` to refresh the links between pages and `html/index.html`.

## What is in the folder

```
README.md            this page
LICENSE              MIT (code);  LICENSE-CONTENT.md: CC BY 4.0 (text and figures)
CITATION.cff         how to cite
requirements.txt     pinned Python packages
playground.html      interactive sampler playground (no install)
notebooks/           executed Jupyter notebooks, <topic>-<number>_<name>.ipynb
html/                the same notebooks as stand-alone web pages; index.html is the course index
figures/             every figure (PNG, 300 dpi) and animation (GIF); roadmap.png is the reading map
bayeskit/            the shared toolkit: problems.py (test problems, grid reference, closed forms), samplers.py
                     (Metropolis, HMC, NUTS, SMC, ensemble Kalman, EKF), diagnostics.py (R-hat, ESS,
                     nonlinearity indices), afterslip.py (the Topic 3 forward model), plotting.py, style.py
build/               src/*.py are the notebook sources; build.py builds them; link_pages.py links the pages and
                     writes html/index.html; roadmap.py draws the map; smoke_test.py and make_data.py check
                     the setup and pre-build the caches
data/                cached forward-model runs (created on first use or by build/make_data.py)
```

The algorithms in `bayeskit` are teaching versions, checked against exact answers on small problems. For research
use, prefer maintained libraries such as NumPyro (NUTS, variational inference).

## License and citation

Code: MIT (`LICENSE`). Text and figures: CC BY 4.0 (`LICENSE-CONTENT.md`). Short quotations from published papers
are reproduced for comment and teaching and remain under their publishers' terms. To cite, see `CITATION.cff`.
