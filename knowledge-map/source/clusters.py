"""One plain-words summary per section of Topic 2 (the counterpart of os-taxonomy's clusters)."""

CLUSTERS = {
    "A": "How to describe what you do not know: random variables and distributions, conditioning and Bayes' rule, why "
         "averages settle and errors look Gaussian, and the matrix tools (covariance, Cholesky, SVD) every inversion uses.",
    "B": "Turning data into a model when the data are not enough: least squares and its limits, Bayesian linear inversion "
         "and the MAP, regularisation as a prior, loss functions and optimisers, choosing a prior, and Gaussian-process "
         "priors on whole fields.",
    "C": "Where errors come from and how to weight data: measurement noise (C_d) versus model error (C_p), and how the "
         "noise in GNSS, InSAR and tsunami records becomes a data covariance.",
    "D": "Computing a posterior that has no formula: Metropolis, gradients and JAX, HMC and NUTS, convergence diagnostics "
         "and Monte Carlo error, tempering and SMC, variational inference, and gradients through physics.",
    "E": "Inversion through time and with ensembles: the Kalman filter, ensemble Kalman methods (EnKF, EKI, EnKI), "
         "measuring nonlinearity, the extended Kalman filter, and localization for small ensembles.",
    "F": "Choosing the knobs and comparing models: entropy and KL divergence, criteria for hyperparameters (L-curve, GCV, "
         "ABIC, AIC, BIC), hierarchical Bayes and data weights, and what objectivity can and cannot mean.",
    "G": "Using and checking the answer: derived quantities, intervals, joint statements, predictive checks, "
         "simulation-based calibration, model comparison by prediction, and one inversion with every choice side by side.",
}
