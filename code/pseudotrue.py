"""analytic pseudo-true parameters for fig02-04.

sample_from_pdf draws each trial from a discrete pmf on a fixed RT grid, so the expected working
log-likelihood is an exact weighted sum over that grid. Its maximizer is the pseudo-true parameter
of the simulated process, free of Monte Carlo error.
"""

import autograd.numpy as anp
import numpy as np
from autograd import grad
from scipy.optimize import minimize

from drift_diffusion.model import pdf


def sampler_grid(t0s):
    """signed RT support used by sample_from_pdf for trials with nondecision times t0s."""

    grid = np.linspace(1e-5 + np.min(t0s), 5 + np.max(t0s), 1000)
    return np.concatenate([-grid, grid])


def pmf(y, a, t0, v, z):
    """sample_from_pdf's per-trial probabilities on grid y."""

    p = pdf(y, a, t0, v, z)
    return p / p.sum()


def pseudotrue(y, weights, f_theta, x0):
    """minimize -sum(weights * log f_theta(theta, y)) over theta, starting from x0."""

    def loss(theta):
        return -anp.sum(weights * anp.log(anp.maximum(f_theta(theta, y), 1e-300)))

    fit = minimize(loss, np.asarray(x0, dtype=float), jac=grad(loss), method="BFGS", options=dict(gtol=1e-10))
    return fit.x
