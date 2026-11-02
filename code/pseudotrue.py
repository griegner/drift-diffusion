"""Monte Carlo pseudo-true parameters for fig04 and fig08"""

# the pseudo-true parameter maximizes the working model's expected log-likelihood under the simulated process;
# estimate it by maximum likelihood on one large simulated sample of N_TRIALS, whose Monte Carlo error is about
# SE(n) / sqrt(N_TRIALS / n), ~0.14 SE at n=1000

import numpy as np
from hddm_wfpt import wfpt
from scipy.optimize import minimize
from sklearn.base import clone

N_TRIALS = 50_000
REP = 10**6  # repeat index of the pseudo-true sample, beyond any simulation repeat


def pseudotrue_4p(ddm, X, y):
    """return the 4p-DDM coefficients of ddm fit to the pseudo-true sample (X, y)"""

    # sample-Hessian covariance is cheapest; only the coefficients are used
    return clone(ddm).set_params(cov_estimator="sample-hessian").fit(X, y).params_


def pseudotrue_7p(y, stim, q0):
    """return HSSM full_ddm parameters (a, t, v_stim, z, sz, sv, st) maximizing the likelihood of (y, stim)"""

    # same likelihood as ddm7p.logp_full_ddm; Nelder-Mead since full_ddm has no gradients
    one = np.ones(len(y))

    def loss(q):
        a, t, v, z, sz, sv, st = q
        if min(a, sv, sz, st) < 0 or not 0 < z - sz / 2 < z + sz / 2 < 1 or st / 2 >= t:
            return 1e10
        logp = wfpt.wiener_logp_array(
            x=y, v=stim * v, sv=sv * one, a=2 * a * one, z=z * one, sz=sz * one, t=t * one, st=st * one,
            err=1e-8, simps_err=1e-4,
        )  # fmt: skip
        return -np.sum(np.where(np.isfinite(logp), logp, -66.1)) / len(y)

    options = dict(xatol=1e-5, fatol=1e-9, maxiter=5000, maxfev=5000, adaptive=True)
    return minimize(loss, np.asarray(q0, dtype=float), method="Nelder-Mead", options=options).x
