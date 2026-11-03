"""Large-sample MLE reference for the 4p models in fig04 and fig08."""

# the pseudo-true parameter maximizes the working model's expected log-likelihood under the simulated process;
# estimate it by maximum likelihood on one large simulated sample of N_TRIALS, whose Monte Carlo error is about
# SE(n) / sqrt(N_TRIALS / n), ~0.14 SE at n=1000

from sklearn.base import clone

N_TRIALS = 50_000
REP = 10**6  # repeat index of the pseudo-true sample, beyond any simulation repeat


def pseudotrue_4p(ddm, X, y):
    """return the 4p-DDM coefficients of ddm fit to the pseudo-true sample (X, y)"""

    # sample-Hessian covariance is cheapest; only the coefficients are used
    return clone(ddm).set_params(cov_estimator="sample-hessian").fit(X, y).params_
