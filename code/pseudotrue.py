"""large-sample MLE reference for fig04 and fig08"""

# approximate the expected-log-likelihood optimum with one large simulated sample
# reference Monte Carlo error: SE(n) * sqrt(n / N_TRIALS), about 0.14 SE at n=1000

from sklearn.base import clone

N_TRIALS = 50_000
REP = 10**6  # reserved reference-sample ID


def pseudotrue_4p(ddm, X, y):
    """fit 4p coefficients to a large simulated sample"""

    # skip score-based covariances; only coefficients are needed
    return clone(ddm).set_params(cov_estimator="sample-hessian").fit(X, y).params_
