"""large-sample MLE reference for fig04 and fig08"""

# approximate the expected-log-likelihood optimum by averaging fits to N_FITS large simulated samples

from sklearn.base import clone

N_TRIALS = 50_000
REP = 10**6  # reserved reference-sample IDs: REP, REP + 1, ..., REP + N_FITS - 1
N_FITS = 5


def pseudotrue_4p(ddm, X, y):
    """fit 4p coefficients to a large simulated sample"""

    # skip score-based covariances; only coefficients are needed
    return clone(ddm).set_params(cov_estimator="sample-hessian").fit(X, y).params_
