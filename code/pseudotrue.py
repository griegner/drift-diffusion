"""large-sample references for fig04 and fig08"""

# approximate each estimator's large-sample limit: one 4p MLE fit to an MLE_TRIALS-trial sample (fig04, fig08);
# the mean of valid 7p MCMC fits to MCMC_TARGET-trial samples (fig08; other MCMC_TRIALS sizes check convergence)

from pathlib import Path

from sklearn.base import clone

MLE_TRIALS, MLE_FITS = 10**6, 1
MCMC_TRIALS, MCMC_FITS = (25_000, 50_000, 75_000, 100_000), 50  # sizes; reference reps REP..REP+MCMC_FITS-1
MCMC_TARGET = 50_000
REP = 10**6  # reserved reference-sample IDs: REP, REP + 1, ...
# saved samples ({fig}-n{n}-rep{rep}.npy) are read from here rather than simulated: sample_from_pdf needs ~0.18 GB per
# 1000 trials at once (~160 GB at MLE_TRIALS); they were drawn in chunks with draws identical to one call
SAMPLES = Path(__file__).resolve().parents[1] / "data/pseudotrue-samples"


def pseudotrue_4p(ddm, X, y):
    """fit 4p coefficients to a large simulated sample"""

    # skip score-based covariances; only coefficients are needed
    return clone(ddm).set_params(cov_estimator="sample-hessian").fit(X, y).params_
