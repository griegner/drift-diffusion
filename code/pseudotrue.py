"""large-sample references for fig04 and fig08"""

# approximate each estimator's large-sample limit: one 4p MLE fit to an MLE_TRIALS-trial sample (fig04, fig08);
# the mean of valid 7p MCMC fits to samples of the largest of MCMC_TRIALS (fig08; smaller sizes check convergence)

from pathlib import Path

from sklearn.base import clone

MLE_TRIALS, MLE_FITS = 10**6, 1
MCMC_TRIALS, MCMC_FITS = (25_000, 50_000), 50  # sizes; reference reps REP..REP+MCMC_FITS-1
REP = 10**6  # reserved reference-sample IDs: REP, REP + 1, ...
# MLE_TRIALS-trial samples ({fig}-n{n}-rep{rep}.npy) are read from here rather than simulated: sample_from_pdf would
# need ~160 GB at once (they were drawn in 10k-trial chunks with draws identical to one call)
SAMPLES = Path(__file__).resolve().parents[1] / "data/pseudotrue-samples"


def pseudotrue_4p(ddm, X, y):
    """fit 4p coefficients to a large simulated sample"""

    # skip score-based covariances; only coefficients are needed
    return clone(ddm).set_params(cov_estimator="sample-hessian").fit(X, y).params_
