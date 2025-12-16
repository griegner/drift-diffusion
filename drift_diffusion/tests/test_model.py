import itertools

import numpy as np
import pandas as pd
import pytest
from joblib import Parallel, delayed
from numpy import testing
from pyddm import gddm
from sklearn.exceptions import NotFittedError
from sklearn.utils.estimator_checks import parametrize_with_checks

from drift_diffusion.model import DriftDiffusionModel, mdf, pdf
from drift_diffusion.sim import sample_from_pdf


@pytest.mark.parametrize(
    "a, t0, v, z",
    itertools.product(
        [1, 2, 3],  # a
        [0, 0.2],  # t0
        [-2, 0, +2],  # v
        [-0.5, 0, +0.5],  # z
    ),
)
def test_pdf_vs_pyddm(a, t0, v, z):
    """test density formula against Anderson and Crank-Nicolson from pyDDM"""
    ddm = gddm(drift=v, bound=a, nondecision=0, starting_position=z, mixture_coef=0)

    # anderson: ddm -> pdf
    and_sol = ddm.solve_analytical()

    # crank-nicolson: ddm -> pdf
    cn_sol = ddm.solve_numerical_cn()

    # navarro: ddm -> pdf
    y = and_sol.t_domain[1:] + t0
    pdf_upr = pdf(y, a, t0, v, z)
    pdf_lwr = pdf(-y, a, t0, v, z)

    # test the densities are close elementwise
    testing.assert_allclose(pdf_upr, and_sol.pdf(choice="upper")[1:], atol=0.08)
    testing.assert_allclose(pdf_lwr, and_sol.pdf(choice="lower")[1:], atol=0.08)
    testing.assert_allclose(pdf_upr, cn_sol.pdf(choice="upper")[1:], atol=0.08)
    testing.assert_allclose(pdf_lwr, cn_sol.pdf(choice="lower")[1:], atol=0.08)


def test_mdf():
    """test mixture of identical distributions is the same as the individual densities"""
    n_samples = 1000
    rt = np.linspace(0.01, 5, n_samples // 2)
    y = np.r_[-rt, rt]

    pdfs = pdf(y, 2, 0, 1, 0)
    mdfs_same = mdf(y, np.array([2, 2, 2]), 0, np.array([1, 1, 1]), 0)
    mdfs_diff = mdf(y, np.array([1, 2, 3]), 0, np.array([-2, 0, +2]), 0)

    assert mdfs_same.shape == y.shape
    # test the densities are close elementwise
    testing.assert_allclose(mdfs_same, pdfs, atol=7e-7)
    # test the densities are different elementwise
    with pytest.raises(AssertionError):
        testing.assert_allclose(mdfs_diff, pdfs, atol=7e-7)


@pytest.mark.skip(reason="5min runtime")
@parametrize_with_checks([DriftDiffusionModel()], legacy=False)
def test_sklearn_compatible_estimator(estimator, check):
    """test estimator is sklearn compatible"""
    check(estimator)


@pytest.mark.parametrize(
    "a, t0, v, z",
    itertools.product(
        [0.5, 0.75, 1],  # a
        [0.1],  # t0
        [-1, 0, +1],  # v
        [0.1],  # z
    ),
)
def test_drift_diffusion_model(a, t0, v, z):
    """test the MLE returns the expected parameters and standard errors"""
    n_repeats = 50
    ddm = DriftDiffusionModel()

    @delayed
    def _fit(repeat):
        y = sample_from_pdf(a, t0, v, z, random_state=repeat)
        X = pd.DataFrame(np.ones(len(y)))
        ddm.fit(X, y)
        stderr = np.sqrt(np.diag(ddm.covariance_))
        return ddm.params_, stderr

    with Parallel(n_jobs=-2) as parallel:
        fits_ = parallel(_fit(repeat) for repeat in range(n_repeats))
        params_, stderrs_ = map(np.asarray, zip(*fits_))

    # test parameters close to true values
    testing.assert_allclose([a, t0, v, z], params_.mean(axis=0), atol=0.2)

    # test standard errors close to true values
    testing.assert_allclose(params_.std(axis=0), stderrs_.mean(axis=0), atol=0.02)


def test_cov_estimator():
    """test all covariance estimates are similar when correctly specified"""
    a, t0, v, z = 1, 0, 0.1, 0
    y = sample_from_pdf(a, t0, v, z)
    X = pd.DataFrame(np.ones(len(y)))
    ddm = DriftDiffusionModel(cov_estimator="all").fit(X, y)
    covariances_ = list(ddm.covariance_.values())

    # test cov_estimator="all" returns four matrices
    assert len(covariances_) == 4

    # test covariance matrix of expected shape and estimates
    for cov in covariances_:
        assert cov.shape == (4, 4)
        testing.assert_allclose(covariances_[0], cov, atol=0.001)


def test_fitted_pdf():
    """test true density within fitted confidence bands"""
    a, t0, v, z = 1.0, 0, 0.372, 0
    y = sample_from_pdf(a, t0, v, z, random_state=0)
    y_range = np.linspace(y.min(), y.max(), len(y))
    X = pd.DataFrame(np.ones(len(y)))
    ddm = DriftDiffusionModel(t0=t0, z=z)

    # test raises expected error
    with pytest.raises(NotFittedError):
        ddm.pdf(X, y_range)

    # test fitted density within 95% confidence bands
    ddm.fit(X, y)
    f = pdf(y_range, a, t0, v, z)
    f_lwr_, f_upr_ = ddm.pdf(X, y_range)
    testing.assert_array_less(f_lwr_, f + 1e-6)
    testing.assert_array_less(f, f_upr_ + 1e-6)


def test_mle_covariates():
    """test estimation of DDM parameters as linear functions of coherence"""

    n_samples = 10000
    coherence = np.array([0.1, 0.3, 0.5, 0.7])
    X = pd.DataFrame({"coherence": np.repeat(coherence, n_samples // 4)})

    # v as linear function of coherence
    v_s = -0.2 + 0.6 * coherence
    y = np.concat([sample_from_pdf(a=1, v=v, n_samples=n_samples // 4, random_state=0) for v in v_s])
    ddm = DriftDiffusionModel(a="+1", t0=0, v="+1 + coherence", z=0).fit(X, y)
    testing.assert_allclose(ddm.params_, [1, -0.2, 0.6], atol=0.018)

    # a as linear function of coherence
    a_s = 0.6 + 1.0 * coherence
    y = np.concat([sample_from_pdf(a=a, v=0, n_samples=n_samples // 4, random_state=0) for a in a_s])
    ddm = DriftDiffusionModel(a="+1 + coherence", t0=0, v="+1", z=0).fit(X, y)
    testing.assert_allclose(ddm.params_, [0.6, 1.0, 0], atol=0.12)
