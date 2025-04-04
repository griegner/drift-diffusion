import itertools

import numpy as np
import pytest
from joblib import Parallel, delayed
from numpy import testing
from pyddm import gddm
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
@parametrize_with_checks([DriftDiffusionModel()], legacy=True)
def test_sklearn_compatible_estimator(estimator, check):
    """test estimator is sklearn compatible"""
    check(estimator)


@pytest.mark.parametrize(
    "a, v, z",
    itertools.product(
        [0.5, 0.75, 1],  # a
        [-1, 0, +1],  # v
        [0.1],  # z
    ),
)
def test_drift_diffusion_model(a, v, z):
    """test the MLE returns the expected parameters and standard errors"""
    n_repeats = 50
    t0 = 0
    ddm = DriftDiffusionModel(t0=t0)

    @delayed
    def _fit(repeat):
        y = sample_from_pdf(a, t0, v, z, random_state=repeat)
        X = np.ones((len(y), 1))
        ddm.fit(X, y)
        stderr = np.sqrt(np.diag(ddm.covariance_))
        return ddm.params_, stderr

    with Parallel(n_jobs=-2) as parallel:
        fits_ = parallel(_fit(repeat) for repeat in range(n_repeats))
        params_, stderrs_ = map(np.asarray, zip(*fits_))

    # test parameters close to true values
    testing.assert_allclose([a, v, z], params_.mean(axis=0), atol=0.2)

    # test standard errors close to true values
    testing.assert_allclose(params_.std(axis=0), stderrs_.mean(axis=0), atol=0.02)


def test_cov_estimator():
    """test all covariance estimates are similar when correctly specified"""
    a, t0, v, z = 1, 0, 0.1, 0
    y = sample_from_pdf(a, t0, v, z)
    X = np.ones((len(y), 1))
    ddm = DriftDiffusionModel(cov_estimator="all")
    ddm.fit(X, y)
    covariances_ = list(ddm.covariance_.values())

    # test cov_estimator="all" returns four matrices
    assert len(covariances_) == 4

    # test covariance matrix of expected shape and estimates
    for cov in covariances_:
        assert cov.shape == (4, 4)
        testing.assert_allclose(covariances_[0], cov, atol=0.001)
