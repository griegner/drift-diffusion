import numpy as np
import pytest
from numpy import testing

from drift_diffusion.sim import sample_from_pdf, sample_from_ssm, sim_ddm


def test_sample_from_ssm():
    """test histograms sample_from_ssm vs sample_from_pdf"""
    n_samples = 10_000
    params = {"a": 1, "t0": 0.1, "v": 0.5, "z": -0.1}
    y_ssm = sample_from_ssm(**params, n_samples=n_samples, random_state=0)
    y_pdf = sample_from_pdf(**params, n_samples=n_samples, random_state=0)

    bin_edges = np.linspace(-4, +4, 100)
    hist_ssm, _ = np.histogram(y_ssm, bins=bin_edges, density=True)
    hist_pdf, _ = np.histogram(y_pdf, bins=bin_edges, density=True)

    testing.assert_allclose(hist_ssm, hist_pdf, atol=0.07)


def test_sim_ddm():
    """test returns array of timesteps"""
    x = sim_ddm(dt=0.01, seed=0)
    assert isinstance(x, np.ndarray)
    assert len(x) > 100

    # test that st, sz, sv introduce variability
    x1 = sim_ddm(dt=0.01, st=0.1, sz=0.1, sv=0.1, seed=0)
    x2 = sim_ddm(dt=0.01, st=1, sz=1, sv=1, seed=0)
    with pytest.raises(AssertionError):
        testing.assert_array_equal(x1, x2)


def test_sim_ddm_raises():
    """test raises value error"""
    with pytest.raises(ValueError, match="invalid error distribution"):
        sim_ddm(dt=0.01, error_dist="invalid", seed=0)


@pytest.mark.parametrize(
    "error_dist",
    ["gaussian", "symmetric_bernoulli", "asymmetric_bernoulli", "t", "mixture"],
)
def test_sim_ddm_error_dist(error_dist):
    """test different error distributions"""
    x = sim_ddm(dt=0.01, error_dist=error_dist, seed=0)
    assert isinstance(x, np.ndarray)
    assert len(x) > 50
