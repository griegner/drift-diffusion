import numpy as np
import pytest

from drift_diffusion.sim import sim_ddm


def test_sim_ddm():
    """test returns array of timesteps"""
    x = sim_ddm(dt=0.01, seed=0)
    assert isinstance(x, np.ndarray)
    assert len(x) > 100


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
