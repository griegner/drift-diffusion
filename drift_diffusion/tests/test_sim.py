import numpy as np

from drift_diffusion.sim import sim_ddm


def test_sim_ddm():
    """tests for simulating seven parameter ddm"""
    x = sim_ddm(tau=0.01)
    assert isinstance(x, np.ndarray)
