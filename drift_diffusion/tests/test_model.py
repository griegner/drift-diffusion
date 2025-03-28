import numpy as np
from numpy import testing

from drift_diffusion.model import DriftDiffusionModel
from drift_diffusion.sim import sample_from_pdf


def test_drift_diffusion_model():

    a, t0, v, z = 1, 0.2, 0.25, 0

    y = sample_from_pdf(a, t0, v, z, random_state=0)
    X = np.ones((len(y), 1))

    ddm = DriftDiffusionModel(z=z, cov_estimator="all")
    ddm.fit(X, y)
    params_ = ddm.params_
    covariance_ = ddm.covariance_.values()

    # test parameters close to true values
    testing.assert_allclose([a, t0, v], params_, atol=0.05)

    # test covariance matrix of expected shape
    for cov in covariance_:
        assert cov.shape == (3, 3)

    # test cov_estimator="all" returns four matrices
    assert len(covariance_) == 4
