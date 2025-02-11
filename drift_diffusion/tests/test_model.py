from numpy import testing

from drift_diffusion.model import DriftDiffusionModel
from drift_diffusion.sim import sample_from_pdf


def test_drift_diffusion_model():
    a, v, z = 1, 0.25, 0
    X, y = sample_from_pdf(a, v, z, random_state=0)
    ddm = DriftDiffusionModel(z=z, cov_estimator="all")
    ddm.fit(X, y)
    params_ = ddm.params_
    testing.assert_allclose([a, v], params_, atol=0.05)
