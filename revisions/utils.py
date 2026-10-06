import numpy as np
from scipy.ndimage import gaussian_filter1d

from drift_diffusion.model import pdf


def slow_traces(rng, n_rows, n_pts, tau_pts):
    noise = rng.normal(size=(n_rows, n_pts))
    smooth = gaussian_filter1d(noise, tau_pts, axis=1, mode="wrap")
    return smooth / smooth.std(axis=1, keepdims=True)


def conditional_accuracy(rt, a, t0, v, z, err=1e-3, *, correct_boundary=1):
    """Model-implied accuracy; correct_boundary is +1 (R) or -1 (L) per trial.

    Correct boundaries are supplied by the task, independently of drift.
    """
    a, t0, v, z, correct_boundary = np.broadcast_arrays(*[np.atleast_1d(p) for p in (a, t0, v, z, correct_boundary)])
    if not np.isin(correct_boundary, [-1, 1]).all():
        raise ValueError("correct_boundary must contain only -1 or +1")

    def _density(response):
        return np.column_stack(
            [np.where(t > t0, pdf(response * np.where(t > t0, t, t0 + 1), a, t0, v, z, err), 0) for t in rt]
        )

    f_upper, f_lower = _density(1), _density(-1)
    f_correct = np.average(np.where(correct_boundary[:, None] == 1, f_upper, f_lower), axis=0)
    f_total = np.average(f_upper + f_lower, axis=0)
    return np.divide(f_correct, f_total, out=np.full_like(f_total, np.nan), where=f_total > 0)
