"""simulate drift diffusion models"""

import numpy as np


def sim_ddm(tau, t=0.1, st=0, z=0, sz=0, v=0, sv=0, a=2, seed=0):
    """Simulate seven parameter drift diffusion model.

    Parameters
    ----------
    tau : float
        time step in seconds
    t : float, optional
        nondecision time, by default 0.1
    st : float, optional
        nondecision time variability (uniform distribution with mean=t, range=st), by default 0
    z : int, optional
        starting point of diffusion, by default 0
    sz : int, optional
        starting point variability (uniform distribution with mean=z, range=sz), by default 0
    v : int, optional
        drift rate, by default 0
    sv : int, optional
        drift rate variability (normal distribution with mean=v, sd=sv)
    a : int, optional
        symmetric threshold separation, by default 2
    seed : int, optional
        random seed, by default 0
    """
    rng = np.random.default_rng(seed=seed)

    # if variability parameters are given, sample t, z, and v
    if st > 0:
        t = rng.uniform(t - st / 2, t + st / 2)
    if sz > 0:
        z = rng.uniform(z - sz / 2, z + sz / 2)
    if sv > 0:
        v = rng.normal(v, sv)

    x = [z for _ in range(int(t / tau))]
    x_t = z
    time = 0

    # sim diffusion process
    while abs(x_t) < a / 2:
        x_t += v * tau + rng.normal(0, np.sqrt(tau))  # random walk step
        x.append(x_t)
        time += tau

    return np.asarray(x)
