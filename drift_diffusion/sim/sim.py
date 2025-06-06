"""simulate drift diffusion models"""

import numpy as np
from sklearn.utils.validation import check_random_state

from drift_diffusion.model import pdf


def sample_from_pdf(a=1, t0=0, v=0, z=0, n_samples=1000, random_state=None):
    """Sample from probability density function (PDF).

    Parameters
    ----------
    a : float, optional
        decision boundary (`a>0`) +a is upper and -a is lower, by default 1
    t0 : float, optional
        nondecision time (`t0>=0`) +t0 is time in seconds, by default 0
    v : float, optional
        drift rate (`-∞<v<+∞`) +v towards +a and -v towards -a, by default 0
    z : float, optional
        starting point (`-1<z<+1`), +1 is +a and -1 is -a, by default 0
    n_samples : int, optional
        number of samples to return, by default 1000
    random_state : int, RandomState instance, or None
        random number generator, by default None

    Returns
    -------
    y : ndarray of shape (n_samples, )
        reaction times (`abs(y)>0`) decision + nondecision time\\
        responses (`sign(y) = {+1, -1}`) +1 is upper and -1 is lower
    """

    random_state = check_random_state(random_state)

    num = 1000  # discretize pdf
    rt_range = (1e-5 + t0, 4 + t0)
    y = np.r_[-np.linspace(*rt_range, num), np.linspace(*rt_range, num)]

    pdfs = pdf(y, a, t0, v, z)
    ps = pdfs / np.sum(pdfs)
    return random_state.choice(y, size=n_samples, p=ps)


def sim_ddm(dt, t=0.1, st=0, z=0, sz=0, v=0, sv=0, a=2, error_dist="gaussian", seed=0):
    """Simulate seven parameter drift diffusion model.

    Parameters
    ----------
    dt : float
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
    error_dist : {'gaussian', 'symmetric_bernoulli', 'asymmetric_bernoulli', 't', 'mixture'}, optional
        error distribution, by default "gaussian"
    seed : int, optional
        random seed, by default 0
    """
    rng = np.random.default_rng(seed=seed)

    # define error distributions
    p = np.clip(0.5 * (1 + v * np.sqrt(dt)), 0, 1)
    error_functions = {
        "gaussian": lambda: rng.normal(loc=v * dt, scale=np.sqrt(dt)),
        "symmetric_bernoulli": lambda: (rng.binomial(n=1, p=0.5) * 2 - 1) * np.sqrt(dt) + v * dt,
        "asymmetric_bernoulli": lambda: (rng.binomial(n=1, p=p) * 2 - 1) * np.sqrt(dt),
        "t": lambda: rng.standard_t(df=5) / np.sqrt(5 / (5 - 2)) * np.sqrt(dt) + v * dt,
        "mixture": lambda: (
            rng.normal(loc=v * dt, scale=np.sqrt((0.5 / 0.9) * dt))
            if rng.binomial(1, 0.9)
            else rng.normal(loc=v * dt, scale=np.sqrt((0.5 / 0.1) * dt))
        ),
    }
    if error_dist not in error_functions:
        raise ValueError("invalid error distribution")

    # if variability parameters are given, sample t, z, and v
    if st > 0:
        t = rng.uniform(t - st / 2, t + st / 2)
    if sz > 0:
        z = rng.uniform(z - sz / 2, z + sz / 2)
    if sv > 0:
        v = rng.normal(v, sv)

    x = [z for _ in range(int(t / dt))]
    x_t = z
    time = 0

    # sim diffusion process
    while abs(x_t) < a / 2:
        e_t = error_functions[error_dist]()
        x_t += e_t
        x.append(x_t)
        time += dt

    return np.asarray(x)
