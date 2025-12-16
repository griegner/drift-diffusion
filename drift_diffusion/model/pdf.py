"""Probability and mixture density functions."""

import autograd.numpy as np


def pdf(y, a, t0, v, z, err=1e-3):
    """Probability density function (PDF).

    Parameters
    ----------
    y : ndarray of shape (n_samples, )
        reaction times (`abs(y)>0`) decision + nondecision time\\
        responses (`sign(y) = {+1, -1}`) +1 is upper and -1 is lower
    a : float or ndarray of shape (n_samples, )
        decision boundary (`a>0`) +a is upper and -a is lower
    t0 : float or ndarray of shape (n_samples, )
        nondecision time (`t0>=0`) +t0 is time in seconds
    v : float or ndarray of shape (n_samples, )
        drift rate (`-∞<v<+∞`) +v towards +a and -v towards -a
    z : float or ndarray of shape (n_samples, )
        starting point (`-1<z<+1`), +1 is +a and -1 is -a
    err : float, optional
        error tolerance, by default 1e-3

    Returns
    -------
    p : ndarray of shape (n_samples, )
        probability densities

    References
    ----------
    .. [1] Navarro & Fuss. Fast and accurate calculations for first-passage times
       in Wiener diffusion models. Journal of Mathematical Psychology 53, 222–230 (2009)
    """

    # decision time = reaction time - nondecision time
    dt, resp = np.abs(y) - t0, np.sign(y)
    outlier_mask = dt < 0
    dt = np.where(outlier_mask, 1e-6, dt)

    # to navarro 2009 notation
    a = 2 * a
    w = 0.5 * (1 - resp * z)
    t = dt / (a**2)

    # k terms for large t
    k_l = np.where(
        np.pi * t * err < 1,
        np.sqrt(-2 * np.log(np.pi * t * err) / (np.pi**2 * t)),
        1 / (np.pi * np.sqrt(t)),
    )
    k_l = np.maximum(k_l, 1 / (np.pi * np.sqrt(t)))

    # k terms for small t
    k_s = np.where(
        2 * np.sqrt(2 * np.pi * t) * err < 1,
        2 + np.sqrt(-2 * t * np.log(2 * np.sqrt(2 * np.pi * t) * err)),
        2,
    )
    k_s = np.maximum(k_s, np.sqrt(t) + 1)

    # f(t|1,0,w)
    mask = k_s < k_l

    # small t approximation
    K_s = np.ceil(k_s).astype(int)
    Ks_s = np.arange(-((K_s.max() - 1) // 2), ((K_s.max() - 1) // 2) + 1)
    exp_s = -((w + 2 * Ks_s[:, None]) ** 2) / (2 * t)
    p_s = np.sum((w + 2 * Ks_s[:, None]) * np.exp(exp_s), axis=0) / np.sqrt(2 * np.pi * t**3)

    # large t approximation
    K_l = np.ceil(k_l).astype(int)
    Ks_l = np.arange(1, K_l.max() + 1)
    exp_l = -(Ks_l[:, None] ** 2) * (np.pi**2) * t / 2
    p_l = np.pi * np.sum(Ks_l[:, None] * np.exp(exp_l) * np.sin(Ks_l[:, None] * np.pi * w), axis=0)

    # combine small/larger t approximations
    p = np.where(mask, p_s, p_l)

    # f(t|a,v,w)
    p = p * np.exp(resp * a * v * w - dt * (v**2) / 2) / (a**2)

    # set density to 0 for outlier trials
    p = np.where(outlier_mask, 0, p)

    return p


def mdf(y, a, t0, v, z, weights=None, err=1e-3):
    """Mixture density function (MDF).

    Parameters
    ----------
    y : ndarray of shape (n_samples, )
        reaction times (`abs(y)>0`) decision + nondecision time\\
        responses (`sign(y) = {+1, -1}`) +1 is upper and -1 is lower
    a : float or ndarray of shape (n_mixtures, )
        decision boundary (`a>0`) +a is upper and -a is lower
    t0 : float or ndarray of shape (n_mixtures, )
        nondecision time (`t0>=0`) +t0 is time in seconds
    v : float or ndarray of shape (n_mixtures, )
        drift rate (`-∞<v<+∞`) +v towards +a and -v towards -a
    z : float or ndarray of shape (n_mixtures, )
        starting point (`-1<z<+1`), +1 is +a and -1 is -a
    weights : None or ndarray of shape (n_mixtures, ), optional
        mixture weights where None weights all mixtures equally, by default None
    err : float, optional
        error tolerance, by default 1e-3

    Returns
    -------
    p : ndarray of shape (n_samples, )
        mixture probability densities
    """
    pdfs = np.stack([pdf(yi, a, t0, v, z, err) for yi in y], axis=1)  # shape (n_mixtures, n_samples)

    if weights is None:
        weights = np.ones(pdfs.shape[0]) / pdfs.shape[0]
    else:
        weights = weights / np.sum(weights)

    p = np.dot(weights, pdfs)  # shape (n_samples,)
    return p
