"""Three parameter probability density function."""

import autograd.numpy as np


def pdf(X, y, a, v, z, err=1e-3):
    """Probability density function (PDF).

    Parameters
    ----------
    X : _type_
        _description_
    y : _type_
        _description_
    a : _type_
        _description_
    v : _type_
        _description_
    z : _type_
        _description_
    err : _type_, optional
        _description_, by default 1e-3

    Returns
    -------
    _type_
        _description_
    """

    x = X if X.ndim == 1 else X[:, 0]

    # to navarro 2009 notation
    a = 2 * a
    v = -y * v
    w = 0.5 * (1 - y * z)

    xt = x / (a**2)  # normalize time

    # k_l: number of terms for large x
    k_l = np.where(
        np.pi * xt * err < 1,
        np.sqrt(-2 * np.log(np.pi * xt * err) / (np.pi**2 * xt)),
        1 / (np.pi * np.sqrt(xt)),
    )
    k_l = np.maximum(k_l, 1 / (np.pi * np.sqrt(xt)))

    # k_s: number of terms for small x
    k_s = np.where(
        2 * np.sqrt(2 * np.pi * xt) * err < 1,
        2 + np.sqrt(-2 * xt * np.log(2 * np.sqrt(2 * np.pi * xt) * err)),
        2,
    )
    k_s = np.maximum(k_s, np.sqrt(xt) + 1)

    # f(x|1,0,w)
    mask_x_s = k_s < k_l

    # small x approximation
    K_s = np.ceil(k_s).astype(int)
    Ks_s = np.arange(-((K_s.max() - 1) // 2), ((K_s.max() - 1) // 2) + 1)
    exp_s = -((w + 2 * Ks_s[:, None]) ** 2) / (2 * xt)
    p_x_s = np.sum((w + 2 * Ks_s[:, None]) * np.exp(exp_s), axis=0) / np.sqrt(
        2 * np.pi * xt**3
    )

    # large x approximation
    K_l = np.ceil(k_l).astype(int)
    Ks_l = np.arange(1, K_l.max() + 1)
    exp_l = -(Ks_l[:, None] ** 2) * (np.pi**2) * xt / 2
    p_x_l = np.pi * np.sum(
        Ks_l[:, None] * np.exp(exp_l) * np.sin(Ks_l[:, None] * np.pi * w), axis=0
    )

    # combine small x and large x approximations
    p = np.where(mask_x_s, p_x_s, p_x_l)

    # f(x|a,v,w)
    p *= np.exp(-v * a * w - (v**2) * x / 2) / (a**2)

    return p
