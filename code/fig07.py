#!/usr/bin/env python

"""Figure 07: pupil-dependent drift and conditional accuracy.

Compare covariate and constant 4p DDMs with an iid drift-variability 5p DDM.
The generating drift has no additional noise term; see revisions.md for details.
"""

import argparse
import json
from pathlib import Path

import hssm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytensor
from matplotlib.colors import to_rgb
from scipy.ndimage import gaussian_filter1d

from drift_diffusion.model import DriftDiffusionModel, pdf
from drift_diffusion.sim import sample_from_pdf


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


def conditional_accuracy_sdv(rt, **params):
    """CAF integrated over iid normal drift; +1 denotes a correct response."""
    from hssm.likelihoods.analytical import logp_ddm_sdv

    rt = np.asarray(rt)
    logp_correct, logp_error = (
        logp_ddm_sdv(np.column_stack((rt, np.full_like(rt, response))), **params).eval() for response in (1, -1)
    )
    return np.exp(logp_correct - np.logaddexp(logp_correct, logp_error))


pytensor.config.cxx = ""
ROOT = Path(__file__).resolve().parents[1]


def main(n_trials=1000, seed=7):
    with (ROOT / "code/config.json").open() as f:
        plt.rcParams.update(json.load(f)["rc-params"])
    plt.rcParams["font.size"] = 12
    trial_dur, pupil_tau = 1.0, 20.0
    rng = np.random.default_rng(seed)

    # one slow pupil trace shared across trials
    pupil = 0.5 + 0.15 * slow_traces(rng, 1, n_trials, pupil_tau / trial_dur)[0]
    pupil_centered = pupil - pupil.mean()
    coherence = rng.choice([-1, 1], n_trials)

    # positive coherence means R is correct; positive drift favors R
    v_min, v_max = 0.1, 2.23 + 1.33 * np.sqrt(3)
    beta_pupil = (v_max - v_min) / np.ptp(pupil)
    beta_coherence = v_min + beta_pupil * (pupil.mean() - pupil.min())
    v_pupil = coherence * (beta_coherence + beta_pupil * pupil_centered)
    params = dict(a=0.63, t0=0.435, v=v_pupil, z=0.008)

    ys = sample_from_pdf(**params, n_samples=n_trials, random_state=0)
    choices, rts = np.sign(ys), np.abs(ys)

    print(f"v range: {v_pupil.min():.3f} to {v_pupil.max():.3f}")
    print(f"accuracy: {(choices == coherence).mean():.3f}")

    X = pd.DataFrame({"coherence": coherence, "pupil_centered": pupil_centered})
    param_names = ("a", "t0", "v", "z")
    models = {}
    for key, pupil_terms in [("constant", ""), ("covariate", " + pupil_centered")]:
        formulas = dict.fromkeys(param_names, "1" + pupil_terms)
        formulas["v"] = "0 + coherence" + (" + coherence:pupil_centered" if pupil_terms else "")
        models[key] = DriftDiffusionModel(**formulas).fit(X, ys)

    covariate = models["covariate"]
    estimates = covariate.params_.reshape(4, 2)
    errors = np.sqrt(np.diag(covariate.covariance_)).reshape(4, 2)
    for name, estimate, error in zip(param_names, estimates[:, 1], errors[:, 1]):
        target = beta_pupil if name == "v" else 0
        print(f"{name} pupil slope: true={target:.3f}, fitted={estimate:.3f} +/- {error:.3f}")

    constant = models["constant"]
    print(
        f"coherence coefficient: true={beta_coherence:.3f}, "
        f"fitted={constant.params_[2]:.3f} +/- {np.sqrt(constant.covariance_[2, 2]):.3f}"
    )

    ddm_hssm = hssm.HSSM(
        data=pd.DataFrame({"rt": rts, "response": choices * coherence}), model="ddm_sdv", p_outlier=1e-12
    )
    ddm_hssm.sample(random_seed=seed)

    params_hssm = {name: float(ddm_hssm.traces.posterior[name].mean()) for name in ("v", "a", "z", "t", "sv")}
    trial = np.arange(1, n_trials + 1)
    rt_grid = np.linspace(max(params["t0"], params_hssm["t"]) + 0.1, params["t0"] + 5, 100)
    v_example = np.random.default_rng(seed).normal(params_hssm["v"], params_hssm["sv"], n_trials)
    accuracy_sim = conditional_accuracy(rt_grid, **params, correct_boundary=coherence)
    accuracy_hssm = conditional_accuracy_sdv(rt_grid, **params_hssm)

    fits = {}
    for key, ddm in models.items():
        bands = ddm.g(X, alpha=0.05)  # Pointwise 95% intervals.
        accuracy = conditional_accuracy(
            rt_grid, **{name: band["g"] for name, band in bands.items()}, correct_boundary=coherence
        )
        fits[key] = (bands["v"], accuracy)

    labels = {
        "true": "True DDM",
        "constant": "Constant 4p-DDM",
        "covariate": "Pupil-Dependent 4p-DDM",
        "variable": "Across-Trial Variability 5p-DDM",
    }
    axes_labels = [
        dict(xlabel="Trial", ylabel=r"$|v|$", title="A. Pupil-Dependent Drift Rate"),
        dict(xlabel="RT", ylabel="ACC", title="B. Conditional Accuracy Function"),
    ]
    strength = 0.3
    colors = {
        key: strength * np.array(to_rgb(color)) + (1 - strength)
        for key, color in [("constant", "b"), ("covariate", "r"), ("variable", "g")]
    }

    fig, axs = plt.subplots(1, 2, figsize=(12, 3.4), width_ratios=[2, 1], layout="constrained")
    axs[0].plot(trial, np.abs(v_example), color=colors["variable"], lw=0.65, zorder=1)
    axs[0].plot(trial, np.abs(v_pupil), "--", color="k", lw=1.6, zorder=3)
    axs[1].plot(rt_grid, accuracy_sim, "--", color="k", lw=1.6, label=labels["true"])
    for key, (band, accuracy) in fits.items():
        lower, upper = band["-"], band["+"]
        abs_lower = np.where(lower * upper <= 0, 0, np.minimum(np.abs(lower), np.abs(upper)))
        abs_upper = np.maximum(np.abs(lower), np.abs(upper))
        axs[0].fill_between(trial, abs_lower, abs_upper, color=colors[key], zorder=2)
        axs[1].plot(rt_grid, accuracy, color=colors[key], lw=2, label=labels[key])
    axs[1].plot(rt_grid, accuracy_hssm, color=colors["variable"], lw=2, label=labels["variable"])
    for ax, text in zip(axs, axes_labels):
        ax.set(**text)
    axs[1].set_ylim(0.5, 1.0)
    axs[1].legend(loc="best")

    fig.savefig(ROOT / "results/fig07.svg")
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()
    if args.n_trials < 2:
        parser.error("--n-trials must be at least 2")
    main(n_trials=args.n_trials, seed=args.seed)
