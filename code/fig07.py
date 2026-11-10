#!/usr/bin/env python

"""figure 07: pupil-dependent drift and conditional accuracy"""

# compare covariate/constant 4p fits with a 7p model of iid variability

import argparse
import json
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ddm7p import CHAINS, ERR, SIMPS_ERR, logp_full_ddm, sample_full_ddm
from matplotlib.colors import to_rgb
from scipy.stats import binomtest
from settings import T0, V_PUPIL, V_PUPIL_MEAN, A, Z, pupil_trace

from drift_diffusion.model import DriftDiffusionModel, pdf
from drift_diffusion.sim import sample_from_pdf


def conditional_accuracy(rt, a, t0, v, z, err=1e-3, *, correct_boundary=1):
    """compute accuracy by RT, with correct_boundary=+1 (R) or -1 (L) per trial"""

    # use task-defined correct boundaries
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


def conditional_accuracy_full(rt, v_coherence, a, z, t, sz, sv, st, *, coherence):
    """compute accuracy by RT over iid v, z, and t0 variability"""

    # stimulus-coded drift; weight each stimulus direction by its frequency
    rt = np.asarray(rt, dtype=float)

    def density(response, c):
        data = np.column_stack((rt, np.full_like(rt, response)))
        return np.exp(logp_full_ddm(data, c * v_coherence, a, z, t, sz, sv, st))  # positional only

    weights = {c: np.mean(coherence == c) for c in (1, -1)}
    f_correct = sum(w * density(c, c) for c, w in weights.items())
    f_error = sum(w * density(-c, c) for c, w in weights.items())
    return f_correct / (f_correct + f_error)


ROOT = Path(__file__).resolve().parents[1]
HSSM_CACHE = ROOT / "data/fig07-hssm.json"
SEED = 58
TUNE, DRAWS = 1000, 3000  # longer than the ddm7p defaults so the single 7p fit reaches r_hat <= 1.01


def main(n_trials=1000, seed=SEED, chains=CHAINS, tune=TUNE, draws=DRAWS):
    """plot and save fig07"""

    with (ROOT / "code/config.json").open() as f:
        plt.rcParams.update(json.load(f)["rc-params"])
    plt.rcParams["font.size"] = 12
    rng = np.random.default_rng(seed)

    # population-scaled pupil, as in fig08; session means vary
    pupil_centered = pupil_trace(rng, 1, n_trials)[0]
    coherence = rng.choice([-1, 1], n_trials)

    # positive coherence makes R correct; positive drift favors R
    beta_pupil = V_PUPIL
    beta_coherence = V_PUPIL_MEAN + beta_pupil * pupil_centered.mean()  # constant model's session drift
    v_pupil = coherence * (V_PUPIL_MEAN + beta_pupil * pupil_centered)
    params = dict(a=A, t0=T0, v=v_pupil, z=Z)

    # --seed varies pupil, stimuli, and RT sampling
    ys = sample_from_pdf(**params, n_samples=n_trials, random_state=seed)
    choices, rts = np.sign(ys), np.abs(ys)

    print(f"|v| range: {np.abs(v_pupil).min():.3f} to {np.abs(v_pupil).max():.3f}")
    print(f"accuracy: {(choices == coherence).mean():.3f}")

    X = pd.DataFrame({"coherence": coherence, "pupil_centered": pupil_centered})
    param_names = ("a", "t0", "v", "z")
    constant = DriftDiffusionModel(a="1", t0="1", v="0 + coherence", z="1", p_outlier=0).fit(X, ys)
    covariate = DriftDiffusionModel(
        a="1 + pupil_centered",
        t0="1 + pupil_centered",
        v="0 + coherence + coherence:pupil_centered",
        z="1 + pupil_centered",
        p_outlier=0,
    ).fit(X, ys)
    models = {"constant": constant, "covariate": covariate}
    estimates = covariate.params_.reshape(4, 2)
    errors = np.sqrt(np.diag(covariate.covariance_)).reshape(4, 2)
    for name, estimate, error in zip(param_names, estimates[:, 1], errors[:, 1]):
        target = beta_pupil if name == "v" else 0
        print(f"{name} pupil slope: true={target:.3f}, fitted={estimate:.3f} +/- {error:.3f}")

    print(
        f"coherence coefficient: true={beta_coherence:.3f}, "
        f"fitted={constant.params_[2]:.3f} +/- {np.sqrt(constant.covariance_[2, 2]):.3f}"
    )

    # reuse 7p posterior means for matching run settings
    names = ("v_coherence", "a", "z", "t", "sz", "sv", "st")
    cache_key = f"n{n_trials}-seed{seed}-c{chains}-t{tune}-d{draws}-err{ERR:g}-simps{SIMPS_ERR:g}"
    cache = json.loads(HSSM_CACHE.read_text()) if HSSM_CACHE.exists() else {}
    if cache_key not in cache:
        ddm_hssm = sample_full_ddm(
            pd.DataFrame({"rt": rts, "response": choices.astype(int), "coherence": coherence}),
            "coherence",
            chains=chains,
            tune=tune,
            draws=draws,
            seed=seed,
        )
        print(az.summary(ddm_hssm.traces, var_names=list(names), round_to=3)[["mean", "sd", "r_hat", "ess_bulk"]])
        cache[cache_key] = {name: float(ddm_hssm.traces.posterior[name].mean()) for name in names}
        HSSM_CACHE.write_text(json.dumps(cache, indent=2))
    params_hssm = cache[cache_key]
    trial = np.arange(1, n_trials + 1)
    rt_grid = np.linspace(max(params["t0"], params_hssm["t"] + params_hssm["st"] / 2) + 0.1, 2, 100)
    v_example = rng.normal(params_hssm["v_coherence"], params_hssm["sv"], n_trials)
    accuracy_sim = conditional_accuracy(rt_grid, **params, correct_boundary=coherence)
    accuracy_hssm = conditional_accuracy_full(rt_grid, **params_hssm, coherence=coherence)

    # RT deciles: accuracy and 95% Wilson intervals
    decile = np.digitize(rts, np.quantile(rts, np.linspace(0, 1, 11))[1:-1])
    correct = choices == coherence
    tests = [binomtest(correct[decile == b].sum(), np.sum(decile == b)) for b in range(10)]
    rt_bins = [rts[decile == b].mean() for b in range(10)]
    accuracy_bins = np.array([test.statistic for test in tests])
    ci_lower, ci_upper = np.array([test.proportion_ci(method="wilson") for test in tests]).T

    fits = {}
    for key, ddm in models.items():
        bands = ddm.g(X, alpha=0.05)  # pointwise 95% CIs
        accuracy = conditional_accuracy(
            rt_grid, **{name: band["g"] for name, band in bands.items()}, correct_boundary=coherence
        )
        fits[key] = (bands["v"], accuracy)

    labels = {
        "true": "True DDM",
        "constant": "Constant 4p-DDM",
        "covariate": "Pupil-Dependent 4p-DDM",
        "variable": "Across-Trial Variability 7p-DDM",
    }
    axes_labels = [
        dict(xlabel="Trial", ylabel=r"$|v|$", title="A. Pupil-Dependent Drift Rate"),
        dict(xlabel="RT", ylabel="ACC", title="B. Conditional Accuracy Function"),
    ]
    strength = 0.3
    colors = {
        key: strength * np.array(to_rgb(color)) + (1 - strength)
        for key, color in [("constant", "k"), ("covariate", "r"), ("variable", "#3b6fb6")]
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
    axs[1].errorbar(
        rt_bins,
        accuracy_bins,
        yerr=[accuracy_bins - ci_lower, ci_upper - accuracy_bins],
        fmt="o",
        color="k",
        ms=4,
        elinewidth=1,
        zorder=0,
        label="Simulated Data, 95% CI",
    )
    for ax, text in zip(axs, axes_labels):
        ax.set(**text)
    axs[1].set_ylim(0.5, 1.0)
    axs[1].legend(loc="lower left", fontsize=9, frameon=False)

    fig.savefig(ROOT / "results/fig07.svg")
    plt.close(fig)


if __name__ == "__main__":
    # set script defaults

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--chains", type=int, default=CHAINS, help="MCMC chains, run in parallel")
    parser.add_argument("--tune", type=int, default=TUNE, help="MCMC (Slice) tuning iterations per chain")
    parser.add_argument("--draws", type=int, default=DRAWS, help="MCMC (Slice) draws per chain")
    args = parser.parse_args()
    if args.n_trials < 2:
        parser.error("--n-trials must be at least 2")
    main(n_trials=args.n_trials, seed=args.seed, chains=args.chains, tune=args.tune, draws=args.draws)
