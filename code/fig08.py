#!/usr/bin/env python

"""figure 08: MLE with session trends, pupil effects, and hidden engagement states"""

# simulate n trials per repeat as n/1000 sessions: pupil traces, hidden states and stimuli are redrawn every repeat;
# the hidden engagement state shifts nondecision time and starting point as in fig02to04's persistent setting
# fit the covariate 4p-DDM MLE (sample-Hessian and autocorrelation-robust SEs) and the seven-parameter HSSM
# full_ddm MCMC to every repeat; targets are Monte Carlo pseudo-true parameters (pseudotrue.py), computed and saved
# before any fit; --plot scores both against them and plots

import argparse
import json
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ddm7p import CHAINS, DRAWS, TUNE, sample_full_ddm
from joblib import Parallel, delayed, effective_n_jobs
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter
from pseudotrue import N_TRIALS, REP, pseudotrue_4p, pseudotrue_7p
from scipy.ndimage import gaussian_filter1d
from sklearn.base import clone

from drift_diffusion.model import DriftDiffusionModel
from drift_diffusion.sim import sample_from_pdf

ROOT = Path(__file__).resolve().parents[1]
SEED = 20261020
SESSION = 1000  # trials per session
TARGETS = ROOT / "data/fig08-pseudotrue.npz"  # one Monte Carlo target for every n
CHECKPOINTS = ROOT / "data/fig08-mcmc"
# |v| spans 0.1 to the fig07 maximum over +-3.5 SD of pupil, fixed across repeats so every repeat shares one target
V_MID, V_SLOPE = (0.1 + 2.23 + 1.33 * np.sqrt(3)) / 2, (2.23 + 1.33 * np.sqrt(3) - 0.1) / (7 * 0.15)
METHODS = ["sample-hessian", "autocorrelation-robust"]
MLE = DriftDiffusionModel(
    a="1 + time",
    t0="1",
    v="0 + stim + stim:pupil_centered",
    z="1",
    p_outlier=0,
    cov_estimator="all",
    autocorrelation_lags=60,
)
PARAM_NAMES = ["a_intercept", "a_time", "t0", "v_stim", "v_stim:pupil", "z"]
# rescale HSSM z -> 2*z - 1 and sz -> 2*sz
MCMC_NAMES = ["a", "t", "v_stim", "z", "sz", "sv", "st"]
MCMC_SCALE, MCMC_OFFSET = np.array([1, 1, 1, 2, 2, 1, 1]), np.array([0, 0, 0, -1, 0, 0, 0])
MCMC_FIELDS = ["runtime", "params", "sd", "r_hat", "ess", "valid"]
# shared targets: first four MCMC_NAMES and contrasts of MLE coefficients
QUANTITIES = ["a_mid", "t0", "v_avg", "z"]
QUANTITY_LABELS = [r"$\bar{a}$", r"$t_0$", r"$\bar{v}$", r"$\bar{z}$"]
# fig07 palette: 4p-DDM in reds (dark for point estimates and AR, light for SH), 7p-DDM in blue
COLORS = {"sample-hessian": "#f19c9d", "autocorrelation-robust": "#8c1515", "mcmc": "#3b6fb6"}
PALE = {key: 0.3 * np.array(to_rgb(color)) + 0.7 for key, color in [("mle", "r"), ("mcmc", "#3b6fb6")]}


def simulation_design(n, rep):
    """pupil traces for n/1000 sessions, drawn per repeat"""

    # standardize each session to mean 0.5 and SD 0.15 so the pupil mean and spread, and with them the targets, are
    # the same in every repeat; centering at the population mean keeps the covariate free of the data
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 17, n, rep]))
    pupil = gaussian_filter1d(rng.normal(size=(n // SESSION, SESSION)), 20, axis=1, mode="wrap")
    centered = (0.15 * (pupil - pupil.mean(axis=1, keepdims=True)) / pupil.std(axis=1, keepdims=True)).ravel()
    session_time = np.tile(np.linspace(-0.5, 0.5, SESSION), n // SESSION)
    # engaged (state 0) and disengaged (state 1) t0 and z, Matzke st and sz scales as in fig02to04
    t0 = 0.435 + np.array([-1, 1]) * 0.183 / np.sqrt(12)
    z = 0.008 + np.array([0, 2]) * (2 * 0.037 / 0.125) / np.sqrt(12)
    return dict(
        time=session_time,
        pupil_centered=centered,
        a=0.63 + 0.252 * session_time,
        t0=t0,
        magnitude=V_MID + V_SLOPE * centered,
        z=z,
    )


def comparable(design):
    """contrasts from the MLE coefficients to QUANTITIES, and their generating values"""

    L = np.zeros((len(QUANTITIES), len(PARAM_NAMES)))
    L[0, [0, 1]] = 1, design["time"].mean()  # a at mean session time
    L[1, 2] = 1
    L[2, [3, 4]] = 1, design["pupil_centered"].mean()  # drift at mean pupil
    L[3, 5] = 1
    truth = [design["a"].mean(), design["t0"].mean(), design["magnitude"].mean(), design["z"].mean()]
    return L, np.array(truth)


def simulate(design, rep):
    """sample RTs with persistent states and independent stimulus directions"""

    n = len(design["time"])
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 2, n, rep]))
    state = np.empty(n, dtype=int)
    state[0] = rng.integers(2)
    state[1:] = (state[0] + np.cumsum(rng.random(n - 1) < 0.05)) % 2
    stim = 2 * rng.integers(2, size=n) - 1
    y = sample_from_pdf(
        a=design["a"],
        t0=design["t0"][state],
        v=stim * design["magnitude"],
        z=design["z"][state],
        n_samples=n,
        random_state=int(rng.integers(2**32)),
    )
    X = pd.DataFrame(dict(time=design["time"], pupil_centered=design["pupil_centered"], stim=stim))
    return X, y, state


def pseudotrue_targets():
    """4p coefficients and 7p parameters (MCMC_NAMES scale) fit to one large simulated sample (pseudotrue.py)"""

    X, y, _ = simulate(simulation_design(N_TRIALS, REP), REP)
    mle = pseudotrue_4p(MLE, X, y)
    mcmc = pseudotrue_7p(y, X["stim"].to_numpy(), q0=[0.63, 0.38, 2.27, 0.59, 0.15, 0.4, 0.1])
    return mle, MCMC_SCALE * mcmc + MCMC_OFFSET


def fit_mle(X, y):
    """fit from the package's default start (a=1, all else 0); time optimization plus covariance estimation"""

    mle = clone(MLE)
    started = time.perf_counter()
    try:
        mle.fit(X, y)
        if not (np.isfinite(mle.fit_.fun) and np.max(abs(mle.fit_.jac)) < 0.01):
            raise ValueError("MLE did not converge")
        runtime = time.perf_counter() - started
        cov = np.array([mle.covariance_[method] for method in METHODS])
        valid = np.isfinite(mle.params_).all() and np.isfinite(cov).all()
        valid = valid and all(np.linalg.eigvalsh((c + c.T) / 2).min() > 0 for c in cov)
        return runtime, mle.params_, cov, bool(valid)
    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return time.perf_counter() - started, np.full(6, np.nan), np.full((2, 6, 6), np.nan), False


def fit_mcmc(X, y, chains, tune, draws, seed):
    """fit seven-parameter full_ddm with stimulus-coded drift and no other covariates (ddm7p settings)"""

    started = time.perf_counter()
    model = sample_full_ddm(
        pd.DataFrame(dict(rt=np.abs(y), response=np.sign(y).astype(int), stim=X["stim"])),
        "stim",
        chains=chains,
        tune=tune,
        draws=draws,
        seed=seed,
    )
    runtime = time.perf_counter() - started
    summary = az.summary(model.traces, var_names=MCMC_NAMES, round_to="none")
    r_hat, ess = summary["r_hat"].to_numpy(), summary["ess_bulk"].to_numpy()
    valid = bool(np.all(r_hat < 1.05) and np.all(ess >= 100))
    mean = MCMC_SCALE * summary["mean"].to_numpy() + MCMC_OFFSET
    sd = MCMC_SCALE * summary["sd"].to_numpy()
    return runtime, mean, sd, r_hat, ess, valid


def run_mcmc_checkpoint(n, rep, chains, tune, draws):
    """save one MCMC fit to CHECKPOINTS so an interrupted run resumes where it stopped"""

    path = CHECKPOINTS / f"n{n}-rep{rep}-c{chains}-t{tune}-d{draws}.npz"
    if not path.exists():
        X, y, _ = simulate(simulation_design(n, rep), rep)
        seed = int(np.random.SeedSequence([SEED, 3, n, rep]).generate_state(1)[0])
        fit = fit_mcmc(X, y, chains, tune, draws, seed)
        np.savez(path, **dict(zip(MCMC_FIELDS, fit)))
        print(f"MCMC rep={rep}: {fit[0]:.0f} s, converged={fit[-1]}", flush=True)
    with np.load(path) as saved:
        return rep, *(saved[name] for name in MCMC_FIELDS)


def run_mcmc(n, n_repeats, n_jobs, chains, tune, draws):
    """run chains in parallel within a fit and fits in parallel across the remaining CPUs"""

    # spawn fits: chain processes cannot nest in loky workers or threads; spawned workers get n as an argument
    CHECKPOINTS.mkdir(exist_ok=True)
    n_fits = max(1, effective_n_jobs(n_jobs) // chains)
    print(f"MCMC: {n_repeats} fits, {n_fits} at a time with {chains} chains each", flush=True)
    with ProcessPoolExecutor(n_fits, mp_context=get_context("spawn")) as pool:
        futures = [pool.submit(run_mcmc_checkpoint, n, rep, chains, tune, draws) for rep in range(n_repeats)]
        records = [future.result() for future in futures]
    return dict(zip(["rep", *MCMC_FIELDS], map(np.array, zip(*records))))


def scores(fits, L, targets):
    """point and SE relative bias and 95% CI coverage (%), with 95% Monte Carlo half-widths from the CLT"""

    def metrics(params_, uncs_, reference):
        # the empirical SD of the point estimates serves as the true SE; each metric's Monte Carlo SE is the SD of
        # its per-repeat influence over sqrt(repeats), by the delta method for the SE ratio
        error, sd, unc = params_ - reference, params_.std(ddof=1, axis=0), uncs_.mean(axis=0)
        covered = abs(error) <= 1.96 * uncs_
        influence = {
            "bias": 100 * error / reference,
            "se_bias": 100 * ((uncs_ - unc) / sd - unc * ((params_ - params_.mean(axis=0)) ** 2 - sd**2) / (2 * sd**3)),
            "coverage": 100 * covered,
        }
        values = {
            "bias": 100 * error.mean(axis=0) / reference,
            "se_bias": 100 * (unc - sd) / sd,
            "coverage": 100 * covered.mean(axis=0),
        }
        return {key: (values[key], 1.96 * influence[key].std(ddof=1, axis=0) / np.sqrt(len(params_))) for key in values}

    mle, mcmc = fits["valid"], fits["mcmc_valid"]
    params_ = fits["params"][mle] @ L.T
    estimates = {
        method: (params_, np.sqrt(np.einsum("ij,rjk,ik->ri", L, fits["cov"][mle, m], L)), L @ targets["mle"])
        for m, method in enumerate(METHODS)
    }
    estimates["mcmc"] = fits["mcmc_params"][mcmc, :4], fits["mcmc_sd"][mcmc, :4], targets["mcmc"][:4]
    result = {}
    for method, (params_, uncs_, reference) in estimates.items():
        result[method] = metrics(params_, uncs_, reference)
        print(
            f"{method} ({len(params_)} repeats):", {key: np.round(value[0], 1) for key, value in result[method].items()}
        )
    return result


def plot_example(fig, spec, fits, n, targets):
    """panel A: one repeat's generating states, pseudo-true 4p trace, 4p fit with 95% AR band, and 7p iid draws"""

    # converged repeat whose 4p and 7p estimates lie jointly closest to their targets
    L, _ = comparable(simulation_design(n, 0))
    estimates = [(fits["params"] @ L.T, L @ targets["mle"]), (fits["mcmc_params"][:, :4], targets["mcmc"][:4])]
    distance = sum(np.sum(((e - ref) / np.nanstd(e, axis=0)) ** 2, axis=1) for e, ref in estimates)
    rep = int(np.argmin(np.where(fits["valid"] & fits["mcmc_valid"], distance, np.inf)))
    design = simulation_design(n, rep)
    beta, cov = fits["params"][rep], fits["cov"][rep, METHODS.index("autocorrelation-robust")]
    a, t0, v, z, sz, sv, st = fits["mcmc_params"][rep]

    one, zero = np.ones(len(design["time"])), np.zeros(len(design["time"]))
    X = {
        "a": np.column_stack([one, design["time"], zero, zero, zero, zero]),
        "t0": np.column_stack([zero, zero, one, zero, zero, zero]),
        "v": np.column_stack([zero, zero, zero, one, design["pupil_centered"], zero]),
        "z": np.column_stack([zero, zero, zero, zero, zero, one]),
    }
    rng = np.random.default_rng(0)
    draws = {
        "a": np.full(len(one), a),
        "t0": rng.uniform(t0 - st / 2, t0 + st / 2, len(one)),
        "v": np.abs(rng.normal(v, sv, len(one))),
        "z": rng.uniform(z - sz / 2, z + sz / 2, len(one)),
    }
    trial = np.arange(1, n + 1)
    axes = [fig.add_subplot(row) for row in spec.subgridspec(4, 1)]
    for ax, key, label in zip(axes, X, [r"$a$", r"$t_0$", r"$|v|$", r"$z$"]):
        mean, half = X[key] @ beta, 1.96 * np.sqrt(np.einsum("ij,jk,ik->i", X[key], cov, X[key]))
        ax.plot(trial, draws[key], color=PALE["mcmc"], lw=0.65, zorder=1)
        ax.fill_between(trial, mean - half, mean + half, color=PALE["mle"], zorder=2)
        ax.plot(trial, X[key] @ targets["mle"], "--", color="k", lw=1.2, zorder=4)
        ax.set_ylabel(label, rotation=0, ha="right", va="center", fontsize=18)
        ax.set_xlim(1, n)
        ax.tick_params(labelbottom=ax is axes[-1])
    state = simulate(design, rep)[2]
    for ax, key in zip(axes[1::2], ["t0", "z"]):
        ax.step(trial, design[key][state], color="0.25", lw=0.6, where="post", zorder=3)
    axes[1].set_ylim(0.33, 0.53)  # covers both state t0 values
    axes[-1].set_xlabel("Trial")
    axes[0].set_title("A. Example Simulation and Model Fits")
    return axes


def plot(fits, n, targets):
    """A: example fit; B: runtime; C-E: point and SE relative bias and coverage, with 95% Monte Carlo intervals"""

    with (ROOT / "code/config.json").open() as f:
        plt.rcParams.update(json.load(f)["rc-params"])
    plt.rcParams["font.size"] = 12
    L, _ = comparable(simulation_design(n, 0))  # session time and pupil means, hence L, are the same in every repeat
    result = scores(fits, L, targets)
    fig = plt.figure(figsize=(12, 7.5), layout="constrained")
    grid = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])
    axes = plot_example(fig, grid[0, :2], fits, n, targets)
    axes += [fig.add_subplot(grid[0, 2])] + [fig.add_subplot(grid[1, k]) for k in range(3)]

    # B: every valid fit's runtime from each saved data/fig08-n{n}.npz, jittered in n
    paths = {int(path.stem.removeprefix("fig08-n")): path for path in (ROOT / "data").glob("fig08-n*.npz")}
    sizes = sorted(paths)
    rng = np.random.default_rng(0)
    for size in sizes:
        with np.load(paths[size]) as saved:
            runtimes = [("autocorrelation-robust", saved["runtime"][saved["valid"]])]
            runtimes += [("mcmc", saved["mcmc_runtime"][saved["mcmc_valid"]])]
        for method, runtime in runtimes:
            jitter = rng.uniform(-250, 250, len(runtime))
            axes[4].plot(size + jitter, runtime, "o", color=COLORS[method], ms=3, mec="white", mew=0.3)
    axes[4].yaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axes[4].xaxis.set_major_formatter(StrMethodFormatter("{x:,.0f}"))
    axes[4].set(xlabel="n Trials", ylabel="Seconds per fit", xticks=sizes, xlim=(0, max(sizes) + 1000))

    panels = [
        ("bias", ["autocorrelation-robust", "mcmc"], "Point relative bias (%)", (-3, 3)),
        ("se_bias", METHODS + ["mcmc"], "SE relative bias (%)", (-45, 45)),
        ("coverage", METHODS + ["mcmc"], "95% CI coverage (%)", (60, 101)),
    ]
    x = np.arange(len(QUANTITIES))
    for ax, (key, methods, label, ylim) in zip(axes[5:], panels):
        for method, offset in zip(methods, np.linspace(-0.2, 0.2, len(methods))):
            value, mcse = result[method][key]
            ax.errorbar(x + offset, value, yerr=mcse, fmt="o", color=COLORS[method], ms=6, mec="white", mew=0.5)
        ax.axhline(95 if key == "coverage" else 0, color="0.35", ls="--" if key == "coverage" else "-", lw=1, zorder=1)
        ax.set(ylabel=label, ylim=ylim, xticks=x, xticklabels=QUANTITY_LABELS, xlim=(-0.5, len(x) - 0.5))
        ax.tick_params(axis="x", labelsize=18)
    half = 196 * np.sqrt(0.95 * 0.05 / fits["mcmc_valid"].sum())  # 95% Monte Carlo band around nominal coverage
    axes[7].axhspan(95 - half, 95 + half, color="0.92", zorder=0)
    titles = ["B. Runtime", "C. Point Estimate Bias", "D. Standard Error Bias", "E. Confidence Interval Coverage"]
    for ax, title in zip(axes[4:], titles):
        ax.set_title(title)
        ax.grid(axis="y", alpha=0.15)
    labels = ["4p-DDM MLE (SH SE)", "4p-DDM MLE (point, AR SE)", "7p-DDM MCMC"]
    handles = [
        Line2D([], [], color=color, marker="o", ms=6, ls="none", label=l) for color, l in zip(COLORS.values(), labels)
    ]
    fig.legend(handles=handles, loc="outside lower center", ncol=3, frameon=False)
    fig.savefig(ROOT / ("results/fig08.svg" if n == SESSION else f"results/fig08-n{n}.svg"))
    plt.close(fig)


def main(n_trials, n_repeats, n_jobs, chains, tune, draws, plot_only):
    """save the pseudo-true targets if missing, then fit all repeats to data/, or with --plot plot the saved fits"""

    if not TARGETS.exists():
        np.savez(TARGETS, **dict(zip(["mle", "mcmc"], pseudotrue_targets())))
        print(f"saved pseudo-true targets to {TARGETS}", flush=True)
    fits_path = ROOT / f"data/fig08-n{n_trials}.npz"
    if not plot_only:

        @delayed
        def run_simulation(rep):
            X, y, _ = simulate(simulation_design(n_trials, rep), rep)
            return rep, *fit_mle(X, y)

        with Parallel(n_jobs=n_jobs) as parallel:
            records = parallel(run_simulation(rep) for rep in range(n_repeats))
        fits = dict(zip(["rep", "runtime", "params", "cov", "valid"], map(np.array, zip(*records))))
        print(f"MLE: {fits['valid'].sum()}/{n_repeats} valid fits", flush=True)
        mcmc = run_mcmc(n_trials, n_repeats, n_jobs, chains, tune, draws)
        print(f"MCMC: {mcmc['valid'].sum()}/{n_repeats} converged fits")
        np.savez_compressed(fits_path, **fits, **{f"mcmc_{key}": value for key, value in mcmc.items()})
        return
    if not fits_path.exists():
        raise SystemExit(f"no saved fits at {fits_path}; run without --plot first")
    with np.load(fits_path) as saved, np.load(TARGETS) as targets:
        plot({key: saved[key] for key in saved}, n_trials, dict(targets))


if __name__ == "__main__":
    # set script defaults

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-trials", type=int, default=SESSION, help=f"trials per repeat, a multiple of {SESSION} (one session each)"
    )
    parser.add_argument(
        "--n-repeats", type=int, default=500, help="number of simulation repeats, each fit by MLE and MCMC"
    )
    parser.add_argument("--n-jobs", type=int, default=-1, help="number of parallel jobs (cpus)")
    parser.add_argument("--chains", type=int, default=CHAINS, help="MCMC chains per fit, run in parallel")
    parser.add_argument("--tune", type=int, default=TUNE, help="MCMC (Slice) tuning iterations per chain")
    parser.add_argument("--draws", type=int, default=DRAWS, help="MCMC (Slice) draws per chain")
    parser.add_argument("--plot", action="store_true", help="plot saved data/fig08-n{n-trials}.npz instead of fitting")
    args = parser.parse_args()
    if args.n_repeats < 3 or args.n_jobs == 0 or args.chains < 2 or args.n_trials < 1 or args.n_trials % SESSION:
        parser.error(f"need at least three repeats, nonzero n-jobs, two chains and n-trials a multiple of {SESSION}")
    main(args.n_trials, args.n_repeats, args.n_jobs, args.chains, args.tune, args.draws, args.plot)
