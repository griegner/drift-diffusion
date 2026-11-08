#!/usr/bin/env python

"""figure 08: MLE with session trends, pupil effects, and hidden engagement states"""

# simulate n trials per repeat as n/1000 sessions: pupil traces, hidden states and stimuli are redrawn every repeat;
# the hidden engagement state shifts nondecision time and starting point as in fig02to04's persistent setting
# fit the covariate 4p-DDM MLE (sample-Hessian and autocorrelation-robust SEs) and the seven-parameter HSSM
# full_ddm MCMC to every repeat; large-sample references use 4p MLE coefficients and 7p posterior means.
# Saved references are reused; --plot scores both against them and plots.

import argparse
import json
import time
import warnings
from pathlib import Path

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ddm7p import CHAINS, DRAWS, TUNE, sample_full_ddm
from fit_process import bounded_fits
from joblib import Parallel, delayed, effective_n_jobs
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter
from pseudotrue import N_TRIALS, REP, pseudotrue_4p
from scipy.ndimage import gaussian_filter1d
from sklearn.base import clone

from drift_diffusion.model import DriftDiffusionModel
from drift_diffusion.sim import sample_from_pdf

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)


ROOT = Path(__file__).resolve().parents[1]
SEED = 20261020
SESSION = 1000  # trials per session
TARGETS = ROOT / "data/fig08-pseudotrue.npz"  # one Monte Carlo target for every n
TARGET_LIKELIHOOD = "hssm-0.5-guard-simps-1e-4"
CHECKPOINTS = ROOT / "data/fig08-mcmc"
# |v| spans 0.1 to the fig07 maximum over +-3.5 SD of pupil, fixed across repeats so every repeat shares one target
V_MID, V_SLOPE = (0.1 + 2.23 + 1.33 * np.sqrt(3)) / 2, (2.23 + 1.33 * np.sqrt(3) - 0.1) / (7 * 0.15)
A_MEAN, T0_MEAN = 0.63, 0.435
Z_MEAN = 0.008 + (2 * 0.037 / 0.125) / np.sqrt(12)
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
MCMC_FIELDS = ["runtime", "params", "sd", "r_hat", "ess", "valid", "status", "error"]
# shared targets: first four MCMC_NAMES and contrasts of MLE coefficients
QUANTITIES = ["a_mid", "t0", "v_avg", "z"]
QUANTITY_LABELS = [r"$\bar{a}$", r"$t_0$", r"$\bar{v}$", r"$\bar{z}$"]
# fig07 palette: 4p-DDM in reds (dark for point estimates and AR, light for SH), 7p-DDM in blue
COLORS = {"sample-hessian": "#f19c9d", "autocorrelation-robust": "#8c1515", "mcmc": "#3b6fb6"}
PALE = {key: 0.3 * np.array(to_rgb(color)) + 0.7 for key, color in [("mle", "r"), ("mcmc", "#3b6fb6")]}


def simulation_design(n, rep):
    """pupil traces for n/1000 sessions, drawn per repeat"""

    # Population mean 0 and SD 0.15; session means and spreads fluctuate across repeats.
    rng = np.random.default_rng(np.random.SeedSequence([SEED, 17, n, rep]))
    pupil = gaussian_filter1d(rng.normal(size=(n // SESSION, SESSION)), 20, axis=1, mode="wrap")
    impulse = np.zeros(SESSION)
    impulse[0] = 1
    population_sd = np.linalg.norm(gaussian_filter1d(impulse, 20, mode="wrap"))
    centered = (0.15 * pupil / population_sd).ravel()
    session_time = np.tile(np.linspace(-0.5, 0.5, SESSION), n // SESSION)
    # engaged (state 0) and disengaged (state 1) t0 and z, Matzke st and sz scales as in fig02to04
    t0 = T0_MEAN + np.array([-1, 1]) * 0.183 / np.sqrt(12)
    z = Z_MEAN + np.array([-1, 1]) * (2 * 0.037 / 0.125) / np.sqrt(12)
    return dict(
        time=session_time,
        pupil_centered=centered,
        a=A_MEAN + 0.252 * session_time,
        t0=t0,
        magnitude=V_MID + V_SLOPE * centered,
        z=z,
    )


def comparable(design):
    """contrasts from the MLE coefficients to QUANTITIES, and fixed generative means"""

    L = np.zeros((len(QUANTITIES), len(PARAM_NAMES)))
    L[0, [0, 1]] = 1, design["time"].mean()  # a at mean session time
    L[1, 2] = 1
    L[2, 3] = 1  # drift at the population pupil mean, shared across repeats
    L[3, 5] = 1
    truth = [A_MEAN, T0_MEAN, V_MID, Z_MEAN]
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
    """4p MLE and production HSSM posterior means on the same large simulated dataset."""

    X, y, _ = simulate(simulation_design(N_TRIALS, REP), REP)
    mle = pseudotrue_4p(MLE, X, y)
    seed = int(np.random.SeedSequence([SEED, 3, N_TRIALS, REP]).generate_state(1)[0])
    _, mean, _, _, _, valid = fit_mcmc(X, y, CHAINS, TUNE, DRAWS, seed)
    if not valid:
        raise RuntimeError("Large-sample HSSM reference failed convergence checks; targets were not saved")
    return mle, mean


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


def mcmc_converged(r_hat, ess, chains):
    """PyMC default diagnostic thresholds for every fitted parameter"""

    return bool(
        np.isfinite(r_hat).all() and np.isfinite(ess).all() and np.all(r_hat <= 1.01) and np.all(ess >= 100 * chains)
    )


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
    valid = mcmc_converged(r_hat, ess, chains)
    mean = MCMC_SCALE * summary["mean"].to_numpy() + MCMC_OFFSET
    sd = MCMC_SCALE * summary["sd"].to_numpy()
    return runtime, mean, sd, r_hat, ess, valid


def checkpoint_path(n, rep, chains, tune, draws):
    return CHECKPOINTS / f"n{n}-rep{rep}-c{chains}-t{tune}-d{draws}.npz"


def load_checkpoint(path, chains):
    with np.load(path) as saved:
        fit = {name: saved[name] for name in MCMC_FIELDS[:6]}
        status, fit["error"] = str(saved["status"]), str(saved["error"])
    # Evaluate saved diagnostics using the current acceptance thresholds.
    completed = status in ("converged", "diagnostic_failure")
    fit["valid"] = completed and mcmc_converged(fit["r_hat"], fit["ess"], chains)
    fit["status"] = ("converged" if fit["valid"] else "diagnostic_failure") if completed else status
    return fit


def mcmc_task(n, rep, chains, tune, draws):
    X, y, _ = simulate(simulation_design(n, rep), rep)
    seed = int(np.random.SeedSequence([SEED, 3, n, rep]).generate_state(1)[0])
    return fit_mcmc(X, y, chains, tune, draws, seed)


def run_mcmc(n, reps, n_jobs, chains, tune, draws, timeout_per_1000=3600):
    """run chains in parallel within a fit and fits in parallel across the remaining CPUs"""

    # spawn fits: chain processes cannot nest in loky workers or threads; spawned workers get n as an argument
    CHECKPOINTS.mkdir(exist_ok=True)
    n_fits = max(1, effective_n_jobs(n_jobs) // chains)
    print(f"MCMC: {len(reps)} fits, {n_fits} at a time with {chains} chains each", flush=True)
    records, tasks = {}, []
    for rep in reps:
        path = checkpoint_path(n, rep, chains, tune, draws)
        if path.exists():
            records[rep] = load_checkpoint(path, chains)
        else:
            tasks.append((rep, (n, rep, chains, tune, draws)))
    for rep, outcome, result, elapsed in bounded_fits(mcmc_task, tasks, n_fits, timeout_per_1000 * n / 1000):
        if outcome == "completed":
            fit = dict(zip(MCMC_FIELDS[:6], result))
            fit.update(status="converged" if fit["valid"] else "diagnostic_failure", error="")
        else:
            fit = dict(
                runtime=elapsed,
                params=np.full(7, np.nan),
                sd=np.full(7, np.nan),
                r_hat=np.full(7, np.nan),
                ess=np.full(7, np.nan),
                valid=False,
                status=outcome,
                error=result,
            )
        path = checkpoint_path(n, rep, chains, tune, draws)
        temp = path.with_suffix(".tmp.npz")
        np.savez(
            temp,
            **fit,
            seed=int(np.random.SeedSequence([SEED, 3, n, rep]).generate_state(1)[0]),
            timeout_seconds=timeout_per_1000 * n / 1000,
        )
        temp.replace(path)  # only supervisor writes; interrupted writes never become checkpoints
        records[rep] = fit
        print(f"MCMC rep={rep}: {fit['runtime']:.0f} s, status={fit['status']}", flush=True)
    records = [(rep, *(records[rep][name] for name in MCMC_FIELDS)) for rep in reps]
    return dict(zip(["rep", *MCMC_FIELDS], map(np.array, zip(*records))))


def collect_results(n, chains, tune, draws):
    """Collect completed checkpoints without launching fits; preserve their repeat IDs."""
    with np.load(ROOT / f"data/fig08-mle-n{n}.npz") as saved:
        fits = dict(saved)
    records = []
    for path in CHECKPOINTS.glob(f"n{n}-rep*-c{chains}-t{tune}-d{draws}.npz"):
        rep = int(path.name.split("-rep")[1].split("-")[0])
        fit = load_checkpoint(path, chains)
        records.append((rep, *(fit[name] for name in MCMC_FIELDS)))
    if not records:
        raise ValueError(f"No completed MCMC checkpoints for n={n}")
    records.sort(key=lambda row: row[0])
    fits.update({f"mcmc_{name}": np.array(values) for name, values in zip(["rep", *MCMC_FIELDS], zip(*records))})
    path = ROOT / f"data/fig08-n{n}.npz"
    temp = path.with_suffix(".tmp.npz")
    np.savez_compressed(temp, **fits)
    temp.replace(path)
    return fits


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
    status = fits["mcmc_status"]
    print(
        "MCMC attempts:",
        {
            str(name): {
                "count": int(np.sum(status == name)),
                "fit_hours": float(np.sum(fits["mcmc_runtime"][status == name]) / 3600),
            }
            for name in np.unique(status)
        },
    )
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

    # converged repeat whose 4p and 7p estimates lie jointly closest to their targets; MCMC extras have no MLE fit
    L, _ = comparable(simulation_design(n, 0))
    lookup = {int(rep): i for i, rep in enumerate(fits["mcmc_rep"])}
    indices = np.array([lookup.get(int(rep), -1) for rep in fits["rep"]])
    safe_indices = np.maximum(indices, 0)
    eligible = fits["valid"] & (indices >= 0) & fits["mcmc_valid"][safe_indices]
    if not eligible.any():
        raise ValueError("No repeat has both valid MLE and MCMC fits")
    estimates = [
        (fits["params"] @ L.T, L @ targets["mle"]),
        (fits["mcmc_params"][safe_indices, :4], targets["mcmc"][:4]),
    ]
    distance = sum(np.sum(((e - ref) / np.nanstd(e, axis=0)) ** 2, axis=1) for e, ref in estimates)
    row = int(np.argmin(np.where(eligible, distance, np.inf)))
    rep = int(fits["rep"][row])
    design = simulation_design(n, rep)
    beta, cov = fits["params"][row], fits["cov"][row, METHODS.index("autocorrelation-robust")]
    a, t0, v, z, sz, sv, st = fits["mcmc_params"][indices[row]]

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
    L, _ = comparable(simulation_design(n, 0))  # fixed population contrasts for every repeat
    result = scores(fits, L, targets)
    fig = plt.figure(figsize=(12, 7.5), layout="constrained")
    fig.suptitle(
        f"n = {n:,} trials; MLE {fits['valid'].sum()}/{len(fits['rep'])}, "
        f"MCMC {fits['mcmc_valid'].sum()}/{len(fits['mcmc_rep'])} completed fits pass"
    )
    grid = fig.add_gridspec(2, 3, height_ratios=[1.15, 1])
    axes = plot_example(fig, grid[0, :2], fits, n, targets)
    axes += [fig.add_subplot(grid[0, 2])] + [fig.add_subplot(grid[1, k]) for k in range(3)]

    # B: valid fit runtimes, explicitly excluding timeouts, jittered in n
    paths = {int(path.stem.removeprefix("fig08-n")): path for path in (ROOT / "data").glob("fig08-n*.npz")}
    sizes = sorted(paths)
    rng = np.random.default_rng(0)
    for size in sizes:
        with np.load(paths[size]) as saved:
            runtimes = [("autocorrelation-robust", saved["runtime"][saved["valid"]])]
            runtime_valid = saved["mcmc_valid"] & (saved["mcmc_status"] != "timeout")
            runtimes += [("mcmc", saved["mcmc_runtime"][runtime_valid])]
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
        lower = min(float(np.min(result[method][key][0] - result[method][key][1])) for method in methods)
        upper = max(float(np.max(result[method][key][0] + result[method][key][1])) for method in methods)
        margin = 0.05 * max(upper - lower, 1)
        ylim = (min(ylim[0], lower - margin), max(ylim[1], upper + margin))
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
    path = ROOT / ("results/fig08.svg" if n == SESSION else f"results/fig08-n{n}.svg")
    fig.savefig(path)
    fig.savefig(path.with_suffix(".png"), dpi=180)
    plt.close(fig)


def main(
    n_trials, n_repeats, n_jobs, chains, tune, draws, plot_only, reps=None, timeout_per_1000=3600, max_attempts=None
):
    """Run checkpointed fits, or plot saved fits against the large-sample reference."""

    if reps is not None:
        # one machine's share: MCMC checkpoints only; a later full run collects them and fits the MLE
        run_mcmc(n_trials, reps, n_jobs, chains, tune, draws, timeout_per_1000)
        return
    if not TARGETS.exists():
        np.savez(TARGETS, **dict(zip(["mle", "mcmc"], pseudotrue_targets())), likelihood=TARGET_LIKELIHOOD)
        print(f"saved pseudo-true targets to {TARGETS}", flush=True)
    with np.load(TARGETS) as saved:
        if "likelihood" not in saved or str(saved["likelihood"]) != TARGET_LIKELIHOOD:
            raise ValueError("Pseudo-true target likelihood does not match the current analysis")
        targets = dict(saved)
    if plot_only:
        plot(collect_results(n_trials, chains, tune, draws), n_trials, targets)
        return

    fits_path = ROOT / f"data/fig08-n{n_trials}.npz"
    mle_path = ROOT / f"data/fig08-mle-n{n_trials}.npz"  # saved before MCMC so an interrupted run keeps the MLE fits

    @delayed
    def run_simulation(rep):
        X, y, _ = simulate(simulation_design(n_trials, rep), rep)
        return rep, *fit_mle(X, y)

    fits = None
    if mle_path.exists():
        with np.load(mle_path) as saved:
            if len(saved["rep"]) == n_repeats:
                fits = dict(saved)
    if fits is None:
        with Parallel(n_jobs=n_jobs) as parallel:
            records = parallel(run_simulation(rep) for rep in range(n_repeats))
        fits = dict(zip(["rep", "runtime", "params", "cov", "valid"], map(np.array, zip(*records))))
        np.savez_compressed(mle_path, **fits)
    print(f"MLE: {fits['valid'].sum()}/{n_repeats} valid fits", flush=True)
    max_attempts = max_attempts or 3 * n_repeats
    if max_attempts < n_repeats:
        raise ValueError("max_attempts must be at least n_repeats")
    mcmc = run_mcmc(n_trials, range(n_repeats), n_jobs, chains, tune, draws, timeout_per_1000)
    # fit the next repeats until exactly n_repeats converge; non-converged fits are kept as extras
    while (missing := n_repeats - mcmc["valid"].sum()) > 0:
        stop = min(len(mcmc["rep"]) + missing, max_attempts)
        if stop == len(mcmc["rep"]):
            break
        extra = run_mcmc(n_trials, range(len(mcmc["rep"]), stop), n_jobs, chains, tune, draws, timeout_per_1000)
        mcmc = {key: np.concatenate([mcmc[key], extra[key]]) for key in mcmc}
    print(
        f"MCMC: {mcmc['valid'].sum()} converged fits from {len(mcmc['rep'])} attempts; "
        f"statuses={dict(zip(*np.unique(mcmc['status'], return_counts=True)))}"
    )
    np.savez_compressed(fits_path, **fits, **{f"mcmc_{key}": value for key, value in mcmc.items()})
    if mcmc["valid"].sum() < n_repeats:
        raise RuntimeError(f"Attempt limit {max_attempts} reached; partial results saved to {fits_path}")


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
    parser.add_argument(
        "--repeats", help="START:STOP: only fit these repeats' MCMC checkpoints, to split a run across machines"
    )
    parser.add_argument("--plot", action="store_true", help="plot saved data/fig08-n{n-trials}.npz instead of fitting")
    parser.add_argument(
        "--timeout-per-1000",
        type=float,
        default=3600,
        help="seconds per 1000 trials (default 3600); inf disables the timeout",
    )
    parser.add_argument("--max-attempts", type=int, help="total MCMC attempt ceiling (default 3*n-repeats)")
    args = parser.parse_args()
    if np.isnan(args.timeout_per_1000) or args.timeout_per_1000 <= 0:
        parser.error("--timeout-per-1000 must be positive (or inf for no timeout)")
    if args.max_attempts is not None and args.max_attempts < args.n_repeats:
        parser.error("--max-attempts must be at least --n-repeats")
    if args.n_repeats < 3 or args.n_jobs == 0 or args.chains < 2 or args.n_trials < 1 or args.n_trials % SESSION:
        parser.error(f"need at least three repeats, nonzero n-jobs, two chains and n-trials a multiple of {SESSION}")
    reps = None
    if args.repeats:
        try:
            start, stop = map(int, args.repeats.split(":"))
        except ValueError:
            parser.error("--repeats must be START:STOP")
        reps = range(start, stop)
        if not 0 <= reps.start < reps.stop or args.plot:
            parser.error("--repeats needs 0 <= START < STOP and no --plot")
    main(
        args.n_trials,
        args.n_repeats,
        args.n_jobs,
        args.chains,
        args.tune,
        args.draws,
        args.plot,
        reps,
        args.timeout_per_1000,
        args.max_attempts,
    )
