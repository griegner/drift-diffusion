#!/usr/bin/env python

"""figure 08: resumable simulations and model fits"""

import argparse
import os
import platform
import time
from collections import Counter
from pathlib import Path

import numpy as np
from ddm7p import CHAINS, DRAWS, TUNE
from fig08 import (
    MCMC_NAMES,
    MCMC_OFFSET,
    MCMC_SCALE,
    METHODS,
    REFERENCE_N,
    REFERENCE_REP,
    ROOT,
    SEED,
    SESSION,
    checkpoint_path,
    load_fit,
    mcmc_seed,
    prepare_folder,
    save_fit,
    scientific_settings,
    trial_inputs,
)
from fit_process import bounded_fits


def simulate(n, rep, config):
    import pandas as pd

    from drift_diffusion.sim import sample_from_pdf

    design, state, stim, seed = trial_inputs(n, rep, config)
    y = sample_from_pdf(
        a=design["a"],
        t0=design["t0"][state],
        v=stim * design["magnitude"],
        z=design["z"][state],
        n_samples=n,
        random_state=seed,
    )
    return pd.DataFrame(dict(time=design["time"], pupil_centered=design["pupil_centered"], stim=stim)), y


def true_7p(s):
    """generative values in HSSM units; two-point t0/z states as uniforms with matching mean and SD"""

    t0_gap = s["T0_STATES"][1] - s["T0_STATES"][0]
    z_gap = (s["Z_STATES"][1] - s["Z_STATES"][0]) / 2  # HSSM z = (z + 1) / 2
    return dict(
        a=s["A"],
        t=np.mean(s["T0_STATES"]),
        v_stim=s["V_PUPIL_MEAN"],
        z=(np.mean(s["Z_STATES"]) + 1) / 2,
        sz=np.sqrt(3) * z_gap,
        sv=s["V_PUPIL"] * s["PUPIL_SD"],
        st=np.sqrt(3) * t0_gap,
    )


def fit_task(method, n, rep, config, cores, reference):
    """fit in a spawned worker; time fitting only"""

    from threadpoolctl import threadpool_limits

    with threadpool_limits(limits=1):
        X, y = simulate(n, rep, config)
        if method == "mle":
            from drift_diffusion.model import DriftDiffusionModel

            mle = DriftDiffusionModel(
                **config["mle"],
                cov_estimator="sample-hessian" if reference else "all",
                autocorrelation_lags=config["ar_lags"],
            )
            started = time.perf_counter()
            mle.fit(X, y)
            runtime = time.perf_counter() - started
            cov = np.array(mle.covariance_) if reference else np.array([mle.covariance_[m] for m in METHODS])
            gradient = np.max(abs(mle.fit_.jac))
            valid = (
                np.isfinite(mle.fit_.fun)
                and gradient < config["gradient_tolerance"]
                and np.isfinite(mle.params_).all()
                and np.isfinite(cov).all()
                and np.linalg.eigvalsh((cov + cov.swapaxes(-1, -2)) / 2).min() > 0
            )
            result = dict(params=mle.params_, cov=cov, objective=mle.fit_.fun, max_gradient=gradient)
        else:
            import arviz as az
            import pandas as pd
            from ddm7p import sample_full_ddm

            started = time.perf_counter()
            model = sample_full_ddm(
                pd.DataFrame(dict(rt=np.abs(y), response=np.sign(y).astype(int), stim=X["stim"])),
                "stim",
                **config["mcmc"],
                seed=mcmc_seed(config, n, rep),
                cores=cores,
                mp_ctx="spawn",
                # reference only: start chains at the generative values instead of HSSM's defaults
                initvals=true_7p(config["simulation"]) if reference else None,
                progressbar=reference,
            )
            runtime = time.perf_counter() - started
            summary = az.summary(model.traces, var_names=MCMC_NAMES, round_to="none")
            r_hat, ess = summary["r_hat"].to_numpy(), summary["ess_bulk"].to_numpy()
            valid = (
                np.isfinite(r_hat).all()
                and np.isfinite(ess).all()
                and np.all(r_hat <= config["max_r_hat"])
                and np.all(ess >= config["ess_per_chain"] * config["mcmc"]["chains"])
            )
            result = dict(
                params=MCMC_SCALE * summary["mean"].to_numpy() + MCMC_OFFSET,
                sd=MCMC_SCALE * summary["sd"].to_numpy(),
                r_hat=r_hat,
                ess=ess,
            )
        return dict(
            **result,
            runtime=runtime,
            valid=bool(valid),
            status="converged" if valid else "diagnostic_failure",
            error="",
        )


def run(args, config):
    prepare_folder(args.output, config)
    n, reps = (REFERENCE_N, [REFERENCE_REP]) if args.pseudo_true else (args.n_trials, args.repeats)
    budget = (os.cpu_count() or 1) if args.n_jobs == -1 else args.n_jobs
    for method in ("mle", "mcmc"):
        cores = min(config["mcmc"]["chains"], budget) if method == "mcmc" else 1
        workers = max(1, budget // cores)
        tasks = [
            (rep, (method, n, rep, config, cores, args.pseudo_true))
            for rep in reps
            if not checkpoint_path(args.output, method, n, rep, args.pseudo_true).exists()
        ]
        print(f"{method}: {len(tasks)} unfinished fits, up to {workers} concurrent ({cores} cores per fit)", flush=True)
        # MCMC timeout includes startup; MLE has no limit
        timeout = args.timeout_per_1000 * n / SESSION if method == "mcmc" else float("inf")
        for rep, outcome, result, elapsed in bounded_fits(fit_task, tasks, workers, timeout):
            if outcome != "completed":
                result = dict(
                    runtime=elapsed,
                    valid=False,
                    status=outcome,
                    error=result,
                    params=np.full(6 if method == "mle" else 7, np.nan),
                )
            save_fit(
                checkpoint_path(args.output, method, n, rep, args.pseudo_true),
                **result,
                n=n,
                rep=rep,
                method=method,
                reference=args.pseudo_true,
                seed=config["seed"],
                mcmc_seed=mcmc_seed(config, n, rep),
                machine=args.machine,
                cores=cores,
                timeout_seconds=timeout,
            )
            print(f"{method} n={n} rep={rep}: {result['status']}, {result['runtime']:.1f} s", flush=True)
        counts = Counter(
            str(load_fit(checkpoint_path(args.output, method, n, rep, args.pseudo_true))["status"]) for rep in reps
        )
        print(f"{method}: {dict(counts)}", flush=True)


def repeat_range(value):
    try:
        start, stop = map(int, value.split(":"))
        if 0 <= start < stop:
            return range(start, stop)
    except ValueError:
        pass
    raise argparse.ArgumentTypeError("need START:STOP with 0 <= START < STOP (STOP is exclusive)")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, default=SESSION, help="trials per repeat (default 1000)")
    parser.add_argument("--repeats", type=repeat_range, default="0:500", help="repeat IDs START:STOP (default 0:500)")
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--n-jobs", type=int, default=-1, help="CPU budget; -1 uses all CPUs")
    parser.add_argument("--chains", type=int, default=CHAINS)
    parser.add_argument("--tune", type=int, default=TUNE)
    parser.add_argument("--draws", type=int, default=DRAWS)
    parser.add_argument(
        "--timeout-per-1000", type=float, default=10800, help="MCMC seconds per 1000 trials; inf disables"
    )
    parser.add_argument("--output", type=Path, default=ROOT / "data/fig08")
    parser.add_argument("--machine", default=platform.node(), help="runtime label (default: hostname)")
    parser.add_argument(
        "--pseudo-true", action="store_true", help="fit the shared 50,000-trial reference instead of repeats"
    )
    args = parser.parse_args(argv)
    if args.n_trials < SESSION or args.n_trials % SESSION or args.seed < 0:
        parser.error(f"need a nonnegative seed and n-trials a positive multiple of {SESSION}")
    if args.n_jobs != -1 and args.n_jobs < 1 or args.chains < 2 or args.tune < 0 or args.draws < 4:
        parser.error("need n-jobs=-1 or positive, chains >= 2, tune >= 0, and draws >= 4")
    if np.isnan(args.timeout_per_1000) or args.timeout_per_1000 <= 0:
        parser.error("timeout-per-1000 must be positive (or inf)")
    return args


if __name__ == "__main__":
    args = parse_args()
    try:
        run(args, scientific_settings(args.seed, args.chains, args.tune, args.draws))
    except (OSError, ValueError) as error:
        raise SystemExit(str(error)) from error
