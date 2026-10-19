#!/usr/bin/env python

"""Figure 08 revision: four-coefficient MLE versus seven-coefficient full DDM.

Reuse fig04's persistent states, stimulus directions, conditional RT sampler,
and seed. Both fits omit the hidden state and estimate a shared drift magnitude
via v = abs_v * stim; the other coefficients are constant across trials.

MLE replications run in parallel; each MCMC replication runs parallel chains.
Both methods get the same CPU budget, without nested parallelism. Runtime in
the figure is total method wall time / replications (including model building,
MCMC tuning, and worker startup), rather than serial per-fit MLE latency.

The parameter RMSE panel uses the four generating marginal means as descriptive
references, NOT pseudo-true targets: neither working model represents the
coupled, serially dependent binary mixture. Uncertainty RMSE compares each
method's reported SDs with its empirical SD across independent replications.
MLE uses fig04's autocorrelation-robust covariance with 60 lags by default.
All seven MCMC estimates, R-hat, and effective sample sizes are saved separately.

Run from any directory in the project's code-ocean/dev environment, e.g.:
    python code/fig08-rev.py
    python code/fig08-rev.py --n-trials 50 --n-repeats 2 --draws 20 --tune 20
The latter is only an implementation check, not an inferential validation.
"""

import argparse
import json
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from fig02to04 import persistent_cdfs, persistent_parameters, sample_persistent
from joblib import Parallel, delayed, effective_n_jobs, parallel_config
from threadpoolctl import threadpool_limits

from drift_diffusion.model import DriftDiffusionModel

ROOT = Path(__file__).resolve().parents[1]
COMMON_PARAMS = ["a", "t0", "abs_v", "z"]
SEED = 720261019


def generating_means():
    """Stationary means, rather than the four-parameter pseudo-true optimum."""
    engaged, disengaged = persistent_parameters()
    return dict(zip(COMMON_PARAMS, [(engaged[p] + disengaged[p]) / 2 for p in ["a", "t0", "v", "z"]]))


def simulate(n, rep, sampling, seed):
    """Identical data for both methods, including fresh states and stimuli."""
    stim, y, _ = sample_persistent(n, rep, *sampling, seed=seed)
    return pd.DataFrame({"stim": stim, "rt": abs(y), "response": np.sign(y).astype(int)})


def fit_mle(data, cov_estimator):
    """Four free coefficients, initialized as in fig04."""
    start = time.perf_counter()
    mle = DriftDiffusionModel(v="-1 + stim", cov_estimator=cov_estimator, autocorrelation_lags=60, p_outlier=1e-12)
    with threadpool_limits(limits=1):
        mle.fit(
            data[["stim"]],
            data["rt"].to_numpy() * data["response"].to_numpy(),
            params0=np.array([0.613275658969, 0.381493834668, 1.478601882646, 0.051482449097]),
        )
    runtime = time.perf_counter() - start
    variances = np.diag(mle.covariance_)
    if np.any(variances < 0) or not np.all(np.isfinite(mle.covariance_)) or not np.all(np.isfinite(mle.params_)):
        raise RuntimeError("MLE produced invalid estimates or covariance")
    rows = [
        dict(param=p, mean=float(mean), sd=float(sd))
        for p, mean, sd in zip(COMMON_PARAMS, mle.params_, np.sqrt(variances))
    ]
    return rows, dict(
        fit_seconds=runtime,
        success=bool(mle.fit_.success),
        message=str(mle.fit_.message),
        gradient_max=float(np.max(abs(mle.fit_.jac))),
    )


def fit_mcmc(data, cores, chains, draws, tune, seed):
    """Seven free coefficients; full_ddm selects HSSM's black-box Slice sampler."""
    import arviz as az
    from hssm import HSSM

    start = time.perf_counter()
    mcmc = HSSM(
        data=data,
        model="full_ddm",
        loglik_kind="blackbox",
        p_outlier=1e-12,
        include=[
            dict(
                name="v",
                formula="v ~ 0 + stim",
                link="identity",
                prior={"stim": {"name": "Normal", "mu": 0.0, "sigma": 3.0}},
            )
        ],
    )
    mcmc.sample(
        sampler="pymc",
        cores=cores,
        chains=chains,
        draws=draws,
        tune=tune,
        blas_cores=cores,
        mp_ctx="spawn",
        quiet=True,
        initvals={"a": 0.63, "t": 0.3, "v_stim": 1.5, "z": 0.55, "sz": 0.1, "sv": 0.5, "st": 0.05},
        random_seed=seed,
        idata_kwargs={"log_likelihood": False},
    )
    runtime = time.perf_counter() - start
    summary = az.summary(mcmc.traces, var_names=["a", "t", "v_stim", "z", "sz", "sv", "st"], round_to="none")
    # HSSM z and sz use [0, 1]; our side bias and its full width use [-1, 1].
    mapping = {
        "a": ("a", 1, 0),
        "t0": ("t", 1, 0),
        "abs_v": ("v_stim", 1, 0),
        "z": ("z", 2, -1),
        "sz": ("sz", 2, 0),
        "sv": ("sv", 1, 0),
        "st": ("st", 1, 0),
    }
    rows = []
    for param, (name, scale, offset) in mapping.items():
        row = summary.loc[name]
        rows.append(
            dict(
                param=param,
                mean=float(scale * row["mean"] + offset),
                sd=float(scale * row["sd"]),
                r_hat=float(row["r_hat"]),
                ess_bulk=float(row["ess_bulk"]),
                ess_tail=float(row["ess_tail"]),
            )
        )
    return rows, dict(fit_seconds=runtime)


def summarize(fits, timings):
    """Retain the original fig08 bias/SD/RMSE metrics, plus explicit references."""
    rows = []
    means = generating_means()
    for (n, method, param), group in fits.groupby(["n", "method", "param"], sort=False):
        estimates = group["mean"].to_numpy()
        empirical_sd = float(np.std(estimates)) if len(group) > 1 else np.nan
        runtime = timings.loc[(timings["n"] == n) & (timings["method"] == method), "seconds_per_repeat"].iloc[0]
        for estimate, values, target in (
            ("param", estimates, means.get(param, np.nan)),
            ("unc", group["sd"].to_numpy(), empirical_sd),
        ):
            rows.append(
                dict(
                    n=n,
                    method=method,
                    estimate=estimate,
                    param=param,
                    runtime=runtime,
                    target=target,
                    reference="generating mean" if estimate == "param" else "empirical SD",
                    bias=float(np.mean(values - target)),
                    sd=float(np.std(values)),
                    rmse=float(np.sqrt(np.mean((values - target) ** 2))),
                    n_repeats=len(group),
                )
            )
    return pd.DataFrame(rows)


def plot_results(summary, timings, output):
    """Compare common parameters; retain variability parameters in the CSV files."""
    with (ROOT / "code/config.json").open() as f:
        plt.rcParams.update(json.load(f)["rc-params"])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), layout="constrained")
    colors = dict(zip(COMMON_PARAMS, ["black", "0.3", "0.55", "0.75"]))
    methods = {"mle": ("MLE (4)", "-"), "mcmc": ("MCMC (7)", "--")}
    ns = list(timings["n"].drop_duplicates())
    positions = {n: i for i, n in enumerate(ns)}
    for method, (label, linestyle) in methods.items():
        timing = timings.loc[timings["method"] == method]
        axes[0].plot(timing["n"].map(positions), timing["seconds_per_repeat"], marker="o", ls=linestyle, label=label)
        for ax, estimate in zip(axes[1:], ["param", "unc"]):
            for param in COMMON_PARAMS:
                group = summary.query("method == @method and estimate == @estimate and param == @param")
                ax.plot(
                    group["n"].map(positions),
                    group["rmse"],
                    marker="o",
                    ls=linestyle,
                    color=colors[param],
                    label=f"{param}, {label}",
                )
    for ax, title in zip(
        axes, ["Wall seconds / replication", "RMSE vs generating mean", "Uncertainty RMSE vs empirical SD"]
    ):
        ax.set(title=title, xlabel="n Trials", xticks=range(len(ns)), xticklabels=ns)
    axes[0].legend(fontsize=10)
    axes[2].legend(fontsize=9, ncol=2)
    fig.savefig(output)
    plt.close(fig)


def main(args):
    cores = effective_n_jobs(args.n_jobs)
    chains = args.chains if args.chains is not None else max(4, cores)
    if chains < max(2, cores):
        raise ValueError("--chains must be >= max(2, CPU budget) so all requested cores can sample")
    args.data_dir.mkdir(parents=True, exist_ok=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.data_dir / "fig08-rev.csv"
    fits_path = args.data_dir / "fig08-rev-fits.csv"
    timing_path = args.data_dir / "fig08-rev-runtime.csv"
    if args.csv:
        summary, timings = pd.read_csv(summary_path), pd.read_csv(timing_path)
    else:
        # Load HSSM once outside timed work; MLE workers never need this import.
        import hssm

        metadata = dict(
            vars(args),
            cores=cores,
            chains=chains,
            hssm_version=hssm.__version__,
            generating_means=generating_means(),
            runtime_definition="method wall seconds / replications; same CPU budget",
            parameter_reference="generating marginal means, not pseudo-true working-model parameters",
        )
        (args.data_dir / "fig08-rev-settings.json").write_text(json.dumps(metadata, default=str, indent=2) + "\n")
        sampling = persistent_cdfs()
        fits, timing_rows = [], []
        print(f"CPU budget: {cores} per method; MCMC: {chains} chains; MLE: parallel replications", flush=True)
        for n in args.n_trials:
            datasets = [simulate(n, rep, sampling, args.seed) for rep in range(args.n_repeats)]
            start = time.perf_counter()
            with parallel_config(backend="loky", inner_max_num_threads=1):
                results = Parallel(n_jobs=cores)(delayed(fit_mle)(data, args.cov_estimator) for data in datasets)
            elapsed = time.perf_counter() - start
            timing_rows.append(
                dict(
                    n=n,
                    method="mle",
                    cores=cores,
                    chains=0,
                    n_repeats=args.n_repeats,
                    wall_seconds=elapsed,
                    seconds_per_repeat=elapsed / args.n_repeats,
                )
            )
            for rep, (rows, diagnostics) in enumerate(results):
                if not diagnostics["success"]:
                    warnings.warn(f"MLE n={n}, rep={rep}: {diagnostics['message']}", stacklevel=2)
                fits.extend(dict(n=n, rep=rep, method="mle", **diagnostics, **row) for row in rows)
            pd.DataFrame(fits).to_csv(fits_path, index=False)
            print(f"n={n}: MLE batch {elapsed:.1f}s", flush=True)
            elapsed = 0.0
            for rep, data in enumerate(datasets):
                start = time.perf_counter()
                seed = int(np.random.SeedSequence([args.seed, n, rep, 1]).generate_state(1)[0])
                rows, diagnostics = fit_mcmc(data, cores, chains, args.draws, args.tune, seed)
                elapsed += time.perf_counter() - start
                fits.extend(dict(n=n, rep=rep, method="mcmc", **diagnostics, **row) for row in rows)
                # Checkpoint every slow MCMC replication, including all seven parameters.
                pd.DataFrame(fits).to_csv(fits_path, index=False)
                print(f"n={n}: MCMC {rep + 1}/{args.n_repeats}, {diagnostics['fit_seconds']:.1f}s", flush=True)
            timing_rows.append(
                dict(
                    n=n,
                    method="mcmc",
                    cores=cores,
                    chains=chains,
                    n_repeats=args.n_repeats,
                    wall_seconds=elapsed,
                    seconds_per_repeat=elapsed / args.n_repeats,
                )
            )
            timings = pd.DataFrame(timing_rows)
            summary = summarize(pd.DataFrame(fits), timings)
            timings.to_csv(timing_path, index=False)
            summary.to_csv(summary_path, index=False)
    output = args.output_dir / "fig08-rev.svg"
    plot_results(summary, timings, output)
    print(f"Saved {output} and {summary_path}", flush=True)


def positive_int(value):
    value = int(value)
    if value < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return value


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n-trials", "--n-samples", nargs="+", type=positive_int, default=[500, 1000, 5000])
    parser.add_argument("--n-repeats", type=positive_int, default=100)
    parser.add_argument("--n-jobs", type=int, default=-1, help="same CPU budget for both methods; -1 uses all cores")
    parser.add_argument("--chains", type=positive_int, help="default: max(4, CPU budget); must cover all cores")
    parser.add_argument("--draws", type=positive_int, default=1000, help="posterior draws per chain")
    parser.add_argument("--tune", type=positive_int, default=1000, help="tuning iterations per chain")
    parser.add_argument("--seed", type=int, default=SEED, help="fig04's persistent simulation seed")
    parser.add_argument(
        "--cov-estimator",
        default="autocorrelation-robust",
        choices=["sample-hessian", "outer-product", "misspecification-robust", "autocorrelation-robust"],
    )
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--csv", action="store_true", help="replot saved summary and runtime CSVs")
    main(parser.parse_args())
