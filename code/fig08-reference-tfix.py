#!/usr/bin/env python

"""figure 08: 7p MCMC references with t fixed at its limit under HSSM's rt > t guard"""

import argparse
import importlib
import platform
import time
from pathlib import Path

import numpy as np
import settings as sim
from fig08 import MCMC_NAMES, MCMC_OFFSET, MCMC_SCALE, REP, ROOT, SEED, mcmc_seed, save_fit, scientific_settings
from fit_process import bounded_fits

T_LIMIT = float(np.min(sim.T0_STATES))  # RT support edge = 7p t limit under the guard
SAMPLED = [name for name in MCMC_NAMES if name != "t"]
OUTPUT = ROOT / "data/fig08-reference-tfix"


def fit_task(n, rep, config):
    """fit 7p HSSM with t fixed, in a spawned worker"""

    import arviz as az
    import hssm
    import pandas as pd
    from ddm7p import logp_full_ddm
    from threadpoolctl import threadpool_limits

    run = importlib.import_module("fig08-run")
    with threadpool_limits(limits=1):
        X, y = run.simulate(n, rep, config)
        data = pd.DataFrame(dict(rt=np.abs(y), response=np.sign(y).astype(int), stim=X["stim"]))
        # same model as ddm7p.sample_full_ddm, plus a fixed t
        model = hssm.HSSM(
            data=data,
            model="full_ddm",
            loglik=logp_full_ddm,
            p_outlier=0,
            include=[
                dict(name="v", formula="v ~ 0 + stim", prior={"stim": {"name": "Normal", "mu": 0.0, "sigma": 3.0}}),
                dict(name="t", prior=T_LIMIT),
            ],
        )
        initvals = {k: v for k, v in run.true_7p(config["simulation"]).items() if k != "t"}
        started = time.perf_counter()
        model.sample(
            **config["mcmc"],
            random_seed=mcmc_seed(config, n, rep),
            cores=config["mcmc"]["chains"],
            mp_ctx="spawn",
            initvals=initvals,
            progressbar=False,
            idata_kwargs={"log_likelihood": False},
        )
        runtime = time.perf_counter() - started
    summary = az.summary(model.traces, var_names=SAMPLED, round_to="none")
    r_hat, ess = summary["r_hat"].to_numpy(), summary["ess_bulk"].to_numpy()
    valid = (
        np.isfinite(r_hat).all()
        and np.isfinite(ess).all()
        and np.all(r_hat <= config["max_r_hat"])
        and np.all(ess >= config["ess_per_chain"] * config["mcmc"]["chains"])
    )
    mean, sd = dict(summary["mean"], t=T_LIMIT), dict(summary["sd"], t=0.0)
    return dict(
        params=MCMC_SCALE * np.array([mean[k] for k in MCMC_NAMES]) + MCMC_OFFSET,
        sd=MCMC_SCALE * np.array([sd[k] for k in MCMC_NAMES]),
        r_hat=r_hat,
        ess=ess,
        runtime=runtime,
        valid=bool(valid),
        status="converged" if valid else "diagnostic_failure",
        error="",
    )


def run_fits(args, config):
    args.output.mkdir(parents=True, exist_ok=True)
    path = lambda rep: args.output / f"n{args.n_trials}-rep{rep}.npz"  # noqa: E731
    tasks = [(rep, (args.n_trials, rep, config)) for rep in args.repeats if not path(rep).exists()]
    workers = max(1, args.n_jobs // config["mcmc"]["chains"])
    print(f"{len(tasks)} unfinished fits, up to {workers} concurrent", flush=True)
    for rep, outcome, result, elapsed in bounded_fits(fit_task, tasks, workers, float("inf")):
        if outcome != "completed":
            result = dict(runtime=elapsed, valid=False, status=outcome, error=result, params=np.full(7, np.nan))
        save_fit(
            path(rep),
            **result,
            n=args.n_trials,
            rep=rep,
            t_fixed=T_LIMIT,
            mcmc=[config["mcmc"][k] for k in ("chains", "tune", "draws")],
            machine=args.machine,
        )
        print(f"n={args.n_trials} rep={rep}: {result['status']}, {result['runtime']:.1f} s", flush=True)


def summary(output):
    """per size: valid count and mean +/- Monte Carlo SE of the reported quantities"""

    fits = [dict(np.load(path)) for path in sorted(output.glob("n*-rep*.npz"))]
    for n in sorted({int(f["n"]) for f in fits}):
        sized = [f for f in fits if int(f["n"]) == n]
        P = np.array([f["params"] for f in sized if f["valid"]])
        print(f"n={n}: {len(P)}/{len(sized)} valid")
        if len(P) > 1:
            se = P.std(axis=0, ddof=1) / np.sqrt(len(P))
            for name, mean, err in zip(MCMC_NAMES, P.mean(axis=0), se):
                print(f"  {name:7s} {mean:.4f} +/- {err:.4f}")


def repeat_range(value):
    start, stop = map(int, value.split(":"))
    return range(start, stop)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-trials", type=int, choices=(25_000, 50_000, 100_000))
    parser.add_argument("--repeats", type=repeat_range, default=range(REP, REP + 10))
    parser.add_argument("--n-jobs", type=int, default=4, help="CPU budget; each fit uses one core per chain")
    parser.add_argument("--tune", type=int, default=1000)
    parser.add_argument("--draws", type=int, default=1000)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    parser.add_argument("--machine", default=platform.node())
    parser.add_argument("--summary", action="store_true", help="print results instead of fitting")
    args = parser.parse_args()
    if args.summary:
        summary(args.output)
    elif args.n_trials is None:
        parser.error("need --n-trials (or --summary)")
    else:
        run_fits(args, scientific_settings(SEED, chains=4, tune=args.tune, draws=args.draws))
