"""figure 08: simulation design, seeds, and saved fits"""

import json
import re
from pathlib import Path

import numpy as np
import settings as sim

ROOT = Path(__file__).resolve().parents[1]
SESSION, SEED = 1000, 20261020
REFERENCE_N, REFERENCE_REP = 50_000, 10**6
METHODS = ["sample-hessian", "outer-product", "misspecification-robust", "autocorrelation-robust"]
PARAM_NAMES = ["a_intercept", "a_time", "t0", "v_stim", "v_stim:pupil", "z"]
MCMC_NAMES = ["a", "t", "v_stim", "z", "sz", "sv", "st"]
# convert HSSM units: z -> 2z - 1, sz -> 2sz
MCMC_SCALE, MCMC_OFFSET = np.array([1, 1, 1, 2, 2, 1, 1]), np.array([0, 0, 0, -1, 0, 0, 0])
QUANTITY_LABELS = [r"$\bar{a}$", r"$t_0$", r"$\bar{v}$", r"$\bar{z}$"]
COLORS = {method: (1, 0, 0, alpha) for method, alpha in zip(METHODS, [0.2, 0.4, 0.6, 0.8])}
COLORS["mcmc"] = "#3b6fb6"
PATTERN = re.compile(r"(?:n\d+-rep\d+|pseudo-true)-(?:mle|mcmc)\.npz")


def scientific_settings(seed, chains, tune, draws):
    """record simulation and fitting settings"""

    return dict(
        format=1,
        seed=seed,
        session=SESSION,
        reference=dict(n=REFERENCE_N, rep=REFERENCE_REP),
        simulation={
            name: np.asarray(getattr(sim, name)).tolist()
            for name in "A T0 V_PUPIL_MEAN A_TREND T0_STATES Z_STATES V_PUPIL P_SWITCH PUPIL_SIGMA PUPIL_SD".split()
        },
        mle=dict(a="1 + time", t0="1", v="0 + stim + stim:pupil_centered", z="1", p_outlier=0),
        ar_lags=60,
        gradient_tolerance=0.01,
        mcmc=dict(chains=chains, tune=tune, draws=draws),
        likelihood="hssm-0.5-guard-simps-1e-4",
        max_r_hat=1.01,
        ess_per_chain=100,
        param_names=PARAM_NAMES,
        mcmc_names=MCMC_NAMES,
    )


def seed_sequence(config, stream, n, rep):
    return np.random.SeedSequence([config["seed"], stream, n, rep])


def mcmc_seed(config, n, rep):
    return int(seed_sequence(config, 3, n, rep).generate_state(1)[0])


def simulation_design(n, rep, config):
    rng = np.random.default_rng(seed_sequence(config, 17, n, rep))
    pupil = sim.pupil_trace(rng, n // SESSION, SESSION).ravel()
    time = np.tile(np.linspace(-0.5, 0.5, SESSION), n // SESSION)
    s = config["simulation"]
    return dict(
        time=time,
        pupil_centered=pupil,
        a=s["A"] * (1 + 2 * s["A_TREND"] * time),
        t0=np.array(s["T0_STATES"]),
        magnitude=s["V_PUPIL_MEAN"] + s["V_PUPIL"] * pupil,
        z=np.array(s["Z_STATES"]),
    )


def trial_inputs(n, rep, config):
    """use identical random draws for fits and example plots"""

    rng = np.random.default_rng(seed_sequence(config, 2, n, rep))
    state = sim.engagement_states(rng, n)
    stim = 2 * rng.integers(2, size=n) - 1
    return simulation_design(n, rep, config), state, stim, int(rng.integers(2**32))


def contrasts(n):
    L = np.eye(6)[[0, 2, 3, 5]]
    L[0, 1] = np.tile(np.linspace(-0.5, 0.5, SESSION), n // SESSION).mean()
    return L


def checkpoint_path(folder, method, n, rep, reference=False):
    return Path(folder) / f"{'pseudo-true' if reference else f'n{n}-rep{rep}'}-{method}.npz"


def save_fit(path, **record):
    """save fits atomically from the supervisor"""

    temporary = path.with_suffix(".tmp.npz")
    np.savez_compressed(temporary, **record)
    temporary.replace(path)


def load_fit(path):
    with np.load(path, allow_pickle=False) as saved:
        fit = dict(saved)
    if str(fit["method"]) == "mle" and not fit["reference"] and "cov" in fit:
        if fit["cov"].shape != (len(METHODS), len(PARAM_NAMES), len(PARAM_NAMES)):
            raise ValueError(f"{path} lacks all four MLE covariances; remove this MLE record and rerun it")
    return fit


def prepare_folder(folder, config):
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "settings.json"
    if path.exists():
        if json.loads(path.read_text()) != config:
            raise ValueError(f"Scientific settings differ in {folder}; use a different output folder")
    else:
        temporary = path.with_suffix(".tmp.json")
        temporary.write_text(json.dumps(config, indent=2) + "\n")
        temporary.replace(path)


def collect_results(folders):
    """collect compatible fits; reject conflicting duplicates"""

    config, records = None, {}
    for folder in map(Path, folders):
        settings = json.loads((folder / "settings.json").read_text())
        expected = scientific_settings(settings["seed"], **settings["mcmc"])
        if settings != expected or (config is not None and settings != config):
            raise ValueError(f"Incompatible scientific settings in {folder}")
        config = settings
        for path in sorted(folder.glob("*.npz")):
            if not PATTERN.fullmatch(path.name):
                continue  # skip temporary files
            fit = load_fit(path)
            key = (bool(fit["reference"]), int(fit["n"]), int(fit["rep"]), str(fit["method"]))
            if (
                checkpoint_path(folder, key[3], key[1], key[2], key[0]) != path
                or int(fit["seed"]) != config["seed"]
                or int(fit["mcmc_seed"]) != mcmc_seed(config, key[1], key[2])
                or (key[0] and key[1:3] != (REFERENCE_N, REFERENCE_REP))
            ):
                raise ValueError(f"Inconsistent fit identity in {path}")
            if key in records:
                previous = records[key]
                equal = previous.keys() == fit.keys() and all(
                    np.array_equal(previous[k], v, equal_nan=v.dtype.kind in "fc") for k, v in fit.items()
                )
                if not equal:
                    raise ValueError(f"Conflicting fit records: {path.name}")
            records[key] = fit
    return config, records
