"""seven-parameter HSSM full_ddm, shared by fig07 and fig08"""

# full_ddm has no gradients, so PyMC samples it with Slice (HSSM's default for blackbox likelihoods)
# sz/st tolerance 1e-4 vs 1e-8: ~70x faster
# Slice updates parameters in sorted order: HSSM's order varies with Python's hash seed, so seeds would not reproduce
# chains start at a generic point independent of the data; HSSM's own (sz = st = 2) has zero likelihood
# piloted on fig08 (n=1000, state-switching t0): 4 chains x (100 tune + 300 draws) passed R-hat < 1.05 and
# ESS >= 100 for all seven parameters in 20/20 repeats; DEMetropolisZ did not converge in sz, sv, st

import sys

import hssm
import pymc as pm
import pytensor
from hddm_wfpt import wfpt
from hssm.likelihoods.blackbox import hddm_to_hssm

if sys.platform == "darwin":
    pytensor.config.cxx = ""  # avoid macOS clang's unsupported -ld64 flag

INITVALS = dict(a=1.0, t=0.2, z=0.5, sz=0.1, sv=0.5, st=0.1)
CHAINS, TUNE, DRAWS = 4, 100, 300


@hddm_to_hssm
def logp_full_ddm(data, v, a, z, t, sz, sv, st):
    """evaluate HSSM's full_ddm likelihood with simps_err=1e-4 and data columns (rt, response)"""

    return wfpt.wiener_logp_array(x=data, v=v, sv=sv, a=a * 2, z=z, sz=sz, t=t, st=st, err=1e-8, simps_err=1e-4)


def sample_full_ddm(data, stimulus, chains=CHAINS, tune=TUNE, draws=DRAWS, seed=None):
    """fit full_ddm with drift v ~ 0 + stimulus to data (rt, response, stimulus) and return the HSSM model"""

    model = hssm.HSSM(
        data=data,
        model="full_ddm",
        loglik_kind="blackbox",
        loglik=logp_full_ddm,
        p_outlier=0,
        include=[
            dict(
                name="v",
                formula=f"v ~ 0 + {stimulus}",
                link="identity",
                prior={stimulus: {"name": "Normal", "mu": 0.0, "sigma": 3.0}},
            )
        ],
    )
    model.sample(
        sampler="pymc",
        step=pm.Slice(vars=sorted(model.pymc_model.value_vars, key=lambda var: var.name), model=model.pymc_model),
        chains=chains,
        cores=chains,
        tune=tune,
        draws=draws,
        blas_cores=1,
        mp_ctx="spawn",
        initvals={**INITVALS, f"v_{stimulus}": 0.0},
        random_seed=seed,
        progressbar=False,
        idata_kwargs={"log_likelihood": False},
    )
    return model
