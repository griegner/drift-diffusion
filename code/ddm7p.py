"""seven-parameter HSSM full_ddm, shared by fig07 and fig08.

HSSM's full_ddm has only a black-box likelihood (no gradients). Defaults here were
piloted on fig08 at n=1000: a looser sz/st integration tolerance (~70x faster) and
DEMetropolisZ chains started near the MLE (~6x the ESS/s of HSSM's default Slice).
"""

import sys

import hssm
import pymc as pm
import pytensor
from hddm_wfpt import wfpt
from hssm.likelihoods.blackbox import hddm_to_hssm

if sys.platform == "darwin":
    pytensor.config.cxx = ""  # macOS clang rejects pytensor's -ld64 link flag; use the python backend


@hddm_to_hssm
def logp_full_ddm(data, v, a, z, t, sz, sv, st):
    """hssm's full_ddm likelihood with simps_err=1e-4 instead of 1e-8; data columns are (rt, response)."""
    return wfpt.wiener_logp_array(x=data, v=v, sv=sv, a=a * 2, z=z, sz=sz, t=t, st=st, err=1e-8, simps_err=1e-4)


def sample_full_ddm(data, initvals, chains=4, tune=1000, draws=12000, seed=None, include=None):
    """fit full_ddm to data (rt, response, covariates) with chains sampling in parallel; returns the HSSM model."""
    model = hssm.HSSM(
        data=data,
        model="full_ddm",
        loglik_kind="blackbox",
        loglik=logp_full_ddm,
        p_outlier=0,
        include=include,
    )
    model.sample(
        sampler="pymc",
        step=pm.DEMetropolisZ(model=model.pymc_model, compile_kwargs={"mode": "cvm"}),
        chains=chains,
        cores=chains,
        tune=tune,
        draws=draws,
        blas_cores=1,
        mp_ctx="spawn",
        initvals=initvals,
        random_seed=seed,
        progressbar=False,
        idata_kwargs={"log_likelihood": False},
    )
    return model
