"""seven-parameter HSSM full_ddm, shared by fig07 and fig08"""

# full_ddm has no gradients; defaults piloted on fig08 (n=1000)
# sz/st tolerance 1e-4 vs 1e-8: ~70x faster
# chains start near the MLE; DEMetropolisZ gives ~6x the ESS/s of HSSM's default Slice

import sys

import hssm
import pymc as pm
import pytensor
from hddm_wfpt import wfpt
from hssm.likelihoods.blackbox import hddm_to_hssm

if sys.platform == "darwin":
    pytensor.config.cxx = ""  # avoid macOS clang's unsupported -ld64 flag


@hddm_to_hssm
def logp_full_ddm(data, v, a, z, t, sz, sv, st):
    """evaluate HSSM's full_ddm likelihood with simps_err=1e-4 and data columns (rt, response)"""

    return wfpt.wiener_logp_array(x=data, v=v, sv=sv, a=a * 2, z=z, sz=sz, t=t, st=st, err=1e-8, simps_err=1e-4)


def sample_full_ddm(data, initvals, chains=4, tune=1000, draws=12000, seed=None, include=None):
    """fit full_ddm to data (rt, response, covariates) with parallel chains and return the HSSM model"""

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
