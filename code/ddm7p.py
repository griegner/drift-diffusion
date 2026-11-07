"""seven-parameter HSSM full_ddm, shared by fig07 and fig08"""

import hssm
from hddm_wfpt import wfpt
from hssm.likelihoods.blackbox import hddm_to_hssm

CHAINS, TUNE, DRAWS = 4, 300, 1000


@hddm_to_hssm
def logp_full_ddm(data, v, a, z, t, sz, sv, st):
    """HSSM full-DDM likelihood."""

    return wfpt.wiener_logp_array(x=data, v=v, sv=sv, a=a * 2, z=z, sz=sz, t=t, st=st, err=1e-8, simps_err=1e-4)


def sample_full_ddm(data, stimulus, chains=CHAINS, tune=TUNE, draws=DRAWS, seed=None):
    """fit full_ddm with drift v ~ 0 + stimulus to data (rt, response, stimulus) and return the HSSM model"""

    model = hssm.HSSM(
        data=data,
        model="full_ddm",
        loglik=logp_full_ddm,
        p_outlier=0,  # default 0.05 adds a lapse mixture
        include=[
            dict(
                name="v",
                formula=f"v ~ 0 + {stimulus}",  # default is v ~ 1
                prior={stimulus: {"name": "Normal", "mu": 0.0, "sigma": 3.0}},  # default N(0, 0.25) too tight for |v|
            )
        ],
    )
    model.sample(
        chains=chains,
        tune=tune,
        draws=draws,
        random_seed=seed,
        progressbar=False,
        idata_kwargs={"log_likelihood": False},  # skip per-trial log-likelihood after sampling
    )
    return model
