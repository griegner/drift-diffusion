"""seven-parameter HSSM model for fig07 and fig08"""

import hssm
from hddm_wfpt import wfpt
from hssm.likelihoods.blackbox import hddm_to_hssm

CHAINS, TUNE, DRAWS = 4, 300, 1000
# density and simpson tolerance defaults: https://hddm.readthedocs.io/en/latest/hddm.models.html#hddm.models.hddm_info.HDDM
ERR, SIMPS_ERR = 1e-4, 1e-3


@hddm_to_hssm
def logp_full_ddm(data, v, a, z, t, sz, sv, st):
    return wfpt.wiener_logp_array(x=data, v=v, sv=sv, a=a * 2, z=z, sz=sz, t=t, st=st, err=ERR, simps_err=SIMPS_ERR)


def sample_full_ddm(data, stimulus, chains=CHAINS, tune=TUNE, draws=DRAWS, seed=None, **kwargs):
    """fit the 7p HSSM model with v ~ 0 + stimulus; kwargs (cores, mp_ctx, initvals, progressbar) go to HSSM.sample"""

    model = hssm.HSSM(
        data=data,
        model="full_ddm",
        loglik=logp_full_ddm,
        p_outlier=0,  # disable the default lapse mixture
        include=[
            dict(
                name="v",
                formula=f"v ~ 0 + {stimulus}",  # replace intercept-only drift
                prior={stimulus: {"name": "Normal", "mu": 0.0, "sigma": 3.0}},  # broaden the default drift prior
            )
        ],
    )
    model.sample(
        chains=chains,
        tune=tune,
        draws=draws,
        random_seed=seed,
        **{"progressbar": False, **kwargs},
        idata_kwargs={"log_likelihood": False},  # omit per-trial log likelihood
    )
    return model
