import numpy as np
import pandas as pd
from scipy.io import loadmat


def _mat_to_pd(mat):
    """load matlab file and select trial variables"""
    mat = loadmat(mat)
    mat = {k: v.squeeze() for k, v in mat.items() if isinstance(v, np.ndarray) and v.shape == mat["RT"].shape}
    return pd.DataFrame(mat)


def preproc_df(path="./datasets/Rat195Vectors_241025.mat"):
    """load and preprocess the dataframe for rat195 from Reinagel 2013"""
    R, L = 0.0, np.pi  # dot movement
    df = (
        _mat_to_pd(path)
        .query("Valid == 1 and RT == RT")  # keep valid trials and non-null RT
        .assign(trialDate=lambda x: pd.to_datetime(x["trialDate"] - 719529, unit="D"))
        .set_index("trialDate")
        .sort_index()
        .loc["2008-12-03":"2009-03-12"]  # select 24h sessions with constant |coh|
        .assign(
            LR=lambda x: x["dotDirection"].map({R: +1, L: -1}) * x["correct"].map({1: 1, 0: -1}),  # +1 R, -1 L choice
            y=lambda x: x["RT"] * x["LR"],  # signed RT
            coherence=lambda x: x["dotDirection"].map({R: +1, L: -1}) * x["coherence"],  # +coh R, -coh L dot movement
            day=lambda x: pd.factorize((x.index - pd.Timedelta(hours=18)).floor("D"), sort=True)[0],  # 6pm-to-6pm
            hour=lambda x: x.index.hour + 1,  # 1 to 24
            trial=lambda x: ((x.groupby("day").cumcount() + 1) // 20) * 20,  # trials in day
        )
        .query("trial < trial.max()-40")  # >40 trials in day for CLT
    )
    return df
