"""shared fig02-08 settings from Matzke & Wagenmakers (2009), Table 3"""

import numpy as np
from scipy.ndimage import gaussian_filter1d

# Table 3 (min, max, mean; Matzke & Wagenmakers, 2009)
S = 0.1
T3 = {
    "a": (0.056, 0.393, 0.125),
    "Ter": (0.206, 0.942, 0.435),
    "v": (0.0, 0.586, 0.223),
    "z/a": (0.272, 0.782, 0.063 / 0.125),
    "st": (0.0, 0.630, 0.183),
    "eta": (0.0, 0.329, 0.133),
    "sz/a": (0.0, 0.900, 0.037 / 0.125),
}

# package units: noise SD 1, boundaries +/-a, starting point z in (-1, 1)
A = T3["a"][2] / (2 * S)
T0 = T3["Ter"][2]
V = T3["v"][2] / S  # fig02-04
ETA = T3["eta"][2] / S
Z = 2 * T3["z/a"][2] - 1
# variability in SD units; uniform width / sqrt(12)
SD_T0 = T3["st"][2] / np.sqrt(12)
SD_Z = 2 * T3["sz/a"][2] / np.sqrt(12)

# simulation design
A_TREND = 0.2  # boundary endpoints A * (1 +/- A_TREND)
P_SWITCH = 0.05  # state switch probability per trial
PUPIL_SIGMA, PUPIL_SD = 20, 0.15  # smoothing width (trials) and population SD

# states 0/1: engaged/disengaged; disengagement shifts t0 up and z right
T0_STATES = T0 + np.array([-1, 1]) * 0.75 * SD_T0  # 0.75-SD mixture spread limits misspecification bias
V_STATES = V + np.array([1, -1]) * ETA  # fig04 only
Z_STATES = Z + np.array([0, 2]) * SD_Z

# pupil designs (fig07/08): mean drift half an eta below the Table 3 mean (~85% accuracy);
# pupil-driven drift SD reaches the Table 3 minimum drift at -PUPIL_TAIL pupil SDs
V_PUPIL_MEAN = (T3["v"][2] - T3["eta"][2] / 2) / S
PUPIL_TAIL = 3
SD_V = (V_PUPIL_MEAN - T3["v"][0] / S) / PUPIL_TAIL
V_PUPIL = SD_V / PUPIL_SD


def engagement_states(rng, n):
    """sample states 0/1 with per-trial switch probability P_SWITCH"""

    state = np.empty(n, dtype=int)
    state[0] = rng.integers(2)
    state[1:] = (state[0] + np.cumsum(rng.random(n - 1) < P_SWITCH)) % 2
    return state


def pupil_trace(rng, n_sessions, length):
    """sample smoothed pupil traces with population mean 0 and SD PUPIL_SD"""

    smooth = gaussian_filter1d(rng.normal(size=(n_sessions, length)), PUPIL_SIGMA, axis=1, mode="wrap")
    impulse = np.zeros(length)
    impulse[0] = 1
    return PUPIL_SD * smooth / np.linalg.norm(gaussian_filter1d(impulse, PUPIL_SIGMA, mode="wrap"))
