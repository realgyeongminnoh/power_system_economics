import numpy as np
from dataclasses import dataclass


@dataclass
class Input_uc:   

    num_units: int
    num_periods: int
    #########################
    demand: np.ndarray
    reserve: np.ndarray
    renewable: np.ndarray
    #########################
    p_min: np.ndarray
    p_max: np.ndarray
    ramp_up: np.ndarray
    ramp_down: np.ndarray
    startup_ramp: np.ndarray
    shutdown_ramp: np.ndarray
    min_up: np.ndarray
    min_down: np.ndarray
    #########################
    cost_lin: np.ndarray
    cost_const: np.ndarray
    #########################
    cost_startup_step: np.ndarray
    num_cooling_steps: np.ndarray
    #########################
    p_prev: np.ndarray
    u_prev: np.ndarray
    min_up_prev: np.ndarray
    min_down_prev: np.ndarray