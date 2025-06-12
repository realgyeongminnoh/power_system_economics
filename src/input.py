import numpy as np
from dataclasses import dataclass


@dataclass
class Input_uc:
    unit_type: np.ndarray
    idx_nuclear: np.ndarray
    idx_coal: np.ndarray
    idx_lng: np.ndarray
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
    cost_startup_step_formulation_0: list
    num_cooling_steps_formulation_0: np.ndarray
    cost_startup_step_formulation_1: np.ndarray
    step_length_formulation_1: np.ndarray
    #########################
    p_prev: np.ndarray
    u_prev: np.ndarray
    min_up_r: np.ndarray
    min_down_r: np.ndarray
    min_up_0: np.ndarray
    min_down_0: np.ndarray
    ##########################
    example_1: bool = False