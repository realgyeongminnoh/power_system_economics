import numpy as np
from dataclasses import dataclass


@dataclass
class Input_uc:
    # meta
    num_units: int
    num_periods: int
    unit_type: np.ndarray
    idx_nuclear: np.ndarray
    idx_coal: np.ndarray
    idx_lng: np.ndarray
    # system
    demand: np.ndarray
        # np.linspace(p_min.sum(), p_max.sum() - (p_max[idx_coal] - p_min[idx_coal]).sum(), num_periods), # demand,
    reserve: np.ndarray
        # np.full((num_periods), (p_max[idx_coal] - p_min[idx_coal]).sum()), # (demand - renewable) * 0.05,
    renewable: np.ndarray
        # np.zeros((num_periods)), # renewable,
    # generator
    p_min: np.ndarray
    p_max: np.ndarray
    ramp_up: np.ndarray
    ramp_down: np.ndarray
    startup_ramp: np.ndarray
    shutdown_ramp: np.ndarray
    min_up: np.ndarray
    min_down: np.ndarray
    # cost function - generation
    cost_lin: np.ndarray
    cost_const: np.ndarray
    # initial conditions
    p_prev: np.ndarray = None
    u_prev: np.ndarray = None
    min_up_r: np.ndarray = None
    min_down_r: np.ndarray = None
    min_up_0: np.ndarray = None
    min_down_0: np.ndarray = None
    # cost function - startup (paper formulation)
    cost_startup_step: np.ndarray = None
    step_length: np.ndarray = None
    # cost function - startup (lecture note formulation)
    cost_startup_step_old: list = None
    num_cooling_steps_old: np.ndarray = None

def get_min_up_down_initials(num_units, min_up, min_down, u_prev):
    # for TA marking: you can check like this easily for init condition correctness
    # for idx_unit, (u_prev_i, min_up_i) in enumerate(zip(u_prev, min_up)):
    #     summm = sum(u_prev_i[-min_up_i:])
    #     if summm not in (0, min_up_i):
    #         print(idx_unit, u_prev_i[-min_up_i:])

    # for idx_unit, (u_prev_i, min_down_i) in enumerate(zip(u_prev, min_down)):
    #     summm = sum(u_prev_i[-min_down_i:])
    #     if summm not in (0, min_down_i):
    #         print(idx_unit, u_prev_i[-min_down_i:])

    def _tail_count(seq, value):
        cnt = 0
        for x in reversed(seq):
            if x == value:
                cnt += 1
            else:
                break
        return cnt

    min_up_r, min_down_r = [], []
    for hist, mu, md in zip(u_prev, min_up, min_down):
        on_tail  = _tail_count(hist, 1)
        off_tail = _tail_count(hist, 0)

        if on_tail:
            min_up_r.append(max(0, mu - on_tail))
            min_down_r.append(0)
        else:
            min_up_r.append(0)
            min_down_r.append(max(0, md - off_tail))

    min_up_r, min_down_r = np.array(min_up_r).astype(np.int64), np.array(min_down_r).astype(np.int64)

    min_up_0 = np.array(
        [_tail_count(u_prev[g], 1) for g in range(num_units)], dtype=np.int64
    )
    min_down_0 = np.array(
        [_tail_count(u_prev[g], 0) for g in range(num_units)], dtype=np.int64
    )

    return min_up_r, min_down_r, min_up_0, min_down_0


@dataclass
class Input_ed:
    pass