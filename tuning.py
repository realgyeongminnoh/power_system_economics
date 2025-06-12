import pickle
from pathlib import Path
import numpy as np
import gurobipy as gp; gp.Model()
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from src.input import Input_uc
from src.output import Output_uc
from src.unit_commitment import solve_uc_formulation_0, solve_uc_formulation_1


def asdf():
    num_units = 122
    num_periods = 24
    time_start, time_end = np.datetime64("2022-07-02T00"), np.datetime64("2022-07-02T23")
    path_folder_processed = Path.cwd() / "data" / "input" / "processed"
    timestamp_2022 = np.load(path_folder_processed / "timestamp_2022.npy")
    idx_time_start, idx_time_end = np.where(timestamp_2022 == time_start)[0][0], np.where(timestamp_2022 == time_end)[0][0]
    idx_time_prev = idx_time_start - 1
    if idx_time_prev < 0:
        raise NotImplementedError("not 2022-01-01T00 for the time start")
    unit_type = np.load(path_folder_processed / "unit_type.npy")
    idx_nuclear = np.load(path_folder_processed / "idx_nuclear.npy")
    idx_coal = np.load(path_folder_processed / "idx_coal.npy")
    idx_lng = np.load(path_folder_processed / "idx_lng.npy")
    p_min = np.load(path_folder_processed / "p_min.npy")
    p_max = np.load(path_folder_processed / "p_max.npy")
    ramp_up = np.load(path_folder_processed / "ramp_up.npy")
    ramp_down = np.load(path_folder_processed / "ramp_down.npy")
    startup_ramp = np.load(path_folder_processed / "startup_ramp.npy")
    shutdown_ramp = np.load(path_folder_processed / "shutdown_ramp.npy")
    min_up = np.load(path_folder_processed / "min_up.npy")
    min_down = np.load(path_folder_processed / "min_down.npy")
    cost_lin = np.load(path_folder_processed / "cost_lin.npy")
    cost_const = np.load(path_folder_processed / "cost_const.npy")
    cost_startup_step_formulation_0 = pickle.load(open(path_folder_processed / "cost_startup_step_formulation_0.pkl", "rb"))
    num_cooling_steps_formulation_0 = np.load(path_folder_processed / "num_cooling_steps_formulation_0.npy")
    cost_startup_step_formulation_1 = np.load(path_folder_processed / "cost_startup_step_formulation_1.npy")
    step_length_formulation_1 = np.load(path_folder_processed / "step_length_formulation_1.npy")
    demand_2022 = np.load(path_folder_processed / "demand_2022.npy")
    demand = demand_2022[idx_time_start:idx_time_end+1]
    # mustoff_2022 = np.load(path_folder_processed / "mustoff_2022.npy")
    renewable_2022 = np.load(path_folder_processed / "renewable_2022.npy")
    renewable = renewable_2022[idx_time_start:idx_time_end+1]
    status_2022 = np.load(path_folder_processed / "status_2022.npy")
    u_prev = status_2022[:, idx_time_start-num_cooling_steps_formulation_0.max():idx_time_start]
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
    def _tail_count(seq, value):
        cnt = 0
        for x in reversed(seq):
            if x == value:
                cnt += 1
            else:
                break
        return cnt

    min_up_0 = np.array(
        [_tail_count(u_prev[g], 1) for g in range(num_units)], dtype=np.int64
    )
    min_down_0 = np.array(
        [_tail_count(u_prev[g], 0) for g in range(num_units)], dtype=np.int64
    )
    idx_u_prev_up = u_prev[:, -1].astype(bool)
    p_prev = np.zeros(num_units)
    p_prev[idx_u_prev_up] = p_min[idx_u_prev_up]

    return Input_uc(
        unit_type=unit_type,
        idx_nuclear=idx_nuclear,
        idx_coal=idx_coal,
        idx_lng=idx_lng,
        num_units=num_units,
        num_periods=num_periods,
        #########################
        demand=demand,
        reserve=(demand - renewable) * 0.1,
        renewable=renewable,
        #########################
        p_min=p_min,
        p_max=p_max,
        ramp_up=ramp_up,
        ramp_down=ramp_down,
        startup_ramp=startup_ramp,
        shutdown_ramp=shutdown_ramp,
        min_up=min_up,
        min_down=min_down,
        #########################
        cost_lin=cost_lin,
        cost_const=cost_const,
        #########################
        cost_startup_step_formulation_0=cost_startup_step_formulation_0,
        num_cooling_steps_formulation_0=num_cooling_steps_formulation_0,
        cost_startup_step_formulation_1=cost_startup_step_formulation_1,
        step_length_formulation_1=step_length_formulation_1,
        #########################
        p_prev=p_prev,
        u_prev=u_prev,
        min_up_r=min_up_r,
        min_down_r=min_down_r,
        min_up_0=min_up_0,
        min_down_0=min_down_0,
    )

def main():
    import argparse
    xd = argparse.ArgumentParser()
    xd.add_argument("--r", type=float, required=True)
    arg = xd.parse_args()
    

    output_uc = Output_uc()
    solve_uc_formulation_1(input_uc=asdf(), output_uc=output_uc, verbose=False, reserve=float(arg.r))


main()