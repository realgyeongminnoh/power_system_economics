import pickle
from pathlib import Path
import numpy as np
import gurobipy as gp; gp.Model()

from .input import Input_uc, get_min_up_down_initials
from .output import Output_uc
from .unit_commitment import solve_uc 


# basically this is same as ipynb file Full UC (+ FULL ED) 
# global vars (2022 7 21 was a better day than my birthday because it was hotter and more solar)
num_units = 122
num_periods = 24
time_start, time_end = np.datetime64("2022-07-21T00"), np.datetime64("2022-07-21T23")


def load_data():
    path_folder_processed = Path(__file__).resolve().parents[1] / "data" / "input" / "processed"
    timestamp_2022 = np.load(path_folder_processed / "timestamp_2022.npy")
    idx_time_start, idx_time_end = np.where(timestamp_2022 == time_start)[0][0], np.where(timestamp_2022 == time_end)[0][0]

    global unit_type, idx_nuclear, idx_coal, idx_lng
    unit_type = np.load(path_folder_processed / "unit_type.npy")
    idx_nuclear = np.load(path_folder_processed / "idx_nuclear.npy")
    idx_coal = np.load(path_folder_processed / "idx_coal.npy")
    idx_lng = np.load(path_folder_processed / "idx_lng.npy")

    global p_min, p_max, ramp_up, ramp_down, startup_ramp, shutdown_ramp, min_up, min_down
    p_min = np.load(path_folder_processed / "p_min.npy")
    p_max = np.load(path_folder_processed / "p_max.npy")
    ramp_up = np.load(path_folder_processed / "ramp_up.npy")
    ramp_down = np.load(path_folder_processed / "ramp_down.npy")
    startup_ramp = np.load(path_folder_processed / "startup_ramp.npy")
    shutdown_ramp = np.load(path_folder_processed / "shutdown_ramp.npy")
    min_up = np.load(path_folder_processed / "min_up.npy")
    min_down = np.load(path_folder_processed / "min_down.npy")

    global cost_lin, cost_const, cost_startup_step_old, num_cooling_steps_old, cost_startup_step, step_length
    cost_lin = np.load(path_folder_processed / "cost_lin.npy")
    cost_const = np.load(path_folder_processed / "cost_const.npy")
    cost_startup_step_old = pickle.load(open(path_folder_processed / "cost_startup_step_old.pkl", "rb"))
    num_cooling_steps_old = np.load(path_folder_processed / "num_cooling_steps_old.npy")
    cost_startup_step = np.load(path_folder_processed / "cost_startup_step.npy")
    step_length = np.load(path_folder_processed / "step_length.npy")

    global demand, renewable
    demand = np.load(path_folder_processed / "demand_2022.npy")[idx_time_start:idx_time_end+1]
    renewable = np.load(path_folder_processed / "renewable_gen_2022.npy")[idx_time_start:idx_time_end+1]



def get_initial_conditions(reserve_margin_ic: float, verbose: bool):
    """
    issue:
    KPG commitment decision data reliance (which i have 0 reason to do so)
    # status_2022 = np.load(path_folder_processed / "status_2022.npy")
    # u_prev_ic = status_2022[:, idx_time_start-num_cooling_steps_old.max():idx_time_start]
    there are just too many unjustifiable modeling design / logic flaws if i use KPG renewables (data.py) not that the data itself is an issue

    solution: 
    black start -96h (= 2 * 48 = 2 * max cooling length) hour
    guaranteed steady state for initial conditions
    at t=-1 (or -48h ~ -1h) (unless some inhumane numerical stability stuffs)
    currently unsure about setting reserve margin requirement even before including NSE
    idk comment may not be fixed but code may be in future too busy rn
    """
    num_periods_ic = 60
    p_prev_ic = np.zeros(num_units)
    u_prev_ic = np.zeros((num_units, num_periods_ic), dtype=np.int64)
    min_up_r_ic, min_down_r_ic, min_up_0_ic, min_down_0_ic = get_min_up_down_initials(num_units, min_up, min_down, u_prev_ic)
    demand_ic = np.full((num_periods_ic), demand[0])
    renewable_ic = np.full((num_periods_ic), renewable[0])
    reserve_ic = (demand_ic - renewable_ic) * reserve_margin_ic

    input_uc_ic = Input_uc(
        # meta
        unit_type=unit_type,
        idx_nuclear=idx_nuclear,
        idx_coal=idx_coal,
        idx_lng=idx_lng,
        num_units=num_units,
        num_periods=num_periods_ic,
        # system
        demand=demand_ic,
        reserve=reserve_ic,
        renewable=renewable_ic,
        # generator
        p_min=p_min,
        p_max=p_max,
        ramp_up=ramp_up,
        ramp_down=ramp_down,
        startup_ramp=startup_ramp,
        shutdown_ramp=shutdown_ramp,
        min_up=min_up,
        min_down=min_down,
        # cost function - generation
        cost_lin=cost_lin,
        cost_const=cost_const,
        # cost function - startup (paper formulation)
        cost_startup_step=cost_startup_step,
        step_length=step_length,
        # cost function - startup (lecture note formulation)
        cost_startup_step_old=cost_startup_step_old,
        num_cooling_steps_old=num_cooling_steps_old,
        # initial conditions
        p_prev=p_prev_ic,
        u_prev=u_prev_ic,
        min_up_r=min_up_r_ic,
        min_down_r=min_down_r_ic,
        min_up_0=min_up_0_ic,
        min_down_0=min_down_0_ic,
    )

    output_uc_ic = Output_uc()
    solve_uc(
        input_uc=input_uc_ic, output_uc=output_uc_ic, verbose=verbose,
    )
    
    global p_prev, u_prev, min_up_r, min_down_r, min_up_0, min_down_0
    p_prev = output_uc_ic.p[:, -1]
    u_prev = output_uc_ic.u[:, -48:]
    min_up_r, min_down_r, min_up_0, min_down_0 = get_min_up_down_initials(num_units, min_up, min_down, u_prev)

    return input_uc_ic, output_uc_ic


def compute(reserve_margin: float, verbose: bool):
    """
    target 24 hour horizon UC computation (quadruple binary decision variable formulation)
    """

    input_uc = Input_uc(
        # meta
        unit_type=unit_type,
        idx_nuclear=idx_nuclear,
        idx_coal=idx_coal,
        idx_lng=idx_lng,
        num_units=num_units,
        num_periods=num_periods,
        # system
        demand=demand,
        reserve=(demand - renewable) * reserve_margin,
        renewable=renewable,
        # generator
        p_min=p_min,
        p_max=p_max,
        ramp_up=ramp_up,
        ramp_down=ramp_down,
        startup_ramp=startup_ramp,
        shutdown_ramp=shutdown_ramp,
        min_up=min_up,
        min_down=min_down,
        # cost function - generation
        cost_lin=cost_lin,
        cost_const=cost_const,
        # cost function - startup (paper formulation)
        cost_startup_step=cost_startup_step,
        step_length=step_length,
        # cost function - startup (lecture note formulation)
        cost_startup_step_old=cost_startup_step_old,
        num_cooling_steps_old=num_cooling_steps_old,
        # initial conditions
        p_prev=p_prev,
        u_prev=u_prev,
        min_up_r=min_up_r,
        min_down_r=min_down_r,
        min_up_0=min_up_0,
        min_down_0=min_down_0,
    )

    output_uc = Output_uc()
    solve_uc(
        input_uc=input_uc, output_uc=output_uc, verbose=verbose, 
    )
    return input_uc, output_uc


def run_uc(reserve_margin: float, reserve_margin_ic: float = None, return_ic: bool = False, verbose: bool = True):
    reserve_margin = float(reserve_margin)
    reserve_margin_ic = reserve_margin if reserve_margin_ic is None else float(reserve_margin_ic)

    load_data()
    input_uc_ic, output_uc_ic = get_initial_conditions(reserve_margin_ic, verbose=verbose)
    input_uc, output_uc = compute(reserve_margin, verbose=verbose)
    if return_ic:
        return input_uc_ic, output_uc_ic, input_uc, output_uc
    return input_uc, output_uc, demand, renewable