import gc
import numpy as np
import gurobipy as gp

from .input import Input_uc, Input_suc


def solve_suc(
    input_uc: Input_uc,
    input_suc: Input_suc,
    verbose: bool = False,
):
    ### ATTRIBUTE LOCALIZATION & GUROBIPY-FRIENDLY TYPE CONVERSION
    # stochastic uc-related
    voll = input_suc.voll # int
    demand_fore = input_suc.demand_fore.tolist() # (num_periods,)
    renewable_fore = input_suc.renewable_fore.tolist() # (num_periods,)
    num_scenarios = input_suc.num_scenarios # int
    demand_scenario = input_suc.demand_scenario.tolist() # (num_scenarios, ) # idx paired with renewable distr.
    renewable_scenario = input_suc.renewable_scenario.tolist() # (num_scenarios, ) # idx paired with renewable distr.
    # meta
    num_units = input_uc.num_units
    num_periods = input_uc.num_periods
    # generator
    p_min = input_uc.p_min.tolist()
    p_max = input_uc.p_max.tolist()
    ramp_up = input_uc.ramp_up.tolist()
    ramp_down = input_uc.ramp_down.tolist()
    startup_ramp = input_uc.startup_ramp.tolist()
    shutdown_ramp = input_uc.shutdown_ramp.tolist()
    min_up = input_uc.min_up.tolist()
    min_down = input_uc.min_down.tolist()
    # cost function - generation
    cost_lin = input_uc.cost_lin.tolist()
    cost_const = input_uc.cost_const.tolist()
    # cost function - startup
    cost_startup_step = input_uc.cost_startup_step.tolist()
    step_length = input_uc.step_length.tolist()
    # initial conditions
    p_tight_prev = (input_uc.p_prev - input_uc.p_min * input_uc.u_prev[:, -1]).tolist()
    u_prev = input_uc.u_prev.tolist()
    min_up_r = input_uc.min_up_r.tolist()
    min_down_r = input_uc.min_down_r.tolist()
    min_down_0 = input_uc.min_down_0.tolist()

    ### MODEL DECLARATION
    model = gp.Model()
    model.setParam("OutputFlag", verbose)
    model.setParam("Symmetry", 2)
    model.setParam("PreDual", 2)
    model.setParam("Presolve", 1)
    model.setParam("PreSparsify", 2)
    model.setParam("Disconnected", 2)
    model.setParam("Heuristics", 1)
    model.setParam("ProjImpliedCuts", 2)

    ### VARIABLE DECLARATION
    # helper - pseudo ub
    u_lb_pseudo = np.zeros((num_units, num_periods, num_scenarios), dtype=np.int64)
    u_ub_pseudo = np.ones((num_units, num_periods, num_scenarios), dtype=np.int64)
    for i in range(num_units):
        # CONSTRAINT - MINIMUM UP-TIME
        for t in range(min_up_r[i]):
            for scen in range(num_scenarios):
                u_lb_pseudo[i, t,  scen] = 1
        # CONSTRAINT - MINIMUM DOWN-TIME
        for t in range(min_down_r[i]):
            for scen in range(num_scenarios):
                u_ub_pseudo[i, t], scen = 0

    p_tight_ub_pseudo = np.tile(np.array(input_uc.p_max - input_uc.p_min)[:, None], reps=num_periods) # i  need to like add one more dimeision here too the scen dimension omg
    r_ub_pseudo = p_tight_ub_pseudo.copy()

    # decision variables
    p_tight = model.addVars(range(num_units), range(num_periods), range(num_scenarios), lb=0, ub=p_tight_ub_pseudo.tolist())
    r = model.addVars(range(num_units), range(num_periods), range(num_scenarios), lb=0, ub=r_ub_pseudo.tolist())
    # if not _is_inside_iter:
    u = model.addVars(range(num_units), range(num_periods), range(num_scenarios), vtype=gp.GRB.BINARY, lb=u_lb_pseudo, ub=u_ub_pseudo)
    v = model.addVars(range(num_units), range(num_periods), range(num_scenarios), vtype=gp.GRB.BINARY)
    w = model.addVars(range(num_units), range(num_periods), range(num_scenarios), vtype=gp.GRB.BINARY)
    delta = model.addVars(range(num_units), range(num_periods), range(3), range(num_scenarios), vtype=gp.GRB.BINARY)
    # else:
    # helper - deletion
    del p_tight_ub_pseudo, r_ub_pseudo, u_lb_pseudo, u_ub_pseudo
    gc.collect()

    ### CONSTRAINTS
    # helper - negative index (initial condition) access function
    def p_tight_minus_proof(i, t_, scen):
        return p_tight[i, t_, scen] if t_ >= 0 else p_tight_prev[i]
    def u_minus_proof(i, t_, scen):
        return u[i, t_, scen] if t_ >= 0 else u_prev[i][t_]
    # helper - startup cost indices precomputation
    T_SU = []
    for i in range(num_units):
        starts = [min_down_0[i]]
        for L in step_length[i][:-1]:
            starts.append(starts[-1] + L)
        T_SU.append(starts)

    #
    #
    #
    #
    #
    #




    ### OBJECTIVE
    # 
    total_cost_generation = gp.quicksum(
        cost_lin[i] * (p_tight[i, t, scen] + p_min[i] * u[i, t, scen]) 
        + cost_const[i] * u[i, t, scen]
        for i in range(num_units)
        for t in range(num_periods)
        for scen in range(num_scenarios)
    )
    #
    total_cost_startup = gp.quicksum(
        cost_startup_step[i][s] * delta[i, t, s, scen]
        for i in range(num_units)
        for t in range(num_periods)
        for s in range(3)
        for scen in range(num_scenarios)
    )
    #
    total_cost_nse = gp.quicksum(
        voll * nse[t, scen]
        for t in range(num_periods)
        for scen in range(num_scenarios)
    )
    total_cost = total_cost_generation + total_cost_startup + total_cost_nse
    model.setObjective(total_cost, gp.GRB.MINIMIZE)
    model.optimize()

    # output registration + getting dual by resolving with fixed binaries we do this later after all is finished in this code