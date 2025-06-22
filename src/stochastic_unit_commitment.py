import gc
import numpy as np
import gurobipy as gp

from .input import Input_uc, Input_suc
from .output import Output_suc


def solve_suc(
    input_uc: Input_uc,
    input_suc: Input_suc,
    output_suc: Output_suc,
    verbose: bool = False,
):
    ############### ATTRIBUTE LOCALIZATION & GUROBIPY-FRIENDLY TYPE CONVERSION
    # stochastic uc-related
    fr_margin_pu = input_suc.fr_margin_pu # float
    voll = input_suc.voll # int
    demand_fore = input_suc.demand_fore.tolist() # (num_periods,)
    renewable_fore = input_suc.renewable_fore.tolist() # (num_periods,)
    num_scenarios = input_suc.num_scenarios # int
    thermal_demand_scenario = input_suc.thermal_demand_scenario.tolist() # (num_periods, num_scenarios)
    scenario_p_weight = input_suc.scenario_p_weight.tolist() # (num_scenarios,)
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

    ############### MODEL DECLARATION
    model = gp.Model()
    model.setParam("OutputFlag", verbose)
    model.setParam("Symmetry", 2)
    model.setParam("PreDual", 2)
    model.setParam("Presolve", 1)
    model.setParam("PreSparsify", 2)
    model.setParam("Disconnected", 2)
    model.setParam("Heuristics", 1)
    model.setParam("ProjImpliedCuts", 2)

    ############### VARIABLE DECLARATION
    # helper - pseudo ub
    u_lb_pseudo = np.zeros((num_units, num_periods), dtype=np.int64)
    u_ub_pseudo = np.ones((num_units, num_periods), dtype=np.int64)
    for i in range(num_units):
        # CONSTRAINT - MINIMUM UP-TIME
        for t in range(min_up_r[i]):
            u_lb_pseudo[i, t] = 1
        # CONSTRAINT - MINIMUM DOWN-TIME
        for t in range(min_down_r[i]):
            u_ub_pseudo[i, t] = 0

    r_ub_pseudo = np.tile(np.array(input_uc.p_max - input_uc.p_min)[:, None], reps=num_periods)
    p_tight_ub_pseudo = np.repeat(
        np.tile((input_uc.p_max - input_uc.p_min)[:, None], num_periods)[..., None],
        num_scenarios, axis=2
    )
    # decision variables
    p_tight = model.addVars(range(num_units), range(num_periods), range(num_scenarios), lb=0, ub=p_tight_ub_pseudo.tolist())
    r = model.addVars(range(num_units), range(num_periods), lb=0, ub=r_ub_pseudo.tolist())
    nse = model.addVars(range(num_periods), range(num_scenarios), lb=0, ub=1000) # simple 1GW continuous sheddable amount (1GW->3B kRW so thats kinda huge not gonna happen too)
    u = model.addVars(range(num_units), range(num_periods), vtype=gp.GRB.BINARY, lb=u_lb_pseudo, ub=u_ub_pseudo)
    v = model.addVars(range(num_units), range(num_periods), vtype=gp.GRB.BINARY)
    w = model.addVars(range(num_units), range(num_periods), vtype=gp.GRB.BINARY)
    delta = model.addVars(range(num_units), range(num_periods), range(3), vtype=gp.GRB.BINARY)
    error_thermal_demand = model.addVars(range(num_periods), range(num_scenarios), lb=0) # its precomputable
    n_minus_one = model.addVars(range(num_periods), lb=0, ub=max(p_max))
    # helper - deletion
    del p_tight_ub_pseudo, r_ub_pseudo, u_lb_pseudo, u_ub_pseudo
    gc.collect()

    ############### CONSTRAINTS
    # helper - negative index (initial condition) access function
    def p_tight_minus_proof(i, t_, k):
        return p_tight[i, t_, k] if t_ >= 0 else p_tight_prev[i]
    def u_minus_proof(i, t_):
        return u[i, t_] if t_ >= 0 else u_prev[i][t_]
    # helper - startup cost indices precomputation
    T_SU = []
    for i in range(num_units):
        starts = [min_down_0[i]]
        for L in step_length[i][:-1]:
            starts.append(starts[-1] + L)
        T_SU.append(starts)

    # TRUE UB (p_tight)
    model.addConstrs(
        p_tight[i, t, k]
        <=
        u[i, t] * (p_max[i] - p_min[i])
        for i in range(num_units)
        for t in range(num_periods)
        for k in range(num_scenarios)
    )
    # TRUE UB (p_tight + reserve)
    model.addConstrs(
        p_tight[i, t, k] + r[i, t]
        <=
        u[i, t] * (p_max[i] - p_min[i])
        for i in range(num_units)
        for t in range(num_periods)
        for k in range(num_scenarios)
    )
    # LOAD GENERATION BALANCE
    model.addConstrs(
        gp.quicksum(
            u[i, t] * p_min[i] 
            +
            p_tight[i, t, k]
            for i in range(num_units)
        )
        + nse[t, k]
        ==
        thermal_demand_scenario[t][k]
        for t in range(num_periods)
        for k in range(num_scenarios)
    )
    # MAX OPERATOR SUBSTITUTION (error between distribution points and forecast) (deterministic)
    model.addConstrs(
        error_thermal_demand[t, k]
        >=
        (thermal_demand_scenario[t][k])
        -
        (demand_fore[t] - renewable_fore[t])
        for t in range(num_periods)
        for k in range(num_scenarios)
    )
    # MAX OPERATOR SUBSTITUTION (NSE is the reside amount after above error is covered by reserve) (non-deterministic)
    model.addConstrs(
        nse[t, k]
        >=
        error_thermal_demand[t, k]
        -
        gp.quicksum(
            r[i, t]
            for i in range(num_units)
        )
        for t in range(num_periods)
        for k in range(num_scenarios)
    )
    # EXPLICIT RESERVE MINIMUM: N - 1 (~1.4GW) + FR MARGIN (thermal demand 2%) & 95% MAXIMUM OPERATING LEVEL (korea)
    # N - 1 RESERVE FIND
    model.addConstrs(
        n_minus_one[t]
        >=
        p_max[i] * u[i, t]
        for i in range(num_units)
        for t in range(num_periods)
    )
    # 95% MAXIMUM OPERATING LEVEL FOR ALL ON UNITS (MINIMUM REQUIREMENT; IT IS NOT SUPERPOSED WITH OTHER CONSTRAINTS)
    model.addConstrs(
        r[i, t]
        >=
        0.05 * p_max[i] * u[i, t]
        for i in range(num_units)
        for t in range(num_periods)
    )
    # FR MARGIN COMPUTE
    fr_margin = [fr_margin_pu* (demand_fore[t] - renewable_fore[t]) for t in range(num_periods)]
    # TOTAL EXPLICIT RESERVE MINIMUM (PSEUDO-DETERMINISTIC N-1 + DETERMINISTIC FRR; NOT SUPERPOSED WITH 95% MAX OP LEVEL OR THE FORECASTING ERROR SOLVER IS SOLVING)
    model.addConstrs(
        gp.quicksum(
            r[i, t]
            for i in range(num_units)
        )
        >=
        n_minus_one[t]
        + fr_margin[t]
        for t in range(num_periods)
    ) 
    # BINARY DECISION VARIABLES
    # STARTUP, SHUTDOWN BINARY VARIABLES RELATION TO COMMITMENT DECISION
    model.addConstrs(
        u[i, t] - u_minus_proof(i, t - 1)
        ==
        v[i, t] - w[i, t]
        for i in range(num_units)
        for t in range(num_periods)
    )
    # MAXIMUM SINGLE DELTA BEING ONE FOR EACH UNIT AND TIME
    model.addConstrs(
        gp.quicksum(
            delta[i, t, s]
            for s in range(3)
        )
        ==
        v[i, t]
        for i in range(num_units)
        for t in range(num_periods)
    )
    # STARTUP COST
    for i in range(num_units):
        for s in range(3):
            t0 = T_SU[i][s]
            t1 = T_SU[i][s+1] if s+1 < len(T_SU[i]) else num_periods

            for t in range(t1, num_periods):
                model.addConstr(
                    delta[i, t, s]
                    <= 
                    gp.quicksum(
                        w[i, t - lag]
                        for lag in range(t0, t1)
                    )
                )
    # RAMP 1
    model.addConstrs(
        p_tight[i, t, k] + r[i, t]
        <=
        u[i, t] * (p_max[i] - p_min[i])
        - v[i, t] * (p_max[i] - startup_ramp[i])
        - w[i, t + 1] * (p_max[i] - shutdown_ramp[i])
        for i in range(num_units)
        for t in range(num_periods - 1)
        for k in range(num_scenarios)
    )
    # RAMP 1
    model.addConstrs(
        p_tight[i, t, k] + r[i, t]
        <=
        u[i, t] * (p_max[i] - p_min[i])
        - v[i, t] * (p_max[i] - startup_ramp[i])
        for i in range(num_units)
        for t in range(num_periods - 1, num_periods)
        for k in range(num_scenarios)
    )
    # RAMP 2
    model.addConstrs(
        p_tight[i, t, k] + r[i, t] - p_tight_minus_proof(i, t - 1, k)
        <=
        ramp_up[i]
        for i in range(num_units)
        for t in range(num_periods)
        for k in range(num_scenarios)
    )
    # RAMP 3
    model.addConstrs(
        - p_tight[i, t, k] + p_tight_minus_proof(i, t - 1, k)
        <=
        ramp_down[i]
        for i in range(num_units)
        for t in range(num_periods)
        for k in range(num_scenarios)
    )
    # MINIMUM UP/DOWN TIME
    # MINIMUM UP TIME (6)
    model.addConstrs(
        gp.quicksum(
            v[i, tau]
            for tau in range(t - min_up[i] + 1, t + 1)
        )
        <=
        u[i, t]
        for i in range(num_units)
        for t in range(min_up[i] - 1, num_periods)
    )
    # MINIMUM DOWN TIME (7)
    model.addConstrs(
        gp.quicksum(
            w[i, tau]
            for tau in range(t - min_down[i] + 1, t + 1)
        )
        <=
        1 - u[i, t]
        for i in range(num_units)
        for t in range(min_down[i] - 1, num_periods)
    )

    ############### OBJECTIVE
    # TOTAL GENERATION (energy) COST
    total_cost_generation = gp.quicksum(
        scenario_p_weight[k]
        *
        (
            cost_lin[i] * (p_tight[i, t, k] + p_min[i] * u[i, t]) 
            + cost_const[i] * u[i, t]
        )
        for i in range(num_units)
        for t in range(num_periods)
        for k in range(num_scenarios)
    )
    # TOTAL STARTUP COST
    total_cost_startup = gp.quicksum(
        cost_startup_step[i][s] * delta[i, t, s]
        for i in range(num_units)
        for t in range(num_periods)
        for s in range(3)
    )
    # TOTAL NSE COST
    total_cost_nse = gp.quicksum(
        voll * nse[t, k] * scenario_p_weight[k]
        for t in range(num_periods)
        for k in range(num_scenarios)
    )
    # TOTAL COST
    total_cost = total_cost_generation + total_cost_startup + total_cost_nse
    model.setObjective(total_cost, gp.GRB.MINIMIZE)
    model.optimize()

    ############### OUTPUT & RE-RUN FOR MARGINAL PRICES
    #
    if model.Status != gp.GRB.OPTIMAL:
        raise NotImplementedError(f"{model.Status}")
    #
    output_suc.r = np.array(model.getAttr("X", r).select()).reshape(num_units, num_periods)
    #
    output_suc.nse_scen = np.array(model.getAttr("X", nse).select()).reshape(num_periods, num_scenarios)
    # 
    output_suc.u = np.array(model.getAttr("X", u).select()).reshape(num_units, num_periods).astype(np.int64)
    output_suc.v = np.array(model.getAttr("X", v).select()).reshape(num_units, num_periods).astype(np.int64)
    output_suc.w = np.array(model.getAttr("X", w).select()).reshape(num_units, num_periods).astype(np.int64)
    output_suc.delta = np.array(model.getAttr("X", delta).select()).reshape(num_units, num_periods, 3).astype(np.int64)
    #
    output_suc.error_thermal_demand = np.array(model.getAttr("X", error_thermal_demand).select()).reshape(num_periods, num_scenarios)
    output_suc.n_minus_one = np.array(model.getAttr("X", n_minus_one).select()).reshape(num_periods)

    p_tight = np.array(model.getAttr("X", p_tight).select()).reshape(num_units, num_periods, num_scenarios)
    output_suc.p_scen = (output_suc.u[:, :, None] * np.array(p_min)[:, None, None] + p_tight).reshape(num_units, num_periods, num_scenarios)

    output_suc.reserve = output_suc.r.sum(axis=0)
    output_suc.total_cost_nse_scen = total_cost_nse.getValue()


def solve_ed_after_suc(
    input_uc: Input_uc,
    input_suc: Input_suc,
    output_suc: Output_suc,
    verbose: bool = False,
):
    ############### ATTRIBUTE LOCALIZATION & GUROBIPY-FRIENDLY TYPE CONVERSION
    # stochastic uc-related
    voll = input_suc.voll # int
    # 
    #
    demand_fore = input_suc.demand_fore.tolist()
    renewable_fore = input_suc.renewable_fore.tolist()
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
    #
    r = gp.tupledict({(i, t): float(output_suc.r[i, t]) for i in range(num_units) for t in range(num_periods)})
    u = gp.tupledict({(i, t): int(output_suc.u[i, t]) for i in range(num_units) for t in range(num_periods)})
    v = gp.tupledict({(i, t): int(output_suc.v[i, t]) for i in range(num_units) for t in range(num_periods)})
    w = gp.tupledict({(i, t): int(output_suc.w[i, t]) for i in range(num_units) for t in range(num_periods)})
    delta = gp.tupledict({(i, t, s): int(output_suc.delta[i, t, s]) for i in range(num_units) for t in range(num_periods) for s in range(3)})


    ############### MODEL DECLARATION
    model = gp.Model()
    model.setParam("OutputFlag", verbose)
    model.setParam("Symmetry", 2)
    model.setParam("PreDual", 2)
    model.setParam("Presolve", 1)
    model.setParam("PreSparsify", 2)
    model.setParam("Disconnected", 2)
    model.setParam("Heuristics", 1)
    model.setParam("ProjImpliedCuts", 2)

    ############### VARIABLE DECLARATION
    # helper - pseudo ub
    p_tight_ub_pseudo = np.tile(np.array(input_uc.p_max - input_uc.p_min)[:, None], reps=num_periods)
    # decision variables
    p_tight = model.addVars(range(num_units), range(num_periods), lb=0, ub=p_tight_ub_pseudo.tolist())
    nse = model.addVars(range(num_periods), lb=0, ub=1000) # same as suc
    # helper - deletion
    del p_tight_ub_pseudo
    gc.collect()

    ### CONSTRAINTS
    # helper - negative index (initial condition) access function
    def p_tight_minus_proof(i, t_):
        return p_tight[i, t_] if t_ >= 0 else p_tight_prev[i]
    # helper - startup cost indices precomputation
    T_SU = []
    for i in range(num_units):
        starts = [min_down_0[i]]
        for L in step_length[i][:-1]:
            starts.append(starts[-1] + L)
        T_SU.append(starts)

    # TRUE UB (p_tight + reserve)
    model.addConstrs(
        p_tight[i, t] + r[i, t]
        <=
        u[i, t] * (p_max[i] - p_min[i])
        for i in range(num_units)
        for t in range(num_periods)
    )
    # LOAD GENERATION BALANCE
    constr_generation = model.addConstrs(
        gp.quicksum(
            u[i, t] * p_min[i] 
            +
            p_tight[i, t]
            for i in range(num_units)
        )
        + nse[t]
        ==
        demand_fore[t] - renewable_fore[t]
        for t in range(num_periods)
    )
    # STARTUP COST (ONLY FOR MEASUREMENT; all fixed)
    for i in range(num_units):
        for s in range(3):
            t0 = T_SU[i][s]
            t1 = T_SU[i][s+1] if s+1 < len(T_SU[i]) else num_periods

            for t in range(t1, num_periods):
                model.addConstr(
                    delta[i, t, s]
                    <= 
                    gp.quicksum(
                        w[i, t - lag]
                        for lag in range(t0, t1)
                    )
                )
    # RAMP 1
    model.addConstrs(
        p_tight[i, t] + r[i, t]
        <=
        u[i, t] * (p_max[i] - p_min[i])
        - v[i, t] * (p_max[i] - startup_ramp[i])
        - w[i, t + 1] * (p_max[i] - shutdown_ramp[i])
        for i in range(num_units)
        for t in range(num_periods - 1)
    )
    # RAMP 1
    model.addConstrs(
        p_tight[i, t] + r[i, t]
        <=
        u[i, t] * (p_max[i] - p_min[i])
        - v[i, t] * (p_max[i] - startup_ramp[i])
        for i in range(num_units)
        for t in range(num_periods - 1, num_periods)
    )
    # RAMP 2
    model.addConstrs(
        p_tight[i, t] + r[i, t] - p_tight_minus_proof(i, t - 1)
        <=
        ramp_up[i]
        for i in range(num_units)
        for t in range(num_periods)
    )
    # RAMP 3
    model.addConstrs(
        - p_tight[i, t] + p_tight_minus_proof(i, t - 1)
        <=
        ramp_down[i]
        for i in range(num_units)
        for t in range(num_periods)
    )

    ############### OBJECTIVE
    # TOTAL GENERATION (energy) COST
    total_cost_generation = gp.quicksum(
        cost_lin[i] * (p_tight[i, t] + p_min[i] * u[i, t]) 
        + cost_const[i] * u[i, t]
        for i in range(num_units)
        for t in range(num_periods)
    )
    # TOTAL STARTUP COST
    total_cost_startup = gp.quicksum(
        cost_startup_step[i][s] * delta[i, t, s]
        for i in range(num_units)
        for t in range(num_periods)
        for s in range(3)
    )
    # TOTAL NSE COST
    total_cost_nse = gp.quicksum(
        voll * nse[t]
        for t in range(num_periods)
    )
    # TOTAL COST
    total_cost = total_cost_generation + total_cost_startup + total_cost_nse
    model.setObjective(total_cost, gp.GRB.MINIMIZE)
    model.optimize()
    ############### OUTPUT & RE-RUN FOR MARGINAL PRICES
    #
    if model.Status != gp.GRB.OPTIMAL:
        raise NotImplementedError(f"{model.Status}")
    #
    p_tight = np.array(model.getAttr("X", p_tight).select()).reshape(num_units, num_periods)
    output_suc.p = (output_suc.u.transpose() * input_uc.p_min + p_tight.transpose()).transpose()
    output_suc.nse = np.array(model.getAttr("X", nse).select()).reshape(num_periods) 
    output_suc.generation = output_suc.p.sum(axis=0)
    # IM GONNA have my head split because of this output regstiration matching the duc one separated by scu and full ed
    output_suc.objval = total_cost.getValue()
    output_suc.total_cost = total_cost.getValue()  

    output_suc.cost_generation = (output_suc.p.transpose() * input_uc.cost_lin + output_suc.u.transpose() * input_uc.cost_const).transpose().sum(axis=0)
    output_suc.total_cost_generation = total_cost_generation.getValue()

    output_suc.cost_startup = np.array([(output_suc.delta[:, t, :] * input_uc.cost_startup_step).sum() for t in range(num_periods)])
    output_suc.total_cost_startup = total_cost_startup.getValue()

    output_suc.cost_nse = voll * output_suc.nse
    output_suc.total_cost_nse = total_cost_nse.getValue()

    output_suc.marginal_price_generation = np.array([constr_generation[t].Pi for t in range(num_periods)])
    output_suc.cost_retailer = (output_suc.p * output_suc.marginal_price_generation).sum(axis=0)
    output_suc.total_cost_retailer = float(output_suc.cost_retailer.sum())


    # ok since dual is not used for reserve price setup 
    # i asked my gpt to rewrite my code below 
    
        # # reserve price (numerical) validation: dual == kpx method with opportunity cost
        # # this is only true because i substituted avg. fuel cost with C1
        # arr_bool = []
        # for t in range(num_periods):
        #     opp_cost_temp = output_uc.marginal_price_generation[t] - input_uc.cost_lin
        #     opp_cost = np.array([max(0, opp_cost_i) for opp_cost_i in opp_cost_temp])
            
        #     indices_with_nonzero_r = np.where(output_uc.r[:, t] != 0)[0]
            
        #     if len(indices_with_nonzero_r) == 0:
        #         if output_uc.marginal_price_reserve[0] == 0:
        #             arr_bool.append(True) # append True if 0 == dual == kpx method with opportunity cost
        #     else:
        #         arr_bool.append(opp_cost[indices_with_nonzero_r[0]] == output_uc.marginal_price_reserve[t])

        # print(np.all(np.array(arr_bool)))

    # this was written by me and its correct because c1 all sorted ascending order     
    # the below code confirmed its correct. i thoughti had heart attack

    # i had my first heart attack when i saw reserve price is just shifted smp
    # but this is good because the "marginal reserve unit" is the very first one, cheapest nuclear unit.
    # thank god 
    
    output_suc.marginal_price_reserve = np.zeros(num_periods)
    output_suc.cost_reserve = np.zeros(num_periods)
    for t in range(num_periods):
        opp_cost = output_suc.marginal_price_generation[t] - np.array(input_uc.cost_lin)
        opp_cost = np.maximum(0, opp_cost)

        indices_with_nonzero_r = np.where(output_suc.r[:, t] > 0 )[0]

        if len(indices_with_nonzero_r) == 0:
            output_suc.marginal_price_reserve[t] = 0.0
        else:
            i_marginal = indices_with_nonzero_r[0]
            output_suc.marginal_price_reserve[t] = opp_cost[i_marginal]

        output_suc.cost_reserve[t] = np.sum(output_suc.r[:, t] * output_suc.marginal_price_reserve[t])

    output_suc.total_cost_reserve = float(np.sum(output_suc.cost_reserve))
    # 
    output_suc.cost = output_suc.cost_generation + output_suc.cost_startup + output_suc.cost_reserve
    output_suc.total_cost += output_suc.total_cost_reserve

    output_suc.reserve_percentage = output_suc.reserve / (input_suc.demand_fore - input_suc.renewable_fore) * 100