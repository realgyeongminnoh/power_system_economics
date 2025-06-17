import gc
import numpy as np
import gurobipy as gp

from .input import Input_uc
from .output import Output_uc


def solve_uc(
    input_uc: Input_uc,
    output_uc: Output_uc,
    verbose: bool = False,
    _is_inside_iter: bool = False,
    # fun experiment
    turn_off_nuclear_reserve: bool = False, 
    turn_off_coal_reserve: bool = False, 
    turn_off_lng_reserve: bool = False,
):
    """
    formulation from Tight and Compact MILP Formulation for the Thermal Unit Commitment Problem
    one-time recursive function call for obtaining dual values
    startup cost cooling time segments fixed to 3
    i did try to make adjustment so that 0 startup cost are not fed into the model
    but it somehow increased compute time so im not looking back at it

    input warning:
    u_prev should be column vector at least (no row vector with shape of (num_units,) but (num_units, 1))
    only the latest hour in u_prev will be accessed
    minimum up and down times must be greater than 1 (for KPG gen. dataset it does)
    """

    ### ATTRIBUTE LOCALIZATION & GUROBIPY-FRIENDLY TYPE CONVERSION
    # meta
    num_units = input_uc.num_units
    num_periods = input_uc.num_periods
    idx_nuclear = input_uc.idx_nuclear
    idx_coal = input_uc.idx_coal
    idx_lng = input_uc.idx_lng
    # system
    demand = input_uc.demand.tolist()
    reserve = input_uc.reserve.tolist()
    renewable = input_uc.renewable.tolist()
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
    model = gp.Model() # below param can ensure better (smaller) objval than no param. the change in objval from default is >0.01% though
    model.setParam("OutputFlag", verbose)
    model.setParam("Symmetry", 2) # okay but its 100% safe at least
    model.setParam("PreDual", 2) # 100% safe and v good
    model.setParam("Presolve", 1) # speed boost v good but obj value increase; this is eaten up by the others
    model.setParam("PreSparsify", 2) # v good with objval decrease
    model.setParam("Disconnected", 2) # okay and 100% safe at least
    model.setParam("Heuristics", 1) # good probably idk at this point; gambling its good 51% times (in speed); objval improvement 100% + feasibility
    model.setParam("ProjImpliedCuts", 2) # gambling 77% ; extremely v good for only large reserve big problem
    # model.setParam("MIPGap", 0)
    # model.setParam("MIPGapAbs", 0)
    # model.setParam("IntFeasTol", 1e-9)
    # model.setParam("FeasibilityTol", 1e-9)
    # model.setParam("OptimalityTol", 1e-9)

    ### VARIABLE DECLARATION
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

    p_tight_ub_pseudo = np.tile(np.array(input_uc.p_max - input_uc.p_min)[:, None], reps=num_periods)
    r_ub_pseudo = p_tight_ub_pseudo.copy()
    if turn_off_nuclear_reserve:
        r_ub_pseudo[idx_nuclear, :] = 0
    if turn_off_coal_reserve:
        r_ub_pseudo[idx_coal, :] = 0
    if turn_off_lng_reserve:
        r_ub_pseudo[idx_lng, :] = 0

    # decision variables
    p_tight = model.addVars(range(num_units), range(num_periods), lb=0, ub=p_tight_ub_pseudo.tolist())
    r = model.addVars(range(num_units), range(num_periods), lb=0, ub=r_ub_pseudo.tolist())
    if not _is_inside_iter:
        u = model.addVars(range(num_units), range(num_periods), vtype=gp.GRB.BINARY, lb=u_lb_pseudo, ub=u_ub_pseudo)
        v = model.addVars(range(num_units), range(num_periods), vtype=gp.GRB.BINARY)
        w = model.addVars(range(num_units), range(num_periods), vtype=gp.GRB.BINARY)
        delta = model.addVars(range(num_units), range(num_periods), range(3), vtype=gp.GRB.BINARY)
    # binary decision variables fix for marginal price computation
    else:
        u = gp.tupledict({(i, t): int(output_uc.u[i, t]) for i in range(num_units) for t in range(num_periods)})
        v = gp.tupledict({(i, t): int(output_uc.v[i, t]) for i in range(num_units) for t in range(num_periods)})
        w = gp.tupledict({(i, t): int(output_uc.w[i, t]) for i in range(num_units) for t in range(num_periods)})
        delta = gp.tupledict({(i, t, s): int(output_uc.delta[i, t, s]) for i in range(num_units) for t in range(num_periods) for s in range(3)})
    # helper - deletion
    del p_tight_ub_pseudo, r_ub_pseudo, u_lb_pseudo, u_ub_pseudo
    gc.collect()

    ### CONSTRAINTS
    # helper - negative index (initial condition) access function
    def p_tight_minus_proof(i, t_):
        return p_tight[i, t_] if t_ >= 0 else p_tight_prev[i]
    def u_minus_proof(i, t_):
        return u[i, t_] if t_ >= 0 else u_prev[i][t_]
    # helper - startup cost indices precomputation
    T_SU = []
    for i in range(num_units):
        starts = [min_down_0[i]]
        for L in step_length[i][:-1]:
            starts.append(starts[-1] + L)
        T_SU.append(starts)

    # 
    model.addConstrs(
        p_tight[i, t]
        <=
        u[i, t] * (p_max[i] - p_min[i])
        for i in range(num_units)
        for t in range(num_periods)
    )
    #
    model.addConstrs(
        p_tight[i, t] + r[i, t]
        <=
        u[i, t] * (p_max[i] - p_min[i])
        for i in range(num_units)
        for t in range(num_periods)
    )
    # 
    constr_generation = model.addConstrs(
        gp.quicksum(
            u[i, t] * p_min[i] 
            +
            p_tight[i, t]
            for i in range(num_units)
        )
        + renewable[t]
        ==
        demand[t]
        for t in range(num_periods)
    )
    #
    if not _is_inside_iter:
        constr_reserve = model.addConstrs(
            gp.quicksum(
                r[i, t]
                for i in range(num_units)
            )
            >=
            reserve[t]
            for t in range(num_periods)
        )
    else:
        output_uc_reserve = output_uc.reserve.tolist() # first run result is used (reserve \geq reserve reg. but its slack so no reserve cost)
        constr_reserve = model.addConstrs(
            gp.quicksum(
                r[i, t]
                for i in range(num_units)
            )
            ==
            output_uc_reserve[t]
            for t in range(num_periods)
        )
    #
    if not _is_inside_iter:
        #
        model.addConstrs(
            u[i, t] - u_minus_proof(i, t - 1)
            ==
            v[i, t] - w[i, t]
            for i in range(num_units)
            for t in range(num_periods)
        )
        #
        model.addConstrs(
            delta[i, t, s]
            <= 
            v[i, t]
            for i in range(num_units)
            for t in range(num_periods)
            for s in range(3)
        )
        #
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
    #
    if not _is_inside_iter:
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
    # RAMP
    model.addConstrs(
        p_tight[i, t] + r[i, t]
        <=
        u[i, t] * (p_max[i] - p_min[i])
        - v[i, t] * (p_max[i] - startup_ramp[i])
        - w[i, t + 1] * (p_max[i] - shutdown_ramp[i])
        for i in range(num_units)
        for t in range(num_periods - 1)
    )
    # RAMP
    model.addConstrs(
        p_tight[i, t] + r[i, t]
        <=
        u[i, t] * (p_max[i] - p_min[i])
        - v[i, t] * (p_max[i] - startup_ramp[i])
        for i in range(num_units)
        for t in range(num_periods - 1, num_periods)
    )
    # RAMP
    model.addConstrs(
        p_tight[i, t] + r[i, t] - p_tight_minus_proof(i, t - 1)
        <=
        ramp_up[i]
        for i in range(num_units)
        for t in range(num_periods)
    )
    # RAMP
    model.addConstrs(
        - p_tight[i, t] + p_tight_minus_proof(i, t - 1)
        <=
        ramp_down[i]
        for i in range(num_units)
        for t in range(num_periods)
    )
    #
    if not _is_inside_iter:
        # (6)
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
        # (7)
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

    ### OBJECTIVE
    # 
    total_cost_generation = gp.quicksum(
        cost_lin[i] * (p_tight[i, t] + p_min[i] * u[i, t]) 
        + cost_const[i] * u[i, t]
        for i in range(num_units)
        for t in range(num_periods)
    )
    #
    total_cost_startup = gp.quicksum(
        cost_startup_step[i][s] * delta[i, t, s]
        for i in range(num_units)
        for t in range(num_periods)
        for s in range(3)
    )
    
    total_cost = total_cost_generation + total_cost_startup
    model.setObjective(total_cost, gp.GRB.MINIMIZE)
    model.optimize()

    ### OUTPUT & RE-RUN FOR MARGINAL PRICES
    #
    if model.Status != gp.GRB.OPTIMAL:
        raise NotImplementedError(f"{model.Status}")
    #
    if not _is_inside_iter:
        output_uc.u = np.array(model.getAttr("X", u).select()).reshape(num_units, num_periods).astype(np.int64)
        p_tight = np.array(model.getAttr("X", p_tight).select()).reshape(num_units, num_periods)
        output_uc.p = (output_uc.u.transpose() * input_uc.p_min + p_tight.transpose()).transpose()
        output_uc.r = np.array(model.getAttr("X", r).select()).reshape(num_units, num_periods)
        output_uc.v = np.array(model.getAttr("X", v).select()).reshape(num_units, num_periods).astype(np.int64)
        output_uc.w = np.array(model.getAttr("X", w).select()).reshape(num_units, num_periods).astype(np.int64)
        output_uc.delta = np.array(model.getAttr("X", delta).select()).reshape(num_units, num_periods, 3).astype(np.int64)
    
        output_uc.generation = output_uc.p.sum(axis=0)
        output_uc.reserve = output_uc.r.sum(axis=0)

        output_uc.objval = total_cost.getValue()
        output_uc.total_cost = total_cost.getValue()

        output_uc.cost_generation = (output_uc.p.transpose() * input_uc.cost_lin + output_uc.u.transpose() * input_uc.cost_const).transpose().sum(axis=0)
        output_uc.total_cost_generation = total_cost_generation.getValue()
    
        output_uc.cost_startup = np.array([(output_uc.delta[:, t, :] * input_uc.cost_startup_step).sum() for t in range(num_periods)])
        output_uc.total_cost_startup = total_cost_startup.getValue()

        del model
        gc.collect()
        solve_uc(
            input_uc=input_uc, output_uc=output_uc, _is_inside_iter=True,
            turn_off_nuclear_reserve=turn_off_nuclear_reserve, 
            turn_off_coal_reserve=turn_off_coal_reserve, 
            turn_off_lng_reserve=turn_off_lng_reserve,
        )
    #
    else:
        output_uc.marginal_price_generation = np.array([constr_generation[t].Pi for t in range(num_periods)])
        output_uc.cost_retailer = (output_uc.p * output_uc.marginal_price_generation).sum(axis=0)
        output_uc.total_cost_retailer = float(output_uc.cost_retailer.sum())

        output_uc.marginal_price_reserve = np.array([constr_reserve[t].Pi for t in range(num_periods)])
        output_uc.cost_reserve = (output_uc.r * output_uc.marginal_price_reserve).sum(axis=0)
        output_uc.total_cost_reserve = float(output_uc.cost_reserve.sum())
        
        output_uc.cost = output_uc.cost_generation + output_uc.cost_startup + output_uc.cost_reserve
        output_uc.total_cost += output_uc.total_cost_reserve

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


def solve_uc_old(
    input_uc: Input_uc,
    output_uc: Output_uc,
    verbose: bool = False,
    _is_inside_iter: bool = False,
):
    """
    formulation from the lecture note

    input warning:
    cost_startup_step_old (list of list) substitutes cost_startup_step_old
    num_cooling_steps_old substitutes step_length
    """

    ### ATTRIBUTE LOCALIZATION & GUROBIPY-FRIENDLY TYPE CONVERSION
    # meta
    num_units = input_uc.num_units
    num_periods = input_uc.num_periods
    # system
    demand = input_uc.demand.tolist()
    reserve = input_uc.reserve.tolist()
    renewable = input_uc.renewable.tolist()
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
    cost_startup_step_old = input_uc.cost_startup_step_old
    num_cooling_steps_old = input_uc.num_cooling_steps_old.tolist()
    # initial conditions
    p_prev = input_uc.p_prev.tolist()
    u_prev = input_uc.u_prev.tolist()
    min_up_r = input_uc.min_up_r.tolist()
    min_down_r = input_uc.min_down_r.tolist()

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

    ###VARIABLE DECLARATION
    # helper - pseudo ub
    p_ub = np.tile(np.array(input_uc.p_max)[:, None], reps=input_uc.num_periods)
    r_ub = p_ub - np.tile(np.array(input_uc.p_min)[:, None], reps=input_uc.num_periods)
    cost_startup_ub = [[float(max(csc_i))] * input_uc.num_periods for csc_i in input_uc.cost_startup_step_old]
    # decision variables
    p = model.addVars(range(num_units), range(num_periods), lb=0, ub=p_ub.tolist())
    r = model.addVars(range(num_units), range(num_periods), lb=0, ub=r_ub.tolist())
    if not _is_inside_iter:
        u = model.addVars(range(num_units), range(num_periods), vtype=gp.GRB.BINARY)
        cost_startup = model.addVars(range(num_units), range(num_periods), lb=0, ub=cost_startup_ub)
    # binary decision variables fix for marginal price computation
    else:
        u = gp.tupledict({(i, t): int(output_uc.u[i, t]) for i in range(num_units) for t in range(num_periods)})
        cost_startup = gp.tupledict({(i, t): float(output_uc._cost_startup_it[i, t]) for i in range(num_units) for t in range(num_periods)})
    # helper - deletion
    del p_ub, r_ub, cost_startup_ub
    gc.collect()

    ### CONSTRAINTS
    # helper - negative index (initial condition) access function
    def p_minus_proof(i, t_):
        return p[i, t_] if t_ >= 0 else p_prev[i]
    def u_minus_proof(i, t_):
        return u[i, t_] if t_ >= 0 else u_prev[i][t_]
    
    #
    model.addConstrs(
        u[i, t] * p_min[i]
        <=
        p[i, t]
        for i in range(num_units)
        for t in range(num_periods)
    )
    #
    model.addConstrs(
        p[i, t] + r[i, t]
        <=
        u[i, t] * p_max[i]
        for i in range(num_units)
        for t in range(num_periods)
    )
    #
    model.addConstrs(
        r[i, t]
        <=
        u[i, t] * (p_max[i] - p_min[i])
        for i in range(num_units)
        for t in range(num_periods)
    )
    #
    constr_generation = model.addConstrs(
        gp.quicksum(
            p[i, t]
            for i in range(num_units)
        )
        + renewable[t]
        ==
        demand[t]
        for t in range(num_periods)
    )
    #
    if not _is_inside_iter:
        constr_reserve = model.addConstrs(
            gp.quicksum(
                r[i, t]
                for i in range(num_units)
            )
            >=
            reserve[t]
            for t in range(num_periods)
        )
    else:
        constr_reserve = model.addConstrs(
            gp.quicksum(
                r[i, t]
                for i in range(num_units)
            )
            ==
            reserve[t]
            for t in range(num_periods)
        )
    # STARTUP COST
    if not _is_inside_iter:
        model.addConstrs(
            cost_startup[i, t]
            >=
            cost_startup_step_old[i][tau - 1] * (
                u[i, t]
                -
                gp.quicksum(
                    u_minus_proof(i, t - t_delta)
                    for t_delta in range(1, tau + 1)
                )
            )
            for i in range(num_units)
            for t in range(num_periods)
            for tau in range(1, num_cooling_steps_old[i] + 1)
        )
    # RAMP
    model.addConstrs(
        p[i, t] + r[i, t]- p_minus_proof(i, t - 1)
        <=
        ramp_up[i] * u_minus_proof(i, t - 1)
        + startup_ramp[i] * (u[i, t] - u_minus_proof(i, t - 1))
        + p_max[i] * (1 - u[i, t])
        for i in range(num_units)
        for t in range(num_periods)        
    )
    # RAMP
    model.addConstrs(
        p[i, t] + r[i, t]
        <=
        p_max[i] * u[i, t + 1]
        + shutdown_ramp[i] * (u[i, t] - u[i, t + 1])
        for i in range(num_units)
        for t in range(num_periods - 1)
    )
    # RAMP
    model.addConstrs(
        p_minus_proof(i, t - 1) - p[i, t]
        <=
        ramp_down[i] * u[i, t]
        + shutdown_ramp[i] * (u_minus_proof(i, t - 1) - u[i, t])
        + p_max[i] * (1 - u_minus_proof(i, t - 1))
        for i in range(num_units)
        for t in range(num_periods - 1)        
    )
    # MINIMUM UP-TIME
    model.addConstrs(
        gp.quicksum(
            1 - u[i, t]
            for t in range(min_up_r[i])
        )
        ==
        0
        for i in range(num_units)
    )
    # MINIMUM UP-TIME
    model.addConstrs(
        gp.quicksum(
            u[i, t_delta]
            for t_delta in range(t, t + min_up[i])
        )
        >=
        min_up[i] * (
            u[i, t] - u_minus_proof(i, t - 1)
        )
        for i in range(num_units)
        for t in range(min_up_r[i], num_periods - min_up[i] + 1)
    )
    # MINIMUM UP-TIME
    model.addConstrs(
        gp.quicksum(
            u[i, t_delta]
            for t_delta in range(t, num_periods)
        )
        >=
        (num_periods - t) * (
            u[i, t] - u_minus_proof(i, t - 1)
        )
        for i in range(num_units)
        for t in range(num_periods - min_up[i] + 1, num_periods)
    )
    # MINIMUM DOWN-TIME
    model.addConstrs(
        gp.quicksum(
            u[i, t]
            for t in range(min_down_r[i])
        )
        ==
        0
        for i in range(num_units)
    )
    # MINIMUM DOWN-TIME
    model.addConstrs(
        gp.quicksum(
            1 - u[i, t_delta]
            for t_delta in range(t, t + min_down[i])
        )
        >=
        min_down[i] * (
            u_minus_proof(i, t - 1) - u[i, t]
        )
        for i in range(num_units)
        for t in range(min_down_r[i], num_periods - min_down[i] + 1)
    )
    # MINIMUM DOWN-TIME
    model.addConstrs(
        gp.quicksum(
            1 - u[i, t_delta]
            for t_delta in range(t, num_periods)
        )
        >=
        (num_periods - t) * (
            u_minus_proof(i, t - 1) - u[i, t]
        )
        for i in range(num_units)
        for t in range(num_periods - min_down[i] + 1, num_periods)
    )

    ### OBJECTIVE
    # 
    total_cost_generation = gp.quicksum(
        cost_lin[i] * p[i, t]
        + cost_const[i] * u[i, t]
        for i in range(num_units)
        for t in range(num_periods)
    )
    #
    total_cost_startup = gp.quicksum(
        cost_startup[i, t]
        for i in range(num_units)
        for t in range(num_periods)
    )
    #
    total_cost = total_cost_generation + total_cost_startup
    model.setObjective(total_cost, gp.GRB.MINIMIZE)
    model.optimize()

    ### OUTPUT & RE-RUN FOR MARGINAL PRICES
    #
    if model.Status != gp.GRB.OPTIMAL:
        raise NotImplementedError(f"{model.Status}")
    #
    if not _is_inside_iter:
        output_uc.u = np.array(model.getAttr("X", u).select()).reshape(num_units, num_periods).astype(np.int64)
        output_uc.p = np.array(model.getAttr("X", p).select()).reshape(num_units, num_periods)
        output_uc.r = np.array(model.getAttr("X", r).select()).reshape(num_units, num_periods)
        output_uc._cost_startup_it = np.array(model.getAttr("X", cost_startup).select()).reshape(num_units, num_periods)

        output_uc.generation = output_uc.p.sum(axis=0)
        output_uc.reserve = output_uc.r.sum(axis=0)

        output_uc.objval = total_cost.getValue()
        output_uc.total_cost = total_cost.getValue()

        output_uc.cost_generation = (output_uc.p.transpose() * input_uc.cost_lin + output_uc.u.transpose() * input_uc.cost_const).transpose().sum(axis=0)
        output_uc.total_cost_generation = total_cost_generation.getValue()

        output_uc.cost_startup = output_uc._cost_startup_it.sum(axis=0)
        output_uc.total_cost_startup = total_cost_startup.getValue()

        del model
        gc.collect()
        solve_uc_old(input_uc=input_uc, output_uc=output_uc, _is_inside_iter=True)
    #
    else:
        output_uc.marginal_price_generation = np.array([constr_generation[t].Pi for t in range(num_periods)])
        output_uc.cost_retailer = (output_uc.p * output_uc.marginal_price_generation).sum(axis=0)
        output_uc.total_cost_retailer = float(output_uc.cost_retailer.sum())

        output_uc.marginal_price_reserve = np.array([constr_reserve[t].Pi for t in range(num_periods)])
        output_uc.cost_reserve = (output_uc.r * output_uc.marginal_price_reserve).sum(axis=0)
        output_uc.total_cost_reserve = float(output_uc.cost_reserve.sum())
        
        output_uc.cost = output_uc.cost_generation + output_uc.cost_startup + output_uc.cost_reserve
        output_uc.total_cost += output_uc.total_cost_reserve