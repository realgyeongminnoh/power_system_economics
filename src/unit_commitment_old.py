import gc
import numpy as np
import gurobipy as gp

from .input import Input_uc
from .output import Output_uc


def solve_uc_formulation_0(
    input_uc: Input_uc,
    output_uc: Output_uc,
    verbose: bool = False,
    _is_inside_iter: bool = False,
):

    p_ub = np.tile(np.array(input_uc.p_max)[:, None], reps=input_uc.num_periods)
    # r_ub = p_ub - np.tile(np.array(input_uc.p_min)[:, None], reps=input_uc.num_periods)
    cost_startup_ub = [[float(max(csc_i))] * input_uc.num_periods for csc_i in input_uc.cost_startup_step]

    num_units = input_uc.num_units
    num_periods = input_uc.num_periods
    #########################
    demand = input_uc.demand.tolist()
    reserve = input_uc.reserve.tolist()
    renewable = input_uc.renewable.tolist()
    #########################
    p_min = input_uc.p_min.tolist()
    p_max = input_uc.p_max.tolist()
    ramp_up = input_uc.ramp_up.tolist()
    ramp_down = input_uc.ramp_down.tolist()
    startup_ramp = input_uc.startup_ramp.tolist()
    shutdown_ramp = input_uc.shutdown_ramp.tolist()
    min_up = input_uc.min_up.tolist()
    min_down = input_uc.min_down.tolist()
    #########################
    cost_lin = input_uc.cost_lin.tolist()
    cost_const = input_uc.cost_const.tolist()
    #########################
    cost_startup_step = input_uc.cost_startup_step.tolist()
    num_cooling_steps = input_uc.num_cooling_steps.tolist()
    #########################
    p_prev = input_uc.p_prev.tolist()
    u_prev = input_uc.u_prev.tolist()
    min_up_r = input_uc.min_up_r.tolist()
    min_down_r = input_uc.min_down_r.tolist()


    #
    model = gp.Model()
    model.setParam("OutputFlag", verbose)

    #
    p = model.addVars(range(num_units), range(num_periods), lb=0, ub=p_ub.tolist())

    # if not input_uc.example_1:
    #     r = model.addVars(range(num_units), range(num_periods), lb=0, ub=r_ub.tolist())
    # else:
    #     r = model.addVars(range(num_units), range(num_periods), lb=0, ub=np.tile(np.array([0, 160, 190, 0])[:, None], num_periods))

    if not _is_inside_iter:
        u = model.addVars(range(num_units), range(num_periods), vtype=gp.GRB.BINARY)
        cost_startup = model.addVars(range(num_units), range(num_periods), lb=0, ub=cost_startup_ub)
    else:
        u = gp.tupledict({(i, t): int(output_uc.u[i, t]) for i in range(num_units) for t in range(num_periods)})
        cost_startup = gp.tupledict({(i, t): float(output_uc.cost_startup[i, t]) for i in range(num_units) for t in range(num_periods)})

    del p_ub, cost_startup_ub #, r_ub
    gc.collect()

    # 
    def p_minus_proof(i, t_):
        return p[i, t_] if t_ >= 0 else p_prev[i]
    
    def u_minus_proof(i, t_):
        return u[i, t_] if t_ >= 0 else u_prev[i][t_]

    #
    model.addConstrs(
        p[i, t]
        >=
        u[i, t] * p_min[i]
        for i in range(num_units)
        for t in range(num_periods)
    )
    model.addConstrs(
        u[i, t] * p_max[i]
        >=
        p[i, t]
        # + r[i, t]
        for i in range(num_units)
        for t in range(num_periods)
    )
    # model.addConstrs(
    #     u[i, t] * (p_max[i] - p_min[i])
    #     >=
    #     r[i, t]
    #     for i in range(num_units)
    #     for t in range(num_periods)
    # )


    # # CONSTRAINTS - STARTUP COST
    # if not _is_inside_iter:
    #     model.addConstrs(
    #         cost_startup[i, t]
    #         >=
    #         cost_startup_step[i][tau - 1] * (
    #             u[i, t]
    #             -
    #             gp.quicksum(
    #                 u_minus_proof(i, t - t_delta)
    #                 for t_delta in range(1, tau + 1)
    #             )
    #         )
    #         for i in range(num_units)
    #         for t in range(num_periods)
    #         for tau in range(1, num_cooling_steps[i] + 1)
    #     )

    # CONSTRAINTS - DEMAND-GENERATION BALANCE
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

    # # CONSTRAINTS - SYSTEM RESERVE REQUIREMENT
    # constr_reserve = model.addConstrs(
    #     gp.quicksum(
    #         r[i, t]
    #         for i in range(num_units)
    #     )
    #     >=
    #     reserve[t]
    #     for t in range(num_periods)
    # )

###################################################################################
    # # CONSTRAINTS - POWER + RESERVE RESPECTING RAMP UP AND STARTUP RAMP
    # model.addConstrs(
    #     p[i, t] + r[i, t] - p_minus_proof(i, t - 1)
    #     <=
    #     ramp_up[i] * u_minus_proof(i, t - 1)
    #     + startup_ramp[i] * (u[i, t] - u_minus_proof(i, t - 1))
    #     + p_max[i] * (1 - u[i, t])
    #     for i in range(num_units)
    #     for t in range(num_periods)        
    # )
    # # CONSTRAINTS - POWER + RESERVE RESPECTING SHUT DOWN
    # model.addConstrs(
    #     p[i, t] + r[i, t]
    #     <=
    #     p_max[i] * u[i, t + 1]
    #     + shutdown_ramp[i] * (u[i, t] - u[i, t + 1])
    #     for i in range(num_units)
    #     for t in range(num_periods - 1)
    # )
    # # CONSTRAINTS - RAMP DOWN AND SHUTDOWN RAMP
    # model.addConstrs(
    #     p_minus_proof(i, t - 1) - p[i, t]
    #     <=
    #     ramp_down[i] * u[i, t]
    #     + shutdown_ramp[i] * (u_minus_proof(i, t - 1) - u[i, t])
    #     + p_max[i] * (1 - u_minus_proof(i, t - 1))
    #     for i in range(num_units)
    #     for t in range(num_periods - 1)        
    # )
###################################################################################
    # CONSTRAINTS - MINIMUM UP-TIME
    model.addConstrs(
        gp.quicksum(
            1 - u[i, t]
            for t in range(min_up_r[i])
        )
        ==
        0
        for i in range(num_units)
    )
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

    # CONSTRAINTS - MINIMUM DOWN-TIME
    model.addConstrs(
        gp.quicksum(
            u[i, t]
            for t in range(min_down_r[i])
        )
        ==
        0
        for i in range(num_units)
    )
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


    total_cost = total_cost_generation + total_cost_startup
    model.setObjective(total_cost, gp.GRB.MINIMIZE)
    model.optimize()

    if model.Status != gp.GRB.OPTIMAL:
        raise NotImplementedError(f"{model.Status}")

    if _is_inside_iter:
        output_uc.marginal_price_generation = np.array([constr_generation[t].Pi for t in range(num_periods)])
        # output_uc.marginal_price_reserve = np.array([constr_reserve[t].Pi for t in range(num_periods)])
        ###

    if not _is_inside_iter:
        output_uc.p = np.array(model.getAttr("X", p).select()).reshape(num_units, num_periods)
        # output_uc.r = np.array(model.getAttr("X", r).select()).reshape(num_units, num_periods)
        output_uc.u = np.array(model.getAttr("X", u).select()).reshape(num_units, num_periods).astype(np.int64)
        output_uc.cost_startup = np.array(model.getAttr("X", cost_startup).select()).reshape(num_units, num_periods)
        ###

        output_uc.total_cost = total_cost.getValue()
        output_uc.total_cost_generation = total_cost_generation.getValue()
        output_uc.total_cost_startup = total_cost_startup.getValue()
        ###

        del model
        gc.collect()
        solve_uc_formulation_0(input_uc=input_uc, output_uc=output_uc, _is_inside_iter=True)


def solve_uc_formulation_1(
    input_uc: Input_uc,
    output_uc: Output_uc,
    verbose: bool = False,
    _is_inside_iter: bool = False,
):
    
    ub_pseudo = np.tile(np.array(input_uc.p_max - input_uc.p_min)[:, None], reps=input_uc.num_periods)


    num_units = input_uc.num_units
    num_periods = input_uc.num_periods

    #########################
    demand = input_uc.demand.tolist()
    reserve = input_uc.reserve.tolist()
    renewable = input_uc.renewable.tolist()
    #########################
    p_min = input_uc.p_min.tolist()
    p_max = input_uc.p_max.tolist()
    ramp_up = input_uc.ramp_up.tolist()
    ramp_down = input_uc.ramp_down.tolist()
    startup_ramp = input_uc.startup_ramp.tolist()
    shutdown_ramp = input_uc.shutdown_ramp.tolist()
    min_up = input_uc.min_up.tolist()
    min_down = input_uc.min_down.tolist()
    #########################
    cost_lin = input_uc.cost_lin.tolist()
    cost_const = input_uc.cost_const.tolist()
    #########################
    p_tight_prev = input_uc.p_tight_prev.tolist()
    u_prev = input_uc.u_prev.tolist()
    min_up_r = input_uc.min_up_r.tolist()
    min_down_r = input_uc.min_down_r.tolist()


    ###############################

    model = gp.Model()
    model.setParam("OutputFlag", verbose)


    p_tight = model.addVars(range(num_units), range(num_periods), lb=0, ub=ub_pseudo)
    # r = model.addVars(range(num_units), range(num_periods), lb=0, ub=ub_pseudo)

    if not _is_inside_iter:
        u = model.addVars(range(num_units), range(num_periods), vtype=gp.GRB.BINARY)
        v = model.addVars(range(num_units), range(num_periods), vtype=gp.GRB.BINARY)
        w = model.addVars(range(num_units), range(num_periods), vtype=gp.GRB.BINARY)
    else:
        u = gp.tupledict({(i, t): int(output_uc.u[i, t]) for i in range(num_units) for t in range(num_periods)})
        v = gp.tupledict({(i, t): int(output_uc.v[i, t]) for i in range(num_units) for t in range(num_periods)})
        w = gp.tupledict({(i, t): int(output_uc.w[i, t]) for i in range(num_units) for t in range(num_periods)})

    del ub_pseudo
    gc.collect()

    def p_tight_minus_proof(i, t_):
        return p_tight[i, t_] if t_ >= 0 else p_tight_prev[i]
    
    def u_minus_proof(i, t_):
        return u[i, t_] if t_ >= 0 else u_prev[i][t_]

    ############################### p_tight , r and gen reserve blaance
    model.addConstrs(
        p_tight[i, t]
        <=
        u[i, t] * (p_max[i] - p_min[i])
        for i in range(num_units)
        for t in range(num_periods)
    )
    # model.addConstrs(
    #     p_tight[i, t] + r[i, t]
    #     <=
    #     u[i, t] * (p_max[i] - p_min[i])
    #     for i in range(num_units)
    #     for t in range(num_periods)
    # )
    # CONSTRAINTS - (4)
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
    # # CONSTRAINTS - (5)
    # constr_reserve = model.addConstrs(
    #     gp.quicksum(
    #         r[i, t]
    #         for i in range(num_units)
    #     )
    #     >=
    #     reserve[t]
    #     for t in range(num_periods)
    # )
###################################################################################
    # # CONSTRAINTS - (11)
    # model.addConstrs(
    #     p_tight[i, t] + r[i, t]
    #     <=
    #     u[i, t] * (p_max[i] - p_min[i])
    #     - v[i, t] * (p_max[i] - startup_ramp[i])
    #     - w[i, t + 1] * (p_max[i] - shutdown_ramp[i])
    #     for i in range(num_units)
    #     for t in range(num_periods - 1)
    # )
    # model.addConstrs(
    #     p_tight[i, t] + r[i, t]
    #     <=
    #     u[i, t] * (p_max[i] - p_min[i])
    #     - v[i, t] * (p_max[i] - startup_ramp[i])
    #     for i in range(num_units)
    #     for t in range(num_periods - 1, num_periods)
    # )
    # # CONSTRAINTS - (12)
    # model.addConstrs(
    #     p_tight[i, t] + r[i, t] - p_tight_minus_proof(i, t - 1)
    #     <=
    #     ramp_up[i]
    #     for i in range(num_units)
    #     for t in range(num_periods)
    # )
    # # CONSTRAINTS - (13)
    # model.addConstrs(
    #     - p_tight[i, t] + p_tight_minus_proof(i, t - 1)
    #     <=
    #     ramp_down[i]
    #     for i in range(num_units)
    #     for t in range(num_periods)
    # )
###################################################################################

    # CONSTRAINTS - U-V-W (8)
    model.addConstrs(
        u[i, t] - u_minus_proof(i, t - 1)
        ==
        v[i, t] - w[i, t]
        for i in range(num_units)
        for t in range(num_periods)
    )

    # CONSTRAINTS - MINIMUM UP-TIME INITIAL
    model.addConstrs(
        gp.quicksum(
            1 - u[i, t]
            for t in range(min_up_r[i])
        )
        ==
        0
        for i in range(num_units)
    )
    # CONSTRAINTS - MINIMUM DOWN-TIME INITIAL
    model.addConstrs(
        gp.quicksum(
            u[i, t]
            for t in range(min_down_r[i])
        )
        ==
        0
        for i in range(num_units)
    )

    # CONSTRAINTS - (6)
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
    # CONSTRAINTS - (7)
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


    #
    total_cost_generation = gp.quicksum(
        cost_lin[i] * (p_tight[i, t] + p_min[i] * u[i, t]) 
        + cost_const[i] * u[i, t]
        for i in range(num_units)
        for t in range(num_periods)
    )

    #
    # total_cost_startup = gp.quicksum(
    #     cost_startup[i, t]
    #     for i in range(num_units)
    #     for t in range(num_periods)
    # )


    total_cost = total_cost_generation # + total_cost_startup
    model.setObjective(total_cost, gp.GRB.MINIMIZE)
    model.optimize()

    if model.Status != gp.GRB.OPTIMAL:
        raise NotImplementedError(f"{model.Status}")

    if _is_inside_iter:
        output_uc.marginal_price_generation = np.array([constr_generation[t].Pi for t in range(num_periods)])
        # output_uc.marginal_price_reserve = np.array([constr_reserve[t].Pi for t in range(num_periods)])
        print(np.allclose(output_uc.p_tight, np.array(model.getAttr("X", p_tight).select()).reshape(num_units, num_periods), atol=1e-2))
        ###

    if not _is_inside_iter:
        output_uc.p_tight = np.array(model.getAttr("X", p_tight).select()).reshape(num_units, num_periods)
        # output_uc.r = np.array(model.getAttr("X", r).select()).reshape(num_units, num_periods)
        output_uc.u = np.array(model.getAttr("X", u).select()).reshape(num_units, num_periods).astype(np.int64)
        output_uc.v = np.array(model.getAttr("X", v).select()).reshape(num_units, num_periods).astype(np.int64)
        output_uc.w = np.array(model.getAttr("X", w).select()).reshape(num_units, num_periods).astype(np.int64)
        output_uc.p = (output_uc.u.transpose() * input_uc.p_min + output_uc.p_tight.transpose()).transpose()
        ###

        output_uc.total_cost = total_cost.getValue()
        output_uc.total_cost_generation = total_cost_generation.getValue()
        # output_uc.total_cost_startup = total_cost_startup.getValue()

        del model
        gc.collect()
        solve_uc_formulation_1(input_uc=input_uc, output_uc=output_uc, _is_inside_iter=True)


