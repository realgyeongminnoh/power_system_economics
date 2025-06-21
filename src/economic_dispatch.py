import numpy as np
import gurobipy as gp

from .input import Input_ed
from .output import Output_ed


def solve_ed(
    input_ed: Input_ed,
    output_ed: Output_ed,
    t: int,
    verbose: bool = False,
):
    ### ATTRIBUTE LOCALIZATION & GUROBIPY-FRIENDLY TYPE CONVERSION
    num_units = input_ed.num_units
    u_t = input_ed.u[:, t]
    cost_lin = input_ed.cost_lin.tolist()
    cost_const = input_ed.cost_const

    ### MODEL DECLARATION
    model = gp.Model()
    model.setParam("OutputFlag", verbose)
    model.setParam("MIPGap", 0)
    model.setParam("MIPGapAbs", 0)
    model.setParam("IntFeasTol", 1e-9)
    model.setParam("FeasibilityTol", 1e-9)
    model.setParam("OptimalityTol", 1e-9)


    ### VARIABLE DECLARATION
    p = model.addVars(
        range(num_units),
        lb=(input_ed.p_min * u_t).tolist(), 
        ub=(input_ed.p_max * u_t).tolist(),
    )

    ### CONSTRAINT DECLARATION
    constr_generation = model.addConstr(gp.quicksum(p[i] for i in range(num_units)) + input_ed.renewable[t] == input_ed.demand[t])
    
    ### OBJECTIVE
    cost_generation = gp.quicksum(cost_lin[i] * p[i] + (cost_const * u_t).tolist()[i] for i in range(num_units))
    model.setObjective(cost_generation, gp.GRB.MINIMIZE)
    model.optimize()

    ### OUTPUT
    if model.Status != gp.GRB.OPTIMAL:
        raise NotImplementedError(f"{model.Status}")
    
    output_ed.t = t
    output_ed.p = np.array(model.getAttr("X", p).select()).reshape(num_units)
    output_ed.cost_generation = cost_generation.getValue()
    output_ed.u_t = input_ed.u
    output_ed.marginal_price_generation = constr_generation.Pi