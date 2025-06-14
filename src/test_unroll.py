import numpy as np
import gurobipy as gp

from .input import Input_ed
from .output import Output_ed


def solve_ed_p_prev(
    input_ed: Input_ed,
    output_ed: Output_ed,
    verbose: bool = False,
):
    ### ATTRIBUTE LOCALIZATION & GUROBIPY-FRIENDLY TYPE CONVERSION
    num_units = input_ed.num_units
    u = input_ed.u
    cost_lin = input_ed.cost_lin.tolist()
    cost_const = input_ed.cost_const

    ### MODEL DECLARATION
    model = gp.Model()
    model.setParam("OutputFlag", verbose)

    ### VARIABLE DECLARATION
    p = model.addVars(
        range(num_units),
        lb=(input_ed.p_min * u).tolist(), 
        ub=(input_ed.p_max * u).tolist(),
    )

    ### CONSTRAINT DECLARATION
    constr_generation = model.addConstr(gp.quicksum(p[i] for i in range(num_units)) + input_ed.renewable == input_ed.demand)
    
    ### OBJECTIVE
    cost_generation = gp.quicksum(cost_lin[i] * p[i] + (cost_const * u).tolist()[i] for i in range(num_units))
    model.setObjective(cost_generation, gp.GRB.MINIMIZE)
    model.optimize()

    ### OUTPUT
    if model.Status != gp.GRB.OPTIMAL:
        raise NotImplementedError(f"{model.Status}")
    
    output_ed.p = np.array(model.getAttr("X", p).select()).reshape(num_units)
    output_ed.cost_generation = cost_generation.getValue()
    output_ed.marginal_price_generation = constr_generation.Pi


def solve_ed_question_1(
    input_ed: Input_ed,
    output_ed: Output_ed,
    verbose: bool = False,
):
    ### ATTRIBUTE LOCALIZATION & GUROBIPY-FRIENDLY TYPE CONVERSION
    num_units = input_ed.num_units
    u = input_ed.u
    cost_lin = input_ed.cost_lin.tolist()
    cost_const = input_ed.cost_const

    ### MODEL DECLARATION
    model = gp.Model()
    model.setParam("OutputFlag", verbose)

    ### VARIABLE DECLARATION
    p = model.addVars(
        range(num_units),
        lb=(input_ed.p_min * u).tolist(), 
        ub=(input_ed.p_max * u).tolist(),
    )

    ### CONSTRAINT DECLARATION
    constr_generation = model.addConstr(gp.quicksum(p[i] for i in range(num_units)) + input_ed.renewable == input_ed.demand)
    
    ### OBJECTIVE
    cost_generation = gp.quicksum(cost_lin[i] * p[i] + (cost_const * u).tolist()[i] for i in range(num_units))
    model.setObjective(cost_generation, gp.GRB.MINIMIZE)
    model.optimize()

    ### OUTPUT
    if model.Status != gp.GRB.OPTIMAL:
        raise NotImplementedError(f"{model.Status}")
    
    output_ed.p = np.array(model.getAttr("X", p).select()).reshape(num_units)
    output_ed.cost_generation = cost_generation.getValue()
    output_ed.marginal_price_generation = constr_generation.Pi