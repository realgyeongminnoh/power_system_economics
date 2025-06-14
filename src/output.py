import numpy as np
from dataclasses import dataclass


@dataclass
class Output_uc:
    #
    objval: float = None
    total_cost: float = None
    total_cost_generation: float = None
    total_cost_startup: float = None
    total_cost_reserve: float = None
    total_cost_retailer: float = None
    #
    cost_generation: np.ndarray = None
    cost_startup: np.ndarray = None
    cost_reserve: np.ndarray = None
    cost_retailer: np.ndarray = None
    marginal_price_generation: np.ndarray = None
    marginal_price_reserve: np.ndarray  = None
    #
    u: np.ndarray = None
    p: np.ndarray = None
    r: np.ndarray = None
    v: np.ndarray = None
    w: np.ndarray = None
    delta: np.ndarray = None
    generation: np.ndarray = None
    reserve: np.ndarray = None


@dataclass
class Output_ed:
    pass