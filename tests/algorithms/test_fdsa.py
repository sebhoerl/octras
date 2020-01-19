import uuid, time
import numpy as np
import scipy.optimize

from ..utils import RoadRailSimulator, RoadRailProblem
from ..utils import QuadraticSumSimulator, RealDimensionalProblem
from octras.simulation import Scheduler
from octras.optimization import Optimizer

from octras.algorithms.random_walk import random_walk_algorithm
from octras.algorithms.scipy import scipy_algorithm
from octras.algorithms.spsa import spsa_algorithm
from octras.algorithms.fdsa import fdsa_algorithm

def test_fdsa():
    np.random.seed(0)

    simulator = QuadraticSumSimulator([2.0, 4.0])
    problem = RealDimensionalProblem(2)

    scheduler = Scheduler(simulator, ping_time = 0.0)
    optimizer = Optimizer(scheduler, problem, maximum_evaluations = 300)

    fdsa_algorithm(optimizer, perturbation_factor = 2e-2, gradient_factor = 0.2)
    assert optimizer.best_objective < 1e-3

def test_fdsa_with_road():
    np.random.seed(0)

    simulator = RoadRailSimulator()
    problem = RoadRailProblem({ "iterations": 200 })

    scheduler = Scheduler(simulator, ping_time = 0.0)
    optimizer = Optimizer(scheduler, problem, maximum_evaluations = 100)

    fdsa_algorithm(optimizer, perturbation_factor = 100.0, gradient_factor = 100.0, perturbation_exponent = 0.9)
    assert optimizer.best_objective < 1e-3
