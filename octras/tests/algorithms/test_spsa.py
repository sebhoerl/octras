import uuid, time
import numpy as np
import scipy.optimize

from octras.tests.utils import RoadRailSimulator, RoadRailProblem
from octras.tests.utils import QuadraticSumSimulator, RealDimensionalProblem
from octras.simulation import Scheduler
from octras.optimization import Optimizer

from octras.algorithms.spsa import spsa_algorithm

def test_spsa():
    np.random.seed(0)

    simulator = QuadraticSumSimulator([2.0, 4.0])
    problem = RealDimensionalProblem(2)

    scheduler = Scheduler(simulator, ping_time = 0.0)
    optimizer = Optimizer(scheduler, problem, maximum_evaluations = 300)

    spsa_algorithm(optimizer, perturbation_factor = 2e-2, gradient_factor = 0.2)
    assert optimizer.best_objective < 1e-3

def test_spsa_with_road():
    np.random.seed(0)

    simulator = RoadRailSimulator()
    problem = RoadRailProblem({ "iterations": 200 })

    scheduler = Scheduler(simulator, ping_time = 0.0)
    optimizer = Optimizer(scheduler, problem, maximum_evaluations = 100)

    spsa_algorithm(optimizer, perturbation_factor = 100.0, gradient_factor = 100.0, perturbation_exponent = 0.7, gradient_exponent = 1.0)
    assert optimizer.best_objective < 1e-3
