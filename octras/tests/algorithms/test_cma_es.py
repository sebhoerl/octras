import uuid, time
import numpy as np
import scipy.optimize

from octras.tests.utils import RoadRailSimulator, RoadRailProblem
from octras.tests.utils import QuadraticSumSimulator, RealDimensionalProblem
from octras.tests.utils import RosenbrockSimulator
from octras.simulation import Scheduler
from octras.optimization import Optimizer

from octras.algorithms.cma_es import cma_es_algorithm

def test_cma_es():
    np.random.seed(0)

    simulator = QuadraticSumSimulator([2.0, 4.0])
    problem = RealDimensionalProblem(2)

    scheduler = Scheduler(simulator, ping_time = 0.0)
    optimizer = Optimizer(scheduler, problem, maximum_evaluations = 200)

    cma_es_algorithm(optimizer, candidate_set_size = 4)
    assert optimizer.best_objective < 1e-3

def test_cma_es_with_rosenbrock():
    np.random.seed(0)

    simulator = RosenbrockSimulator()
    problem = RealDimensionalProblem(3)

    scheduler = Scheduler(simulator, ping_time = 0.0, default_parameters = { "dimensions": 3 })
    optimizer = Optimizer(scheduler, problem, maximum_evaluations = 10000)

    cma_es_algorithm(optimizer)
    assert optimizer.best_objective < 1e-3

def test_cma_es_with_road():
    np.random.seed(0)

    simulator = RoadRailSimulator()
    problem = RoadRailProblem()

    default_parameters = {
        "iterations": 200
    }

    scheduler = Scheduler(simulator, ping_time = 0.0, default_parameters = default_parameters)
    optimizer = Optimizer(scheduler, problem, maximum_evaluations = 200)

    cma_es_algorithm(optimizer, candidate_set_size = 4)
    assert optimizer.best_objective < 1e-3
