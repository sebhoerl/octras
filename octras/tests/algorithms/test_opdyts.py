import numpy as np

from octras.tests.utils import RoadRailSimulator, RoadRailProblem
from octras.simulation import Scheduler
from octras.optimization import Optimizer

from octras.algorithms.opdyts import opdyts_algorithm
from octras.algorithms.random_walk import random_walk_algorithm
from octras.algorithms.scipy import scipy_algorithm

def test_opdyts():
    np.random.seed(0)

    simulator = RoadRailSimulator()
    problem = RoadRailProblem()

    scheduler = Scheduler(simulator, ping_time = 0.0)
    optimizer = Optimizer(scheduler, problem, maximum_cost = 10000)

    opdyts_algorithm(optimizer, perturbation_length = 2.0, transition_iterations = 10, number_of_transitions = 20, candidate_set_size = 10)
    assert optimizer.best_objective < 1e-3
