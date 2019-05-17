import numpy as np

from octras.tests.utils import ExponentialSimulator, ExponentialProblem
from octras.simulation import Scheduler
from octras.optimization import Optimizer

from octras.algorithms.opdyts import opdyts_algorithm
from octras.algorithms.random_walk import random_walk_algorithm
from octras.algorithms.scipy import scipy_algorithm

def test_opdyts():
    np.random.seed(0)

    simulator = ExponentialSimulator([1.0, 2.0, 1.0], [-0.1, -0.05, -0.2], int(1e6))
    problem = ExponentialProblem(3)

    scheduler = Scheduler(simulator, ping_time = 0.0)
    optimizer = Optimizer(scheduler, problem)

    parameters, objective = opdyts_algorithm(optimizer, perturbation_factor = 0.05, transition_iterations = 50, number_of_transitions = 100, maximum_transitions = int(1e3))
    assert objective < 0.02
