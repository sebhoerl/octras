import uuid, time
import numpy as np
import scipy.optimize

from octras.tests.utils import QuadraticSumSimulator, RealDimensionalProblem
from octras.simulation import Scheduler
from octras.optimization import Optimizer

from octras.algorithms.random_walk import random_walk_algorithm
from octras.algorithms.scipy import scipy_algorithm
from octras.algorithms.spsa import spsa_algorithm
from octras.algorithms.fdsa import fdsa_algorithm

def test_spsa():
    np.random.seed(0)

    simulator = QuadraticSumSimulator([2.0, 4.0])
    problem = RealDimensionalProblem(2)

    scheduler = Scheduler(simulator, ping_time = 0.0)
    optimizer = Optimizer(scheduler, problem)

    parameters, objective = spsa_algorithm(optimizer)
    assert objective < 0.01
