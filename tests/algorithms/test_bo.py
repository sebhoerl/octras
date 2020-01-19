import uuid, time
import numpy as np
import scipy.optimize
import warnings

from ..utils import RoadRailSimulator, RoadRailProblem
from ..utils import QuadraticSumSimulator, RealDimensionalProblem
from octras.simulation import Scheduler
from octras.optimization import Optimizer

from octras.algorithms.bo import bo_algorithm

import pytest

def test_bo_single_fidelity():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for method in ("gpbucb", "mes",):
            np.random.seed(0)

            simulator = QuadraticSumSimulator([2.0, 4.0])
            problem = RealDimensionalProblem(2)

            scheduler = Scheduler(simulator, ping_time = 0.0)
            optimizer = Optimizer(scheduler, problem, maximum_evaluations = 100)

            bo_algorithm(optimizer, method = method)
            assert optimizer.best_objective < 2e-3

def test_bo_single_fidelity_with_road():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for method in ("mes", "gpbucb"):
            np.random.seed(0)

            simulator = RoadRailSimulator()
            problem = RoadRailProblem({ "iterations": 200 })

            scheduler = Scheduler(simulator, ping_time = 0.0)
            optimizer = Optimizer(scheduler, problem, maximum_evaluations = 100)

            bo_algorithm(optimizer, method = method)
            assert optimizer.best_objective < 1e-3

@pytest.mark.skip(reason = "Not working yet!")
def test_bo_multi_fidelity():
    # @ Anastasia: This is the unit test for multi-fidelity

    np.random.seed(0)

    simulator = RoadRailSimulator()
    problem = RoadRailProblem()

    scheduler = Scheduler(simulator, ping_time = 0.0)

    # @ Anastasia, this is how you provide the path to the output *.p file in the unit tests. You can
    # use, e.g.
    #   python3 use_case/plotting/plot_problem.py /path/to/mfmes.p
    # to see the progress of the optimization. But as I wrote in the algorithm
    # file, you can also write custom information into the p file for each sample
    # and then analyze/visualize it.
    optimizer = Optimizer(scheduler, problem, maximum_evaluations = 100, log_path = "/home/shoerl/mfmes.p")

    # @ Anastasia: This is how the fidelity levels are defined. Basically, they get a name,
    # a cost, and a set of parameters that are specific to this fidelity level.
    fidelities = [
        { "parameters": { "iterations": 50 }, "name": "it50", "cost": 50 },
        { "parameters": { "iterations": 100 }, "name": "it100", "cost": 100 },
        { "parameters": { "iterations": 200 }, "name": "it200", "ccost": 200 }
    ]

    bo_algorithm(optimizer, method = "mfmes", fidelities = fidelities)
    assert optimizer.best_objective < 1e-3
