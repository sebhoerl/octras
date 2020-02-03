import numpy as np
import warnings

from ..utils import RoadRailSimulator, RoadRailProblem
from ..utils import QuadraticSumSimulator, RealDimensionalProblem
from octras.simulation import Scheduler
from octras.optimization import Optimizer

from octras.algorithms.bo import bo_algorithm, subdomain_bo_algorithm

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
            assert optimizer.best_objective < 4e-1


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


# @pytest.mark.skip(reason = "Not working yet!")
def test_bo_multi_fidelity():

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
    #optimizer = Optimizer(scheduler, problem, maximum_evaluations = 100, log_path = "/home/anmakaro/matsim/oc-matsim/tests/mfmes.p")
    optimizer = Optimizer(scheduler, problem, maximum_evaluations = 100)

    # @Sebastian: maybe we need a special class for fidelities to standartise it?
    # fidelity is defines by name, cost, and a set of parameters that are specific to this fidelity level
    fidelities = [
        { "parameters": { "iterations": 50 }, "name": "it50", "cost": 50 },
        # { "parameters": { "iterations": 100 }, "name": "it100", "cost": 100 },
        { "parameters": { "iterations": 200 }, "name": "it200", "cost": 200 }
    ]

    bo_algorithm(optimizer, method = "mfmes", fidelities = fidelities)
    assert optimizer.best_objective < 1e-3


def test_subdomain_bo_single_fidelity():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for method in ("gpbucb", "mes",):
            np.random.seed(0)

            simulator = QuadraticSumSimulator([2.0, 4.0])
            problem = RealDimensionalProblem(2)

            scheduler = Scheduler(simulator, ping_time = 0.0)
            optimizer = Optimizer(scheduler, problem, maximum_evaluations = 100)

            subdomain_bo_algorithm(optimizer, method = method)
            assert optimizer.best_objective < 4e-1


def test_subdomain_bo_single_fidelity_with_road():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")

        for method in ("mes", "gpbucb"):
            np.random.seed(0)

            simulator = RoadRailSimulator()
            problem = RoadRailProblem({ "iterations": 200 })

            scheduler = Scheduler(simulator, ping_time = 0.0)
            optimizer = Optimizer(scheduler, problem, maximum_evaluations = 100)

            subdomain_bo_algorithm(optimizer, method = method)
            assert optimizer.best_objective < 1e-3


def test_subdomain_bo_multi_fidelity():

    np.random.seed(0)

    simulator = RoadRailSimulator()
    problem = RoadRailProblem()

    scheduler = Scheduler(simulator, ping_time = 0.0)
    #optimizer = Optimizer(scheduler, problem, maximum_evaluations = 100, log_path = "/home/anmakaro/matsim/oc-matsim/tests/mfmes.p")
    optimizer = Optimizer(scheduler, problem, maximum_evaluations = 100)

    fidelities = [
        { "parameters": { "iterations": 50 }, "name": "it50", "cost": 50 },
        # { "parameters": { "iterations": 100 }, "name": "it100", "cost": 100 },
        { "parameters": { "iterations": 200 }, "name": "it200", "cost": 200 }
    ]

    subdomain_bo_algorithm(optimizer, method = "mfmes", fidelities = fidelities)
    assert optimizer.best_objective < 1e-3
