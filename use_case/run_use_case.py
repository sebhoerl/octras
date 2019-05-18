from octras.simulation import Scheduler
from octras.optimization import Optimizer

from matsim import MATSimSimulator
from problems import ModeShareProblem, TravelTimeProblem

import logging
logging.basicConfig(level = logging.INFO)

import numpy as np
import sys

np.random.seed(0)

problem = sys.argv[1]
algorithm = sys.argv[2]

simulator = MATSimSimulator({
    "java_path": "/usr/java/jdk1.8.0_144/bin/java",
    "working_directory": "temp",
    "simulation_path": "simulation",
    "number_of_threads": 4
})

default_parameters = {
    "sample_size": "1pm",
    "iterations": 40
}

scheduler = Scheduler(
    simulator, default_parameters,
    number_of_runners = 1
)

# Use mode share problem
if problem == "mode_share":
    problem = ModeShareProblem("simulation/reference.csv")

# Use travel time problem
elif problem == "travel_time":
    five_minute_bounds = np.arange(1, 13) * 5.0
    problem = TravelTimeProblem("simulation/reference.csv", five_minute_bounds)

else:
    raise RuntimeError()

optimizer = Optimizer(scheduler, problem, history_path = "temp/history_%s.p" % algorithm)

# Use random walk
if algorithm == "random_walk":
    from octras.algorithms.random_walk import random_walk_algorithm
    bounds = [(-10.0, 10.0)] * 3
    random_walk_algorithm(optimizer, bounds)

# Use FDSA
elif algorithm == "fdsa":
    from octras.algorithms.fdsa import fdsa_algorithm

    perturbation_factor = 1.0
    perturbation_exponent = 0.9
    gradient_factor = 1.0

    fdsa_algorithm(optimizer, perturbation_factor, perturbation_exponent, gradient_factor)

# Use SPSA
elif algorithm == "spsa":
    from octras.algorithms.fdsa import fdsa_algorithm

    perturbation_factor = 1.0
    perturbation_exponent = 0.9
    gradient_factor = 1.0

    fdsa_algorithm(optimizer, perturbation_factor, perturbation_exponent, gradient_factor)

# Use Opdyts
elif algorithm == "opdyts":
    from octras.algorithms.opdyts import opdyts_algorithm

    candidate_set_size = 8
    perturbation_factor = 1.0
    transition_iterations = 5
    number_of_transitions = 8
    adaptation_weight = 0.9

    opdyts_algorithm(optimizer, candidate_set_size, perturbation_factor, transition_iterations, number_of_transitions, adaptation_weight)
