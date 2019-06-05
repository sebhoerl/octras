from octras.simulation import Scheduler
from octras.optimization import Optimizer

from matsim import MATSimSimulator
from problems import ModeShareProblem, TravelTimeProblem

import logging
logging.basicConfig(level = logging.INFO)

import numpy as np
import sys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--problem", choices = ["mode_share", "travel_time"], required = True)
parser.add_argument("--algorithm", choices = ["random_walk", "fdsa", "spsa", "opdyts", "cma_es"], required = True)
parser.add_argument("--perturbation_factor", default = 1.0, type = float)
parser.add_argument("--perturbation_exponent", default = 0.9, type = float)
parser.add_argument("--gradient_factor", default = 1.0, type = float)
parser.add_argument("--candidate_set_size", default = 4, type = int)
parser.add_argument("--transition_iterations", default = 5, type = int)
parser.add_argument("--number_of_transitions", default = 8, type = int)
parser.add_argument("--adaptation_weight", default = 0.9, type = float)
parser.add_argument("--number_of_threads", default = 4, type = int)
parser.add_argument("--default_number_of_iterations", default = 40, type = int)
parser.add_argument("--default_sample_size", default = "1pm", choices = ["1pm", "1pct", "10pct", "25pct"])
parser.add_argument("--reference_sample_size", default = "25pct", choices = ["1pm", "1pct", "10pct", "25pct"])
parser.add_argument("--number_of_runners", default = 1, type = int)
parser.add_argument("--working_directory", default = "temp")
parser.add_argument("--java_path", default = "java")
parser.add_argument("--java_memory", default = "10G")
parser.add_argument("--simulation_path", default = "simulation")
parser.add_argument("--log_path", default = None)
parser.add_argument("--maximum_evaluations", default = np.inf, type = float)
parser.add_argument("--maximum_cost", default = np.inf, type = float)
parser.add_argument("--initial_step_size", default = 0.3, type = float)
parser.add_argument("--random_seed", default = 0, type = int)
parser.add_argument("--bounds", default = 2.0, type = float)
cmd = parser.parse_args()

np.random.seed(cmd.random_seed)

simulator = MATSimSimulator({
    "java_path": cmd.java_path,
    "java_memory": cmd.java_memory,
    "working_directory": cmd.working_directory,
    "simulation_path": cmd.simulation_path,
    "number_of_threads": cmd.number_of_threads
})

default_parameters = {
    "sample_size": cmd.default_sample_size,
    "iterations": cmd.default_number_of_iterations
}

scheduler = Scheduler(
    simulator, default_parameters,
    number_of_runners = cmd.number_of_runners
)

# Use mode share problem
if cmd.problem == "mode_share":
    problem = ModeShareProblem(cmd.simulation_path, cmd.reference_sample_size)

# Use travel time problem
elif cmd.problem == "travel_time":
    problem = TravelTimeProblem(cmd.simulation_path, cmd.reference_sample_size)

else:
    raise RuntimeError("Unknown problem")

optimizer = Optimizer(
    scheduler, problem,
    log_path = cmd.log_path,
    maximum_evaluations = cmd.maximum_evaluations,
    maximum_cost = cmd.maximum_cost
)

# Use random walk
if cmd.algorithm == "random_walk":
    from octras.algorithms.random_walk import random_walk_algorithm
    #bounds = [(-cmd.bounds, cmd.bounds)] * 3
    bounds = [(-1.0, 1.0), (-1.0, 1.0), (0.0, 2.0)]
    random_walk_algorithm(optimizer, bounds)

# Use FDSA
elif cmd.algorithm == "fdsa":
    from octras.algorithms.fdsa import fdsa_algorithm

    perturbation_factor = cmd.perturbation_factor
    perturbation_exponent = cmd.perturbation_exponent
    gradient_factor = cmd.gradient_factor

    fdsa_algorithm(optimizer, perturbation_factor, perturbation_exponent, gradient_factor)

# Use SPSA
elif cmd.algorithm == "spsa":
    from octras.algorithms.spsa import spsa_algorithm

    perturbation_factor = cmd.perturbation_factor
    perturbation_exponent = cmd.perturbation_exponent
    gradient_factor = cmd.gradient_factor

    spsa_algorithm(optimizer, perturbation_factor, perturbation_exponent, gradient_factor)

# Use Opdyts
elif cmd.algorithm == "opdyts":
    from octras.algorithms.opdyts import opdyts_algorithm

    candidate_set_size = cmd.candidate_set_size
    perturbation_factor = cmd.perturbation_factor
    transition_iterations = cmd.transition_iterations
    number_of_transitions = cmd.number_of_transitions
    adaptation_weight = cmd.adaptation_weight

    opdyts_algorithm(optimizer, candidate_set_size, perturbation_factor, transition_iterations, number_of_transitions, adaptation_weight)

# Use CMA-ES
elif cmd.algorithm == "cma_es":
    from octras.algorithms.cma_es import cma_es_algorithm

    candidate_set_size = cmd.candidate_set_size
    initial_step_size = cmd.initial_step_size

    cma_es_algorithm(optimizer, candidate_set_size, initial_step_size)

else:
    raise RuntimeError("Unknown algorithm")
