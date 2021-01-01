from octras import Evaluator, Loop
from octras.matsim import MATSimSimulator
from octras.tracker import PickleTracker

from octras.algorithms import RandomWalk, CMAES, SPSA, NelderMead #, BatchBayesianOptimization

from problem import ParisProblem
from analyzer import ParisAnalyzer

import numpy as np

import logging
logging.basicConfig(level = logging.INFO)

total_threads = 4
threads_per_simulation = 4
iterations = 10
output_path = "optimization.p"

simulator = MATSimSimulator(
    working_directory = "work",
    class_path = "resources/ile_de_france-1.2.0.jar",
    main_class = "org.eqasim.ile_de_france.RunSimulation"
)

analyzer = ParisAnalyzer(
    threshold = 0.05,
    number_of_bounds = 40,
    cutoff_distance = 4 * 1e3,
    reference_path = "resources/hts_trips.csv"
)

problem = ParisProblem(analyzer,
    threads = threads_per_simulation,
    iterations = iterations,
    config_path = "resources/scenario/config.xml"
)

evaluator = Evaluator(
    problem = problem,
    simulator = simulator,
    parallel = np.floor(total_threads / threads_per_simulation)
)

algorithm = SPSA(
    evaluator = evaluator,
    perturbation_factor = 0.1,
    gradient_factor = 0.5,
    perturbation_exponent = 0.101,
    gradient_exponent = 0.602,
    gradient_offset = 0
)

tracker = PickleTracker(output_path)

Loop().run(
    evaluator = evaluator,
    algorithm = algorithm, tracker = tracker
)
