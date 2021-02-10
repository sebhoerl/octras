from octras import Evaluator, Loop
from octras.matsim import MATSimSimulator
from octras.tracker import PickleTracker

from octras.algorithms import RandomWalk, CMAES, SPSA, NelderMead #, BatchBayesianOptimization

from flow_problem import ParisFlowProblem
from flow_analyzer import ParisDailyFlowAnalyzer

import logging
logging.basicConfig(level = logging.INFO)

threads_per_simulation = 4
parallel_simulations = 1
iterations = 40

output_path = "optimization.p"

reference_path = "resources/hourly_reference.csv"
config_path = "resources/scenario/config.xml"

simulator = MATSimSimulator(
    working_directory = "work",
    class_path = "resources/ile_de_france-1.2.0.jar",
    main_class = "org.eqasim.ile_de_france.RunSimulation"
)

analyzer = ParisDailyFlowAnalyzer(
    reference_path = reference_path
)

problem = ParisFlowProblem(analyzer,
    threads = threads_per_simulation,
    iterations = iterations,
    config_path = config_path,
    reference_path = reference_path
)

evaluator = Evaluator(
    problem = problem,
    simulator = simulator,
    parallel = parallel_simulations
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
