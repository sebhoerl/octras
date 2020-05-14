from octras import Evaluator, Loop
from octras.matsim import MATSimSimulator
from octras.tracker import PickleTracker

from octras.algorithms import RandomWalk, CMAES, BatchBayesianOptimization

from toy_example_problem import ToyExampleProblem

import logging
logging.basicConfig(level = logging.INFO)

output_path = "optimization_output.p"

# First, build the project in toy_example with
#  mvn package

simulator = MATSimSimulator(
    working_directory = "/home/shoerl/bo/test2", # Change to an existing empty directory
    class_path = "toy_example/target/optimization_toy_example-1.0.0.jar",
    main_class = "ch.ethz.matsim.optimization_toy_example.RunToyExample",
    iterations = 200,
    java = "/home/shoerl/.java/jdk8u222-b10/bin/java"
)

problem = ToyExampleProblem(0.5)

evaluator = Evaluator(
    problem = problem,
    simulator = simulator,
    parallel = 4
)

algorithm = CMAES(
    evaluator = evaluator,
    initial_step_size = 0.1,
    seed = 0
)

#algorithm = BatchBayesianOptimization(
#    evaluator = evaluator,
#    batch_size = 4
#)

#algorithm = RandomWalk(
#    evaluator = evaluator
#)

tracker = PickleTracker(output_path)

Loop().run(
    evaluator = evaluator,
    algorithm = algorithm, tracker = tracker
)
