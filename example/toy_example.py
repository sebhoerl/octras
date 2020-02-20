from octras import Evaluator, Loop
from matsim import MATSimSimulator
from flow_capacity import FlowCapacityProblem

from octras.algorithms import CMAES

import logging
logging.basicConfig(level = logging.INFO)

simulator = MATSimSimulator(
    working_directory = "/home/shoerl/bo/test2",
    class_path = "toy_example/target/optimization_toy_example-1.0.0.jar",
    main_class = "ch.ethz.matsim.optimization_toy_example.RunToyExample",
    iterations = 20
)

problem = FlowCapacityProblem(0.8)

evaluator = Evaluator(
    problem = problem,
    simulator = simulator,
    parallel = 3
)

algorithm = CMAES(
    evaluator = evaluator,
    initial_step_size = 0.1
)

print(Loop(maximum_cost = 20).run(
    evaluator = evaluator,
    algorithm = algorithm
))
