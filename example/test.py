from matsim import MATSimSimulator
from mode_share import ModeShareProblem
from flow_capacity import FlowCapacityProblem
from evaluator import Evaluator
from random_walk import RandomWalk
from loop import Loop

import time, logging

logging.basicConfig(level = logging.INFO)

loop = Loop(maximum_cost = 20)

simulator = MATSimSimulator(
    working_directory = "/home/shoerl/bo/test2",
    class_path = "toy_example/target/optimization_toy_example-1.0.0.jar",
    main_class = "ch.ethz.matsim.optimization_toy_example.RunToyExample",
    iterations = 20
)

#problem = ModeShareProblem(0.3, 0.3)
problem = FlowCapacityProblem(0.8)

evaluator = Evaluator(
    problem = problem,
    simulator = simulator,
    parallel = 3,
    tracker = loop
)

algorithm = RandomWalk(
    evaluator = evaluator,
    parallel = 3
)

loop.run(
    evaluator = evaluator,
    algorithm = algorithm
)
