import matsim
from mode_share import ModeShareProblem
from octras.algorithms import CMAES

simulator = matsim.Simulator(
    working_directory = "optimization_temp",
    class_path = "/oath/to/berlin.jar",
    main_class = "org.matsim.RunBerlin",
    arguments = ["/path/to/berlin_config.xml"]
)

scheduler = octras.Scheduler(
    simulator = simulator,
    maximum_runs = 4
)

problem = ModeShareProblem(
    car_reference = 0.3,
    pt_reference = 0.3
)

algorithm = CMAES(scheduler, problem)

optimizer = octras.Optimizer(
    problem = problem,
    algorithm = algorithm
)

optimizer.run(maximum_cost = 20)
