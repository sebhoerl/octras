from simulation import Simulator
from calibration import CalibrationProblem, Calibrator
from problems import TravelTimeProblem
import reference

import time
import numpy as np

number_of_iterations = 2

simulator = Simulator({
    "java_path": "/usr/java/jdk1.8.0_144/bin/java",
    "working_directory": "temp",
    "class_path": "simulation/astra_2018_002-1.0.0.jar",
    "config_path": "simulation/zurich_{sample_size}/zurich_config.xml",
    "number_of_threads": 4,
    "number_of_parallel_runs": 1
})

travel_time_reference, travel_time_bounds = reference.get_travel_time_reference("simulation/reference.csv", 5)
problem = TravelTimeProblem(travel_time_reference, travel_time_bounds, default_parameters = {
    "iterations": number_of_iterations,
    "sample_size": "1pm"
})

calibrator = Calibrator(problem)

perturbation_factor = 1.0

best_parameters = np.zeros((problem.number_of_parameters,))
best_objective = np.inf

while not calibrator.converged:
    # Update parameters with perturbation in random direction
    direction = 2.0 * (np.random.random((problem.number_of_parameters,)) - 0.5)
    parameters = best_parameters + perturbation_factor * direction

    # Schedule and wait for sample
    identifier = simulator.schedule(problem.get_simulation_parameters(parameters))
    state = problem.get_state(simulator.get(identifier))
    objective = problem.get_objective(state)
    simulator.cleanup(identifier)

    # Update calibrator
    calibrator.add_sample(parameters, state, objective, number_of_iterations)
    calibrator.save("output/random_walk.p")
    calibrator.plot("output/random_walk.pdf")

    # Accept as new starting point if objective is better
    if objective < best_objective:
        best_parameters = parameters
        best_objective = objective
