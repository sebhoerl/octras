from simulation import Simulator
from calibration import CalibrationProblem, Calibrator
from problems import ModeShareProblem

import time
import numpy as np

simulator = Simulator({
    "java_path": "/usr/java/jdk1.8.0_144/bin/java",
    "working_directory": "temp",
    "class_path": "simulation/astra_2018_002-1.0.0.jar",
    "config_path": "simulation/sa/sa_config.xml"
})

number_of_iterations = 20

problem = ModeShareProblem(number_of_iterations)
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
