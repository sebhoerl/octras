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
perturbation_exponent = 0.5
gradient_factor = 1.0

parameters = np.zeros((problem.number_of_parameters,))
spsa_iteration = 0

while not calibrator.converged:
    spsa_iteration += 1

    # Sample direction from Rademacher distribution
    direction = np.random.randint(0, 2, problem.number_of_parameters) - 0.5
    perturbation_length = perturbation_factor / (spsa_iteration ** perturbation_exponent)

    # Schedule samples
    positive_parameters = np.copy(parameters)
    positive_parameters += direction * perturbation_length
    positive_identifier = simulator.schedule(problem.get_simulation_parameters(positive_parameters))

    negative_parameters = np.copy(parameters)
    negative_parameters -= direction * perturbation_length
    negative_identifier = simulator.schedule(problem.get_simulation_parameters(negative_parameters))

    # Wait for gradient run results
    simulator.wait()

    positive_state = problem.get_state(simulator.get(positive_identifier))
    positive_objective = problem.get_objective(positive_state)
    calibrator.add_sample(positive_parameters, positive_state, positive_objective, number_of_iterations)
    simulator.cleanup(positive_identifier)

    negative_state = problem.get_state(simulator.get(negative_identifier))
    negative_objective = problem.get_objective(negative_state)
    calibrator.add_sample(negative_parameters, negative_state, negative_objective, number_of_iterations)
    simulator.cleanup(negative_identifier)

    gradient = (positive_objective - negative_objective) / (2.0 * perturbation_length * direction)

    # Update parameters
    gradient_length = gradient_factor / spsa_iteration
    parameters -= gradient_length * gradient

    calibrator.save("output/spsa.p")
    calibrator.plot("output/spsa.pdf")
