from simulation import Simulator
from calibration import CalibrationProblem, Calibrator
from problems import ModeShareProblem
import reference

import time
import numpy as np

number_of_iterations = 20

simulator = Simulator({
    "java_path": "/usr/java/jdk1.8.0_144/bin/java",
    "working_directory": "temp",
    "class_path": "simulation/astra_2018_002-1.0.0.jar",
    "config_path": "simulation/zurich_{sample_size}/zurich_config.xml",
    "number_of_threads": 4,
    "number_of_parallel_runs": 1
})

mode_share_reference = reference.get_mode_share_reference("simulation/reference.csv")
problem = ModeShareProblem(mode_share_reference, default_parameters = {
    "iterations": number_of_iterations,
    "sample_size": "1pm"
})

calibrator = Calibrator(problem)

perturbation_factor = 1.0
perturbation_exponent = 0.5
gradient_factor = 1.0

parameters = np.zeros((problem.number_of_parameters,))
fdsa_iteration = 0

while not calibrator.converged:
    fdsa_iteration += 1

    # I) Calculate gradients
    gradient = np.zeros((problem.number_of_parameters,))

    perturbation_length = perturbation_factor / (fdsa_iteration ** perturbation_exponent)
    gradient_identifiers = []

    # Schedule all necessary runs
    for d in range(problem.number_of_parameters):
        positive_parameters = np.copy(parameters)
        positive_parameters[d] += perturbation_length
        positive_identifier = simulator.schedule(problem.get_simulation_parameters(positive_parameters))

        negative_parameters = np.copy(parameters)
        negative_parameters[d] -= perturbation_length
        negative_identifier = simulator.schedule(problem.get_simulation_parameters(negative_parameters))

        gradient_identifiers.append((positive_parameters, positive_identifier, negative_parameters, negative_identifier))

    # Wait for gradient run results
    simulator.wait()

    for d, item in enumerate(gradient_identifiers):
        positive_parameters, positive_identifier, negative_parameters, negative_identifier = item

        positive_state = problem.get_state(simulator.get(positive_identifier))
        positive_objective = problem.get_objective(positive_state)
        calibrator.add_sample(positive_parameters, positive_state, positive_objective, number_of_iterations)
        simulator.cleanup(positive_identifier)

        negative_state = problem.get_state(simulator.get(negative_identifier))
        negative_objective = problem.get_objective(negative_state)
        calibrator.add_sample(negative_parameters, negative_state, negative_objective, number_of_iterations)
        simulator.cleanup(negative_identifier)

        gradient[d] = (positive_objective - negative_objective) / (2.0 * perturbation_length)

    # II) Update parameters
    gradient_length = gradient_factor / fdsa_iteration
    parameters -= gradient_length * gradient

    calibrator.save("output/fdsa.p")
    calibrator.plot("output/fdsa.pdf")
