import numpy as np

def fdsa_algorithm(calibrator, maximum_iterations = 1000, initial_parameters = None, perturbation_factor = 1.0, perturbation_exponent = 0.5, gradient_factor = 1.0):
    best_objective = np.inf
    best_parameters = None
    fdsa_iteration = 1

    if initial_parameters is None:
        initial_parameters = np.zeros((calibrator.problem.number_of_parameters,))

    parameters = np.copy(initial_parameters)

    while fdsa_iteration < maximum_iterations:
        # I) Calculate gradients
        gradient = np.zeros((calibrator.problem.number_of_parameters,))

        perturbation_length = perturbation_factor / (fdsa_iteration ** perturbation_exponent)
        gradient_identifiers = []

        # Schedule all necessary runs
        for d in range(calibrator.problem.number_of_parameters):
            positive_parameters = np.copy(parameters)
            positive_parameters[d] += perturbation_length
            positive_identifier = calibrator.schedule(positive_parameters)

            negative_parameters = np.copy(parameters)
            negative_parameters[d] -= perturbation_length
            negative_identifier = calibrator.schedule(negative_parameters)

            gradient_identifiers.append((positive_parameters, positive_identifier, negative_parameters, negative_identifier))

        # Wait for gradient run results
        calibrator.wait()

        for d, item in enumerate(gradient_identifiers):
            positive_parameters, positive_identifier, negative_parameters, negative_identifier = item

            positive_objective, positive_state = calibrator.get(positive_identifier)
            calibrator.cleanup(positive_identifier)

            negative_objective, negative_state = calibrator.get(negative_identifier)
            calibrator.cleanup(negative_identifier)

            gradient[d] = (positive_objective - negative_objective) / (2.0 * perturbation_length)

            if positive_objective < best_objective:
                print("Iteration %d, Objective %f" % (fdsa_iteration, positive_objective))
                best_objective = positive_objective
                best_parameters = positive_parameters

            if negative_objective < best_objective:
                print("Iteration %d, Objective %f" % (fdsa_iteration, negative_objective))
                best_objective = negative_objective
                best_parameters = negative_parameters

        # II) Update parameters
        gradient_length = gradient_factor / fdsa_iteration
        parameters -= gradient_length * gradient

        fdsa_iteration += 1

    return best_parameters, best_objective
