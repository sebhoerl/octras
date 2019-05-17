import numpy as np

def spsa_algorithm(calibrator, maximum_iterations = 1000, initial_parameters = None, perturbation_factor = 1.0, perturbation_exponent = 0.5, gradient_factor = 1.0):
    best_objective = np.inf
    best_parameters = None
    spsa_iteration = 1

    if initial_parameters is None:
        initial_parameters = np.zeros((calibrator.problem.number_of_parameters,))

    parameters = np.copy(initial_parameters)

    while spsa_iteration < maximum_iterations:
        # Sample direction from Rademacher distribution
        direction = np.random.randint(0, 2, calibrator.problem.number_of_parameters) - 0.5
        perturbation_length = perturbation_factor / (spsa_iteration ** perturbation_exponent)

        # Schedule samples
        positive_parameters = np.copy(parameters)
        positive_parameters += direction * perturbation_length
        positive_identifier = calibrator.schedule(positive_parameters)

        negative_parameters = np.copy(parameters)
        negative_parameters -= direction * perturbation_length
        negative_identifier = calibrator.schedule(negative_parameters)

        # Wait for gradient run results
        calibrator.wait(verbose = False)

        positive_objective, positive_state = calibrator.get(positive_identifier)
        calibrator.cleanup(positive_identifier)

        negative_objective, negative_state = calibrator.get(negative_identifier)
        calibrator.cleanup(negative_identifier)

        gradient = (positive_objective - negative_objective) / (2.0 * perturbation_length * direction)

        # Update parameters
        gradient_length = gradient_factor / spsa_iteration
        parameters -= gradient_length * gradient

        if positive_objective < best_objective:
            print("Iteration %d, Objective %f" % (spsa_iteration, positive_objective))
            best_objective = positive_objective
            best_parameters = positive_parameters

        if negative_objective < best_objective:
            print("Iteration %d, Objective %f" % (spsa_iteration, negative_objective))
            best_objective = negative_objective
            best_parameters = negative_parameters

        spsa_iteration += 1

    return best_parameters, best_objective
