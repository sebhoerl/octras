import numpy as np

import logging
logger = logging.getLogger(__name__)

def fdsa_algorithm(calibrator, perturbation_factor = 1.0, perturbation_exponent = 0.5, gradient_factor = 1.0, compute_objective = True):
    fdsa_iteration = 0
    parameters = np.copy(calibrator.problem.initial_parameters)

    while not calibrator.finished:
        fdsa_iteration += 1
        logger.info("Starting FDSA iteration %d." % fdsa_iteration)

        # Update lengths
        perturbation_length = perturbation_factor / (fdsa_iteration ** perturbation_exponent)
        gradient_length = gradient_factor / fdsa_iteration

        annotations = {
            "perturbation_length" : perturbation_length,
            "gradient_length": gradient_length
        }

        # I) Calculate gradients
        gradient = np.zeros((calibrator.problem.number_of_parameters,))
        gradient_identifiers = []

        # Schedule all necessary runs
        for d in range(calibrator.problem.number_of_parameters):
            annotations.update({ "dimension": d })

            positive_parameters = np.copy(parameters)
            positive_parameters[d] += perturbation_length
            annotations.update({ "type": "positive_gradient" })
            positive_identifier = calibrator.schedule(positive_parameters, annotations = annotations)

            negative_parameters = np.copy(parameters)
            negative_parameters[d] -= perturbation_length
            annotations.update({ "sign": "negative_gradient" })
            negative_identifier = calibrator.schedule(negative_parameters, annotations = annotations)

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

        # II) Update state
        parameters -= gradient_length * gradient

        if compute_objective:
            annotations.update({ "type": "objective" })
            identifier = calibrator.schedule(parameters, annotations = annotations)

            calibrator.wait()

            objective, state = calibrator.get(identifier)
            calibrator.cleanup(identifier)
