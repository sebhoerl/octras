import numpy as np

import logging
logger = logging.getLogger(__name__)

def spsa_algorithm(calibrator, perturbation_factor = 1.0, perturbation_exponent = 0.5, gradient_factor = 1.0, compute_objective = True):
    spsa_iteration = 0
    parameters = np.copy(calibrator.problem.initial_parameters)

    while not calibrator.finished:
        spsa_iteration += 1
        logger.info("Starting SPSA iteration %d." % spsa_iteration)

        # Update step lengths
        gradient_length = gradient_factor / spsa_iteration
        perturbation_length = perturbation_factor / (spsa_iteration ** perturbation_exponent)

        # Sample direction from Rademacher distribution
        direction = np.random.randint(0, 2, calibrator.problem.number_of_parameters) - 0.5

        annotations = {
            "perturbation_length": perturbation_length,
            "gradient_length": gradient_length,
            "direction": direction
        }

        # Schedule samples
        positive_parameters = np.copy(parameters)
        positive_parameters += direction * perturbation_length
        annotations.update({ "type": "positive_gradient" })
        positive_identifier = calibrator.schedule(positive_parameters, annotations = annotations)

        negative_parameters = np.copy(parameters)
        negative_parameters -= direction * perturbation_length
        annotations.update({ "type": "negative_gradient" })
        negative_identifier = calibrator.schedule(negative_parameters, annotations = annotations)

        # Wait for gradient run results
        calibrator.wait()

        positive_objective, positive_state = calibrator.get(positive_identifier)
        calibrator.cleanup(positive_identifier)

        negative_objective, negative_state = calibrator.get(negative_identifier)
        calibrator.cleanup(negative_identifier)

        gradient = (positive_objective - negative_objective) / (2.0 * perturbation_length * direction)

        # Update state
        parameters -= gradient_length * gradient

        if compute_objective:
            annotations.update({ "type": "objective" })
            identifier = calibrator.schedule(parameters, annotations = annotations)

            calibrator.wait()

            objective, state = calibrator.get(identifier)
            calibrator.cleanup(identifier)
