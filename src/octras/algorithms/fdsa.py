import numpy as np

import logging
logger = logging.getLogger(__name__)

# https://www.jhuapl.edu/spsa/PDF-SPSA/Spall_Implementation_of_the_Simultaneous.PDF
def fdsa_algorithm(calibrator, perturbation_factor, gradient_factor, perturbation_exponent = 0.101, gradient_exponent = 0.602, gradient_offset = 0, compute_objective = False):
    iteration = 0
    parameters = [p["initial"] for p in calibrator.problem.parameters]

    while not calibrator.finished:
        logger.info("Starting FDSA iteration %d." % iteration)

        # Update lengths
        gradient_length = gradient_factor / (iteration + 1 + gradient_offset)**gradient_exponent
        perturbation_length = perturbation_factor / (iteration + 1)**perturbation_exponent

        annotations = {
            "gradient_length" : gradient_length,
            "perturbation_length": perturbation_length
        }

        # I) Calculate gradients
        gradient = np.zeros((len(parameters),))
        gradient_information = []

        # Schedule all necessary runs
        for d in range(len(parameters)):
            annotations.update({ "dimension": d })

            positive_parameters = np.copy(parameters)
            positive_parameters[d] += perturbation_length
            annotations.update({ "type": "positive_gradient" })
            positive_identifier = calibrator.schedule(positive_parameters, annotations = annotations)

            negative_parameters = np.copy(parameters)
            negative_parameters[d] -= perturbation_length
            annotations.update({ "sign": "negative_gradient" })
            negative_identifier = calibrator.schedule(negative_parameters, annotations = annotations)

            gradient_information.append((positive_parameters, positive_identifier, negative_parameters, negative_identifier))

        # Wait for gradient run results
        calibrator.wait()

        for d, item in enumerate(gradient_information):
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

        iteration += 1
