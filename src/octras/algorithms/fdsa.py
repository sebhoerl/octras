import numpy as np
import deep_merge

import logging
logger = logging.getLogger(__name__)

# https://www.jhuapl.edu/spsa/PDF-SPSA/Spall_Implementation_of_the_Simultaneous.PDF

class FDSA:
    def __init__(self, evaluator, perturbation_factor, gradient_factor, perturbation_exponent = 0.101, gradient_exponent = 0.602, gradient_offset = 0, compute_objective = False):
        self.evaluator = evaluator

        self.perturbation_factor = perturbation_factor
        self.perturbation_exponent = perturbation_exponent

        self.gradient_factor = gradient_factor
        self.gradient_exponent = gradient_exponent
        self.gradient_offset = gradient_offset

        self.compute_objective = compute_objective

        self.iteration = 0

        if not hasattr(self.evaluator.problem, "initial"):
            raise RuntimeError("Initial parameters must be provided by problem for FDSA")

        self.parameters = None

    def advance(self):
        self.iteration += 1
        logger.info("Starting FDSA iteration %d." % self.iteration)

        if self.parameters is None:
            self.parameters = self.evaluator.problem.initial

        # Update lengths
        gradient_length = self.gradient_factor / (self.iteration + self.gradient_offset)**self.gradient_exponent
        perturbation_length = self.perturbation_factor / self.iteration**self.perturbation_exponent

        annotations = {
            "gradient_length" : gradient_length,
            "perturbation_length": perturbation_length
        }

        # I) Calculate gradients
        gradient = np.zeros((len(self.parameters),))
        gradient_information = []

        # Schedule all necessary runs
        for d in range(len(self.parameters)):
            annotations = deep_merge.merge(annotations, { "dimension": d })

            positive_parameters = np.copy(self.parameters)
            positive_parameters[d] += perturbation_length
            annotations = deep_merge.merge(annotations, { "type": "positive_gradient" })
            positive_identifier = self.evaluator.submit(positive_parameters, annotations = annotations)

            negative_parameters = np.copy(self.parameters)
            negative_parameters[d] -= perturbation_length
            annotations = deep_merge.merge(annotations, { "sign": "negative_gradient" })
            negative_identifier = self.evaluator.submit(negative_parameters, annotations = annotations)

            gradient_information.append((positive_parameters, positive_identifier, negative_parameters, negative_identifier))

        # Wait for gradient run results
        self.evaluator.wait()

        for d, item in enumerate(gradient_information):
            positive_parameters, positive_identifier, negative_parameters, negative_identifier = item

            positive_objective, positive_state = self.evaluator.get(positive_identifier)
            self.evaluator.clean(positive_identifier)

            negative_objective, negative_state = self.evaluator.get(negative_identifier)
            self.evaluator.clean(negative_identifier)

            gradient[d] = (positive_objective - negative_objective) / (2.0 * perturbation_length)

        # II) Update state
        self.parameters -= gradient_length * gradient

        if self.compute_objective:
            annotations = deep_merge.merge({ "type": "objective" })
            identifier = self.evaluator.submit(self.parameters, annotations = annotations)

            objective, state = self.evaluator.get(identifier)
            self.evaluator.clean(identifier)
