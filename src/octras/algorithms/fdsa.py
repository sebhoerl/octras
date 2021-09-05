import numpy as np
from deepmerge import always_merger

import logging
logger = logging.getLogger("octras")

from octras import Evaluator

# https://www.jhuapl.edu/spsa/PDF-SPSA/Spall_Implementation_of_the_Simultaneous.PDF

class FDSA:
    def __init__(self, problem, perturbation_factor, gradient_factor, perturbation_exponent = 0.101, gradient_exponent = 0.602, gradient_offset = 0, compute_objective = True):
        self.perturbation_factor = perturbation_factor
        self.perturbation_exponent = perturbation_exponent

        self.gradient_factor = gradient_factor
        self.gradient_exponent = gradient_exponent
        self.gradient_offset = gradient_offset

        self.compute_objective = compute_objective

        self.iteration = 0

        problem_information = problem.get_information()

        if not "number_of_parameters" in problem_information:
            raise RuntimeError("FDSA expects number_of_parameters in problem information.")

        if not "initial_values" in problem_information:
            raise RuntimeError("FDSA expects initial_values in problem information.")

        self.number_of_parameters = problem_information["number_of_parameters"]
        self.parameters = np.array(problem_information["initial_values"])

    def advance(self, evaluator: Evaluator):
        self.iteration += 1
        logger.info("Starting FDSA iteration %d." % self.iteration)

        # Calculate objective
        if self.compute_objective:
            annotations = { "type": "objective" }
            objective_identifier = evaluator.submit(self.parameters, annotations = annotations)

        # Update lengths
        gradient_length = self.gradient_factor / (self.iteration + self.gradient_offset)**self.gradient_exponent
        perturbation_length = self.perturbation_factor / self.iteration**self.perturbation_exponent

        annotations = {
            "gradient_length" : gradient_length,
            "perturbation_length": perturbation_length,
            "type": "gradient"
        }

        # I) Calculate gradients
        gradient = np.zeros((len(self.parameters),))
        gradient_information = []

        # Schedule all necessary runs
        for d in range(len(self.parameters)):
            annotations = always_merger.merge(annotations, { "dimension": d })

            positive_parameters = np.copy(self.parameters)
            positive_parameters[d] += perturbation_length
            annotations = always_merger.merge(annotations, { "type": "positive_gradient" })
            positive_identifier = evaluator.submit(positive_parameters, annotations = annotations)

            negative_parameters = np.copy(self.parameters)
            negative_parameters[d] -= perturbation_length
            annotations = always_merger.merge(annotations, { "sign": "negative_gradient" })
            negative_identifier = evaluator.submit(negative_parameters, annotations = annotations)

            gradient_information.append((positive_parameters, positive_identifier, negative_parameters, negative_identifier))

        # Wait for gradient run results
        evaluator.wait()

        if self.compute_objective:
            evaluator.clean(objective_identifier)

        for d, item in enumerate(gradient_information):
            positive_parameters, positive_identifier, negative_parameters, negative_identifier = item

            positive_objective, positive_state = evaluator.get(positive_identifier)
            evaluator.clean(positive_identifier)

            negative_objective, negative_state = evaluator.get(negative_identifier)
            evaluator.clean(negative_identifier)

            gradient[d] = (positive_objective - negative_objective) / (2.0 * perturbation_length)

        # II) Update state
        self.parameters -= gradient_length * gradient
