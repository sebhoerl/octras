import numpy as np
import deep_merge

import logging
logger = logging.getLogger("octras")

# https://www.jhuapl.edu/spsa/PDF-SPSA/Spall_Implementation_of_the_Simultaneous.PDF

from octras import Evaluator

class SPSA:
    def __init__(self, problem, perturbation_factor, gradient_factor, perturbation_exponent = 0.101, gradient_exponent = 0.602, gradient_offset = 0, compute_objective = True, seed = 0):
        self.perturbation_factor = perturbation_factor
        self.perturbation_exponent = perturbation_exponent

        self.gradient_factor = gradient_factor
        self.gradient_exponent = gradient_exponent
        self.gradient_offset = gradient_offset

        self.compute_objective = compute_objective

        self.iteration = 0

        self.random = np.random.RandomState(seed)

        problem_information = problem.get_information()

        if not "number_of_parameters" in problem_information:
            raise RuntimeError("SPSA expects number_of_parameters in problem information.")

        if not "initial_values" in problem_information:
            raise RuntimeError("SPSA expects initial_values in problem information.")

        self.number_of_parameters = problem_information["number_of_parameters"]
        self.parameters = np.array(problem_information["initial_values"])

    def advance(self, evaluator: Evaluator):
        self.iteration += 1
        logger.info("Starting SPSA iteration %d." % self.iteration)

        if self.parameters is None:
            self.parameters = evaluator.problem.initial

        # Calculate objective
        if self.compute_objective:
            annotations = { "type": "objective" }
            objective_identifier = evaluator.submit(self.parameters, annotations = annotations)

        # Update step lengths
        gradient_length = self.gradient_factor / (self.iteration + self.gradient_offset)**self.gradient_exponent
        perturbation_length = self.perturbation_factor / self.iteration**self.perturbation_exponent

        # Sample direction from Rademacher distribution
        direction = self.random.randint(0, 2, len(self.parameters)) - 0.5

        annotations = {
            "gradient_length": gradient_length,
            "perturbation_length": perturbation_length,
            "direction": direction,
            "type": "gradient"
        }

        # Schedule samples
        positive_parameters = np.copy(self.parameters)
        positive_parameters += direction * perturbation_length
        annotations = deep_merge.merge(annotations, { "type": "positive_gradient" })
        positive_identifier = evaluator.submit(positive_parameters, annotations = annotations)

        negative_parameters = np.copy(self.parameters)
        negative_parameters -= direction * perturbation_length
        annotations = deep_merge.merge(annotations, { "type": "negative_gradient" })
        negative_identifier = evaluator.submit(negative_parameters, annotations = annotations)

        # Wait for gradient run results
        evaluator.wait()

        if self.compute_objective:
            evaluator.clean(objective_identifier)

        positive_objective, positive_state = evaluator.get(positive_identifier)
        evaluator.clean(positive_identifier)

        negative_objective, negative_state = evaluator.get(negative_identifier)
        evaluator.clean(negative_identifier)

        g_k = (positive_objective - negative_objective) / (2.0 * perturbation_length)
        g_k *= direction**-1

        # Update state
        self.parameters -= gradient_length * g_k
