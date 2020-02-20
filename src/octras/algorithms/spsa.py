import numpy as np
import deep_merge

import logging
logger = logging.getLogger(__name__)

# https://www.jhuapl.edu/spsa/PDF-SPSA/Spall_Implementation_of_the_Simultaneous.PDF

class SPSA:
    def __init__(self, evaluator, perturbation_factor, gradient_factor, perturbation_exponent = 0.101, gradient_exponent = 0.602, gradient_offset = 0, compute_objective = False, seed = None):
        self.evaluator = evaluator

        self.perturbation_factor = perturbation_factor
        self.perturbation_exponent = perturbation_exponent

        self.gradient_factor = gradient_factor
        self.gradient_exponent = gradient_exponent
        self.gradient_offset = gradient_offset

        self.compute_objective = compute_objective

        self.iteration = 0

        self.seed = seed
        self.random = np.random.RandomState(self.seed)

        if not hasattr(self.evaluator.problem, "initial"):
            raise RuntimeError("Initial parameters must be provided by problem for SPSA")

        self.parameters = None

    def advance(self):
        self.iteration += 1
        logger.info("Starting SPSA iteration %d." % self.iteration)

        if self.parameters is None:
            self.parameters = self.evaluator.problem.initial

        # Update step lengths
        gradient_length = self.gradient_factor / (self.iteration + self.gradient_offset)**self.gradient_exponent
        perturbation_length = self.perturbation_factor / self.iteration**self.perturbation_exponent

        # Sample direction from Rademacher distribution
        direction = self.random.randint(0, 2, len(self.parameters)) - 0.5

        annotations = {
            "gradient_length": gradient_length,
            "perturbation_length": perturbation_length,
            "direction": direction
        }

        # Schedule samples
        positive_parameters = np.copy(self.parameters)
        positive_parameters += direction * perturbation_length
        annotations = deep_merge.merge(annotations, { "type": "positive_gradient" })
        positive_identifier = self.evaluator.submit(positive_parameters, annotations = annotations)

        negative_parameters = np.copy(self.parameters)
        negative_parameters -= direction * perturbation_length
        annotations = deep_merge.merge(annotations, { "type": "negative_gradient" })
        negative_identifier = self.evaluator.submit(negative_parameters, annotations = annotations)

        # Wait for gradient run results
        self.evaluator.wait()

        positive_objective, positive_state = self.evaluator.get(positive_identifier)
        self.evaluator.clean(positive_identifier)

        negative_objective, negative_state = self.evaluator.get(negative_identifier)
        self.evaluator.clean(negative_identifier)

        g_k = (positive_objective - negative_objective) / (2.0 * perturbation_length)
        g_k *= direction**-1

        # Update state
        self.parameters -= gradient_length * g_k

        if self.compute_objective:
            annotations = deep_merge.merge(annotations, { "type": "objective" })
            identifier = self.evaluator.submit(parameters, annotations = annotations)

            self.evaluator.wait()

            objective, state = self.evaluator.get(identifier)
            self.evaluator.clean(identifier)
