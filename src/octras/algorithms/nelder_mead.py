import numpy as np

import logging
logger = logging.getLogger(__name__)

def has_duplicates(values):
    for k in range(len(values)):
        if np.sum(np.sum(values[k] == values, axis = 1) == 2) > 1:
            return True

    return False

class NelderMead:
    def __init__(self, evaluator, alpha = 1.0, gamma = 2.0, rho = 0.5, sigma = 0.5, seed = 0, bounds = None):
        self.evaluator = evaluator

        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma
        self.seed = seed
        self.bounds = bounds

        self.iteration = 0

        if not hasattr(self.evaluator.problem, "bounds") and bounds is None:
            raise RuntimeError("Bounds must be provided for Nelder Mead")

        self.simplex = None
        self.values = None

    def advance(self):
        self.iteration += 1
        logger.info("Starting Nelder-Mead iteration %d." % self.iteration)

        if self.simplex is None:
            logger.info("Initializing simplex ...")

            if hasattr(self.evaluator.problem, "bounds"):
                bounds = np.array(self.evaluator.problem.bounds)
            else:
                bounds = np.array(self.bounds)

            random = np.random.RandomState(self.seed)
            found_duplicates = True

            while found_duplicates:
                simplex = []

                for k in range(self.evaluator.problem.number_of_parameters):
                    simplex.append(bounds[
                        k, random.randint(0, 2, self.evaluator.problem.number_of_parameters + 1)
                    ])

                self.simplex = np.array(simplex).T
                found_duplicates = has_duplicates(self.simplex)

            identifiers = [
                self.evaluator.submit(parameters)
                for parameters in self.simplex
            ]

            self.evaluator.wait()

            self.values = np.array([
                self.evaluator.get(identifier)[0]
                for identifier in identifiers
            ])

            for identifier in identifiers:
                self.evaluator.clean(identifier)

            logger.info("Initialization finished.")

        # 1) Sort simplex
        logger.info("Sorting simplex ...")
        sorter = np.argsort(self.values)
        self.values = self.values[sorter]
        self.simplex = self.simplex[sorter]

        # 2) Calculate centroid
        logger.info("Calculating centroid ...")
        centroid = np.mean(self.simplex[:-1], axis = 0)

        # 3) Reflection
        logger.info("Reflection ...")
        reflection = centroid + self.alpha * (centroid - self.simplex[-1])
        identifier = self.evaluator.submit(reflection)
        reflection_value = self.evaluator.get(identifier)[0]
        self.evaluator.clean(identifier)

        if self.values[0] <= reflection_value and reflection_value <= self.values[-2]:
            self.simplex[-1] = reflection
            self.values[-1] = reflection_value

            logger.info("  Accepted reflection")
            return

        # 4) Expansion
        logger.info("Expansion ...")
        if reflection_value < self.values[0]:
            expansion = centroid + self.gamma * (reflection - centroid)
            identifier = self.evaluator.submit(expansion)
            expansion_value = self.evaluator.get(identifier)[0]
            self.evaluator.clean(identifier)

            if expansion_value < reflection_value:
                self.simplex[-1] = expansion
                self.values[-1] = expansion_value
                logger.info("  Accepted expansion")
            else:
                self.simplex[-1] = reflection
                self.values[-1] = reflection_value
                logger.info("  Accepted reflection")

            return

        # 5) Contraction
        logger.info("Contraction ...")
        contraction = centroid + self.rho * (self.simplex[-1] - centroid)
        identifier = self.evaluator.submit(contraction)
        contraction_value = self.evaluator.get(identifier)[0]
        self.evaluator.clean(identifier)

        if contraction_value < self.values[-1]:
            self.simplex[-1] = contraction
            self.values[-1] = contraction_value
            logger.info("  Accepted contraction")

            return

        # 6) Shrink
        logger.info("Shrinking simplex ...")
        self.simplex[1:] = self.simplex[0] + self.sigma * (self.simplex[1:] - self.simplex[0])

        identifiers = [
            self.evaluator.submit(parameters)
            for parameters in self.simplex[1:]
        ]

        self.evaluator.wait()

        self.values[1:] = np.array([
            self.evaluator.get(identifier)[0]
            for identifier in identifiers
        ])

        for identifier in identifiers:
            self.evaluator.clean(identifier)
