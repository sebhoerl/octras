import numpy as np

import logging
logger = logging.getLogger("octras")

from octras import Evaluator

def has_duplicates(values):
    for k in range(len(values)):
        if np.sum(np.sum(values[k] == values, axis = 1) == 2) > 1:
            return True

    return False

class NelderMead:
    def __init__(self, problem, alpha = 1.0, gamma = 2.0, rho = 0.5, sigma = 0.5, seed = 0):
        self.alpha = alpha
        self.gamma = gamma
        self.rho = rho
        self.sigma = sigma

        self.iteration = 0

        problem_information = problem.get_information()

        if not "bounds" in problem_information:
            raise RuntimeError("Nelder-Mead expects bounds in problem information.")

        if not "number_of_parameters" in problem_information:
            raise RuntimeError("Nelder-Mead expects number_of_parameters in problem information.")

        self.number_of_parameters = problem_information["number_of_parameters"]
        self.bounds = np.array(problem_information["bounds"])

        self.simplex = None
        self.values = None

        self.random = np.random.RandomState(seed)

    def advance(self, evaluator):
        self.iteration += 1
        logger.info("Starting Nelder-Mead iteration %d." % self.iteration)

        if self.simplex is None:
            logger.info("Initializing simplex ...")

            found_duplicates = True

            while found_duplicates:
                simplex = []

                for k in range(self.number_of_parameters):
                    simplex.append(self.bounds[
                        k, self.random.randint(0, 2, self.number_of_parameters + 1)
                    ])

                self.simplex = np.array(simplex).T
                found_duplicates = has_duplicates(self.simplex)

            identifiers = [
                evaluator.submit(parameters)
                for parameters in self.simplex
            ]

            evaluator.wait()

            self.values = np.array([
                evaluator.get(identifier)[0]
                for identifier in identifiers
            ])

            for identifier in identifiers:
                evaluator.clean(identifier)

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
        identifier = evaluator.submit(reflection)
        reflection_value = evaluator.get(identifier)[0]
        evaluator.clean(identifier)

        if self.values[0] <= reflection_value and reflection_value <= self.values[-2]:
            self.simplex[-1] = reflection
            self.values[-1] = reflection_value

            logger.info("  Accepted reflection")
            return

        # 4) Expansion
        logger.info("Expansion ...")
        if reflection_value < self.values[0]:
            expansion = centroid + self.gamma * (reflection - centroid)
            identifier = evaluator.submit(expansion)
            expansion_value = evaluator.get(identifier)[0]
            evaluator.clean(identifier)

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
        identifier = evaluator.submit(contraction)
        contraction_value = evaluator.get(identifier)[0]
        evaluator.clean(identifier)

        if contraction_value < self.values[-1]:
            self.simplex[-1] = contraction
            self.values[-1] = contraction_value
            logger.info("  Accepted contraction")

            return

        # 6) Shrink
        logger.info("Shrinking simplex ...")
        self.simplex[1:] = self.simplex[0] + self.sigma * (self.simplex[1:] - self.simplex[0])

        identifiers = [
            evaluator.submit(parameters)
            for parameters in self.simplex[1:]
        ]

        evaluator.wait()

        self.values[1:] = np.array([
            evaluator.get(identifier)[0]
            for identifier in identifiers
        ])

        for identifier in identifiers:
            evaluator.clean(identifier)
