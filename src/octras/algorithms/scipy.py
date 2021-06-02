import scipy.optimize

import logging
logger = logging.getLogger("octras")

from octras import Evaluator

class ScipyAlgorithm:
    def __init__(self, problem, **arguments):
        self.arguments = arguments

        problem_information = problem.get_information()

        if not "initial_values" in problem_information:
            raise RuntimeError("Random Walk expects initial_values in problem information.")

        self.initial = problem_information["initial_values"]
        self.evaluator = None

    def _worker(self, x):
        identifier = self.evaluator.submit(x)
        objective, state = self.evaluator.get(identifier)
        self.evaluator.clean(identifier)
        return objective

    def advance(self, evaluator: Evaluator):
        logger.info("Starting optimization with scipy.optimize.minimize")

        self.evaluator = evaluator

        return scipy.optimize.minimize(
            fun = self._worker,
            x0 = self.initial,
            **self.arguments
        )
