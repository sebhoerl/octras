import scipy.optimize

import logging
logger = logging.getLogger(__name__)

class ScipyAlgorithm:
    def __init__(self, evaluator, **arguments):
        self.evaluator = evaluator
        self.arguments = arguments

        if not hasattr(self.evaluator.problem, "initial"):
            raise RuntimeError("Initial parameters must be provided by problem for SciPy")

    def _worker(self, x):
        identifier = self.evaluator.submit(x)
        objective, state = self.evaluator.get(identifier)
        self.evaluator.clean(identifier)
        return objective

    def advance(self):
        logger.info("Starting optimization with scipy.optimize.minimize")

        return scipy.optimize.minimize(
            fun = self._worker,
            x0 = self.evaluator.problem.initial,
            **self.arguments
        )
