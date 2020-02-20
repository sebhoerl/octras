import numpy as np

import logging
logger = logging.getLogger(__name__)

class RandomWalk:
    def __init__(self, evaluator, parallel = 1):
        self.evaluator = evaluator
        self.problem = self.evaluator.problem

        self.parallel = parallel

        self.iteration = 0

        if not hasattr(self.problem, "bounds"):
            raise RuntimeError("Problem needs to provide bounds if RandomWalk is used.")

    def advance(self):
        self.iteration += 1
        logger.info("Starting Random Walk iteration %d" % self.iteration)

        parameters = [np.array([
            bounds[0] + np.random.random() * (bounds[1] - bounds[0]) # TODO: Not demterinistic!
            for bounds in self.problem.bounds
        ]) for k in range(self.parallel)]

        identifiers = [self.evaluator.submit(p) for p in parameters]

        self.evaluator.wait(identifiers)
        self.evaluator.clean()
