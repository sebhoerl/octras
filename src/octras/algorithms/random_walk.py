import numpy as np

import logging
logger = logging.getLogger("octras")

from octras import Evaluator

class RandomWalk:
    def __init__(self, problem, parallel = 1, seed = 0):
        self.parallel = parallel

        self.iteration = 0
        self.random = np.random.RandomState(seed)

        problem_information = problem.get_information()

        if not "bounds" in problem_information:
            raise RuntimeError("Random Walk expects bounds in problem information.")

        self.bounds = problem_information["bounds"]

    def advance(self, evaluator: Evaluator):
        self.iteration += 1
        logger.info("Starting Random Walk iteration %d" % self.iteration)

        parameters = [np.array([
            bounds[0] + self.random.random() * (bounds[1] - bounds[0]) # TODO: Not demterinistic!
            for bounds in self.bounds
        ]) for k in range(self.parallel)]

        identifiers = [evaluator.submit(p) for p in parameters]

        evaluator.wait(identifiers)
        evaluator.clean()
