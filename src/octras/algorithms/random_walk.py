import numpy as np

import logging
logger = logging.getLogger(__name__)

def random_walk_algorithm(calibrator, parallel_samples = 1):
    iteration = 0

    while not calibrator.finished:
        iteration += 1
        logger.info("Starting Random Walk iteration %d." % iteration)

        parameters = [np.array([
            p["bounds"][0] + np.random.random() * (p["bounds"][1] - p["bounds"][0])
            for p in calibrator.problem.parameters
        ]) for k in range(parallel_samples)]

        identifiers = [calibrator.schedule(p) for p in parameters]
        calibrator.wait(identifiers)

        for identifier in identifiers:
            calibrator.get(identifier)
            calibrator.cleanup(identifier)
