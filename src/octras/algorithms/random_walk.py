import numpy as np

import logging
logger = logging.getLogger(__name__)

def random_walk_algorithm(calibrator):
    iteration = 0

    while not calibrator.finished:
        iteration += 1
        logger.info("Starting Random Walk iteration %d." % iteration)

        parameters = np.array([
            p["bounds"][0] + np.random.random() * (p["bounds"][1] - p["bounds"][0])
            for p in calibrator.problem.parameters
        ])

        identifier = calibrator.schedule(parameters)
        objective, state = calibrator.get(identifier)
        calibrator.cleanup(identifier)
