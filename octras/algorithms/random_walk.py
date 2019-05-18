import numpy as np

import logging
logger = logging.getLogger(__name__)

def random_walk_algorithm(calibrator, ranges):
    iteration = 0

    while not calibrator.finished:
        iteration += 1
        logger.info("Starting Random Walk iteration %d." % iteration)

        parameters = np.array([
            ranges[i][0] + np.random.random() * (ranges[i][1] - ranges[i][0])
            for i in range(calibrator.problem.number_of_parameters)
        ])

        identifier = calibrator.schedule(parameters)
        objective, state = calibrator.get(identifier)
        calibrator.cleanup(identifier)
