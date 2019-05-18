import numpy as np
import scipy.optimize

import logging
logger = logging.getLogger(__name__)

def scipy_worker(x, calibrator):
    identifier = calibrator.schedule(x)
    objective, state = calibrator.get(identifier)
    calibrator.cleanup(identifier)
    return objective

def scipy_algorithm(calibrator, **arguments):
    logger.info("Starting optimization with scipy.optimize.minimize")
    return scipy.optimize.minimize(fun = scipy_worker, x0 = calibrator.problem.initial_parameters, args = (calibrator,), **arguments)
