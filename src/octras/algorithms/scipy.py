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
    x0 = [p["initial"] for p in calibrator.problem.parameters]
    return scipy.optimize.minimize(fun = scipy_worker, x0 = x0, args = (calibrator,), **arguments)
