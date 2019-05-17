import numpy as np
import scipy.optimize

def scipy_worker(x, calibrator):
    identifier = calibrator.schedule(x)
    objective, state = calibrator.get(identifier)
    calibrator.cleanup(identifier)

    return objective

def scipy_algorithm(calibrator, initial_parameters = None, **arguments):
    if initial_parameters is None:
        initial_parameters = np.zeros((calibrator.problem.number_of_parameters,))

    return scipy.optimize.minimize(fun = scipy_worker, x0 = initial_parameters, args = (calibrator,), **arguments)
