import numpy as np

def random_walk_algorithm(calibrator, ranges, maximum_iterations = 1000):
    best_objective = np.inf
    best_parameters = None
    iteration = 0

    while iteration < maximum_iterations:
        parameters = np.array([
            ranges[i][0] + np.random.random() * (ranges[i][1] - ranges[i][0])
            for i in range(calibrator.problem.number_of_parameters)
        ])

        identifier = calibrator.schedule(parameters)
        objective, state = calibrator.get(identifier)
        calibrator.cleanup(identifier)

        if objective < best_objective:
            print("Iteration %d, Objective %f" % (iteration, objective))
            best_objective = objective
            best_parameters = parameters

        iteration += 1

    return best_parameters, best_objective
