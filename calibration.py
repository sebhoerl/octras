import numpy as np
import matplotlib.pyplot as plt
import json
import time
import pickle

class Calibrator:
    def __init__(self, problem, maximum_iterations = None):
        self.problem = problem
        self.maximum_iterations = maximum_iterations

        self.samples = []
        self.current_iterations = 0

        self.best_objective = np.inf
        self.best_sample = None

        self.converged = False

        self.initial_time = time.time()

    def load(self, path, ignore_missing_file = False):
        if os.path.isfile(path):
            with open(path, "rb") as f:
                self.samples, self.current_iterations = pickle.load(f)
        elif not ignore_missing_file:
            raise RuntimeError("File does not exist:", path)

    def save(self, path):
        with open(path, "wb+") as f:
            pickle.dump((self.samples, self.current_iterations), f)

    def add_sample(self, parameters, state, objective, iterations, annotations = {}):
        if len(parameters) != self.problem.number_of_parameters:
            raise RuntimeError("Wrong number of parameters")

        if len(state) != self.problem.number_of_states:
            raise RuntimeError("Wrong number of states")

        self.current_iterations += iterations

        self.samples.append({
            "parameters": parameters,
            "state": state,
            "objective": objective,
            "calibration_iterations": self.current_iterations,
            "sample_iterations": iterations,
            "annotations": annotations,
            "time": time.time() - self.initial_time
        })

        if objective < self.best_objective:
            absolute_difference = objective - self.best_objective
            relative_difference = objective / self.best_objective - 1.0

            self.best_objective = objective
            self.best_sample = self.samples[-1]

            self.converged = self.problem.is_converged(state, objective, absolute_difference, relative_difference)

        print("[Calibrator] Received sample. Objective: %f (best %f), Iterations: %d" % (objective, self.best_objective, self.current_iterations))

        if self.maximum_iterations is not None and self.current_iterations >= self.maximum_iterations:
            print("[Calibrator] Reached maximum number of iterations.")
            self.converged = True

    def plot(self, path):
        # Format data for plotting
        iterations = np.zeros((len(self.samples),))

        objectives = np.zeros((len(self.samples),))
        best_objectives = np.zeros((len(self.samples),))

        parameters = np.zeros((len(self.samples), self.problem.number_of_parameters))
        best_parameters = np.zeros((len(self.samples), self.problem.number_of_parameters))

        states = np.zeros((len(self.samples), self.problem.number_of_states))
        best_states = np.zeros((len(self.samples), self.problem.number_of_states))

        current_best_objective = np.inf
        current_best_parameters = None
        current_best_states = None

        for k, sample in enumerate(self.samples):
            iterations[k] = sample["calibration_iterations"]
            objectives[k] = sample["objective"]
            parameters[k] = sample["parameters"]
            states[k] = sample["state"]

            if sample["objective"] < current_best_objective:
                current_best_objective = sample["objective"]
                current_best_parameters = sample["parameters"]
                current_best_states = sample["state"]

            best_objectives[k] = current_best_objective
            best_parameters[k] = current_best_parameters
            best_states[k] = current_best_states


        plt.figure(dpi = 120, figsize = (6, 4 * 3))

        # Objective
        plt.subplot(3,1,1)

        plt.plot(iterations, objectives, ".", label = "Samples", color = "C0")
        plt.plot(iterations, best_objectives, label = "Best", color = "C0")

        plt.xlabel("Iterations")
        plt.ylabel("Objective")

        plt.grid()
        plt.legend(loc = "best")

        # Parameters
        plt.subplot(3,1,2)

        for k in range(self.problem.number_of_parameters):
            plt.plot(iterations, parameters[:,k], ".", color = "C%d" % k)
            plt.plot(iterations, best_parameters[:,k], color = "C%d" % k, label = self.problem.parameter_names[k])

        plt.xlabel("Iterations")
        plt.ylabel("Parameter")

        plt.grid()
        plt.legend(loc = "best")

        # Parameters
        plt.subplot(3,1,3)

        for k in range(self.problem.number_of_states):
            plt.plot(iterations, states[:,k], ".", color = "C%d" % k)
            plt.plot(iterations, best_states[:,k], color = "C%d" % k, label = self.problem.state_names[k])

        plt.xlabel("Iterations")
        plt.ylabel("State")

        plt.grid()
        plt.legend(loc = "best")

        plt.savefig(path)

    #def run(self, simulator, algorithm):
    #    generator = algorithm.run(self.problem, simulator)
    #
    #    while not self.converged:
    #        sample = generator()
    #
    #        if len(sample) == 4:
    #            annotations = {}
    #            parameters, state, objective, iterations = sample
    #        elif len(sample) == 5:
    #            parameters, state, objective, iterations, annotations = sample
    #        else:
    #            raise RuntimeError()
    #
    #        self.add_sample(parameters, state, objective, iterations, annotations)

class CalibrationProblem:
    def __init__(self, number_of_parameters, number_of_states, parameter_names = None, state_names = None):
        self.number_of_parameters = number_of_parameters
        self.number_of_states = number_of_states

        if parameter_names is None:
            parameter_names = [ "param_%d" % i for i in range(number_of_parameters) ]
        elif len(parameter_names) != number_of_parameters:
            raise RuntimeError("Wrong number of parameter names")

        self.parameter_names = parameter_names

        if state_names is None:
            state_names = [ "state_%d" % i for i in range(number_of_states) ]
        elif len(state_names) != number_of_states:
            raise RuntimeError("Wrong number of state names")

        self.state_names = state_names

    def get_simulation_parameters(self, parameters):
        raise NotImplementedError()

    def get_state(self, simulation):
        raise NotImplementedError()

    def get_objective(self, state):
        raise NotImplementedError()

    def is_converged(self, state, objective, absolute_difference, relative_difference):
        return False

#class CalibrationAlgorithm:
#    def run(self):
#        raise NotImplementedError()
