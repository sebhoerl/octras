from .generic import GenericTestSimulator
from octras import Problem

from scipy.optimize import rosen as rosenbrock_function
import numpy as np

class RosenbrockSimulator(GenericTestSimulator):
    def run(self, identifier, parameters):
        self.results[identifier] = rosenbrock_function(parameters["x"])

class RosenbrockProblem(Problem):
    def __init__(self, dimensions):
        self.number_of_parameters = dimensions
        self.initial = [0.0] * dimensions
        self.bounds = np.array([
            [-2, 2] for n in range(dimensions)
        ])

    def get_information(self):
        return {
            "number_of_parameters": self.number_of_parameters,
            "initial_values": self.initial,
            "bounds": self.bounds
        }

    def parameterize(self, x):
        return dict(x = x)

    def evaluate(self, x, response):
        return response
