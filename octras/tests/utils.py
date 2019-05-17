import octras.simulation
import octras.optimization
import numpy as np

class QuadraticSumSimulator(octras.simulation.Simulator):
    def __init__(self, zeros):
        self.results = {}
        self.zeros = zeros

    def run(self, identifier, parameters):
        x = parameters["x"]

        if len(self.zeros) != len(x) or len(x) == 0:
            raise RuntimeError()

        value = 0.0

        for zero, xi in zip(self.zeros, x):
            value += (xi - zero)**2

        self.results[identifier] = value

    def is_running(self, identifier):
        return False

    def get_result(self, identifier):
        return self.results[identifier]

class RosenbrockSimulator(octras.simulation.Simulator):
    def __init__(self):
        self.results = {}

    def run(self, identifier, parameters):
        dimensions = parameters["dimensions"]
        x = parameters["x"]

        value = sum([100 * (x[i + 1] - x[i]**2)**2 + (x[i] - 1)**2 for i in range(dimensions - 1)])
        self.results[identifier] = value

    def is_running(self, identifier):
        return False

    def get_result(self, identifier):
        return self.results[identifier]

class ExponentialSimulator(octras.simulation.Simulator):
    def __init__(self, C, l, step_size = 1e-3, default_iterations = 100):
        self.results = {}
        self.iterations = {}

        self.l = l
        self.C = C
        self.step_size = step_size
        self.default_iterations = default_iterations

    def run(self, identifier, parameters):
        if "iterations" in parameters:
            iterations = parameters["iterations"]
        else:
            iterations = self.default_iterations

        if "initial_identifier" in parameters:
            print("initial %d -> %d" % (iterations, iterations + self.iterations[parameters["initial_identifier"]]))
            iterations += self.iterations[parameters["initial_identifier"]]

        x = parameters["x"]

        self.results[identifier] = [
            Ci * np.exp(li * iterations * self.step_size) + xi
            for Ci, li, xi in zip(self.C, self.l, x)
        ]

        self.iterations[identifier] = iterations

    def is_running(self, identifier):
        return False

    def get_result(self, identifier):
        return self.results[identifier]

class ExponentialProblem(octras.optimization.OptimizationProblem):
    def __init__(self, dimensions):
        octras.optimization.OptimizationProblem.__init__(self, dimensions, dimensions, np.zeros((dimensions,)))
        self.dimensions = dimensions

    def get_simulator_parameters(self, parameters):
        return { "x": parameters }

    def compute_state(self, parameters, simulator_result):
        return simulator_result

    def compute_objective(self, parameters, state):
        return np.sum(np.abs(state))

class RealDimensionalProblem(octras.optimization.OptimizationProblem):
    def __init__(self, dimensions, variable_name = "x", dimensions_name = "dimensions"):
        octras.optimization.OptimizationProblem.__init__(self, dimensions, 1, np.zeros((dimensions,)))
        self.dimensions = dimensions
        self.variable_name = variable_name
        self.dimensions_name = dimensions_name

    def get_simulator_parameters(self, parameters):
        return { self.variable_name: parameters, self.dimensions_name: self.dimensions }

    def compute_state(self, parameters, simulator_result):
        return simulator_result
