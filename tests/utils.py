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
        return np.array([self.results[identifier]])

class RosenbrockSimulator(octras.simulation.Simulator):
    def __init__(self):
        self.results = {}

    def run(self, identifier, parameters):
        dimensions = parameters["dimensions"]
        x = parameters["x"]

        if len(x) != dimensions:
            raise RuntimeError("Wrong dimension")

        value = sum([
            100 * (x[i + 1] - x[i]**2)**2 + (1 - x[i])**2
            for i in range(dimensions - 1)
        ])
        self.results[identifier] = value

    def is_running(self, identifier):
        return False

    def get_result(self, identifier):
        return np.array([self.results[identifier]])

class RoadRailSimulator(octras.simulation.Simulator):
    def __init__(self):
        self.number_of_travellers = 1000

        self.alpha_road = 0.5 * np.random.normal(size = (self.number_of_travellers,))
        self.alpha_rail = 0.5 * 2.0 * (np.zeros(shape = (self.number_of_travellers,)) - 0.5)

        self.beta_road = -np.random.random(size = (self.number_of_travellers,)) * 0.1
        self.beta_rail = -np.random.random(size = (self.number_of_travellers,)) * 0.1
        self.beta_money = -np.random.random(size = (self.number_of_travellers,)) * 0.1

        self.results = {}
        self.states = {}
        self.iterations = {}

    def run(self, identifier, parameters):
        iterations = parameters["iterations"]
        self.iterations[identifier] = iterations

        toll = parameters["toll"]
        road_count = None

        travel_time_road = 60.0
        travel_time_rail = 30.0

        road_selector = np.random.random(size = (self.number_of_travellers,)) > 0.5

        if "initial_identifier" in parameters:
            road_selector = self.states[parameters["initial_identifier"]]

        for k in range(iterations):
            road_count = np.sum(road_selector)
            travel_time_road = 30.0 * (1.0 - road_count / self.number_of_travellers)

            utilities_road = self.alpha_road + self.beta_road * travel_time_road + self.beta_money * toll + 0.01 * np.random.normal(size = (self.number_of_travellers,))
            utilities_rail = self.alpha_rail + self.beta_rail * travel_time_rail + 0.01 * np.random.normal(size = (self.number_of_travellers,))

            replanning_selector = np.random.random(size = (self.number_of_travellers,)) < 0.07
            road_selector[replanning_selector] = (utilities_road > utilities_rail)[replanning_selector]

        self.results[identifier] = road_count / self.number_of_travellers
        self.states[identifier] = road_selector

    def is_running(self, identifier):
        return False

    def get_result(self, identifier):
        return np.array([self.results[identifier]])

    def get_cost(self, identifier):
        return self.iterations[identifier]

class RoadRailProblem(octras.optimization.OptimizationProblem):
    def __init__(self, default_parameters = {}):
        parameters = [{ "name": "toll", "initial": 0.0, "bounds": (0.0, 50.0) }]
        octras.optimization.OptimizationProblem.__init__(self, 1, parameters)
        self.default_parameters = default_parameters

    def get_simulator_parameters(self, values):
        parameters = {}
        parameters.update(self.default_parameters)
        parameters.update({ "toll": values[0] })
        return parameters

    def evaluate(self, parameters, simulator_result):
        return np.sum(np.abs(simulator_result - 0.75)**2), simulator_result

    def get_reference_state(self):
        return np.array([0.75])

class RealDimensionalProblem(octras.optimization.OptimizationProblem):
    def __init__(self, dimensions):
        parameters = [{ "name": "x%d" % d, "initial": 0.0, "bounds": (-5.0, 5.0) } for d in range(dimensions)]
        octras.optimization.OptimizationProblem.__init__(self, 1, parameters)
        self.dimensions = dimensions

    def get_simulator_parameters(self, parameters):
        return { "x": parameters, "dimensions": self.dimensions }

    def evaluate(self, parameters, simulator_result):
        return simulator_result[0], simulator_result

    def get_reference_state(self):
        return np.array([0.0])
