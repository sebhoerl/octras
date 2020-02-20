from octras import Simulator, Problem
from scipy.optimize import rosen as rosenbrock_function
import numpy as np

class TestSimulator(Simulator):
    __test__ = False

    def __init__(self):
        self.results = {}

    def ready(self, identifier):
        return True

    def get(self, identifier):
        return self.results[identifier]

    def clean(self, identifier):
        del self.results[identifier]

class RosenbrockSimulator(TestSimulator):
    def run(self, identifier, parameters):
        self.results[identifier] = rosenbrock_function(parameters["x"])

class RosenbrockProblem(Problem):
    def __init__(self, dimensions):
        self.number_of_parameters = dimensions

    def prepare(self, x):
        return dict(x = x)

    def evaluate(self, x, result):
        return result

class QuadraticSimulator(TestSimulator):
    def run(self, identifier, parameters):
        us, xs = parameters["u"], parameters["x"]
        self.results[identifier] = sum([(x - u)**2 for u, x in zip(us, xs)])

class QuadraticProblem(Problem):
    def __init__(self, u = [0.0], initial = [0.0]):
        self.number_of_parameters = len(u)
        self.u = u

        self.initial = initial
        self.bounds = [[-10, 10]] * len(u)

    def prepare(self, x):
        return dict(x = x, u = self.u)

    def evaluate(self, x, result):
        return result

class SISSimulator(TestSimulator):
    """
        Integrates a SIS epidemic model with infection rate beta and
        recovery rate gamma.
    """
    def run(self, identifier, parameters):
        N, I, S = 1000, 200, 800

        if "restart" in parameters:
            restart = self.get(parameters["restart"])
            S, I = N * restart[0], N * restart[1]

        steps = parameters["steps"] if "steps" in parameters else 5000
        dt = parameters["dt"] if "dt" in parameters else 1e-2

        beta = parameters["beta"]
        gamma = parameters["gamma"]

        for k in range(steps):
            dSdt = -beta * (S * I) / N + gamma * I
            dIdt = beta * (S * I) / N - gamma * I

            I += dIdt * dt
            S += dSdt * dt

            I = max(min(I, N), 0)
            S = max(min(S, N), 0)

        self.results[identifier] = (S / N, I / N)

class SISProblem(Problem):
    def __init__(self, target, gamma = 0.2):
        self.target = target
        self.gamma = gamma
        self.number_of_parameters = 1

        self.bounds = [[0.0, 1.0]]

    def prepare(self, x):
        return dict(beta = x[0], gamma = self.gamma)

    def evaluate(self, x, result):
        return (result[1] - self.target)**2 + 1e-3 * x[0] # Minimize infected while keeping beta small

class CongestionSimulator(TestSimulator):
    def __init__(self):
        super().__init__()

        self.states = {}
        self.random_states = {}

    def run(self, identifier, parameters):
        random = np.random.RandomState(0)

        number_of_travellers = 1000
        replanning_rate = 0.01

        freeflow_travel_time_road = 30.0
        freeflow_travel_time_rail = 30.0

        beta_road = -0.2
        beta_rail = -0.3

        iterations = parameters["iterations"] if "iterations" in parameters else 1000
        capacity = parameters["capacity"]

        tastes = random.random_sample(size = (number_of_travellers,)) * 2.0
        road_selector = random.random(size = (number_of_travellers,)) > 0.5

        if "restart" in parameters:
            road_selector = self.states[parameters["restart"]]
            random.set_state(self.random_states[parameters["restart"]])

        road_count = np.sum(road_selector)

        for k in range(iterations):
            travel_time_road = freeflow_travel_time_road * (1.0 + 0.15 * (road_count / capacity)**4)
            travel_time_rail = freeflow_travel_time_rail

            utilities_road = beta_road * travel_time_road
            utilities_rail = tastes + beta_rail * travel_time_rail

            replanning_selector = random.random_sample(size = (number_of_travellers,)) < replanning_rate
            road_selector[replanning_selector] = (utilities_road > utilities_rail)[replanning_selector]
            road_count = np.sum(road_selector)

        self.states[identifier] = road_selector
        self.random_states[identifier] = random.get_state()
        self.results[identifier] = road_count / number_of_travellers

class CongestionProblem(Problem):
    def __init__(self, target_share, iterations = 1000):
        self.number_of_parameters = 1
        self.number_of_states = 2

        self.target_share = target_share
        self.iterations = iterations

        self.initial = [600]

    def prepare(self, x):
        return dict(capacity = max(1, x[0]), iterations = self.iterations), self.iterations

    def evaluate(self, x, result):
        return (result - self.target_share)**2, [result, 1.0 - result]
