from .generic import GenericTestSimulator
from octras import Problem

class QuadraticSimulator(GenericTestSimulator):
    def run(self, identifier, parameters):
        us, xs = parameters["u"], parameters["x"]
        self.results[identifier] = sum([(x - u)**2 for u, x in zip(us, xs)])

class QuadraticProblem(Problem):
    def __init__(self, u = [0.0], initial = [0.0]):
        self.u = u
        self.initial = initial
        self.bounds = [[-10.0, 10.0]] * len(u)

    def get_information(self):
        return {
            "number_of_parameters": len(self.u),
            "initial_values": self.initial,
            "bounds": self.bounds
        }

    def parameterize(self, x):
        return dict(x = x, u = self.u)

    def evaluate(self, x, response):
        return response
