import numpy as np

class OptimizationProblem:
    def __init__(self, number_of_parameters, number_of_states, initial_parameters):
        self.number_of_parameters = number_of_parameters
        self.number_of_states = number_of_states
        self.initial_parameters = initial_parameters

    def get_simulator_parameters(self, parameters):
        raise RuntimeError()

    def compute_state(self, parameters, simulator_result):
        pass

    def compute_objective(self, parameters, state):
        return state

class Optimizer:
    def __init__(self, scheduler, problem):
        self.scheduler = scheduler
        self.problem = problem

        self.registry = {}
        self.history = []

        self.best_objective = np.inf
        self.best_index = None

    def schedule(self, parameters, _simulator_parameters = {}):
        simulator_parameters = self.problem.get_simulator_parameters(parameters)
        simulator_parameters.update(_simulator_parameters)
        identifier = self.scheduler.schedule(simulator_parameters)
        self.registry[identifier] = parameters
        return identifier

    def wait(self, identifiers = None, verbose = True):
        return self.scheduler.wait(identifiers, verbose)

    def get(self, identifier, verbose = False):
        simulator_result = self.scheduler.get(identifier, verbose)["result"]
        state = self.problem.compute_state(self.registry[identifier], simulator_result)
        objective = self.problem.compute_objective(self.registry[identifier], state)

        self.history.append({
            "state": state,
            "objective": objective,
            "simulator_result": simulator_result,
            "parameters": self.registry[identifier],
            "identifier": identifier
        })

        if objective < self.best_objective:
            self.best_objective = objective
            self.best_index = len(self.history) - 1

        return objective, state

    def cleanup(self, identifier):
        self.scheduler.cleanup(identifier)
