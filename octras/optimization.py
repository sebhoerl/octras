import numpy as np
import logging
import pickle
import copy

logger = logging.getLogger(__name__)

class OptimizationProblem:
    def __init__(self, number_of_parameters, number_of_states, initial_parameters):
        self.number_of_parameters = number_of_parameters
        self.number_of_states = number_of_states
        self.initial_parameters = initial_parameters

    def get_simulator_parameters(self, parameters):
        raise NotImplementedError()

    def evaluate(self, parameters, result):
        raise NotImplementedError()

class Optimizer:
    def __init__(self, scheduler, problem, tolerance = -1.0, maximum_evaluations = np.inf, maximum_cost = np.inf, history_path = None):
        self.scheduler = scheduler
        self.problem = problem

        self.simulator_parameters = {}
        self.parameters = {}
        self.results = {}
        self.annotations = {}

        self.history_path = history_path
        self.history = []
        self.history_identifiers = set()

        self.finished = False
        self.success = False
        self.exit_criterion = None

        self.evaluations = 0.0
        self.maximum_evaluations = maximum_evaluations

        self.total_cost = 0.0
        self.maximum_cost = maximum_cost

        self.best_objective = np.inf
        self.best_parameters = None
        self.tolerance = tolerance

    def schedule(self, parameters, simulator_parameters = {}, annotations = {}):
        _simulator_parameters = copy.deepcopy(self.problem.get_simulator_parameters(parameters))
        _simulator_parameters.update(simulator_parameters)

        identifier = self.scheduler.schedule(_simulator_parameters)

        self.simulator_parameters[identifier] = _simulator_parameters
        self.parameters[identifier] = copy.deepcopy(parameters)
        self.annotations[identifier] = copy.deepcopy(annotations)

        return identifier

    def wait(self, identifiers = None):
        return self.scheduler.wait(identifiers)

    def get(self, identifier, transient = False):
        simulator_result = self.scheduler.get_result(identifier)
        objective, state = self.problem.evaluate(self.parameters[identifier], simulator_result)

        if not identifier in self.history_identifiers:
            self.history_identifiers.add(identifier)
            self.evaluations += 1

            cost = self.scheduler.get_cost(identifier)
            self.total_cost += cost

            self.history.append({
                "state": state, "objective": objective,
                "parameters": self.parameters[identifier],
                "identifier": identifier, "annotations": self.annotations[identifier],
                "evaluations": self.evaluations, "total_cost": self.total_cost, "cost": cost
            })

            if self.evaluations > self.maximum_evaluations:
                self.finished = True
                self.exit_criterion = "Maximum evaluations reached."
                logger.warning("Maximum evaluations reached.")

            if self.total_cost > self.maximum_cost:
                self.finished = True
                self.exit_criterion = "Maximum cost reached."
                logger.warning("Maximum cost reached.")

        if not transient and objective < self.best_objective:
            delta = self.best_objective - objective
            self.best_objective = objective
            self.best_parameters = self.parameters[identifier]

            logger.info("Best objective: %.2e (Δ %.2e) (evaluations: %d, cost: %f)" % (
                objective, delta, self.evaluations, self.total_cost
            ))

            if delta <= self.tolerance:
                self.finished = True
                self.success = True
                self.exit_criterion = "Tolerance reached."
                logger.info("Tolerance reached. Successful optimization.")

        if self.history_path is not None:
            self.save(self.history_path)

        return objective, state

    def save(self, path):
        with open(path, "wb+") as f:
            pickle.dump(self.history, f)

    def cleanup(self, identifier):
        self.scheduler.cleanup(identifier)
