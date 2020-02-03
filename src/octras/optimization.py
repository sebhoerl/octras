import numpy as np
import logging
import pickle
import copy

logger = logging.getLogger(__name__)

class OptimizationProblem:
    def __init__(self, number_of_states, parameters):
        self.parameters = parameters
        self.number_of_parameters = len(self.parameters)
        self.number_of_states = number_of_states

    def get_simulator_parameters(self, parameters):
        raise NotImplementedError()

    def evaluate(self, parameters, result):
        raise NotImplementedError()

    def get_reference_state(self):
        raise NotImplementedError()

    def get_info(self):
        return {}

"""
@Sebastian: I doubt that Optimizer is the best way to call it.  
In fact, there is no optimization happening, but rather scheduling, evaluation and logging. No?
The name Optimizer make it confusing, because all the 'optimization machinery' 
actually happens in bo.py, sma-_es.py,...
Maybe ObjectiveEvaluator? 
"""
class Optimizer:
    def __init__(self, scheduler, problem, tolerance = -1.0, maximum_evaluations = np.inf, maximum_cost = np.inf, log_path = None):
        self.scheduler = scheduler
        self.problem = problem

        self.simulator_parameters = {}
        self.parameters = {}
        self.results = {}
        self.annotations = {}
        self.method_details = {}

        self.log_path = log_path
        self.log = []
        self.log_identifiers = set()

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

        if not identifier in self.log_identifiers:
            self.log_identifiers.add(identifier)
            self.evaluations += 1

            cost = self.scheduler.get_cost(identifier)
            self.total_cost += cost

            self.log.append({
                "state": state, "objective": objective,
                "parameters": self.parameters[identifier],
                "identifier": identifier, "annotations": self.annotations[identifier],
                "evaluations": self.evaluations, "total_cost": self.total_cost, "cost": cost,
                "transient": transient
            })

            if self.evaluations > self.maximum_evaluations:
                self.finished = True
                self.exit_criterion = "Maximum evaluations reached."
                logger.warning("Maximum evaluations reached.")

            if self.total_cost > self.maximum_cost:
                self.finished = True
                self.exit_criterion = "Maximum cost reached."
                logger.warning("Maximum cost reached.")

        if not self.finished and not transient and objective < self.best_objective:
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

        if self.log_path is not None:
            self.save(self.log_path)

        return objective, state

    def save(self, path):
        with open(path, "wb+") as f:
            pickle.dump({
                "log": self.log,
                "reference": self.problem.get_reference_state(),
                "problem": self.problem.get_info(),
                "method_details": self.method_details
            }, f)

    def cleanup(self, identifier):
        self.scheduler.cleanup(identifier)
