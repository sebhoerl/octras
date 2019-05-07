from simulation import Simulator
from calibration import CalibrationProblem, Calibrator
from problems import ModeShareProblem
import scipy.optimize as opt
import itertools

import time
import numpy as np
import json

class ApproximateSelectionProblem:
    def __init__(self, v, w, deltas, objectives):
        self.deltas = deltas
        self.objectives = objectives
        self.w = w
        self.v = v

    def get_uniformity_gap(self, alpha):
        return np.sum(alpha**2)

    def get_equilibrium_gap(self, alpha):
        return np.sqrt(np.sum((alpha[:, np.newaxis] * self.deltas)**2))

    def get_transient_performance(self, alpha):
        return np.sum(alpha * self.objectives)

    def get_objective(self, alpha):
        objective = self.get_transient_performance(alpha)
        objective += self.v * self.get_equilibrium_gap(alpha)
        objective += self.w * self.get_uniformity_gap(alpha)
        return objective

    def solve(self):
        alpha = np.ones((len(self.objectives),)) / len(self.objectives)
        result = opt.minimize(self.get_objective, alpha, constraints = [
            { "type": "eq", "fun": lambda alpha: np.sum(alpha) - 1.0 },
        ], bounds = [(0.0, 1.0)] * len(self.objectives), options = { "disp": False })

        if not result.success:
            print("Deltas:", self.deltas)
            print("Objectives:", self.objectives)
            print("v, w:", self.v, self.w)
            raise RuntimeError("Could not solve Approximate Selection Problem")

        return result.x

class AdaptationProblem:
    def __init__(self, weight, selection_performance, transient_performance, equilibrium_gap, uniformity_gap):
        self.weight = weight
        self.selection_performance = selection_performance
        self.transient_performance = transient_performance
        self.uniformity_gap = uniformity_gap
        self.equilibrium_gap = equilibrium_gap

    def get_objective(self, vw):
        R = len(self.selection_performance)
        v, w = vw

        objective = 0.0

        for r in range(R):
            local_objective = np.abs(self.transient_performance[r] - self.selection_performance[r])
            local_objective -= (v * self.equilibrium_gap[r] + w * self.uniformity_gap[r])
            local_objective = np.sum(local_objective**2)
            objective += self.weight**(R - r) * local_objective

        return objective

    def solve(self):
        vw = np.array([0.0, 0.0])

        result = opt.minimize(self.get_objective, vw, bounds = [
            (0.0, 1.0), (0.0, 1.0)
        ], options = { "disp": False })

        if not result.success:
            raise RuntimeError("Could not solve Adaptation Problem")

        return result.x

simulator = Simulator({
    "java_path": "/usr/java/jdk1.8.0_144/bin/java",
    "working_directory": "temp",
    "class_path": "simulation/astra_2018_002-1.0.0.jar",
    "config_path": "simulation/sa/sa_config.xml"
})

candidate_set_size = 4
perturbation_factor = 1.0
transition_iterations = 5
number_of_transitions = 4
adaptation_weight = 0.9

candidate_set_size = 2
transition_iterations = 2
number_of_transitions = 2

problem = ModeShareProblem(transition_iterations)
calibrator = Calibrator(problem)

initial_parameters = np.zeros((problem.number_of_parameters,))

# Run one iteration to get the initial state
simulation_parameters = problem.get_simulation_parameters(initial_parameters)
simulation_parameters.update({ "iterations": 1 })

initial_identifier = simulator.schedule(simulation_parameters)
simulator.wait([initial_identifier])

initial_state = problem.get_state(simulator.get(initial_identifier))
initial_objective = problem.get_objective(initial_state)

calibrator.add_sample(initial_parameters, initial_state, initial_objective, 1)

v, w = 0.0, 0.0
opdyts_iteration = 0

adaptation_transient_performance = []
adaptation_equilibrium_gap = []
adaptation_uniformity_gap = []
adaptation_selection_performance = []

while not calibrator.converged:
    opdyts_iteration += 1

    # Create new set of candidate parameters
    candidate_parameters = np.zeros((candidate_set_size, problem.number_of_parameters))

    for c in range(candidate_set_size):
        direction = np.random.randint(0, 2, problem.number_of_parameters) - 0.5
        candidate_parameters[c] = initial_parameters + direction * np.random.random() * perturbation_factor

    # Find initial candidate states
    candidate_identifiers = []
    candidate_states = np.zeros((candidate_set_size, problem.number_of_states))
    candidate_deltas = np.zeros((candidate_set_size, problem.number_of_states))
    candidate_objectives = np.zeros((candidate_set_size,))
    candidate_transitions = np.ones((candidate_set_size,))

    for c in range(candidate_set_size):
        simulation_parameters = problem.get_simulation_parameters(candidate_parameters[c])

        path = simulator.registry[initial_identifier]["path"]
        simulation_parameters["config"].update({ "plans.inputPlansFile": "%s/output/output_plans.xml.gz" % path })

        candidate_identifiers.append(simulator.schedule(simulation_parameters))

    simulator.wait()

    annotations = {
        "v": v, "w": w, "opdyts_iteration": opdyts_iteration, "opdyts_type": "initial"
    }

    for c in range(candidate_set_size):
        candidate_states[c] = problem.get_state(simulator.get(candidate_identifiers[c]))
        candidate_objectives[c] = problem.get_objective(candidate_states[c])
        candidate_deltas[c] = candidate_states[c] - initial_state

        annotations.update({ "candidate": c })
        calibrator.add_sample(candidate_parameters[c], candidate_states[c], candidate_objectives[c], transition_iterations, annotations)

    local_adaptation_transient_performance = []
    local_adaptation_equilibrium_gap = []
    local_adaptation_uniformity_gap = []

    # Advance candidates
    while np.max(candidate_transitions) < number_of_transitions:
        # Approximate selection problem
        selection_problem = ApproximateSelectionProblem(v, w, candidate_deltas, candidate_objectives)
        alpha = selection_problem.solve()

        print("[Opdyts] Transient performance:", selection_problem.get_transient_performance(alpha), "Equilibrium gap:", selection_problem.get_equilibrium_gap(alpha), "Uniformity gap:", selection_problem.get_uniformity_gap(alpha))

        cumulative_alpha = np.cumsum(alpha)
        c = np.sum(np.random.random() > cumulative_alpha)
        print("[Opdyts] Transition candidate", c)

        # Advance selected candidate
        simulation_parameters = problem.get_simulation_parameters(candidate_parameters[c])

        path = simulator.registry[candidate_identifiers[c]]["path"]
        simulation_parameters["config"].update({ "plans.inputPlansFile": "%s/output/output_plans.xml.gz" % path })

        identifier = simulator.schedule(simulation_parameters)
        simulator.wait([identifier])

        simulator.cleanup(candidate_identifiers[c])
        candidate_identifiers[c] = identifier

        state = problem.get_state(simulator.get(identifier))
        candidate_deltas[c] = state - candidate_states[c]
        candidate_states[c] = problem.get_state(simulator.get(identifier))
        candidate_objectives[c] = problem.get_objective(candidate_states[c])
        candidate_transitions[c] += 1

        annotations = {
            "v": v, "w": w, "opdyts_iteration": opdyts_iteration, "type": "transition",
            "transient_performance": selection_problem.get_transient_performance(alpha),
            "uniformity_gap": selection_problem.get_uniformity_gap(alpha),
            "equilibrium_gap": selection_problem.get_equilibrium_gap(alpha),
            "candidate": c
        }

        calibrator.add_sample(candidate_parameters[c], candidate_states[c], candidate_objectives[c], transition_iterations, annotations)

        local_adaptation_transient_performance.append(selection_problem.get_transient_performance(alpha))
        local_adaptation_equilibrium_gap.append(selection_problem.get_equilibrium_gap(alpha))
        local_adaptation_uniformity_gap.append(selection_problem.get_uniformity_gap(alpha))

    index = np.argmax(candidate_transitions)
    print("[Opdyts] Finished selection problem with candidate", index)

    for c in range(candidate_set_size):
        if c != index:
            simulator.cleanup(candidate_identifiers[c])

    simulator.cleanup(initial_identifier)
    initial_identifier = candidate_identifiers[index]
    initial_state = candidate_states[index]
    initial_parameters = candidate_parameters[index]

    adaptation_selection_performance.append(candidate_objectives[index])
    adaptation_transient_performance.append(np.array(local_adaptation_transient_performance))
    adaptation_equilibrium_gap.append(np.array(local_adaptation_equilibrium_gap))
    adaptation_uniformity_gap.append(np.array(local_adaptation_uniformity_gap))

    adaptation_problem = AdaptationProblem(adaptation_weight, adaptation_selection_performance, adaptation_transient_performance, adaptation_equilibrium_gap, adaptation_uniformity_gap)
    v, w = adaptation_problem.solve()

    print("[Opdyts] Adaptation Problem solved: v = ", v, ", w = ", w)

    # Output
    calibrator.save("output/opdyts.p")
    calibrator.plot("output/opdyts.pdf")
