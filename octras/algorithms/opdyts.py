import numpy as np
import scipy.optimize as opt

import logging
logger = logging.getLogger(__name__)

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

def opdyts_algorithm(calibrator, perturbation_length, transition_iterations, number_of_transitions, candidate_set_size, adaptation_weight = 0.9):
    opdyts_iteration = 0
    v, w = 0.0, 0.0

    adaptation_transient_performance = []
    adaptation_equilibrium_gap = []
    adaptation_uniformity_gap = []
    adaptation_selection_performance = []

    if candidate_set_size % 2 != 0:
        raise RuntimeError("Opdyts expects candiate set size as a multiple of 2")

    initial_parameters = np.copy(calibrator.problem.initial_parameters)

    # Run one iteration to get the initial state
    logger.info("Initializing Opdyts.")
    initial_identifier = calibrator.schedule(initial_parameters, { "iterations": 1 }, { "type": "initial", "transient": True })
    initial_objective, initial_state = calibrator.get(initial_identifier, transient = True)

    while not calibrator.finished:
        opdyts_iteration += 1
        logger.info("Starting Opdyts iteration %d." % opdyts_iteration)

        # Create new set of candidate parameters
        candidate_parameters = np.zeros((candidate_set_size, calibrator.problem.number_of_parameters))

        for c in range(0, candidate_set_size, 2):
            direction = np.random.random(size = (calibrator.problem.number_of_parameters,)) * 2.0 - 1.0
            candidate_parameters[c] = initial_parameters + direction * perturbation_length
            candidate_parameters[c + 1] = initial_parameters + direction * perturbation_length

        # Find initial candidate states
        candidate_identifiers = []
        candidate_states = np.zeros((candidate_set_size, calibrator.problem.number_of_states))
        candidate_deltas = np.zeros((candidate_set_size, calibrator.problem.number_of_states))
        candidate_objectives = np.zeros((candidate_set_size,))
        candidate_transitions = np.ones((candidate_set_size,))

        annotations = {
            "type": "candidates",
            "v": v, "w": w, "transient": True, "opdyts_iteration": opdyts_iteration
        }

        for c in range(candidate_set_size):
            annotations.update({ "candidate": c })
            candidate_identifiers.append(calibrator.schedule(candidate_parameters[c], {
                "iterations": transition_iterations,
                "initial_identifier": initial_identifier
            }, annotations))

        calibrator.wait()

        for c in range(candidate_set_size):
            candidate_objectives[c], candidate_states[c] = calibrator.get(candidate_identifiers[c], transient = True)
            candidate_deltas[c] = candidate_states[c] - initial_state

        # Advance candidates
        local_adaptation_transient_performance = []
        local_adaptation_equilibrium_gap = []
        local_adaptation_uniformity_gap = []

        while np.max(candidate_transitions) < number_of_transitions:
            # Approximate selection problem
            selection_problem = ApproximateSelectionProblem(v, w, candidate_deltas, candidate_objectives)
            alpha = selection_problem.solve()

            transient_performance = selection_problem.get_transient_performance(alpha)
            equilibrium_gap = selection_problem.get_equilibrium_gap(alpha)
            uniformity_gap = selection_problem.get_uniformity_gap(alpha)

            local_adaptation_transient_performance.append(transient_performance)
            local_adaptation_equilibrium_gap.append(equilibrium_gap)
            local_adaptation_uniformity_gap.append(uniformity_gap)

            logger.info(
                "Transient performance: %f, Equilibirum gap: %f, Uniformity_gap: %f",
                transient_performance, equilibrium_gap, uniformity_gap)

            cumulative_alpha = np.cumsum(alpha)
            c = np.sum(np.random.random() > cumulative_alpha)

            logger.info("Transitioning candidate %d", c)
            candidate_transitions[c] += 1
            transient = candidate_transitions[c] < number_of_transitions

            annotations.update({
                "type": "transition",
                "candidate": c, "transient_performance": transient_performance,
                "equilibrium_gap": equilibrium_gap, "uniformity_gap": uniformity_gap,
                "transient": transient
            })

            # Advance selected candidate
            identifier = calibrator.schedule(candidate_parameters[c], {
                "iterations": transition_iterations,
                "initial_identifier": candidate_identifiers[c]
            }, annotations)

            new_objective, new_state = calibrator.get(identifier, transient = transient)
            calibrator.cleanup(candidate_identifiers[c])

            candidate_deltas[c] = new_state - candidate_states[c]
            candidate_states[c], candidate_objectives[c] = new_state, new_objective
            candidate_identifiers[c] = identifier

        index = np.argmax(candidate_transitions)
        logger.info("Solved selection problem with candidate %d", index)

        for c in range(candidate_set_size):
            if c != index:
                calibrator.cleanup(candidate_identifiers[c])

        calibrator.cleanup(initial_identifier)
        initial_identifier = candidate_identifiers[index]
        initial_state = candidate_states[index]
        initial_parameters = candidate_parameters[index]

        adaptation_selection_performance.append(candidate_objectives[index])
        adaptation_transient_performance.append(np.array(local_adaptation_transient_performance))
        adaptation_equilibrium_gap.append(np.array(local_adaptation_equilibrium_gap))
        adaptation_uniformity_gap.append(np.array(local_adaptation_uniformity_gap))

        adaptation_problem = AdaptationProblem(adaptation_weight, adaptation_selection_performance, adaptation_transient_performance, adaptation_equilibrium_gap, adaptation_uniformity_gap)
        v, w = adaptation_problem.solve()

        logger.info("Solved Adaptation Problem. v = %f, w = %f", v, w)
