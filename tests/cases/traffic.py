from .generic import GenericTestSimulator
from octras import Problem

import numpy as np

class TrafficSimulator(GenericTestSimulator):
    def __init__(self):
        super().__init__()

        self.selection = {}
        self.random_states = {}

    def run(self, identifier, parameters):
        capacities = parameters["capacities"]
        assert len(capacities) == 2

        scaling = 1.0

        if "scaling" in parameters:
            scaling = parameters["scaling"]

        selection = None
        random = None

        if "restart" in parameters:
            selection = self.selection[parameters["restart"]]
            random = self.random_states[parameters["restart"]]

        result, selection, random = simulate([
            capacities[0], capacities[1], 100.0
        ], scaling, selection, parameters["iterations"], random)

        self.results[identifier] = result[:2]
        self.selection[identifier] = selection
        self.random_states[identifier] = random

class TrafficProblem(Problem):
    def __init__(self, reference = [455.0, 1545.0], multi_fidelity = False, iterations = 400):
        self.bounds = [[0, 800]] * 2
        self.initial = [500.0, 500.0]

        self.reference = np.array(reference)

        self.multi_fidelity = multi_fidelity

        self.iterations = iterations

    def use_bounds(self, values):
        self.bounds = values

    def get_information(self):
        information = {
            "number_of_parameters": 2,
            "number_of_states": 2,
            "initial_values": self.initial,
            "reference_values": self.reference,
            "fidelities": 3
        }

        if not self.bounds is None:
            information["bounds"] = self.bounds

        return information

    def parameterize(self, x):
        parameters = {}
        parameters["capacities"] = [x[0], x[1]]
        parameters["iterations"] = self.iterations

        if self.multi_fidelity:
            scaling_levels = [1.0, 0.1, 0.025]
            parameters["scaling"] = scaling_levels[x[2]]

        return parameters

    def evaluate(self, x, result):
        return np.sum(np.abs(self.reference - result)), result

def simulate(capacities, scaling, selection = None, iterations = 400, random = None):
    random = np.random.RandomState(0) if random is None else random

    N = int(2000 * scaling)
    selection = np.array([0] * N) if selection is None else selection
    capacities = np.array(capacities) * scaling
    capacities = np.maximum(capacities, 0.0)

    share = 0.01

    history = []
    converged = False

    for k in range(iterations):
        counts = np.array([
            np.count_nonzero(selection == 0),
            np.count_nonzero(selection == 1),
            np.count_nonzero(selection == 2)
        ])

        travel_times = np.array([
            90.0 + 0.15 * (counts[0] / capacities[0])**4,
            60.0 + 0.15 * (counts[1] / capacities[1])**4,
            120.0 + 0.15 * (counts[2] / capacities[2])**4,
        ])

        selected_indices = np.arange(N)
        selected_indices = selected_indices[random.random_sample(N) < share]

        fastest_index = np.argmin(travel_times)
        selection[selected_indices] = fastest_index

    return counts / scaling, selection, random
