import octras.optimization
import numpy as np
import pandas as pd

class TravelTimeDistribution:
    def __init__(self, bounds, mode):
        self.bounds = bounds
        self.mode = mode

    def __call__(self, df):
        df = df.copy()

        if not "weight" in df:
            df["weight"] = 1.0

        df = df[df["crowfly_distance"] > 0]
        df = df[df["preceedingPurpose"] != "outside"]
        df = df[df["followingPurpose"] != "outside"]

        df = df[df["mode"] == mode]

        df["class"] = np.digitize(df["travel_time"], self.bounds)
        counts = np.array([np.sum(df[df["class"] == k]["weight"]) for k in range(len(self.bounds) + 1)])

        return counts / np.sum(counts)

class TotalModeShare:
    def __init__(self, modes):
        self.modes = modes

    def __call__(self, df):
        df = df.copy()

        if not "weight" in df:
            df["weight"] = 1.0

        df = df[df["crowfly_distance"] > 0]
        df = df[df["preceedingPurpose"] != "outside"]
        df = df[df["followingPurpose"] != "outside"]

        shares = np.array([
            df[df["mode"] == mode]["weight"].sum()
            for mode in self.modes
        ])

        return shares / np.sum(shares)

class ModeShareByDistance:
    def __init__(self, mode_bounds):
        self.mode_bounds = mode_bounds

    def __call__(self, df):
        df = df.copy()

        if not "weight" in df:
            df["weight"] = 1.0

        df = df[df["crowfly_distance"] > 0]
        df = df[df["preceedingPurpose"] != "outside"]
        df = df[df["followingPurpose"] != "outside"]

        values = []

        for item in self.mode_bounds:
            df["class"] = np.digitize(df["network_distance"], item["bounds"])

            f = df["mode"] == item["mode"]
            values += [np.sum(df[df["class"] == k]["weight"]) for k in range(len(item["bounds"]) + 1)]

        values = np.array(values)
        return values / np.sum(values)

def l2_distance(simulation_state, reference_state):
    return np.sqrt(np.sum((simulation_state - reference_state)**2))

def hellinger_distance(simulation_state, reference_state):
    return np.sqrt(np.sum((np.sqrt(simulation_state) - np.sqrt(reference_state))**2)) / np.sqrt(2)

class TripBasedProblem(octras.optimization.OptimizationProblem):
    def __init__(self, problem_name, state_calculator, state_names, objective_calculator, parameters, reference_path):
        number_of_parameters = len(parameters)
        number_of_states = len(state_names)
        initial_parameters = [parameter["initial"] for parameter in parameters]

        octras.optimization.OptimizationProblem.__init__(self, number_of_parameters, number_of_states, initial_parameters)

        self.info = dict(
            problem_name = problem_name,
            state_names = state_names,
            parameters = parameters
        )

        self.state_calculator = state_calculator
        self.objective_calculator = objective_calculator

        self.parameters = parameters

        self.reference_state = self.compute_reference_state(reference_path)

    def compute_reference_state(self, reference_path):
        df_reference = pd.read_csv(reference_path, sep = ";")
        return self.state_calculator(df_reference)

    def get_simulator_parameters(self, values):
        return {
            "mode-parameters": {
                parameter["name"]: values[index]
                for index, parameter in enumerate(self.parameters)
            }
        }

    def evaluate(self, parameters, df_simulation):
        simulation_state = self.state_calculator(df_simulation)
        objective = self.objective_calculator(simulation_state, self.reference_state)
        return objective, simulation_state

    def get_reference_state(self):
        return self.reference_state

    def get_info(self):
        return self.info
