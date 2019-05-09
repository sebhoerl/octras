from simulation import Simulator
from calibration import CalibrationProblem, Calibrator
import time
import numpy as np
import pandas as pd

class ModeShareProblem(CalibrationProblem):
    def __init__(self, reference, default_parameters = {}):
        CalibrationProblem.__init__(self, number_of_parameters = 3, number_of_states = 4, parameter_names = [
            "alpha_car", "alpha_bike", "alpha_walk"
        ], state_names = [
            "car", "pt", "bike", "walk"
        ])

        if not len(reference) == 4:
            raise RuntimeError()

        self.reference = reference
        self.default_parameters = default_parameters

    def get_simulation_parameters(self, parameters):
        new_parameters = {}
        new_parameters.update(self.default_parameters)
        new_parameters.update({
            "utilities": {
                "car.alpha": parameters[0],
                "pt.alpha": 0.0,
                "bike.alpha": parameters[1],
                "walk.alpha": parameters[2]
            },
            "config": {}
        })
        return new_parameters

    def get_state(self, simulation_state):
        df_trips = simulation_state
        df_trips = df_trips[df_trips["crowfly_distance"] > 0]

        state = np.array([
            np.count_nonzero(df_trips["mode"] == "car"),
            np.count_nonzero(df_trips["mode"] == "pt"),
            np.count_nonzero(df_trips["mode"] == "bike"),
            np.count_nonzero(df_trips["mode"] == "walk")
        ], dtype = np.float)
        state /= np.sum(state)

        return state

    def get_objective(self, state):
        return np.sum((self.reference - state)**2)

class TravelTimeProblem(CalibrationProblem):
    def __init__(self, reference, bounds, default_parameters = {}):
        CalibrationProblem.__init__(self, number_of_parameters = 3, number_of_states = len(bounds), parameter_names = [
            "alpha_car", "alpha_bike", "alpha_walk"
        ], state_names = [
            "q%d" % i for i in range(1, len(bounds) + 1)
        ])

        self.reference = reference
        self.bounds = bounds
        self.default_parameters = default_parameters

    def get_simulation_parameters(self, parameters):
        new_parameters = {}
        new_parameters.update(self.default_parameters)
        new_parameters.update({
            "utilities": {
                "car.alpha": parameters[0],
                "pt.alpha": 0.0,
                "bike.alpha": parameters[1],
                "walk.alpha": parameters[2]
            },
            "config": {}
        })
        return new_parameters

    def get_state(self, simulation_state):
        df_trips = simulation_state
        df_trips = df_trips[df_trips["crowfly_distance"] > 0]

        classes = np.digitize(df_trips["travel_time"], self.bounds)
        counts = np.array([np.sum(classes == i) for i in range(len(self.bounds))], dtype = np.float)
        counts /= np.sum(counts)

        return counts

    def get_objective(self, state):
        return np.sum((self.reference - state)**2)
