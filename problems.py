from simulation import Simulator
from calibration import CalibrationProblem, Calibrator
import time
import numpy as np

class ModeShareProblem(CalibrationProblem):
    def __init__(self, number_of_iterations):
        CalibrationProblem.__init__(self, number_of_parameters = 3, number_of_states = 4, parameter_names = [
            "alpha_car", "alpha_bike", "alpha_walk"
        ], state_names = [
            "car", "pt", "bike", "walk"
        ])

        self.number_of_iterations = number_of_iterations

    def get_simulation_parameters(self, parameters):
        return {
            "iterations": self.number_of_iterations,
            "utilities": {
                "car.alpha": parameters[0],
                "pt.alpha": 0.0,
                "bike.alpha": parameters[1],
                "walk.alpha": parameters[2]
            },
            "config": {}
        }

    def get_state(self, simulation_state):
        df_trips = simulation_state

        car_share = np.count_nonzero(df_trips["mode"] == "car") / len(df_trips)
        pt_share = np.count_nonzero(df_trips["mode"] == "pt") / len(df_trips)
        bike_share = np.count_nonzero(df_trips["mode"] == "bike") / len(df_trips)
        walk_share = np.count_nonzero(df_trips["mode"] == "walk") / len(df_trips)

        return np.array([car_share, pt_share, bike_share, walk_share])

    def get_objective(self, state):
        reference = np.array([0.34, 0.20, 0.09, 0.29])
        return np.sum((reference - state)**2)
