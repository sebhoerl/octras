import octras.optimization
import numpy as np
import pandas as pd

class ModeShareProblem(octras.optimization.OptimizationProblem):
    def __init__(self, reference_path):
        octras.optimization.OptimizationProblem.__init__(self, 3, 4, np.zeros((3,)))

        df_reference = pd.read_csv(reference_path, sep = ";")
        self.reference = self._compute_shares(df_reference)

    def _compute_shares(self, df):
        df = pd.DataFrame(df, copy = True)

        if not "weight" in df:
            df["weight"] = 1.0

        df = df[df["crowfly_distance"] > 0]

        shares = np.array([
            np.sum(df.loc[df["mode"] == "car", "weight"]),
            np.sum(df.loc[df["mode"] == "pt", "weight"]),
            np.sum(df.loc[df["mode"] == "bike", "weight"]),
            np.sum(df.loc[df["mode"] == "walk", "weight"])
        ])
        shares /= np.sum(shares)

        return shares

    def get_simulator_parameters(self, parameters):
        return {
            "utilities": {
                "car.alpha": parameters[0],
                "pt.alpha": 0.0,
                "bike.alpha": parameters[1],
                "walk.alpha": parameters[2]
            }
        }

    def evaluate(self, parameters, result):
        shares = self._compute_shares(result)
        objective = np.sqrt(np.sum((self.reference - shares)**2))
        return objective, shares

class TravelTimeProblem(octras.optimization.OptimizationProblem):
    def __init__(self, reference_path, bounds):
        octras.optimization.OptimizationProblem.__init__(self, 3, len(bounds), np.zeros((3,)))
        self.bounds = bounds

        df_reference = pd.read_csv(reference_path, sep = ";")
        self.reference = self._compute_distribution(df_reference)

        exit()

    def _compute_distribution(self, df):
        df = pd.DataFrame(df, copy = True)

        if not "weight" in df:
            df["weight"] = 1.0

        df = df[df["crowfly_distance"] > 0]
        df = df[df["mode"] == "car"]

        df["minutes"] = df["travel_time"] / 60.0
        df["classes"] = np.digitize(df["minutes"], self.bounds)
        counts = np.array([np.sum(df["classes"] == k) for k in range(len(self.bounds))])

        return counts / np.sum(counts)

    def get_simulator_parameters(self, parameters):
        return {
            "utilities": {
                "car.alpha": parameters[0],
                "pt.alpha": 0.0,
                "bike.alpha": parameters[1],
                "walk.alpha": parameters[2]
            }
        }

    def evaluate(self, parameters, result):
        distribution = self._compute_distribution(result)
        objective = np.sqrt(np.sum((np.sqrt(distribution) - np.sqrt(self.reference))**2)) / np.sqrt(2.0)
        return objective, distribution
