import octras.optimization
import numpy as np
import pandas as pd

class ModeShareProblem(octras.optimization.OptimizationProblem):
    def __init__(self, simulation_path, reference_sample_size):
        octras.optimization.OptimizationProblem.__init__(self, 3, 4, np.zeros((3,)))

        if not reference_sample_size in ("1pm", "1pct", "10pct", "25pct"):
            raise RuntimeError("Reference sample size should be one of: 1pm, 1pct, 10pct, 25pct")

        reference_path = "%s/zurich_%s/zurich_%s_reference.csv" % (simulation_path, reference_sample_size, reference_sample_size)
        df_reference = pd.read_csv(reference_path, sep = ";")
        self.reference = self._compute_shares(df_reference)

    def _compute_shares(self, df):
        df = pd.DataFrame(df, copy = True)

        if not "weight" in df:
            df["weight"] = 1.0

        df = df[df["crowfly_distance"] > 0]
        df = df[df["preceedingPurpose"] != "outside"]
        df = df[df["followingPurpose"] != "outside"]

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

    def get_reference_state(self):
        return self.reference

class TravelTimeProblem(octras.optimization.OptimizationProblem):
    def __init__(self, simulation_path, reference_sample_size, bounds = None, number_of_bounds = 5, maximum_travel_time = 60.0):
        if bounds is not None: number_of_bounds = len(bounds)
        octras.optimization.OptimizationProblem.__init__(self, 3, number_of_bounds, np.zeros((3,)))

        if not reference_sample_size in ("1pm", "1pct", "10pct", "25pct"):
            raise RuntimeError("Reference sample size should be one of: 1pm, 1pct, 10pct, 25pct")

        reference_path = "%s/zurich_%s/zurich_%s_reference.csv" % (simulation_path, reference_sample_size, reference_sample_size)
        df_reference = pd.read_csv(reference_path, sep = ";")

        if bounds is not None:
            self.bounds = bounds
        else:
            self.bounds = self._compute_bounds(df_reference, number_of_bounds, maximum_travel_time)

        self.reference = self._compute_distribution(df_reference)

    def _compute_bounds(self, df, number_of_bounds, maximum_travel_time):
        df = pd.DataFrame(df, copy = True)

        if not "weight" in df:
            df["weight"] = 1.0

        df = df[df["mode"] == "car"]
        df = df[df["travel_time"] / 60 <= maximum_travel_time]

        df = df[df["crowfly_distance"] > 0]
        df = df[df["preceedingPurpose"] != "outside"]
        df = df[df["followingPurpose"] != "outside"]

        values = df["travel_time"].values / 60
        weights = df["weight"].values

        sorter = np.argsort(values)
        values = values[sorter]
        weights = weights[sorter]

        cdf = np.cumsum(weights)
        cdf /= cdf[-1]

        probabilities = np.arange(1, number_of_bounds + 1) / number_of_bounds
        percentiles = np.array([values[np.count_nonzero(cdf < p)] for p in probabilities])

        return percentiles

    def _compute_distribution(self, df):
        df = pd.DataFrame(df, copy = True)

        if not "weight" in df:
            df["weight"] = 1.0

        df = df[df["crowfly_distance"] > 0]
        df = df[df["preceedingPurpose"] != "outside"]
        df = df[df["followingPurpose"] != "outside"]

        df = df[df["mode"] == "car"]

        df["minutes"] = df["travel_time"] / 60.0
        df["classes"] = np.digitize(df["minutes"], self.bounds)
        counts = np.array([np.sum(df[df["classes"] == k]["weight"]) for k in range(len(self.bounds))])

        return counts / np.sum(counts)

    def get_simulator_parameters(self, parameters):
        return {
            "utilities": {
                "car.betaTravelTime": parameters[0],
                #"pt.alpha": 0.0,
                "bike.betaTravelTime": parameters[1],
                "walk.betaTravelTime": parameters[2]
            }
        }

    def evaluate(self, parameters, result):
        distribution = self._compute_distribution(result)
        objective = np.sqrt(np.sum((np.sqrt(distribution) - np.sqrt(self.reference))**2)) / np.sqrt(2.0)
        return objective, distribution

    def get_reference_state(self):
        return self.reference

    def get_info(self):
        return {
            "bounds": self.bounds
        }
