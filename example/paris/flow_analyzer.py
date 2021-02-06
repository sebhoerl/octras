import pandas as pd
import numpy as np

class ParisDailyFlowAnalyzer:
    def __init__(self, sampling_rate, threshold, reference_path, objective = "sum"):
        self.threshold = threshold
        self.sampling_rate = sampling_rate
        self.reference_path = reference_path
        self.objective = objective

    def prepare_reference(self, reference_path):
        df = pd.read_csv(reference_path, sep = ";")
        df = df.groupby("link_id").sum().reset_index()[["link_id", "flow"]]
        df = df.rename(columns = { "flow": "reference_count" })
        return df

    def prepare_simulation(self, output_path):
        df = pd.read_csv("%s/flows.csv" % output_path, sep = ";")
        df = df.groupby("link_id").sum().reset_index()[["link_id", "count"]]
        df = df.rename(columns = { "count": "simulation_count" })
        df["simulation_count"] /= self.sampling_rate
        return df

    def calculate_objective(self, df):
        objective = 0.0

        simulation_values = df["simulation_count"].values
        reference_values = df["reference_count"].values
        relative_errors = np.abs((simulation_values - reference_values) / reference_values)

        if self.objective == "sum":
            objective = np.sum(np.maximum(self.threshold,
                relative_errors
            ) - self.threshold)

        elif self.objective == "max":
            objective = np.max(np.maximum(self.threshold,
                relative_errors
            ) - self.threshold)

        return objective

    def execute(self, output_path):
        df_reference = self.prepare_reference(self.reference_path)
        df_simulation = self.prepare_simulation(output_path)
        df_comparison = pd.merge(df_reference, df_simulation, on = "link_id")
        df_comparison = df_comparison[df_comparison["reference_count"] > 0.0]

        objective = self.calculate_objective(df_comparison)

        return {
            "comparison": df_comparison,
            "objective": objective
        }

if __name__ == "__main__":
    analyzer = ParisDailyFlowAnalyzer(
        threshold = 0.05,
        sampling_rate = 0.001,
        reference_path = "/home/shoerl/backup/gpe/matching/hourly_reference.csv")

    result = analyzer.execute("/home/shoerl/backup/gpe/output_1pm/simulation_output")
    print(result)
