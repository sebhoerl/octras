import pandas as pd
import numpy as np
import scipy.optimize as opt

class ParisDailyFlowAnalyzer:
    def __init__(self, reference_path):
        self.reference_path = reference_path

    def prepare_reference(self, reference_path):
        df = pd.read_csv(reference_path, sep = ";")
        df = df[df["osm"].isin(["motorway", "trunk", "primary", "secondary"])]
        df = df.groupby("link_id").sum().reset_index()[["link_id", "flow"]]
        df = df.rename(columns = { "flow": "reference_count" })
        return df

    def prepare_simulation(self, output_path):
        df = pd.read_csv("%s/flows.csv" % output_path, sep = ";")
        df = df.groupby("link_id").sum().reset_index()[["link_id", "count"]]
        df = df.rename(columns = { "count": "simulation_count" })
        return df

    def calculate_factor(self, df):
        f = lambda x: np.sum(np.abs((df["simulation_count"] * x - df["reference_count"]))**2)
        result = opt.minimize_scalar(f)
        return result.x

    def calculate_objective(self, df, factor):
        objective = 0.0

        simulation_values = df["simulation_count"].values * factor
        reference_values = df["reference_count"].values
        relative_errors = np.abs((simulation_values - reference_values) / reference_values)

        return np.mean(relative_errors <= 0.2)

    def execute(self, output_path):
        df_reference = self.prepare_reference(self.reference_path)
        df_simulation = self.prepare_simulation(output_path)
        df_comparison = pd.merge(df_reference, df_simulation, on = "link_id")
        df_comparison = df_comparison[df_comparison["reference_count"] > 0.0]

        factor = self.calculate_factor(df_comparison)
        objective = self.calculate_objective(df_comparison, factor)

        return {
            "comparison": df_comparison,
            "objective": objective,
            "factor": factor,
        }

if __name__ == "__main__":
    analyzer = ParisDailyFlowAnalyzer(
        reference_path = "/home/shoerl/backup/gpe/matching/hourly_reference.csv")

    result = analyzer.execute("/home/shoerl/backup/gpe/output_1pm/simulation_output")
    print(result)
