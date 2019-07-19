import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

input_path = sys.argv[1]
output_path = sys.argv[2]

df = pd.read_csv(input_path)

for cost_column in ("serial_cost", "parallel_cost"):
    for case in ("mode_share", "travel_time"):
        plt.figure(dpi = 120, figsize = (8, 6))
        f_case = df["case"] == case

        for i, algorithm in enumerate(("random_walk", "cma_es", "opdyts")):
            f_algorithm = df["algorithm"] == algorithm

            indices = np.sort(df[f_case & f_algorithm][cost_column].unique())
            df_algorithm = []

            for seed in (1000, 2000, 3000, 4000):
                f_seed = df["seed"] == seed
                df_seed = df[f_case & f_algorithm & f_seed]

                plt.plot(
                    df_seed[cost_column].values,
                    df_seed["objective"].values,
                    linewidth = 1.0, color = "C%d" % i, alpha = 0.2,
                    marker = ".", markersize = 2.0
                )

                df_algorithm.append(
                    df_seed.sort_values(
                        by = [cost_column, "objective"], ascending = [True, False]
                    ).drop_duplicates(
                        cost_column, keep = "last"
                    ).set_index(cost_column).reindex(indices, method = "ffill")
                )

            df_algorithm = pd.concat(df_algorithm).groupby(cost_column).aggregate({
                "objective": ["mean", "std"]
            }).reset_index()

            cost = df_algorithm[cost_column].values
            mean = df_algorithm[("objective", "mean")].values
            std = df_algorithm[("objective", "std")].values

            plt.fill_between(cost, mean - std, mean + std, color = "C%d" % i, alpha = 0.1, linewidth = 0.0)
            plt.plot(cost, mean, color = "C%d" % i)

            plt.plot([np.nan] * 2, [np.nan] * 2, color = "C%d" % i, label = algorithm)

        if cost_column == "serial_cost":
            plt.xlabel("Number of MATSim iterations (serial)")
        else:
            plt.xlabel("Number of MATSim iterations (parallel)")

        plt.ylabel("Objective")

        plt.legend(title = "Algorithm", loc = "best")
        plt.grid()
        plt.savefig("%s/%s_%s.pdf" % (output_path, case, cost_column))
