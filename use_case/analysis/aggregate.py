import pickle
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import sys

def get_best_objectives(log):
    best_objective = np.inf
    objectives = []

    for sample in log:
        if not sample["transient"]:
            if sample["objective"] < best_objective:
                best_objective = sample["objective"]

        objectives.append(best_objective)

    return np.array(objectives)

def compute_serial_cost(log):
    cost = 0
    cost_series = []

    for sample in log:
        cost += sample["cost"]
        cost_series.append(cost)

    return np.array(cost_series)

def compute_parallel_cost_opdyts(log):
    cost = 0
    cost_series = []

    transient_tracked = False

    for sample in log:
        if sample["transient"] and not transient_tracked:
            cost += sample["cost"]
            transient_tracked = True
            cost_series.append(cost)
        else:
            transient_tracked = False
            cost += sample["cost"]
            cost_series.append(cost)

    return np.array(cost_series)

def compute_parallel_cost(log, threads):
    cost = 0
    cost_series = []

    for index, sample in enumerate(log):
        if sample["transient"]:
            raise RuntimeError()

        if index > 0 and index % threads == 0:
            cost += sample["cost"]
            cost_series += [cost] * threads

    cost += log[-1]["cost"]
    cost_series += [cost] * (len(log) - len(cost_series))

    return np.array(cost_series)

def read_run(path, algorithm, threads):
    with open(path, "rb") as f:
        data = pickle.load(f)

    objectives = get_best_objectives(data["log"])
    serial_cost = compute_serial_cost(data["log"])

    if algorithm == "opdyts":
        parallel_cost = compute_parallel_cost_opdyts(data["log"])
    else:
        parallel_cost = compute_parallel_cost(data["log"], threads)

    transient = np.array([sample["transient"] for sample in data["log"]], dtype = np.bool)

    objectives = objectives[~transient]
    serial_cost = serial_cost[~transient]
    parallel_cost = parallel_cost[~transient]

    return pd.DataFrame({
        "serial_cost": serial_cost,
        "parallel_cost": parallel_cost,
        "objective": objectives
    })

input_path = sys.argv[1]
output_path = sys.argv[2]

THREADS = 5
full_df = []

for case in ("mode_share", "travel_time"):
    plt.figure(dpi = 120, figsize = (8, 6))

    for algorithm in ("random_walk", "cma_es", "opdyts"):
        for seed in (1000, 2000, 3000, 4000):
            df = read_run("%s/1pm_%s_%s_%d.p" % (input_path, case, algorithm, seed), algorithm, THREADS)

            df["seed"] = seed
            df["algorithm"] = algorithm
            df["case"] = case

            full_df.append(df)

full_df = pd.concat(full_df)
full_df.to_csv(output_path, index = None)
