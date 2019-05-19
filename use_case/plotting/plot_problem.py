import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import sys

log_path = sys.argv[1]
problem = sys.argv[2] if len(sys.argv) > 2 else "generic"

with open(log_path, "rb") as f:
    data = pickle.load(f)

reference = data["reference"]
log = data["log"]

costs = []
objectives = []
best_objectives = []
improvement_indices = []

states = []
best_states = []

parameters = []
best_parameters = []

for entry in log:
    if not entry["transient"]:
        entry_objective = entry["objective"]
        entry_state = entry["state"]
        entry_parameters = entry["parameters"]
        entry_cost = entry["total_cost"]

        objectives.append(entry_objective)
        states.append(entry_state)
        parameters.append(entry_parameters)
        costs.append(entry_cost)

        if len(best_objectives) == 0 or entry_objective < best_objectives[-1]:
            best_objectives.append(entry_objective)
            best_states.append(entry_state)
            best_parameters.append(entry_parameters)
            improvement_indices.append(len(best_parameters) - 1)
        else:
            best_objectives.append(best_objectives[-1])
            best_states.append(best_states[-1])
            best_parameters.append(best_parameters[-1])

number_of_parameters = len(parameters[0])
number_of_states = len(states[0])

plt.figure(dpi = 120, figsize = (6, 8))

plt.subplot(3, 1, 1)
plt.plot(costs, objectives, ".", color = "C0", alpha = 0.5)
plt.plot(costs, best_objectives, color = 'C0')

plt.grid()
plt.ylabel("Objective")

plt.subplot(3, 1, 2)

if problem == "travel_time":
    number_of_improvements = len(improvement_indices)
    display_count = min(5, number_of_improvements)
    indices = np.round(np.linspace(0, number_of_improvements - 1, display_count)).astype(np.int)
    indices = np.array(improvement_indices)[indices]
    bounds = data["problem"]["bounds"]

    for k, index in enumerate(indices):
        plt.plot(bounds, np.cumsum(best_states[index]), alpha = (k + 1) / display_count, color = "C0", label = "%.2f" % costs[index])

    plt.plot(bounds, np.cumsum(reference), "--", color = "k")

    plt.grid()
    plt.legend(loc = "best", title = "Simulation cost")
    plt.xlabel("Travel time [min]")
    plt.ylabel("CDF")

else:
    states = np.array(states)
    best_states = np.array(best_states)

    for i in range(number_of_states):
        plt.plot(costs, states[:,i], ".", color = "C%d" % i, alpha = 0.5)
        plt.plot(costs, best_states[:,i], color = "C%d" % i)
        plt.plot([costs[0], costs[-1]], [reference[i]] * 2, color = "C%d" % i, linestyle = "--")

    plt.ylabel("State")
    plt.grid()

plt.subplot(3, 1, 3)

parameters = np.array(parameters)
best_parameters = np.array(best_parameters)

if problem in ["travel_time", "mode_share"]:
    names = ["car", "bike", "walk"]
else:
    names = ["Parameter %d" % (i + 1) for i in range(number_of_parameters)]

for i, name in enumerate(names):
    plt.plot(costs, parameters[:,i], ".", color = "C%d" % i, alpha = 0.5)
    plt.plot(costs, best_parameters[:,i], color = "C%d" % i, label = name)

plt.legend(loc = "best")
plt.ylabel("Parameter")
plt.xlabel("Simulation cost")
plt.grid()

plt.tight_layout()
plt.show()
