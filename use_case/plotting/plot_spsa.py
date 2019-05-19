import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys

log_path = sys.argv[1]

with open(log_path, "rb") as f:
    data = pickle.load(f)

log = data["log"]

positive_costs = []
positive_objectives = []

negative_costs = []
negative_objectives = []

objective_costs = []
objectives = []

costs = []
gradient_length = []
perturbation_length = []

for item in log:
    if item["annotations"]["type"] == "positive_gradient":
        positive_costs.append(item["total_cost"])
        positive_objectives.append(item["objective"])
    elif item["annotations"]["type"] == "negative_gradient":
        negative_costs.append(item["total_cost"])
        negative_objectives.append(item["objective"])
    elif item["annotations"]["type"] == "objective":
        objective_costs.append(item["total_cost"])
        objectives.append(item["objective"])

    costs.append(item["total_cost"])
    gradient_length.append(item["annotations"]["gradient_length"])
    perturbation_length.append(item["annotations"]["perturbation_length"])

plt.figure(dpi = 120, figsize = (6, 4))

plt.subplot(2,1,1)
plt.plot(positive_costs, positive_objectives, label = "Positive gradient")
plt.plot(negative_costs, negative_objectives, label = "Negative gradient")
plt.plot(objective_costs, objectives, 'k.-', label = "Objective")
plt.grid()
plt.legend(loc = "best")
plt.ylabel("Objective")

plt.subplot(2,1,2)
plt.plot(costs, gradient_length, label = "Gradient length")
plt.plot(costs, perturbation_length, label = "Perturbation length")
plt.grid()
plt.xlabel("Simulation cost")
plt.legend(loc = "best")

plt.tight_layout()
plt.show()
