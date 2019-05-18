import pickle
import numpy as np
import matplotlib.pyplot as plt
import sys

history_path = sys.argv[1]

with open(history_path, "rb") as f:
    history = pickle.load(f)

positive_costs = []
positive_objectives = []

negative_costs = []
negative_objectives = []

objective_costs = []
objectives = []

costs = []
gradient_length = []
perturbation_length = []

for h in history:
    if h["annotations"]["type"] == "positive_gradient":
        positive_costs.append(h["total_cost"])
        positive_objectives.append(h["objective"])
    elif h["annotations"]["type"] == "negative_gradient":
        negative_costs.append(h["total_cost"])
        negative_objectives.append(h["objective"])
    elif h["annotations"]["type"] == "objective":
        objective_costs.append(h["total_cost"])
        objectives.append(h["objective"])

    costs.append(h["total_cost"])
    gradient_length.append(h["annotations"]["gradient_length"])
    perturbation_length.append(h["annotations"]["perturbation_length"])

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
plt.xlabel("Transitions")
plt.legend(loc = "best")

plt.tight_layout()
plt.show()
