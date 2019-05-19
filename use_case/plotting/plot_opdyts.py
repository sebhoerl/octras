import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import sys

log_path = sys.argv[1]

with open(log_path, "rb") as f:
    data = pickle.load(f)

log = data["log"]

number_of_candidates = np.max([
    item["annotations"]["candidate"] for item in log if "candidate" in item["annotations"]
]) + 1

opdyts_iterations = np.max([
    item["annotations"]["opdyts_iteration"] for item in log if "opdyts_iteration" in item["annotations"]
])

plt.figure(dpi = 120, figsize = (6, 8))
plt.subplot(3,1,1)

for o in range(1, opdyts_iterations + 1):
    for c in range(number_of_candidates):
        cost = []
        objective = []

        for item in log:
            if "candidate" in item["annotations"] and item["annotations"]["candidate"] == c:
                if "opdyts_iteration" in item["annotations"] and item["annotations"]["opdyts_iteration"] == o:
                    cost.append(item["total_cost"])
                    objective.append(item["objective"])

        plt.plot(cost, objective, '-', marker = ".", markersize = 5, color = "C%d" % c)

cost = []
objective = []

for item in log:
    if not item["annotations"]["transient"]:
        cost.append(item["total_cost"])
        objective.append(item["objective"])

plt.plot(cost, objective, '.', markersize = 10, color = "k")
plt.grid()
plt.ylabel("Objective")


cost = []
equlibirum_gap = []
uniformity_gap = []

for item in log:
    annotations = item["annotations"]

    if "equilibrium_gap" in annotations:
        cost.append(item["total_cost"])
        equlibirum_gap.append(annotations["equilibrium_gap"])

    if "uniformity_gap" in annotations:
        uniformity_gap.append(annotations["uniformity_gap"])

plt.subplot(3,1,2)
plt.plot(cost, equlibirum_gap, label = "Equlibirum gap")
plt.plot(cost, uniformity_gap, label = "Uniformity gap")
plt.grid()
plt.legend(loc = "best")

cost = []
v = []
w = []

for item in log:
    annotations = item["annotations"]

    if "v" in annotations:
        cost.append(item["total_cost"])
        v.append(annotations["v"])

    if "w" in annotations:
        w.append(annotations["w"])

plt.subplot(3,1,3)
plt.plot(cost, v, label = "v")
plt.plot(cost, w, label = "w")
plt.grid()
plt.xlabel("Simulation cost")
plt.legend(loc = "best")

plt.tight_layout()
plt.show()
