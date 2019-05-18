import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as tck
import sys

history_path = sys.argv[1]

with open(history_path, "rb") as f:
    history = pickle.load(f)

number_of_candidates = np.max([
    h["annotations"]["candidate"] for h in history if "candidate" in h["annotations"]
]) + 1

opdyts_iterations = np.max([
    h["annotations"]["opdyts_iteration"] for h in history if "opdyts_iteration" in h["annotations"]
])

plt.figure(dpi = 120, figsize = (6, 8))
plt.subplot(3,1,1)

for o in range(1, opdyts_iterations + 1):
    for c in range(number_of_candidates):
        cost = []
        objective = []

        for h in history:
            if "candidate" in h["annotations"] and h["annotations"]["candidate"] == c:
                if "opdyts_iteration" in h["annotations"] and h["annotations"]["opdyts_iteration"] == o:
                    cost.append(h["total_cost"])
                    objective.append(h["objective"])

        plt.plot(cost, objective, '-', marker = ".", markersize = 5, color = "C%d" % c)

cost = []
objective = []

for h in history:
    if not h["annotations"]["transient"]:
        cost.append(h["total_cost"])
        objective.append(h["objective"])

plt.plot(cost, objective, '.', markersize = 10, color = "k")
plt.grid()
plt.ylabel("Objective")


cost = []
equlibirum_gap = []
uniformity_gap = []

for h in history:
    annotations = h["annotations"]

    if "equilibrium_gap" in annotations:
        cost.append(h["total_cost"])
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

for h in history:
    annotations = h["annotations"]

    if "v" in annotations:
        cost.append(h["total_cost"])
        v.append(annotations["v"])

    if "w" in annotations:
        w.append(annotations["w"])

plt.subplot(3,1,3)
plt.plot(cost, v, label = "v")
plt.plot(cost, w, label = "w")
plt.grid()
plt.xlabel("Transitions")
plt.legend(loc = "best")

plt.tight_layout()
plt.show()
