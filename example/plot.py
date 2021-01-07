import matplotlib.pyplot as plt
import pickle
import numpy as np
import sys

path = "optimization_output_nelder_mead.p"

if len(sys.argv) > 1:
    path = sys.argv[1]

data = pickle.load(open(path, "rb"))

objectives = []
parameters = []

best_objectives = []
best_parameters = []

for row in data:
    objectives.append(row["objective"])
    parameters.append(row["x"])

    if len(best_objectives) == 0 or objectives[-1] < best_objectives[-1]:
        best_objectives.append(objectives[-1])
        best_parameters.append(parameters[-1])
    else:
        best_objectives.append(best_objectives[-1])
        best_parameters.append(best_parameters[-1])

plt.figure(figsize = (8, 3), dpi = 120)

plt.subplot(1,2,1)
plt.plot(objectives, color = "C0", alpha = 0.25, marker = ".", linestyle = "none")
plt.plot(best_objectives, color = "C0")
plt.grid()
plt.xlabel("Iteration")
plt.ylabel("Objective")

plt.subplot(1,2,2)

parameters = np.array(parameters).T

for index, row in enumerate(parameters):
    plt.plot(row, color = "C%d" % index, alpha = 0.25, marker = ".", linestyle = "none")

best_parameters = np.array(best_parameters).T

for index, row in enumerate(best_parameters):
    plt.plot(row, color = "C%d" % index)

plt.grid()
plt.xlabel("Iteration")
plt.ylabel("Parameter")

plt.tight_layout()

plt.show()

perturbation_lengths = []
gradient_lengths = []

for row in data:
    perturbation_factors.append(row["annotation"]["perturbation_length"])
    gradient_lengths.append(row["annotation"]["gradient_length"])
