import matplotlib.pyplot as plt
import pickle
import numpy as np
import sys

path = "optimization_output_nelder_mead.p"

if len(sys.argv) > 1:
    path = sys.argv[1]

data = pickle.load(open(path, "rb"))

perturbation_lengths = []
gradient_lengths = []

for row in data:
    if row["annotations"]["type"] == "negative_gradient":
        perturbation_lengths.append(row["annotations"]["perturbation_length"])
        gradient_lengths.append(row["annotations"]["gradient_length"])

plt.figure()

plt.subplot(1, 2, 1)
plt.plot(perturbation_lengths)
plt.title("Perturbation lengths")

plt.subplot(1, 2, 2)
plt.plot(gradient_lengths)
plt.title("Gradient lengths")

plt.tight_layout()
plt.show()
