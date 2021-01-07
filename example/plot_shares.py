import matplotlib.pyplot as plt
import pickle
import numpy as np
import sys
import palettable

path = "optimization.p"

if len(sys.argv) > 1:
    path = sys.argv[1]

data = pickle.load(open(path, "rb"))

objective = np.inf
COLORS = COLORS =  palettable.cartocolors.qualitative.Vivid_8.mpl_colors
modes = ["car", "pt", "bike", "walk"]

for row in data:
    if row["objective"] < objective:
        objective = row["objective"]
        index = row["evaluator_runs"]
        info = row["information"]

        plt.figure()
        plt.subplot(1, 2, 1)

        plt.title("Region (%d)" % index)

        for mode, color in zip(modes, COLORS):
            plt.plot(np.log10(info["region_bounds"][1:]), info["region_reference_shares"][mode], linestyle = ":", color = color)
            plt.plot(np.log10(info["region_bounds"][1:]), info["region_simulation_shares"][mode], linestyle = "-", color = color)

        plt.subplot(1, 2, 2)

        plt.title("Paris (%d)" % index)

        for mode, color in zip(modes, COLORS):
            plt.plot(np.log10(info["paris_bounds"][1:]), info["paris_reference_shares"][mode], linestyle = ":", color = color)
            plt.plot(np.log10(info["paris_bounds"][1:]), info["paris_simulation_shares"][mode], linestyle = "-", color = color)

        plt.tight_layout()
        plt.savefig("shares/%05d.png" % index)
