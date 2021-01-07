import pickle
import matplotlib.pyplot as plt

with open("optimization.p", "rb") as f:
    data = pickle.load(f)

    for index, mode in enumerate(["car", "pt", "bike", "walk"]):
        plt.plot(data[0]["information"]["region_reference_shares"][mode], color = "C%d" % index, linestyle = ":")
        plt.plot(data[0]["information"]["region_simulation_shares"][mode], color = "C%d" % index, linestyle = "-")

plt.show()
