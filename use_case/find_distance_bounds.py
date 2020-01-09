import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

reference_path = "/home/shoerl/bo/reference/reference.csv"
df = pd.read_csv(reference_path, sep = ";")

df = df[df["crowfly_distance"] > 0]
df = df[df["preceedingPurpose"] != "outside"]
df = df[df["followingPurpose"] != "outside"]

number_of_quantiles = 5

plt.figure()
for mode in ("car", "pt", "bike", "walk"):
    f = df["mode"] == mode

    quantiles = np.arange(1, number_of_quantiles) / number_of_quantiles
    values = [df[f]["network_distance"].quantile(q) for q in quantiles]

    plt.plot(values, quantiles, label = mode)
    print(mode, values)

plt.legend()
plt.show()
