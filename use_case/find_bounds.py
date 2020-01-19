import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

reference_path = "/home/shoerl/bo/reference/reference.csv"
df = pd.read_csv(reference_path, sep = ";")

df = df[df["crowfly_distance"] > 0]
df = df[df["preceedingPurpose"] != "outside"]
df = df[df["followingPurpose"] != "outside"]

number_of_quantiles = 5
counts = []

plt.figure()
for mode in ("car", "pt", "bike", "walk"):
    f = df["mode"] == mode

    quantiles = np.arange(1, number_of_quantiles) / number_of_quantiles
    values = [df[f]["network_distance"].quantile(q) for q in quantiles]

    classes = np.digitize(df[f]["network_distance"], values)
    counts += [np.sum(classes == k) for k in range(len(quantiles) + 1)]

    plt.plot(values, quantiles, label = mode)
    print(mode, values)

counts = np.array(counts)
print(counts / np.sum(counts))

plt.legend()
plt.show()

#plt.figure()

quantiles = np.arange(1, number_of_quantiles) / number_of_quantiles
values = [df[df["mode"] == "car"]["travel_time"].quantile(q) for q in quantiles]
#plt.plot(values, quantiles)
print("car_travel_time", values)
#plt.show()
