import pandas as pd
import numpy as np

def get_mode_share_reference(reference_path):
    df_reference = pd.read_csv(reference_path, sep = ";")
    df_reference = df_reference[df_reference["crowfly_distance"] > 0]

    reference = np.array([
        np.sum(df_reference.loc[df_reference["mode"] == "car", "weight"]),
        np.sum(df_reference.loc[df_reference["mode"] == "pt", "weight"]),
        np.sum(df_reference.loc[df_reference["mode"] == "bike", "weight"]),
        np.sum(df_reference.loc[df_reference["mode"] == "walk", "weight"])
    ])
    reference /= np.sum(reference)

    return reference

def get_travel_time_reference(reference_path, number_of_bins):
    df_reference = pd.read_csv(reference_path, sep = ";")
    df_reference = df_reference[df_reference["crowfly_distance"] > 0]

    bins = [df_reference["travel_time"].quantile(b / number_of_bins) for b in range(1, number_of_bins + 1)]
    bins[-1] = np.inf

    classes = np.digitize(df_reference["travel_time"], bins)
    counts = np.array([np.sum(classes == i) for i in range(number_of_bins)], dtype = np.float)
    counts /= np.sum(counts)

    return counts, bins
