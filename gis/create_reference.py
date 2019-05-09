import pandas as pd
import shapely.geometry as geo
import geopandas as gpd
import tqdm

SCENARIO_SHAPEFILE_PATH = "zurich_20km.shp"
TRIPS_PATH = "/run/media/sebastian/shoerl_data/astra1802/analysis/reference/trips.csv"
PERSONS_PATH = "/run/media/sebastian/shoerl_data/astra1802/analysis/reference/persons.csv"
OUTPUT_PATH = "../simulation/reference.csv"

df = pd.read_csv(TRIPS_PATH, sep = ";")
df_persons = pd.read_csv(PERSONS_PATH, sep = ";")[["person_id", "person_weight"]]
df = pd.merge(df, df_persons, on = "person_id").rename(columns = {"person_weight": "weight"})

df_shapes = gpd.read_file(SCENARIO_SHAPEFILE_PATH)
df_shapes.crs = {"init": "EPSG:2056"}
df_shapes["scenario"] = df_shapes["scenario"].astype("category")

df_od = pd.DataFrame(df[["person_id", "trip_id", "origin_x", "destination_x", "origin_y", "destination_y"]], copy = True)
df_od["origin_geometry"] = [geo.Point(*xy) for xy in tqdm.tqdm(zip(df_od["origin_x"], df_od["origin_y"]), total = len(df))]
df_od["destination_geometry"] = [geo.Point(*xy) for xy in tqdm.tqdm(zip(df_od["destination_x"], df_od["destination_y"]), total = len(df))]
df_od = gpd.GeoDataFrame(df_od, crs = {"init": "EPSG:2056"})

df_od = df_od.set_geometry("origin_geometry")
df_od = gpd.sjoin(df_od, df_shapes.rename({"scenario": "origin_scenario"}, axis = 1), op = "within")
del df_od["index_right"]

df_od = df_od.set_geometry("destination_geometry")
df_od = gpd.sjoin(df_od, df_shapes.rename({"scenario": "destination_scenario"}, axis = 1), op = "within")
del df_od["index_right"]

df_od = df_od[["person_id", "trip_id", "origin_scenario", "destination_scenario"]]
df = pd.merge(df, df_od, how = "left", on = ["person_id", "trip_id"])

df["travel_time"] = df["arrival_time"] - df["departure_time"]
df = df[["mode", "crowfly_distance", "travel_time", "weight"]]
df = df[df["crowfly_distance"] > 0]

df.to_csv(OUTPUT_PATH, sep = ";", index = None)
