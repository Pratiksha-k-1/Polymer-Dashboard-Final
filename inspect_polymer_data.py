import pandas as pd

df = pd.read_csv("all_data/polymer_spot_timeseries.csv")
print(df.columns.tolist())
print(df.head())
