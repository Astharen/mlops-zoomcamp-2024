import pandas as pd


print(pd.read_parquet('data/yellow_tripdata_2023-03.parquet').shape[0])