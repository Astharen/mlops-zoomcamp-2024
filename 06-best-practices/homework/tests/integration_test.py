import os
import pandas as pd
from datetime import datetime

import sys

sys.path.append(os.path.join(os.getcwd()))

import batch


year = 2023
month = 1

endpoint_url = os.getenv('S3_ENDPOINT_URL')
input_pattern= os.getenv('INPUT_FILE_PATTERN')
input_pattern = input_pattern.format(year=year, month=month)
output_pattern = os.getenv('OUTPUT_FILE_PATTERN')
output_pattern = output_pattern.format(year=year, month=month)

options = {
    'client_kwargs': {
        'endpoint_url': endpoint_url
    }
}


def dt(hour, minute, second=0):
    return datetime(2023, 1, 1, hour, minute, second)

data = [
    (None, None, dt(1, 1), dt(1, 10)),
    (1, 1, dt(1, 2), dt(1, 10)),
    (1, None, dt(1, 2, 0), dt(1, 2, 59)),
    (3, 4, dt(1, 2, 0), dt(2, 2, 1)),      
]

columns = ['PULocationID', 'DOLocationID', 'tpep_pickup_datetime', 'tpep_dropoff_datetime']
categorical = ['PULocationID', 'DOLocationID']
df = pd.DataFrame(data, columns=columns)
 
print('input data:', input_pattern)
batch.save_data(df, input_pattern)

batch.main(year=year, month=month)

df_results = batch.read_data(output_pattern)

assert round(df_results.predicted_duration.sum(axis=0), 2) == 36.28
