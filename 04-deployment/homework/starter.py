#!/usr/bin/env python
# coding: utf-8


import sys
import pickle
import pandas as pd



def read_data(filename, categorical):
    df = pd.read_parquet(filename)
    
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60

    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()

    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


def predict(df, categorical):

    with open('./model.bin', 'rb') as f_in:
        dv, model = pickle.load(f_in)

    dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)
    return y_pred


def save_predictions(df, y_pred, year, month):
    ride_id = pd.DataFrame(f'{year:04d}/{month:02d}_' + df.index.astype('str'), columns=['ride_id'])

    output_file = f'yellow_taxi_prediction_{year:04d}_{month:02d}.parquet'
    y_pred = pd.DataFrame(y_pred, columns=['predictions'])
    df_result = pd.DataFrame(pd.concat([ride_id, y_pred]))
    df_result.to_parquet(
        output_file,
        engine='pyarrow',
        compression=None,
        index=False
    )


def main():
    year = int(sys.argv[1])
    month = int(sys.argv[2])

    print(year, month)

    categorical = ['PULocationID', 'DOLocationID']

    df = read_data(f'https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_{year:04d}-{month:02d}.parquet', 
                   categorical)
    y_pred = predict(df, categorical)
    print(f'The mean of the predictions is: {y_pred.mean()}')
    print(f'The standard deviations of the predictions is: {y_pred.std()}')

    save_predictions(df, y_pred, year, month)


if __name__ == '__main__':
    main()
