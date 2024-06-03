import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from mlops.utils.data_preparation.feature_selector import select_features

# Data cleaning
df = pd.read_parquet('./data/yellow_tripdata_2023-03.parquet')

df.tpep_dropoff_datetime = pd.to_datetime(df.tpep_dropoff_datetime)
df.tpep_pickup_datetime = pd.to_datetime(df.tpep_pickup_datetime)

df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
df.duration = df.duration.dt.total_seconds() / 60

df = df[(df.duration >= 1) & (df.duration <= 60)]

categorical = ['PULocationID', 'DOLocationID']
df[categorical] = df[categorical].astype(str)

print(df.shape[0])

df = df.iloc[:100]

# ---------------------------------------- homework 1 ----------------------------------------
categorical = ['PULocationID', 'DOLocationID']
train_dicts = df[categorical].to_dict(orient='records')
dv = DictVectorizer()
X_train = dv.fit_transform(train_dicts)

target = 'duration'
y_train = df[target].values


lr = LinearRegression()
lr.fit(X_train, y_train)

y_pred = lr.predict(X_train)

print(lr.intercept_)


import os
import pickle
import mlflow


# os.system('mlflow server --backend-store-uri sqlite:///backend.db')

mlflow.set_tracking_uri("http://0.0.0.0:5000")
print(mlflow.get_tracking_uri())
mlflow.set_experiment('lin_reg_taxi')
print(mlflow.search_experiments())

with mlflow.start_run():
    mlflow.sklearn.log_model(lr, artifact_path="models")

    if not os.path.isdir('models'):
        os.mkdir('models')

    with open("models/preprocessor.b", "wb") as f_out:
        pickle.dump(dv, f_out)

    mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")