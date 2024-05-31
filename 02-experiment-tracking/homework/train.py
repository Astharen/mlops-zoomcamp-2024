import os
import pickle
import click
import mlflow

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)


@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)
def run_train(data_path: str):

    with mlflow.start_run():
        train_data_path = os.path.join(data_path, "train.pkl")
        valid_data_path = os.path.join(data_path, "val.pkl")
        X_train, y_train = load_pickle(train_data_path)
        X_val, y_val = load_pickle(valid_data_path)
        # mlflow.log_param("train-data-path", train_data_path)
        # mlflow.log_param("valid-data-path", valid_data_path)

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse = mean_squared_error(y_val, y_pred, squared=False)
        # mlflow.log_metric("rmse", rmse)
        # mlflow.sklearn.log_model(rf, artifact_path="models")
        print(f"default artifacts URI: '{mlflow.get_artifact_uri()}'")


if __name__ == '__main__':
    mlflow.autolog()
    mlflow.set_experiment("homework-week-2")
    run_train()
