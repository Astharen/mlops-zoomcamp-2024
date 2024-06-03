import os
import pickle
import mlflow


if 'data_exporter' not in globals():
    from mage_ai.data_preparation.decorators import data_exporter


@data_exporter
def export_data(models, *args, **kwargs):

    print('Server set')
    mlflow.set_tracking_uri("http://mlflow:5000")
    print(mlflow.get_tracking_uri())
    mlflow.set_experiment('lin_reg_taxi')
    print(mlflow.search_experiments())

    lr, dv = models
    with mlflow.start_run():
        mlflow.sklearn.log_model(lr, artifact_path="models")

        if not os.path.isdir('models'):
            os.mkdir('models')

        with open("models/preprocessor.b", "wb") as f_out:
            pickle.dump(dv, f_out)

        mlflow.log_artifact("models/preprocessor.b", artifact_path="preprocessor")
