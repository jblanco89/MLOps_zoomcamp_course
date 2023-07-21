from mlflow.tracking import MlflowClient
from mlflow.models.signature import infer_signature

import pickle
import mlflow
import numpy as np
import pandas as pd
import mlflow.sklearn

if __name__ == '__main__':
    # mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_tracking_uri("http://34.175.211.162:5000/")
    EXPERIMENT_NAME = "LSTM_Hyperparameter_Experiment"
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    print(f"experiment_id={experiment.experiment_id}")
    # Get the best performing run based on rmse
    
    
    
    client = MlflowClient()
    runs = client.search_runs(experiment.experiment_id)
    runs_data = [(run.data.metrics['rmse'], run) for run in runs]
    sorted_runs_data = sorted(runs_data, key=lambda x: x[0])
    sorted_runs_df = pd.DataFrame(sorted_runs_data, columns=['rmse', 'run'])
    best_run = sorted_runs_df.iloc[0]['run']
    print(best_run)

    model_uri = f"runs:/{best_run.info.run_id}/lstm_model"
    registered_model = mlflow.register_model(model_uri, "LSTM Model")
    print(f"Registered Model Version: {registered_model.version}")