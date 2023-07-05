from hyperparameters import run_trials_lstm_model
from mlflow.models.signature import infer_signature

import pickle
import mlflow
import numpy as np
import pandas as pd
import mlflow.sklearn

if __name__ == '__main__':
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    EXPERIMENT_NAME = "LSTM_Hyperparameter_Experiment"
    mlflow.set_experiment(EXPERIMENT_NAME)
    experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
    print(f"experiment_id={experiment.experiment_id}")

    # grid_search_lstm_model(df_path='./data/AAPL_20230601.csv')
    run_trials_lstm_model(df_path = './data/AAPL_20230601.csv', n_trials=15)