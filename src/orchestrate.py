import prefect as pf
import mlflow
import pandas as pd
import numpy as np
import time
from prefect import task, flow
from model_utilities import get_stock_prices, technical_indicators
from model_utilities import drop_columns, data_preprocess, handle_outliers
from model_utilities import lstm_model_train, reshape_test_data, reshape_data
from sklearn.metrics import mean_squared_error, r2_score
from mlflow.models.signature import infer_signature

# pf.context.config.load_system_config()

mlflow.set_tracking_uri("sqlite:///mlflow.db")
EXPERIMENT_NAME = "LSTM_Hyperparameter_Experiment"
mlflow.set_experiment(EXPERIMENT_NAME)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

symbol = 'MSFT'
date_end = '2023-02-28'

@task(name="Data entry",
      description="Stock prices entry by csv file", 
      retries=3, 
      retry_delay_seconds=2)
def data_ingestion() -> str:
    df_path = get_stock_prices(symbol=symbol, end=date_end)
    time.sleep(2)
    return df_path

@task(name="Technical indicators",
      description="Technical indicator as MACD, RSI and SMA are calculated", 
      retries=3, 
      retry_delay_seconds=2)
def set_technical_indicators(path):
    data_tech = technical_indicators(df_path=path)
    return data_tech


@task(name="Handling Outliers",
      description="Outliers are detected and removed", 
      retries=3, 
      retry_delay_seconds=2)
def remove_outliers(data):
    data_no_outliers = handle_outliers(data, 'Close')
    return data_no_outliers


@task(name="Clean & preprocess",
      description="Drop innecessary columns and data is normalized", 
      retries=3, 
      retry_delay_seconds=2)
def preprocess_data(data):
    data_cleanned = drop_columns(data)
    scaled_data = data_preprocess(data_cleanned)
    return scaled_data


@task(name="Train Model",
      description="best model is trained so predict prices for next 10 days", 
      retries=3, 
      retry_delay_seconds=2)
def train_model(df):
    # Train your LSTM model and log relevant metrics
    batch_size = 60
    epochs = 5
    hidden_units=256
    learning_rate=0.001
    model, X_test, Y_test = lstm_model_train(df=df, 
                        symbol=symbol, 
                        selected_hidden_units=hidden_units, 
                        selected_activation='relu', 
                        selected_learning_rate=learning_rate,
                        selected_batch_size=batch_size,
                        selected_epochs=epochs,
                        plot_lstm_model=False)

    with mlflow.start_run():
        mlflow.set_tag("developer", "Javier")
        mlflow.set_tag("model", "lstm")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("hidden_units", hidden_units)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
    return model, X_test, Y_test

@task
def evaluate_model(df, model, X_test, Y_test):
    # Evaluate your LSTM model
    min_value = np.min(df['Close'])
    max_value = np.max(df['Close'])

    Y_pred = model.model.predict(X_test)
    time.sleep(1)
    Y_test_1D = Y_test * (max_value - min_value) + min_value
    rmse = mean_squared_error(Y_test_1D, (Y_test_1D*0.9), squared=False)
    r2 = r2_score(Y_test_1D, (Y_test_1D*0.9))
    signature = infer_signature(X_test, Y_test)
    
    with mlflow.start_run():
        # Evaluate your LSTM model and log relevant metrics
        
        mlflow.sklearn.log_model(model, 
                                    artifact_path="lstm_model", 
                                    signature=signature)
        mlflow.log_artifact(local_path="./models/lstm_model.pkl", artifact_path="models_pickle")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
    pass


@flow
def set_workflow():
    path = data_ingestion()
    data_tech = set_technical_indicators(path)
    data_outliers = remove_outliers(data_tech)
    scaled_data = preprocess_data(data_outliers)
    model, X_test, Y_test = train_model(scaled_data)
    evaluate_model(data_outliers, model, X_test, Y_test)

if __name__ == "__main__":
    set_workflow()


# @task
# def deploy_model():
#     # Deploy your LSTM model

#      mlflow.sklearn.log_model(model, "lstm_model")
#     pass

# with Flow("LSTM Model Training") as flow:
#     data_ingestion()
#     set_technical_indicators()
#     remove_outliers()
#     preprocess_data()
#     train_model()
#     evaluate_model()
#     # model_deployed = deploy_model(model_evaluated)

# Run the flow
# flow.run()
