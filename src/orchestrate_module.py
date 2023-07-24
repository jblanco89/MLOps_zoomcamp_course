import prefect as pf
import mlflow
import pandas as pd
import numpy as np
import time
from prefect import task, flow
from evidently import ColumnMapping
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset, DataQualityPreset
from evidently.metric_preset import TargetDriftPreset, RegressionPreset
from model_utilities import get_stock_prices, technical_indicators
from model_utilities import drop_columns, data_preprocess, handle_outliers
from model_utilities import lstm_model_train, reshape_test_data, reshape_data
from sklearn.metrics import mean_squared_error, r2_score
from mlflow.models.signature import infer_signature
from reports import generate_report_to_bq


# pf.context.config.load_system_config()

# mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_tracking_uri("http://34.175.211.162:5000/")
EXPERIMENT_NAME = "LSTM_Hyperparameter_Experiment"
mlflow.set_experiment(EXPERIMENT_NAME)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

@task(name="Data entry",
      description="Stock prices entry by csv file", 
      retries=3, 
      retry_delay_seconds=2)
def data_ingestion(symbol, date_end) -> str:
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
      retries=1, 
      retry_delay_seconds=1)
def train_model(df, symbol):
    # Train your LSTM model and log relevant metrics
    batch_size = 10
    epochs = 20
    hidden_units=128
    learning_rate=0.001
    model, X_test, Y_test = lstm_model_train(df=df, 
                        symbol=symbol, 
                        selected_hidden_units=hidden_units, 
                        selected_activation='relu', 
                        selected_learning_rate=learning_rate,
                        selected_batch_size=batch_size,
                        selected_epochs=epochs,
                        plot_lstm_model=False)

    return model, X_test, Y_test 

@task
def evaluate_model(df, model, X_test, Y_test):
    batch_size = 10
    epochs = 20
    hidden_units=128
    learning_rate=0.001

    df = df.tail(180) #last six months
    min_value = np.min(df['Close'])
    max_value = np.max(df['Close'])

    num_time_steps = 10  # Example: Use 10 time steps
    num_features = 1  # Example: Use 1 feature (Close price)
    n = np.int((len(df)/num_time_steps))
    subset_df = df.tail(n)
    X_test = np.reshape(df['Close'].values, (len(subset_df), num_time_steps, num_features))
    # print(X)
    X_test = np.repeat(X_test, 9, axis=-1)

    Y_pred = model.model.predict(X_test)
    Y_pred = Y_pred[0]
    Y_test = Y_test[-10:]

    Y_pred = Y_pred * (max_value - min_value) + min_value
    Y_test = Y_test * (max_value - min_value) + min_value

    time.sleep(1)
    print(Y_pred)
    print(Y_test)
    print(Y_test.shape)
    
    rmse = mean_squared_error(Y_pred, Y_test, squared=False)
    r2 = r2_score(Y_pred, Y_pred)
    signature = infer_signature(X_test, Y_test)

    
    with mlflow.start_run():
        # Evaluate your LSTM model and log relevant metrics
        mlflow.set_tag("developer", "Javier")
        mlflow.set_tag("model", "lstm")
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("hidden_units", hidden_units)
        mlflow.log_param("learning_rate", learning_rate)
        mlflow.log_param("batch_size", batch_size)
        mlflow.sklearn.log_model(model, 
                                    artifact_path="lstm_model", 
                                    signature=signature)
        mlflow.log_artifact(local_path="./models/lstm_model.pkl", artifact_path="models_pickle")
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
    
    return Y_pred, Y_test

@task
def generate_report(Y_pred, Y_test):
    report = Report(metrics=[
    DataDriftPreset()
    ])

    Y_pred_flattened = np.ravel(Y_pred)

    current = pd.DataFrame({'Close': Y_test})
    df_reference = pd.DataFrame({'Close': Y_pred_flattened})
    df_reset = df_reference.reset_index(drop=True)
    current_reset = current.reset_index(drop=True)

    df_reset = pd.DataFrame(df_reset, columns=['Close'])
    current_reset = pd.DataFrame(current_reset, columns=['Close'])
    report.run(reference_data=df_reset.tail(10), current_data=current_reset.tail(10))

    # report.save_html("./reports/dataReport.html")
    report.save_json("./reports/dataReport.json")


@flow
def set_workflow(symbol, date_end):
    path = data_ingestion(symbol, date_end)
    data_tech = set_technical_indicators(path)
    data_outliers = remove_outliers(data_tech)
    scaled_data = preprocess_data(data_outliers)
    model, X_test, Y_test = train_model(scaled_data, symbol)
    Y_pred, Y_test = evaluate_model(data_outliers, model, X_test, Y_test)
    # print(Y_pred)
    # print(Y_test)
    generate_report(Y_pred=Y_pred, Y_test=Y_test)
    generate_report_to_bq()


