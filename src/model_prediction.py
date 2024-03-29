'''
This script loads a trained LSTM model using MLflow and makes predictions on a dataset. 

'''

from model_utilities import technical_indicators, handle_outliers, drop_columns, data_preprocess, lstm_model_train
import mlflow
import mlflow.pyfunc
import pandas as pd
mlflow.set_tracking_uri("sqlite:///mlflow.db")
logged_model = 'runs:/f9c2ed1d39fc40f49291aabd2c248913/lstm_model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

print(loaded_model)

df_path = './data/GOOG_20230130.csv'

data_raw = technical_indicators(df_path=df_path)
data = handle_outliers(data_raw, 'Close')
data = drop_columns(data)
scaled_data = data_preprocess(data)



print(loaded_model.predict(scaled_data.tail(50)))

