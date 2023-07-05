
# TO TEST LATER
# DO NOT EXECUTE YET

import mlflow
import mlflow.pyfunc
import pandas as pd
mlflow.set_tracking_uri("sqlite:///mlflow.db")
logged_model = 'runs:/9aa4885427bf48c2837b1c608c799945/lstm_model'

# Load model as a PyFuncModel.
loaded_model = mlflow.pyfunc.load_model(logged_model)

print(loaded_model)

data_raw = technical_indicators(df_path=df_path)
data = handle_outliers(data_raw, 'Close')
data = drop_columns(data)
scaled_data = data_preprocess(data)


print(loaded_model.predict(scaled_data.drop('Close', axis=1)))

