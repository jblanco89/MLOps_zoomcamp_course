from model import get_stock_prices, technical_indicators, handle_outliers, data_preprocess, drop_columns
from model import reshape_test_data, predictions_as_array, show_data_result

import pickle
import mlflow

mlflow.set_tracking_uri("sqlite:///mlflow.db")

EXPERIMENT_NAME = "nyc-taxi-experiment"
mlflow.set_experiment(EXPERIMENT_NAME)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
print(f"experiment_id={experiment.experiment_id}")

# Loading the model
with open('./models/lstm_model.pkl', 'rb') as file:
    lstm_loaded_model = pickle.load(file)

data_raw = technical_indicators(df_path='./data/MSFT_20230601.csv')
data = handle_outliers(data_raw, 'Close')
data = drop_columns(data)
scaled_data = data_preprocess(data)

test_data = reshape_test_data(scaled_data)

data_predicted = predictions_as_array(data_raw, test_data, model=lstm_loaded_model)


show_data_result(data_raw, data_predicted, n=30, make_plot = True)

print(data_predicted)


