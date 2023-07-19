# cloud_function.py
import pickle
import pandas as pd
import numpy as np
from google.cloud import bigquery
from google.cloud import storage
import gcsfs

def predict_stock_prices():
    # Load the LSTM model from Google Cloud Storage
    model_bucket = 'lstm_model_test'
    gcs_client = storage.Client()
    bucket = gcs_client.get_bucket(model_bucket)
    data_path = 'MSFT_20230425.csv'


    with open("./models/lstm_model.pkl", "rb") as f:
        lstm_model = pickle.load(f)

    # Read the daily stock prices data from CSV (assumes the file is in the event data)
    # csv_data = "./data/MSFT_20230425.csv"
    csv_data = f"gs://{model_bucket}/{data_path}"
    data = pd.read_csv(csv_data, sep=",")

    df = data[['Date','Close']]
    df = df.set_index('Date')
    df = df.tail(360)
    # Reshape the input data
    num_time_steps = 10  # Example: Use 10 time steps
    num_features = 1  # Example: Use 1 feature (Close price)
    n = np.int((len(df)/num_time_steps))
    subset_df = df.tail(n)
    X = np.reshape(df['Close'].values, (len(subset_df), num_time_steps, num_features))
    # print(X)
    X = np.repeat(X, 9, axis=-1)


    # Perform predictions using the LSTM model
    predictions = lstm_model.predict(X)

    min_value = np.min(df['Close'])
    max_value = np.max(df['Close'])
    predictions = predictions[1]
    predictions = predictions * (max_value - min_value) + min_value
    
    y_predicted = np.squeeze(predictions)
    start_date = df.index[-1]
    dates = pd.date_range(start=pd.to_datetime(start_date) + pd.DateOffset(days=1), 
                            periods=len(y_predicted), 
                            freq='D')
    new_data = pd.DataFrame({'Close': y_predicted}, index=dates)
    data_result = pd.concat([df, new_data])
    data_result.index = pd.to_datetime(data_result.index)
    data_result.index = data_result.index.tz_localize(None)
    print(data_result)

    # Save the predictions to BigQuery
    bq_client = bigquery.Client()
    dataset_id = 'stock_output'
    table_id = 'predicted_prices'
    table_ref = bq_client.dataset(dataset_id).table(table_id)

    table_string = f"{table_ref.project}.{table_ref.dataset_id}.{table_ref.table_id}"

    data_result = data_result.reset_index(drop=False)

    data_result.to_gbq(destination_table=table_string, project_id='ambient-decoder-391319', if_exists='replace')

    # return print(predictions)

def main():
  predict_stock_prices()

if __name__ == "__main__":
    main()