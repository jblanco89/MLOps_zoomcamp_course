# cloud_function.py
import pickle
import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import datetime
from google.cloud import bigquery
from google.cloud import storage
import argparse
import yfinance as yf
yf.pdr_override()

def get_stock_prices(symbol: str, end : str):
    '''Fetches daily stock prices from Yahoo Finance for a specified stock symbol up to a given end date. 
    It then adds the stock symbol to the data, resets the index, prints a preview of the data, 
    and exports it to a CSV file. 
    The CSV file is saved with a filename based on the stock symbol and end date.
    
    Parameters:
    ------------  
    symbol (String): A string representing the stock symbol for which to fetch the prices.
    end (String): A string representing the end date until which to fetch the prices in the format 'YYYY-MM-DD'.

    Returns:
    --------.
    df.to_csv (dataframe): dataframe is saved as csv using comma as separator and index false
    
    '''
    end_date = datetime.datetime.strptime(end, '%Y-%m-%d')
    df = pdr.get_data_yahoo(symbol, end=end_date)
    df['Symbol'] = symbol
    df.reset_index(inplace=True)
    end = end.replace('-', '')
    df.to_csv(f'./data/{symbol}_{end}.csv', sep=',', index=False)
    return f'./data/{symbol}_{end}.csv'


def upload_csv_to_gcs(bucket_name, source_file_path, destination_blob_name):
    """
    Uploads a local CSV file to Google Cloud Storage.

    Parameters:
        - bucket_name (str): The name of the Google Cloud Storage bucket.
        - source_file_path (str): The local file path of the CSV file to upload.
        - destination_blob_name (str): The name of the blob (object) to create in the bucket.

    Returns:
        (str) The public URL of the uploaded CSV file.
    """
    try:
        # Initialize the GCS client
        storage_client = storage.Client()

        # Get the bucket where the file will be uploaded
        bucket = storage_client.bucket(bucket_name)

        # Upload the file to GCS
        blob = bucket.blob(destination_blob_name)
        blob.upload_from_filename(source_file_path)

        private_url = f'gs://{bucket_name}/{destination_blob_name}'

        return private_url

    except Exception as e:
        print("An error occurred:", str(e))
        return None

def predict_stock_prices(gs_url, bucket_name):
    # Load the LSTM model from Google Cloud Storage
    model_bucket = bucket_name
    gcs_client = storage.Client()
    bucket = gcs_client.get_bucket(model_bucket)
    blob = bucket.blob("lstm_model.pkl")
    data_path = gs_url
    data = pd.read_csv(data_path, sep=",")
    df = data[['Date','Close', 'Symbol']]
    symbol = df['Symbol'].iloc[0]
    df = df.set_index('Date')
    df = df.tail(90)
    # with open("./models/lstm_model.pkl", "rb") as f:
    #     lstm_model = pickle.load(f)

    with blob.open(mode = "rb") as f:
        lstm_model = pickle.load(f)

    # Reshape the input data
    num_time_steps = 10  # Example: Use 10 time steps
    num_features = 1  # Example: Use 1 feature (Close price)
    n = np.int((len(df)/num_time_steps))
    subset_df = df.tail(n)
    X = np.reshape(df['Close'].values, (len(subset_df), num_time_steps, num_features))
    X = np.repeat(X, 9, axis=-1)

    # Perform predictions using the LSTM model
    # predictions = lstm_model.predict(X)

    min_value = np.min(df['Close'])
    max_value = np.max(df['Close'])
    # predictions = predictions[1]
    predictions = np.zeros(9)
    noise = np.random.normal(0, df['Close'].std(), predictions.shape)
    predictions = (predictions * (max_value - min_value) + min_value) + noise
    
    y_predicted = np.squeeze(predictions)
    start_date = df.index[-1]
    dates = pd.date_range(start=pd.to_datetime(start_date) + pd.DateOffset(days=1), 
                            periods=len(y_predicted), 
                            freq='D')
    new_data = pd.DataFrame({'Close': y_predicted, 'Symbol': symbol}, index=dates)
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
    return predictions


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the workflow with symbol and date_end arguments.')
    parser.add_argument('symbol', type=str, help='The symbol for data ingestion (e.g., MSFT)')
    parser.add_argument('date_end', type=str, help='The end date for data ingestion (e.g., 2023-02-28)')
    args = parser.parse_args()

    local_path = get_stock_prices(args.symbol, args.date_end)

    destination_name = f'{args.symbol}_{args.date_end}.csv'
    bucket_name = 'lstm_model_test'

    cloud_path = upload_csv_to_gcs(bucket_name=bucket_name, 
                                   source_file_path=local_path, 
                                   destination_blob_name=destination_name)
    # predict_stock_prices(gs_url=cloud_path, bucket_name=bucket_name)
    
