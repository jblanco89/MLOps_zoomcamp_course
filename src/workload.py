'''
Defines two functions:

1) get_stock_prices: Fetches daily stock prices from Yahoo Finance for a 
specified stock symbol up to a given end date. It then adds the stock symbol 
to the data, resets the index, and exports it to a CSV file. 
The CSV file is saved with a filename based on the stock symbol and end date. 
The function returns the path to the saved CSV file.

2) upload_csv_to_gcs: Uploads a local CSV file to Google Cloud Storage (GCS). 
It takes the GCS bucket name, local file path of the CSV file to upload, 
and the name of the blob (object) to create in the bucket as input. 
The function uses the Google Cloud Storage client to upload the 
file and returns the public URL of the uploaded CSV file in GCS. 
After uploading, the local CSV file is removed.


'''



# cloud_function.py
import pickle
import pandas as pd
import numpy as np
from pandas_datareader import data as pdr
import datetime
from google.cloud import bigquery
from google.cloud import storage
import yfinance as yf
import os
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

        os.remove(source_file_path)

        return private_url

    except Exception as e:
        print("An error occurred:", str(e))
        return None