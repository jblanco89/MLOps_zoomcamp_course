from workload import get_stock_prices
from workload import upload_csv_to_gcs
import argparse

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
    