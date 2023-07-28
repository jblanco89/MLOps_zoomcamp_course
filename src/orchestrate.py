'''
Defines a command-line interface (CLI) for running data processing workflow 
defined in the "orchestrate_module" module. 
The workflow requires two input arguments: "symbol" and "date_end." 
The user can provide these arguments when executing the script.

'''

from orchestrate_module import set_workflow
import argparse

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Run the workflow with symbol and date_end arguments.')
    parser.add_argument('symbol', type=str, help='The symbol for data ingestion (e.g., MSFT)')
    parser.add_argument('date_end', type=str, help='The end date for data ingestion (e.g., 2023-02-28)')
    args = parser.parse_args()

    set_workflow(args.symbol, args.date_end)

    