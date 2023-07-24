from google.cloud import bigquery
import pandas as pd
import pandas_gbq
import json


def generate_report_to_bq():
    # Replace 'dataReport.json' with the correct file path
    file_path = './reports/dataReport.json'

    # Read the JSON data from the file
    with open(file_path, 'r') as json_file:
        json_data = json.load(json_file)


    metrics_data = json_data["metrics"]
    timestamp = json_data["timestamp"]

    # Create an empty list to store rows of data
    rows = {}

    # Iterate through the metrics_data and extract the relevant information
    for metric_data in metrics_data:
        metric_name = metric_data["metric"]
        result_data = metric_data["result"]

        if metric_name not in rows:
            rows[metric_name] = {
                "timestamp": timestamp,
                "metric_name": metric_name,
            }
        
        if "drift_by_columns" in result_data:
            drift_by_columns = result_data["drift_by_columns"]
            
            for column_name, column_data in drift_by_columns.items():
                current_data = column_data["current"]["small_distribution"]
                reference_data = column_data["reference"]["small_distribution"]
                # Convert lists to JSON strings
                current_x = json.dumps(current_data["x"])
                current_y = json.dumps(current_data["y"])
                reference_x = json.dumps(reference_data["x"])
                reference_y = json.dumps(reference_data["y"])
                rows[metric_name].update({
                    "timestamp": timestamp,
                    "metric_name": metric_name,
                    "column_name": column_name,
                    "column_type": column_data["column_type"],
                    "stattest_name": column_data["stattest_name"],
                    "stattest_threshold": column_data["stattest_threshold"],
                    "drift_score": column_data["drift_score"],
                    "drift_detected": column_data["drift_detected"],
                    "current_x": current_x,
                    "current_y": current_y,
                    "reference_x": reference_x,
                    "reference_y": reference_y,

                })

    final_rows = list(rows.values())


    df = pd.DataFrame(final_rows)

    # print(df)

    # Set your GCP project ID
    project_id = "ambient-decoder-391319"

    # Set the BigQuery dataset and table name
    dataset_name = "evidently_report001"
    table_name = "daily_report"

    # Upload the DataFrame to BigQuery
    return pandas_gbq.to_gbq(df, f"{dataset_name}.{table_name}", project_id=project_id, if_exists="append")




