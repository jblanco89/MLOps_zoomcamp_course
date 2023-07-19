from google.cloud import bigquery
import json

# Replace 'dataReport.json' with the correct file path
file_path = './reports/dataReport.json'

# Read the JSON data from the file
with open(file_path, 'r') as json_file:
    json_data = json.load(json_file)

# Set your project ID and dataset ID
project_id = 'ambient-decoder-391319'
dataset_id = 'evidently_report001'
table_id = 'daily_report'

# Initialize the BigQuery client
client = bigquery.Client(project=project_id)

# Define the table schema
schema = [
    bigquery.SchemaField("version", "STRING"),
    bigquery.SchemaField("timestamp", "STRING"),
    bigquery.SchemaField("DatasetDriftMetric_drift_share", "FLOAT"),
    bigquery.SchemaField("DatasetDriftMetric_number_of_columns", "INTEGER"),
    bigquery.SchemaField("DatasetDriftMetric_number_of_drifted_columns", "INTEGER"),
    # Add more fields for each key in the JSON data
    # Example:
    bigquery.SchemaField("DataDriftTable_number_of_columns", "INTEGER"),
    bigquery.SchemaField("DataDriftTable_number_of_drifted_columns", "INTEGER"),
    # Add fields for nested structures
    # Example:
    bigquery.SchemaField("DataDriftTable_drift_by_columns_Close_column_name", "STRING"),
    bigquery.SchemaField("DataDriftTable_drift_by_columns_Close_column_type", "STRING"),
    bigquery.SchemaField("DataDriftTable_drift_by_columns_Close_stattest_name", "STRING"),
    # Add more fields for 'current' and 'reference' nested structures
    # Example:
    bigquery.SchemaField("DataDriftTable_drift_by_columns_Close_current_small_distribution_x", "FLOAT", mode="REPEATED"),
    bigquery.SchemaField("DataDriftTable_drift_by_columns_Close_current_small_distribution_y", "FLOAT", mode="REPEATED"),
    bigquery.SchemaField("DataDriftTable_drift_by_columns_Close_reference_small_distribution_x", "FLOAT", mode="REPEATED"),
    bigquery.SchemaField("DataDriftTable_drift_by_columns_Close_reference_small_distribution_y", "FLOAT", mode="REPEATED"),
    # Add more fields for other metrics
    # Example:
    bigquery.SchemaField("DatasetSummaryMetric_almost_duplicated_threshold", "FLOAT"),
    bigquery.SchemaField("DatasetSummaryMetric_current_target", "STRING"),
    bigquery.SchemaField("DatasetSummaryMetric_current_prediction", "STRING"),

    bigquery.SchemaField("DatasetMissingValuesMetric_share_of_rows_with_missing_values", "INTEGER")

    # Add more fields for other metrics and nested structures as needed
]

# Get the dataset reference
dataset_ref = client.dataset(dataset_id)

# Create the dataset if it does not exist
try:
    client.get_dataset(dataset_ref)
except Exception as e:
    dataset = bigquery.Dataset(dataset_ref)
    dataset = client.create_dataset(dataset)

# Get the table reference
table_ref = dataset_ref.table(table_id)

# Create the table if it does not exist
try:
    table = client.get_table(table_ref)
except Exception as e:
    table = bigquery.Table(table_ref, schema=schema)
    table = client.create_table(table)

data_row = {
    "version": json_data["version"],
    "timestamp": json_data["timestamp"],
}

for metric in json_data["metrics"]:
    metric_name = metric["metric"]
    result = metric["result"]
    for key, value in result.items():
        if isinstance(value, dict):
            for sub_key, sub_value in value.items():
                if isinstance(sub_value, dict):
                    for sub_sub_key, sub_sub_value in sub_value.items():
                        # Add nested fields
                        column_name = f"{metric_name}_{key}_{sub_key}_{sub_sub_key}"
                        data_row[column_name] = sub_sub_value
                else:
                    # Add fields for nested structures
                    column_name = f"{metric_name}_{key}_{sub_key}"
                    data_row[column_name] = sub_value
        else:
            data_row[f"{metric_name}_{key}"] = value

# Insert the row into the table
# errors = client.insert_rows_json(table, [data_row])

if len(data_row) > 0:
    try:
        errors = client.insert_rows_json(table, [data_row])
        print(len(data_row))
    except:
        pass

# if errors:
#     print("Encountered errors while inserting rows: {}".format(errors))
# else:
#     print("Data inserted successfully into BigQuery table.")
