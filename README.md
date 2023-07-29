# Stock Prices Prediction based on Deep Learning model (MLOps Zoomcamp Project)

## Table Of Content

1. [Project Description](#project-description)
2. [Solution Architecture (Google Cloud Platform)](#solution-architecture)
3. [System in Production](#system-in-production)
4. [User Guide](#user-guide)
5. [Model Management (MLFlow)](#model-management-mlflow)
6. [Orchestration (Prefect 2.0)](#orchestration-prefect-20)
7. [Monitoring (Evidently AI)](#monitoring-evidently-ai)
8. [Dashboard](#dashboard)
9. [Makefile and Dockefile](#makefile-and-dockefile)
10. [Recomendations](#recomendations)
11. [References](#references)
12. [Annexes](#annexes)
13. [Exploratory Data Analysis](./EDA/stock_market_analysis.ipynb)


## Project Description
This project aims to develop and deploy a stock price prediction system using an LSTM model, which predicts stock prices with a 10-day lag based on historical OHLC and technical indicators such as MACD, RSI, SMA and EMA. By providing accurate predictions, the system empowers traders to make well-informed decisions, optimize investment strategies, and minimize risks while capitalizing on market fluctuations.

For this project cloud-based approach was chosen. By utilizing Google Cloud Platform (GCP) and MLflow integration, efficiency and scalability of the system is significatibly enhanced. Deploying the LSTM model on GCP also enables seamless batch processing of large datasets, ensuring timely predictions. Additionally, MLflow's model tracking and versioning features facilitate efficient monitoring and maintenance of the model's performance over time. Furthermore, the Prefect framework orchestrates the workflow, automating data processing, scheduling, and error handling, resulting in a streamlined and reliable end-to-end solution.

To sum, this stock price prediction system addresses the specific needs of traders and investors by providing accurate predictions, timely insights, and efficient model management through cloud technology. With this solution, traders could stay ahead in the fast-paced financial markets, make informed decisions, and enhance their overall financial outcomes.


## Solution Architecture

**Important**: *Linux (Ubuntu 22.04 LTS) is the recommended development environment.*

**LSTM model** was deployed as <u>*Batch Workload*</u> using some GCP tools such as:

1. Cloud Storage
2. Cloud Scheduler/Trigger
3. Cloud Functions
4. BigQuery
5. Google Looker Studio (previously known as Data Studio)

As we can see, the workflow solution implemented in Google Cloud Platform (GCP) leverages serverless batch processing using Cloud Storage, Cloud Functions, BigQuery, and Google Looker Studio to predict stock prices based on historical OHLC (Open, High, Low, Close) data.

The process begins when users upload a CSV file containing OHLC price data for their desired stock to **Google Cloud Storage**. As soon as the file is uploaded, the **Cloud Function** is triggered to execute the LSTM (Long Short-Term Memory) model, which is designed to forecast the stock's prices for the next 10 days (*see code below*). 

```python
# This function runs as Google Cloud Function inside GCP

def predict_stock_prices(gs_url: str, bucket_name: str) -> np.array():
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

    with blob.open(mode = "rb") as f:
        lstm_model = pickle.load(f)

    # Reshape the input data
    num_time_steps = 10  # setting 10 time steps
    num_features = 1  # setting 1 feature (Close price)
    n = np.int((len(df)/num_time_steps))
    subset_df = df.tail(n)
    X = np.reshape(df['Close'].values, (len(subset_df), num_time_steps, num_features))
    X = np.repeat(X, 9, axis=-1)

    # Perform predictions using the LSTM model
    predictions = lstm_model.predict(X)

    min_value = np.min(df['Close'])
    max_value = np.max(df['Close'])
    predictions = predictions[1]
    predictions = np.zeros(9)

    predictions = (predictions * (max_value - min_value) + min_value)
    
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

```

The results of the LSTM model are then stored in a dedicated **BigQuery** table named `predicted_prices`. The data in the table feeds into a **Looker Studio** Dashboard, where users can visualize and explore the forecasted stock prices in a user-friendly and intuitive manner.

The integration of Cloud Functions, BigQuery, and Looker Studio streamlines the entire process, making it a powerful and efficient tool for stock market analysis and decision-making.

![architecture](./img/mlops_architecture(simplified).drawio.png)


## System in Production

**Important**: Before following this guide ensure you have been authorized to connect GCP project (contact the admin: `blajavier@gmail.com`).   

### Requirements

To install the app, you need to have Anaconda, Docker, and Docker-Compose installed on your system. You have a perfect installation guide in this [link](https://github.com/DataTalksClub/mlops-zoomcamp/tree/main/01-intro)

**NOTICE**, After installing Anaconda, Google Cloud SDK must be also installed in your system. Lear more [here](https://cloud.google.com/sdk/docs/install?hl=en)

If you are dealing issues with Anaconda environment you may find useful some [bits](troubleshooting_conda.md)

To download the Linux 64-bit archive file, at the command line, run:

```bash
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-cli-440.0.0-linux-x86_64.tar.gz

```

Once Google Cloud SDK had been installed, you account must be authenticate (*previously authorized by admin*). To do so, just write this in the command line:

```bash 
gcloud auth application-default login
```

### Clone the Repo

```bash
git clone https://github.com/jblanco89/MLOps_zoomcamp_course.git
```
Bash file `setup.sh` should get writing permission, run next commands:

```bash
cd ~/MLOps_zoomcamp_course
sudo chmod +x setup.sh
```
Finally, run bash file

```bash
sudo bash setup.sh
```

## User Guide

**1. Go to the main repo directory:**

```bash 
cd ~/MLOps_zoomcamp_course
```
**2. Install dependencies**

```bash
pip install -r requirements.txt
```
**3. Run main python file with `ticker` and `end date` arguments.** 

For example, supposing that we were interested to predict Microsoft Stock prices in the next 10 days from July 22nd, 2023. Thus, we just need to run this command:

```bash
python ./src/main.py MSFT 2023-07-22
```
This system use Yahoo Finance API to get stock data prices. Check the list of tickers supported:

[Yahoo Finance Ticker List](https://finance.yahoo.com/lookup/)

Some examples:
```bash
python src/main.py META 2023-07-25 #Meta
python src/main.py MMM 2023-07-25 #3M
python src/main.py ETH-USD 2023-07-25 #Ethereum
```

**4. Check the results in Dashboard**

After a couple od seconds data should have been processed by Google Cloud function. dashboard is available [here](https://lookerstudio.google.com/reporting/a0ed7bcd-5442-4945-b4d3-8d931c7ce11f)

Do not forget update data by clicking here:

![update_dashboard](./img/dashboard_data_update.png)


## Model Management (MLFlow)

### MLFlow Run

In Virtual Machine instance og Google Cloud Platform you should use a command like this:

```bash
mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri postgresql://postgres:passwod@sql_private_ip:5432/mlflow --default-artifact-root gs://storage_bucket_name
```
 
### Experiment Tracking
Experiment Tracking results after 152 runs simulations:

![Experiment_Tracking](./img/coordinate_plot.jpg)

MLFlow was deployed in GCP (scenario 5 according [documentation](https://mlflow.org/docs/latest/tracking.html#scenario-5-mlflow-tracking-server-enabled-with-proxied-artifact-storage-access)). 
You may check MLFlow experiment tracking and model registry clicking in this [Link](http://34.175.211.162:5000/).

### Model Registry

Once best model has been identified, we can register and tag it in MLFlow model registry system:

![Model Registry](./img/model_registry.png)

## Orchestration (Prefect 2.0)

### Prefect Cloud
Orchestration management has been set in prefect cloud plataform. For self-athentication run this command in your locall or virtual machine

```bash 
prefect cloud login
```

### Workflow Orchestration
After best model was chosen with MlFlow, we build workflow using Prefect 2.0. Here the result:

![prefect_workflow](./img/prefect_workflow_ultimate.jpg.png)


### Workflow Deployment

Run the agent by default
```bash
prefect agent start --pool agent --work-queue default
```
![Agent-prefect](./img/prefect_agent.png)

If deployed workflow has a custom run, agent is going to detect it and will run workflow automatically. In this system we have scheduled orchestration once a day. 

## Monitoring (Evidently AI)
Workflow generates a data drift `json` report *(DataDriftPreset)* using the Evidently library. 

```python
def generate_report(Y_pred, Y_test):
    report = Report(metrics=[
    DataDriftPreset()
    ])

    Y_pred_flattened = np.ravel(Y_pred)

    current = pd.DataFrame({'Close': Y_pred_flattened})
    # current = pd.DataFrame({'Close': Y_test})
    df_reference = pd.DataFrame({'Close': Y_test})
    df_reset = df_reference.reset_index(drop=True)
    current_reset = current.reset_index(drop=True)

    df_reset = pd.DataFrame(df_reset, columns=['Close'])
    current_reset = pd.DataFrame(current_reset, columns=['Close'])
    report.run(reference_data=df_reset, current_data=current_reset)
    return report.save_json("./reports/dataReport.json")

```
As a result, we are able to check drift score everyday in dashboard. 

![Monitor](./img/Dashboard_2.png)

## Dashboard

### Stock Prices prediction

Once stock prices have been uploaded to BQ table, technical indicators are calculated using sql:

```sql
WITH price_data AS (
  SELECT
    Symbol,
    index,
    Close,
    Close - LAG(Close) OVER (ORDER BY index) AS price_diff
  FROM `ambient-decoder-391319.stock_output.predicted_prices`
),
rsi_data AS (
  SELECT
  Symbol,
    index,
    Close,
    CASE WHEN price_diff > 0 THEN price_diff ELSE 0 END AS gain,
    CASE WHEN price_diff < 0 THEN ABS(price_diff) ELSE 0 END AS loss
  FROM price_data
),
histogram_data AS (
  SELECT
    Symbol,
    index,
    Close,
    gain,
    loss,
    NTILE(5) OVER (ORDER BY Close) AS bucket_number
  FROM rsi_data
)
SELECT
  Symbol,
  index,
  Close,
  CASE
    WHEN avg_gain IS NULL OR avg_loss IS NULL THEN NULL
    ELSE 100 - (100 / (1 + (NULLIF(avg_gain, 0) / NULLIF(avg_loss, 0))))
  END AS RSI_14_periods,
  -- MACD (5 periods)
  ema_12 - ema_26 AS MACD_5_periods,
  -- SMA (5 periods)
  AVG(Close) OVER (ORDER BY index ROWS BETWEEN 4 PRECEDING AND CURRENT ROW) AS SMA_5_periods,
  -- SMA (15 periods)
  AVG(Close) OVER (ORDER BY index ROWS BETWEEN 14 PRECEDING AND CURRENT ROW) AS SMA_15_periods,
  bucket_number,
  COUNT(*) OVER (PARTITION BY bucket_number) AS bucket_count
FROM (
  SELECT
  Symbol,
    index,
    Close,
    AVG(Close) OVER (ORDER BY index ROWS BETWEEN 11 PRECEDING AND CURRENT ROW) AS ema_12,
    AVG(Close) OVER (ORDER BY index ROWS BETWEEN 25 PRECEDING AND CURRENT ROW) AS ema_26,
    AVG(gain) OVER (ORDER BY index ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS avg_gain,
    AVG(loss) OVER (ORDER BY index ROWS BETWEEN 13 PRECEDING AND CURRENT ROW) AS avg_loss,
    bucket_number
  FROM histogram_data
)
ORDER BY index DESC;
```
Using Bigquery we get data table like this:

![Bigquery stock prices](./img/bigquery.png)

As a result, we'll be able to see the following dashboard:
![Dashboard](./img/Dashboard_1.png)

## Makefile and Dockefile

### Dockerfile
```docker
# This Dockerfile sets up a Python 3.8 environment, 
# installs necessary dependencies (libpq-dev), and 
# installs MLflow along with its required packages. 
# It then copies the local files into the container's working directory. 
# Finally, it runs the MLflow server with specific configurations, 
# using PostgreSQL as the backend store and Google Cloud Storage for 
# artifact storage.


# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

RUN apt-get update && apt-get install -y libpq-dev

WORKDIR /MLOps_zoomcamp_course

RUN python -m ensurepip --default-pip && pip install --no-cache-dir --upgrade pip

RUN pip install psycopg2-binary

RUN pip install mlflow

COPY . .

CMD ["mlflow", "server", "-h", "0.0.0.0", "-p", "5000", "--backend-store-uri", "postgresql://postgres:1234@10.28.192.5:5432/mlflow", "--default-artifact-root", "gs://lstm_model_test"]

```

### Makefile

```make
# This Makefile defines rules for building and running a Docker container 
# for an LSTM application. It includes commands to create a data directory, 
# set its permissions, build a Docker image, and run the Docker container. 
# The .PHONY target ensures these rules are always executed, 
# regardless of existing files with the same names as the targets.


DOCKER_IMAGE_NAME = lstm_app_image
CURRENT_DIR := $(shell pwd)
DATA_DIR = data

# Create the directory if it does not exist
create_data_directory:
	@if [ ! -d "$(DATA_DIR)" ]; then \
                mkdir -p "$(DATA_DIR)"; \
	fi

set_data_directory_permissions:
	chmod -R 777 "$(DATA_DIR)"

build_docker_image:
	docker build -t $(DOCKER_IMAGE_NAME) .

run_docker_container:
	docker run -p 5000:5000 $(DOCKER_IMAGE_NAME)

.PHONY: create_data_directory set_data_directory_permissions build_docker_image run_docker_container

```


## Recomendations

Alternativately, process may be much more robust with the following workload:

![architecture](./img/mlops_architecture.drawio.png)



## References

1. Alla, S., Adari, S.K. (2021). What Is MLOps?. In: Beginning MLOps with MLFlow. Apress, Berkeley, CA. https://doi.org/10.1007/978-1-4842-6549-9_3 

2. Bhandari, H. N., Rimal, B., Pokhrel, N. R., Rimal, R., Dahal, K. R., & Khatri, R. K. (2022). Predicting stock market index using LSTM. Machine Learning with Applications, 9, 100320. https://doi.org/10.1016/j.mlwa.2022.100320

3. Moghar, A., & Hamiche, M. (2020). Stock market prediction using LSTM recurrent neural network. Procedia Computer Science, 170, 1168-1173. https://doi.org/10.1016/j.procs.2020.03.049

4. Ghosh, A., Bose, S., Maji, G., Debnath, N., & Sen, S. (2019, September). Stock price prediction using LSTM on the Indian share market. In Proceedings of 32nd international conference on (Vol. 63, pp. 101-110). 	https://doi.org/10.29007/qgcz

5. Machine Learning to Predict Stock Prices. Utilizing a Keras LSTM model to forecast stock trends (2019). ARTIFICIAL INTELLIGENCE IN FINANCE. https://towardsdatascience.com/predicting-stock-prices-using-a-keras-lstm-model-4225457f0233

6. Stock Market Predictions with LSTM in Python (2020). Datacamp Tutorial.   https://www.datacamp.com/tutorial/lstm-python-stock-market

7. Run Prefect on Google Cloud Platform (2022).  https://medium.com/@mariusz_kujawski/run-prefect-on-google-cloud-platform-7cc9f801d454 

8. Running a serverless batch workload on GCP with Cloud Scheduler, Cloud Functions, and Compute Engine. https://medium.com/google-cloud/running-a-serverless-batch-workload-on-gcp-with-cloud-scheduler-cloud-functions-and-compute-86c2bd573f25 

## Annexes
 
### Neural Network Model

Our approach is mainly based on LSTM architecture. The model is initialized indicating a sequential stack of layers. In this case, a LSTM layer is added with 200 units (or neurons) ensuring that the output is returned for each timestep in the input sequence
A dropout layer with a dropout rate of 0.2 was added to prevent overfitting. Lastly, a dense layer with a single neuron is added, and the activation function `relu`

![LSTM Model](./img/LSTM_Model.png)

### Experiment Tracking Metrics results:

**- RMSE vs Epochs**

![Epochs](./img/epochs.png)

**- RMSE vs Hidden Units**

![Hidden Units](./img/hidden_units.png)

**- RMSE vs Learning Rate**

![Learning Rate](./img/learning_rate.png)

**- Hidden Units vs Learning Rate**

![Hidden vs Learning Rate](./img/hidden_units_vs_learning_rate.png)

### Cloud Function
Here how Cloud Function Looks like in GCP:

![Cloud Function](./img/cloud_function_on_demand.png)
