# Stock Prices Prediction based on Deep Learning model (MLOps Zoomcamp Project)

## Project Description
This project aims to develop and deploy a stock price prediction system using an LSTM model, which predicts stock prices with a 10-day lag based on historical OHLC prices and technical indicators such as MACD, RSI, SMA and EMA. By providing accurate predictions, the system empowers traders to make well-informed decisions, optimize investment strategies, and minimize risks while capitalizing on market fluctuations.

For this project cloud-based approach was chosen. By utilizing Google Cloud Platform (GCP) and MLflow integration, efficiency and scalability of the system is significatibly enhanced. Deploying the LSTM model on GCP also enables seamless batch processing of large datasets, ensuring timely predictions. Additionally, MLflow's model tracking and versioning features facilitate efficient monitoring and maintenance of the model's performance over time. Furthermore, the Prefect framework orchestrates the workflow, automating data processing, scheduling, and error handling, resulting in a streamlined and reliable end-to-end solution.

To sum, this stock price prediction system addresses the specific needs of traders and investors by providing accurate predictions, timely insights, and efficient model management through cloud technology. With this solution, traders could stay ahead in the fast-paced financial markets, make informed decisions, and enhance their overall financial outcomes.


## Solution Architecture

**LSTM model** was deployed as Batch Workload using some GCP tools such as:

1. Cloud Storage
2. Cloud Scheduler/Trigger
3. Cloud Functions
4. BigQuery
5. Google Looker Studio (previously known as Data Studio)

As we can see, the workflow solution implemented in Google Cloud Platform (GCP) leverages serverless batch processing using Cloud Storage, Cloud Functions, BigQuery, and Google Looker Studio to predict stock prices based on historical OHLC (Open, High, Low, Close) data.

The process begins when users upload a CSV file containing OHLC price data for their desired stock to **Google Cloud Storage**. As soon as the file is uploaded, the **Cloud Function** is triggered to execute the LSTM (Long Short-Term Memory) model, which is designed to forecast the stock's prices for the next 10 days. The LSTM model's predictive capabilities enable traders and investors to make informed decisions and optimize their investment strategies based on reliable forecasts.

The results of the LSTM model are then stored in a dedicated **BigQuery** table named `predicted_prices`. The data in the table feeds into a **Looker Studio** Dashboard, where users can visualize and explore the forecasted stock prices in a user-friendly and intuitive manner.

The integration of Cloud Functions, BigQuery, and Looker Studio streamlines the entire process, making it a powerful and efficient tool for stock market analysis and decision-making.


![architecture](./img/mlops_architecture(simplified).drawio.png)


Alternativately, process may be much more robust with the following workload:

![architecture](./img/mlops_architecture.drawio.png)

But it was no applied in this project


## Requirements

To install the app, you need to have Anaconda, Docker, and Docker-Compose installed on your system. You have a perfect installation guide in this [link ](https://github.com/DataTalksClub/mlops-zoomcamp/tree/main/01-intro)


**Important**: Linux (Ubuntu 22.04 LTS) is the recommended development environment. 

Additionaly, we'll need:

```bash 
    mlflow
    scikit-learn
    pandas
    pandas_datareader
    pandas_ta
    mplfinance
    tensorflow==2.11.0
    yfinance
    seaborn
    hyperopt
    fastparquet
    prefect
    evidently
    google-cloud-bigquery
    google-cloud-storage
    gcsfs
    pandas-gbq
```

## Docker Image

```bash

 To Do

```

## Installation

```bash
    pip install -r requirements.txt

```

## MlFlow Tool

### MLflow running

Locally use this commnad:

```bash 
    mlflow ui --backend-store-uri sqlite:///mlflow.db 
```
from Virtual Machine instance in Google Cloud Platform you should use:

```bash
    mlflow server -h 0.0.0.0 -p 5000 --backend-store-uri postgresql://postgres:passwod@sql_private_ip:5432/mlflow --default-artifact-root gs://storage_bucket_name
```
 
### Experiment Tracking
Experiment Tracking results afrer 152 runs simulations

![Experiment_Tracking](./img/coordinate_plot.jpg)

You may check MlFlow experiment tracking and model registry here: [MLFlow](http://34.175.211.162:5000/).


## Prefect

### Prefect Cloud

For self-athentication run this command in your locall or virtual machine

```bash 
    prefect cloud login
```

### Prefect Agent

```bash
    prefect agent start --pool agent --work-queue default

```
![Agent-prefect](./img/prefect_agent.png)


### Workflow Orchestration

![prefect_workflow](./img/prefect_workflow_ultimate.jpg.png)

## SSH Connection

```bash

ssh -i mlops-instance01-key-vscode mlops-instance01-local@34.175.211.162

```
## Results

### Dashboard

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

As a result, we'll be able to see the following dashboard:
![Dashboard](./img/Dashboard_1.png)


### Model Monitoring

![Monitor](./img/Dashboard_2.png)

## References

1. Alla, S., Adari, S.K. (2021). What Is MLOps?. In: Beginning MLOps with MLFlow. Apress, Berkeley, CA. https://doi.org/10.1007/978-1-4842-6549-9_3 

2. Bhandari, H. N., Rimal, B., Pokhrel, N. R., Rimal, R., Dahal, K. R., & Khatri, R. K. (2022). Predicting stock market index using LSTM. Machine Learning with Applications, 9, 100320. https://doi.org/10.1016/j.mlwa.2022.100320

3. Moghar, A., & Hamiche, M. (2020). Stock market prediction using LSTM recurrent neural network. Procedia Computer Science, 170, 1168-1173. https://doi.org/10.1016/j.procs.2020.03.049

4. Ghosh, A., Bose, S., Maji, G., Debnath, N., & Sen, S. (2019, September). Stock price prediction using LSTM on the Indian share market. In Proceedings of 32nd international conference on (Vol. 63, pp. 101-110). 	https://doi.org/10.29007/qgcz

5. Machine Learning to Predict Stock Prices. Utilizing a Keras LSTM model to forecast stock trends (2019). ARTIFICIAL INTELLIGENCE IN FINANCE. https://towardsdatascience.com/predicting-stock-prices-using-a-keras-lstm-model-4225457f0233

6. Stock Market Predictions with LSTM in Python (2020). Datacamp Tutorial.   https://www.datacamp.com/tutorial/lstm-python-stock-market

7. Run Prefect on Google Cloud Platform (2022).  https://medium.com/@mariusz_kujawski/run-prefect-on-google-cloud-platform-7cc9f801d454 

8. Running a serverless batch workload on GCP with Cloud Scheduler, Cloud Functions, and Compute Engine. https://medium.com/google-cloud/running-a-serverless-batch-workload-on-gcp-with-cloud-scheduler-cloud-functions-and-compute-86c2bd573f25 