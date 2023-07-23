# MLOps_zoomcamp_course
The MLOps ZoomCamp course provides course material, subjects, notes, and results. This repository is part of my learning path for Data Engineering career


## Architecture

![architecture](./img/mlops_architecture.drawio.png)

## Requirements
Anaconda
Docker
Docker-Compose

Additionaly:

```bash 
mlflow
jupyter
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
protobuf==3.20.*

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

You may check MlFlow experiment tracking and model registry here:


[MLFlow](http://34.175.211.162:5000/).


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

![prefect_workflow](./img/prefect_workflow_2.jpg)

## SSH Connection

```bash

ssh -i mlops-instance01-key-vscode mlops-instance01-local@34.175.211.162

```