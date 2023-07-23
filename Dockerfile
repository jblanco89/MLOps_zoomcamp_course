# syntax=docker/dockerfile:1

FROM python:3.8-slim-buster

RUN apt-get update && apt-get install -y libpq-dev

WORKDIR /MLOps_zoomcamp_course

RUN python -m ensurepip --default-pip && pip install --no-cache-dir --upgrade pip

RUN pip install psycopg2-binary

RUN pip install mlflow

COPY . .

CMD ["mlflow", "server", "-h", "0.0.0.0", "-p", "5000", "--backend-store-uri", "postgresql://postgres:1234@10.28.192.5:5432/mlflow", "--default-artifact-root", "gs://lstm_model_test"]


