from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime, timedelta
import sys

sys.path.insert(0, '/opt/airflow')

from ingestion.fetch_weather import fetch_weather, save_weather
from ingestion.fetch_rte import fetch_consumption, save_consumption
from storage.upload_to_minio import upload_file

default_args = {
    "owner": "mehdi",
    "retries": 2,
    "retry_delay": timedelta(minutes=5),
    "email_on_failure": False,
}

def task_fetch_weather():
    df = fetch_weather()
    save_weather(df)

def task_fetch_consumption():
    df = fetch_consumption()
    save_consumption(df)

def task_upload_to_minio():
    upload_file("data/raw/weather.parquet", "raw", "weather/weather.parquet")
    upload_file("data/raw/consumption.parquet", "raw", "consumption/consumption.parquet")

with DAG(
    dag_id="energy_pipeline",
    description="Pipeline météo + consommation électrique",
    schedule="0 6 * * *",
    start_date=datetime(2024, 1, 1),
    catchup=False,
    default_args=default_args,
    tags=["energy", "meteo"]
) as dag:

    fetch_weather_task = PythonOperator(
        task_id="fetch_weather",
        python_callable=task_fetch_weather,
    )

    fetch_consumption_task = PythonOperator(
        task_id="fetch_consumption",
        python_callable=task_fetch_consumption,
    )

    upload_task = PythonOperator(
        task_id="upload_to_minio",
        python_callable=task_upload_to_minio,
    )

    [fetch_weather_task, fetch_consumption_task] >> upload_task