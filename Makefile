.PHONY: start api stop

start:
	docker compose up -d
	sleep 5
	python ingestion/fetch_weather.py
	python ingestion/fetch_rte.py
	python storage/upload_to_minio.py
	python transform/spark_transform.py
	python transform/load_to_postgres.py
	cd dbt && dbt run && cd ..

api:
	uvicorn api.main:app --reload --port 8000

stop:
	docker compose down