import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
import boto3
from botocore.client import Config
from loguru import logger
from dotenv import load_dotenv
import io
import os

load_dotenv()

def get_minio_client():
    return boto3.client(
        "s3",
        endpoint_url=os.getenv("MINIO_ENDPOINT", "http://localhost:9000"),
        aws_access_key_id=os.getenv("MINIO_ROOT_USER", "minioadmin"),
        aws_secret_access_key=os.getenv("MINIO_ROOT_PASSWORD", "minioadmin123"),
        config=Config(signature_version="s3v4"),
        region_name="us-east-1"
    )

def read_parquet_from_minio(bucket: str, prefix: str) -> pd.DataFrame:
    client = get_minio_client()
    response = client.list_objects_v2(Bucket=bucket, Prefix=prefix)
    files = [
        obj["Key"] for obj in response.get("Contents", [])
        if obj["Key"].endswith(".parquet")
    ]
    logger.info(f"Fichiers trouvés : {files}")
    dfs = []
    for key in files:
        obj = client.get_object(Bucket=bucket, Key=key)
        df = pd.read_parquet(io.BytesIO(obj["Body"].read()))
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    logger.success(f"{len(combined)} lignes lues depuis MinIO")
    return combined

def load_features_to_postgres():
    logger.info("Lecture des features depuis MinIO...")
    df = read_parquet_from_minio("processed", "features/features.parquet/")

    # Convertit les colonnes timestamp en string pour psycopg2
    for col in df.select_dtypes(include=["datetime64", "datetimetz"]).columns:
        df[col] = df[col].astype(str)

    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", 5432),
        dbname=os.getenv("POSTGRES_DB", "energy_db"),
        user=os.getenv("POSTGRES_USER", "energy_user"),
        password=os.getenv("POSTGRES_PASSWORD", "energy_pass")
    )
    cur = conn.cursor()

    # Crée la table
    cols = df.columns.tolist()
    col_defs = []
    for col in cols:
        dtype = str(df[col].dtype)
        if "int" in dtype:
            pg_type = "BIGINT"
        elif "float" in dtype:
            pg_type = "FLOAT"
        else:
            pg_type = "TEXT"
        col_defs.append(f'"{col}" {pg_type}')

    cur.execute("DROP TABLE IF EXISTS raw_features CASCADE")
    cur.execute(f"CREATE TABLE raw_features ({', '.join(col_defs)})")
    logger.info("Table raw_features créée")

    # Insère les données
    rows = [tuple(row) for row in df.itertuples(index=False)]
    insert_sql = f"""
        INSERT INTO raw_features ({', '.join(f'"{c}"' for c in cols)})
        VALUES %s
    """
    execute_values(cur, insert_sql, rows, page_size=500)
    conn.commit()

    cur.execute("SELECT COUNT(*) FROM raw_features")
    count = cur.fetchone()[0]
    logger.success(f"Table 'raw_features' créée : {count} lignes dans PostgreSQL")

    cur.close()
    conn.close()

if __name__ == "__main__":
    load_features_to_postgres()