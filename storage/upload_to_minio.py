import boto3
from botocore.client import Config
from botocore.exceptions import ClientError
from pathlib import Path
from loguru import logger
from dotenv import load_dotenv
import os

load_dotenv()

def get_minio_client():
    """Crée et retourne un client MinIO."""
    endpoint = os.getenv("MINIO_ENDPOINT", "http://localhost:9000")
    client = boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.getenv("MINIO_ROOT_USER", "minioadmin"),
        aws_secret_access_key=os.getenv("MINIO_ROOT_PASSWORD", "minioadmin123"),
        config=Config(signature_version="s3v4"),
        region_name="us-east-1"
    )
    return client


def upload_file(local_path: str, bucket: str, key: str) -> bool:
    """
    Upload un fichier local vers MinIO.
    
    local_path : chemin du fichier sur ta machine
    bucket     : nom du bucket MinIO (ex: 'raw')
    key        : nom du fichier dans MinIO (ex: 'weather/weather.parquet')
    """
    client = get_minio_client()
    local_path = Path(local_path)

    if not local_path.exists():
        logger.error(f"Fichier introuvable : {local_path}")
        return False

    try:
        client.upload_file(str(local_path), bucket, key)
        logger.success(f"Upload réussi : {local_path} → s3://{bucket}/{key}")
        return True
    except ClientError as e:
        logger.error(f"Erreur upload : {e}")
        return False


def list_files(bucket: str) -> list:
    """Liste tous les fichiers dans un bucket MinIO."""
    client = get_minio_client()
    try:
        response = client.list_objects_v2(Bucket=bucket)
        files = [obj["Key"] for obj in response.get("Contents", [])]
        return files
    except ClientError as e:
        logger.error(f"Erreur listing : {e}")
        return []


if __name__ == "__main__":
    logger.info("Upload des fichiers Parquet vers MinIO...")

    upload_file("data/raw/weather.parquet", "raw", "weather/weather.parquet")
    upload_file("data/raw/consumption.parquet", "raw", "consumption/consumption.parquet")

    logger.info("Fichiers dans le bucket 'raw' :")
    for f in list_files("raw"):
        print(f"  - {f}")

def download_file(bucket: str, key: str, local_path: str) -> bool:
    """Télécharge un fichier depuis MinIO vers le disque local."""
    client = get_minio_client()
    Path(local_path).parent.mkdir(parents=True, exist_ok=True)
    try:
        client.download_file(bucket, key, local_path)
        logger.success(f"Download réussi : s3://{bucket}/{key} → {local_path}")
        return True
    except ClientError as e:
        logger.error(f"Erreur download : {e}")
        return False