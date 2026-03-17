import httpx
import pandas as pd
from pathlib import Path
from datetime import date
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

def fetch_weather(
    lat: float = 48.85,
    lon: float = 2.35,
    start_date: str = "2023-01-01",
    end_date: str = None
) -> pd.DataFrame:
    """
    Récupère les données météo horaires depuis l'API Open-Meteo.
    Par défaut : Paris, depuis le 1er janvier 2023.
    """
    if end_date is None:
        end_date = str(date.today())

    logger.info(f"Récupération météo du {start_date} au {end_date}...")

    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": lat,
        "longitude": lon,
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,windspeed_10m,cloudcover,shortwave_radiation",
        "timezone": "Europe/Paris"
    }

    try:
        response = httpx.get(url, params=params, timeout=30)
        response.raise_for_status()
    except httpx.TimeoutException:
        logger.error("Timeout : l'API met trop de temps à répondre")
        raise
    except httpx.HTTPStatusError as e:
        logger.error(f"Erreur HTTP {e.response.status_code}")
        raise

    data = response.json()["hourly"]
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])

    logger.success(f"{len(df)} lignes récupérées")
    return df


def save_weather(df: pd.DataFrame) -> Path:
    """Sauvegarde le DataFrame en Parquet dans data/raw/"""
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "weather.parquet"
    df.to_parquet(output_path, index=False)

    logger.success(f"Fichier sauvegardé : {output_path}")
    return output_path


if __name__ == "__main__":
    df = fetch_weather()
    save_weather(df)
    print(df.head())
    print(f"\nColonnes : {list(df.columns)}")
    print(f"Période : {df['time'].min()} → {df['time'].max()}")