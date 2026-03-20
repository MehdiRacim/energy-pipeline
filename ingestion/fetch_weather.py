import httpx
import pandas as pd
from pathlib import Path
from datetime import date
from loguru import logger
from dotenv import load_dotenv

load_dotenv()

# 5 villes représentatives avec leur poids démographique
VILLES = [
    {"nom": "Paris",      "lat": 48.85, "lon": 2.35,  "poids": 0.35},
    {"nom": "Lyon",       "lat": 45.75, "lon": 4.85,  "poids": 0.15},
    {"nom": "Marseille",  "lat": 43.30, "lon": 5.37,  "poids": 0.12},
    {"nom": "Toulouse",   "lat": 43.60, "lon": 1.44,  "poids": 0.10},
    {"nom": "Lille",      "lat": 50.63, "lon": 3.07,  "poids": 0.08},
]


def fetch_city_weather(ville: dict, start_date: str, end_date: str) -> pd.DataFrame:
    """Récupère la météo horaire d'une ville."""
    url = "https://archive-api.open-meteo.com/v1/archive"
    params = {
        "latitude": ville["lat"],
        "longitude": ville["lon"],
        "start_date": start_date,
        "end_date": end_date,
        "hourly": "temperature_2m,windspeed_10m,cloudcover,shortwave_radiation",
        "timezone": "Europe/Paris"
    }
    response = httpx.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()["hourly"]
    df = pd.DataFrame(data)
    df["time"] = pd.to_datetime(df["time"])
    df["ville"] = ville["nom"]
    df["poids"] = ville["poids"]
    return df


def fetch_weather(
    start_date: str = "2023-01-01",
    end_date: str = None
) -> pd.DataFrame:
    """
    Récupère la météo de 5 villes françaises et calcule
    la moyenne pondérée par population.
    """
    if end_date is None:
        end_date = str(date.today())

    logger.info(f"Récupération météo multi-villes du {start_date} au {end_date}...")

    dfs = []
    for ville in VILLES:
        logger.info(f"  → {ville['nom']} (poids: {ville['poids']})")
        df = fetch_city_weather(ville, start_date, end_date)
        dfs.append(df)

    # Combine toutes les villes
    all_df = pd.concat(dfs, ignore_index=True)

    # Moyenne pondérée par heure
    weighted = all_df.groupby("time").apply(
        lambda g: pd.Series({
            "temperature_2m": (g["temperature_2m"] * g["poids"]).sum() / g["poids"].sum(),
            "windspeed_10m":  (g["windspeed_10m"]  * g["poids"]).sum() / g["poids"].sum(),
            "cloudcover":     (g["cloudcover"]      * g["poids"]).sum() / g["poids"].sum(),
            "shortwave_radiation": (g["shortwave_radiation"] * g["poids"]).sum() / g["poids"].sum(),
        })
    ).reset_index()

    logger.success(f"{len(weighted)} lignes récupérées (moyenne pondérée 5 villes)")
    return weighted


def save_weather(df: pd.DataFrame) -> Path:
    """Sauvegarde le DataFrame en Parquet."""
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "weather.parquet"
    df.to_parquet(
        output_path,
        index=False,
        coerce_timestamps="us",
        allow_truncated_timestamps=True
    )
    logger.success(f"Fichier sauvegardé : {output_path}")
    return output_path


if __name__ == "__main__":
    df = fetch_weather()
    save_weather(df)
    print(df.head())
    print(f"\nColonnes : {list(df.columns)}")
    print(f"Période : {df['time'].min()} → {df['time'].max()}")