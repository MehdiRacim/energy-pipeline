import httpx
import pandas as pd
from pathlib import Path
from loguru import logger


def fetch_consumption(
    start_date: str = "2023-01-01",
    end_date: str = "2026-03-17"
) -> pd.DataFrame:
    """
    Récupère la consommation électrique française depuis l'API RTE (éCO2mix).
    Données disponibles par tranche de 30 minutes.
    """
    logger.info(f"Récupération conso électrique du {start_date} au {end_date}...")

    url = "https://odre.opendatasoft.com/api/explore/v2.1/catalog/datasets/eco2mix-national-cons-def/exports/json"
    params = {
        "where": f"date_heure >= '{start_date}' AND date_heure <= '{end_date}'",
        "limit": 50000,
        "select": "date_heure,consommation,prevision_j1,prevision_j"
    }

    try:
        logger.info("Appel API RTE (peut prendre 30-60 secondes)...")
        response = httpx.get(url, params=params, timeout=120)
        response.raise_for_status()
    except httpx.TimeoutException:
        logger.error("Timeout : l'API RTE est lente, réessaie dans quelques minutes")
        raise
    except httpx.HTTPStatusError as e:
        logger.error(f"Erreur HTTP {e.response.status_code}")
        raise

    records = response.json()
    if not records:
        logger.warning("Aucune donnée reçue")
        return pd.DataFrame()

    df = pd.DataFrame(records)
    df["date_heure"] = pd.to_datetime(df["date_heure"])
    df = df.sort_values("date_heure").reset_index(drop=True)

    df = df.dropna(subset=["consommation"])
    df["consommation"] = pd.to_numeric(df["consommation"], errors="coerce")

    logger.success(f"{len(df)} lignes récupérées")
    return df


def save_consumption(df: pd.DataFrame) -> Path:
    """Sauvegarde le DataFrame en Parquet dans data/raw/"""
    output_dir = Path("data/raw")
    output_dir.mkdir(parents=True, exist_ok=True)

    output_path = output_dir / "consumption.parquet"
    df.to_parquet(output_path, index=False)

    logger.success(f"Fichier sauvegardé : {output_path}")
    return output_path


if __name__ == "__main__":
    df = fetch_consumption()
    save_consumption(df)
    print(df.head())
    print(f"\nColonnes : {list(df.columns)}")
    print(f"Période : {df['date_heure'].min()} → {df['date_heure'].max()}")
    print(f"Conso min : {df['consommation'].min()} MW")
    print(f"Conso max : {df['consommation'].max()} MW")