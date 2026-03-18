import httpx
from datetime import date, timedelta
from loguru import logger


def fetch_tomorrow_forecast(lat: float = 48.85, lon: float = 2.35) -> dict:
    """
    Récupère les prévisions météo de demain depuis Open-Meteo.
    Retourne les moyennes journalières dont le modèle a besoin.
    """
    tomorrow = str(date.today() + timedelta(days=1))
    logger.info(f"Récupération des prévisions météo pour {tomorrow}...")

    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "hourly": "temperature_2m,windspeed_10m,cloudcover",
        "start_date": tomorrow,
        "end_date": tomorrow,
        "timezone": "Europe/Paris"
    }

    response = httpx.get(url, params=params, timeout=30)
    response.raise_for_status()
    data = response.json()["hourly"]

    temps = data["temperature_2m"]
    winds = data["windspeed_10m"]
    clouds = data["cloudcover"]

    # Calcule les moyennes journalières
    result = {
        "date": tomorrow,
        "temp_moy": round(sum(temps) / len(temps), 1),
        "temp_min": round(min(temps), 1),
        "temp_max": round(max(temps), 1),
        "vent_moy": round(sum(winds) / len(winds), 1),
        "nuages_moy": round(sum(clouds) / len(clouds), 1),
        "est_weekend": 1 if date.fromisoformat(tomorrow).weekday() >= 5 else 0,
        "mois": date.fromisoformat(tomorrow).month
    }

    logger.success(f"Prévisions récupérées : {result['temp_moy']}°C, vent {result['vent_moy']} km/h")
    return result


if __name__ == "__main__":
    forecast = fetch_tomorrow_forecast()
    print(forecast)