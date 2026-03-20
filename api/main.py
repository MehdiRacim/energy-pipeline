from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import joblib
import numpy as np
import json
import psycopg2
import os
import sys
from pathlib import Path
from loguru import logger
from datetime import date, timedelta
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))
from ingestion.fetch_forecast import fetch_tomorrow_forecast

app = FastAPI(
    title="Energy Prediction API",
    description="Prédit la consommation électrique française J+1",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

MODEL_PATH = Path("ml/models/energy_predictor.pkl")
METRICS_PATH = Path("ml/models/metrics.json")

model = None
metrics = None


@app.on_event("startup")
def load_model():
    """Charge le modèle au démarrage de l'API."""
    global model, metrics
    if not MODEL_PATH.exists():
        logger.error(f"Modèle introuvable : {MODEL_PATH}")
        return
    model = joblib.load(MODEL_PATH)
    with open(METRICS_PATH) as f:
        metrics = json.load(f)
    logger.success(f"Modèle chargé — MAE: {metrics['mae']:,.0f} MWh")


class PredictionRequest(BaseModel):
    temp_moy: float = Field(..., description="Température moyenne prévue (°C)", example=8.5)
    temp_min: float = Field(..., description="Température minimale prévue (°C)", example=3.0)
    temp_max: float = Field(..., description="Température maximale prévue (°C)", example=14.0)
    vent_moy: float = Field(..., description="Vitesse moyenne du vent (km/h)", example=15.0)
    nuages_moy: float = Field(..., description="Couverture nuageuse moyenne (%)", example=60.0)
    est_weekend: int = Field(..., description="1 si weekend, 0 sinon", example=0)
    mois: int = Field(..., description="Mois (1-12)", example=3)
    conso_semaine_precedente: float = Field(..., description="Conso totale il y a 7 jours (MWh)", example=950000.0)


class PredictionResponse(BaseModel):
    prediction_mwh: float
    prediction_gwh: float
    date_prediction: str
    modele_mae_mwh: float
    message: str


@app.get("/")
def root():
    return {
        "service": "Energy Prediction API",
        "version": "1.0.0",
        "status": "ok",
        "endpoints": ["/predict", "/predict/tomorrow", "/health", "/docs"]
    }


@app.get("/health")
def health():
    """Vérifie que l'API et le modèle sont opérationnels."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")
    return {
        "status": "ok",
        "model_loaded": True,
        "model_mae_mwh": round(metrics["mae"], 0),
        "model_mape_pct": round(metrics["mape"], 1)
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    """Prédit la consommation électrique pour demain."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    features = np.array([[
        request.temp_moy,
        request.temp_min,
        request.temp_max,
        request.vent_moy,
        request.nuages_moy,
        request.est_weekend,
        request.mois,
        request.conso_semaine_precedente
    ]])

    prediction = float(model.predict(features)[0])
    tomorrow = str(date.today() + timedelta(days=1))

    logger.info(f"Prédiction pour {tomorrow} : {prediction:,.0f} MWh")

    return PredictionResponse(
        prediction_mwh=round(prediction, 0),
        prediction_gwh=round(prediction / 1000, 1),
        date_prediction=tomorrow,
        modele_mae_mwh=round(metrics["mae"], 0),
        message=f"Consommation prédite pour le {tomorrow}"
    )


@app.get("/predict")
def predict_get(
    temp_moy: float = 10.0,
    temp_min: float = 5.0,
    temp_max: float = 15.0,
    vent_moy: float = 20.0,
    nuages_moy: float = 50.0,
    est_weekend: int = 0,
    mois: int = 3,
    conso_semaine_precedente: float = 950000.0
):
    """Version GET pour tester facilement depuis le navigateur."""
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    features = np.array([[
        temp_moy, temp_min, temp_max,
        vent_moy, nuages_moy, est_weekend,
        mois, conso_semaine_precedente
    ]])

    prediction = float(model.predict(features)[0])
    tomorrow = str(date.today() + timedelta(days=1))

    return {
        "prediction_mwh": round(prediction, 0),
        "prediction_gwh": round(prediction / 1000, 1),
        "date_prediction": tomorrow,
        "modele_mae_mwh": round(metrics["mae"], 0),
        "parametres": {
            "temp_moy": temp_moy,
            "est_weekend": est_weekend,
            "mois": mois
        }
    }


@app.get("/predict/tomorrow")
def predict_tomorrow():
    if model is None:
        raise HTTPException(status_code=503, detail="Modèle non chargé")

    try:
        forecast = fetch_tomorrow_forecast()
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Erreur météo : {str(e)}")

    try:
        conn = psycopg2.connect(
            host=os.getenv("POSTGRES_HOST", "localhost"),
            port=os.getenv("POSTGRES_PORT", 5432),
            dbname=os.getenv("POSTGRES_DB", "energy_db"),
            user=os.getenv("POSTGRES_USER", "energy_user"),
            password=os.getenv("POSTGRES_PASSWORD", "energy_pass")
        )
        cur = conn.cursor()

        # Conso J-7
        cur.execute("""
            SELECT conso_totale_mwh FROM daily_energy_weather
            ORDER BY jour DESC OFFSET 6 LIMIT 1
        """)
        row = cur.fetchone()
        conso_j7 = float(row[0]) if row else 950000.0

        # Conso J-1
        cur.execute("""
            SELECT conso_totale_mwh FROM daily_energy_weather
            ORDER BY jour DESC LIMIT 1
        """)
        row = cur.fetchone()
        conso_j1 = float(row[0]) if row else 950000.0

        # Conso J-2
        cur.execute("""
            SELECT conso_totale_mwh FROM daily_energy_weather
            ORDER BY jour DESC OFFSET 1 LIMIT 1
        """)
        row = cur.fetchone()
        conso_j2 = float(row[0]) if row else 950000.0

        # Conso J-14
        cur.execute("""
            SELECT conso_totale_mwh FROM daily_energy_weather
            ORDER BY jour DESC OFFSET 13 LIMIT 1
        """)
        row = cur.fetchone()
        conso_j14 = float(row[0]) if row else 950000.0

        conn.close()
    except Exception:
        conso_j7 = conso_j1 = conso_j2 = conso_j14 = 950000.0

    # Température ressentie
    temp_ressentie = forecast["temp_moy"] - (forecast["vent_moy"] * 0.3)

    # DataFrame avec les noms de colonnes — Ridge en a besoin
    import pandas as pd
    features = pd.DataFrame([{
        "temp_moy":                 forecast["temp_moy"],
        "temp_min":                 forecast["temp_min"],
        "temp_max":                 forecast["temp_max"],
        "vent_moy":                 forecast["vent_moy"],
        "nuages_moy":               forecast["nuages_moy"],
        "est_weekend":              forecast["est_weekend"],
        "est_ferie":                0,
        "mois":                     forecast["mois"],
        "conso_semaine_precedente": conso_j7,
        "temp_ressentie":           temp_ressentie,
        "conso_j1":                 conso_j1,
        "conso_j2":                 conso_j2,
        "conso_j14":                conso_j14,
    }])

    prediction = float(model.predict(features)[0])

    return {
        "date_prediction": forecast["date"],
        "prediction_mwh": round(prediction, 0),
        "prediction_gwh": round(prediction / 1000, 1),
        "modele_mae_mwh": round(metrics["mae"], 0),
        "modele_nom": metrics.get("model_name", "Ridge"),
        "meteo_prevue": {
            "temperature_moy": forecast["temp_moy"],
            "temperature_min": forecast["temp_min"],
            "temperature_max": forecast["temp_max"],
            "vent_moy": forecast["vent_moy"],
            "nuages_moy": forecast["nuages_moy"],
            "est_weekend": forecast["est_weekend"]
        },
        "conso_semaine_precedente_mwh": round(conso_j7, 0),
        "message": f"Prédiction automatique pour le {forecast['date']}"
    }