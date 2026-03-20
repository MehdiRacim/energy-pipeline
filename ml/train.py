import pandas as pd
import numpy as np
import psycopg2
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from sklearn.preprocessing import StandardScaler
from loguru import logger
from pathlib import Path
import os
from dotenv import load_dotenv

load_dotenv()

FEATURES = [
    "temp_moy",
    "temp_min",
    "temp_max",
    "vent_moy",
    "nuages_moy",
    "est_weekend",
    "est_ferie",
    "mois",
    "conso_semaine_precedente"
]

TARGET = "conso_totale_mwh"


def load_data() -> pd.DataFrame:
    """Charge les données depuis PostgreSQL."""
    conn = psycopg2.connect(
        host=os.getenv("POSTGRES_HOST", "localhost"),
        port=os.getenv("POSTGRES_PORT", 5432),
        dbname=os.getenv("POSTGRES_DB", "energy_db"),
        user=os.getenv("POSTGRES_USER", "energy_user"),
        password=os.getenv("POSTGRES_PASSWORD", "energy_pass")
    )

    df = pd.read_sql(
        "SELECT * FROM daily_energy_weather ORDER BY jour",
        conn
    )
    conn.close()

    logger.success(f"{len(df)} jours chargés depuis PostgreSQL")
    return df


def prepare_data(df: pd.DataFrame):
    """Prépare les features et la cible."""
    df = df.dropna(subset=FEATURES + [TARGET])

    X = df[FEATURES]
    y = df[TARGET]

    logger.info(f"Features : {FEATURES}")
    logger.info(f"Cible : {TARGET}")
    logger.info(f"Données : {len(X)} lignes après nettoyage")
    return X, y, df


def train_with_cross_validation(X, y):
    """
    Recherche des meilleurs hyperparamètres avec GridSearchCV
    puis validation croisée temporelle.
    """
    from sklearn.model_selection import GridSearchCV

    tscv = TimeSeriesSplit(n_splits=5)

    param_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [3, 4, 5],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
    }

    logger.info("Recherche des meilleurs hyperparamètres (GridSearchCV)...")
    logger.info(f"Nombre de combinaisons : {3*3*3*2} × 5 folds = {3*3*3*2*5} fits")

    base_model = GradientBoostingRegressor(random_state=42)

    grid_search = GridSearchCV(
        base_model,
        param_grid,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1
    )

    grid_search.fit(X, y)

    best_params = grid_search.best_params_
    best_mae = -grid_search.best_score_

    logger.success(f"Meilleurs paramètres : {best_params}")
    logger.success(f"Meilleur MAE (CV) : {best_mae:,.0f} MWh")

    # Validation fold par fold avec les meilleurs paramètres
    maes = []
    mapes = []

    logger.info("Validation croisée avec les meilleurs paramètres...")
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]

        model = GradientBoostingRegressor(**best_params, random_state=42)
        model.fit(X_train, y_train)
        preds = model.predict(X_val)

        mae = mean_absolute_error(y_val, preds)
        mape = mean_absolute_percentage_error(y_val, preds) * 100
        maes.append(mae)
        mapes.append(mape)

        logger.info(f"Fold {fold+1} — MAE: {mae:,.0f} MWh | MAPE: {mape:.1f}%")

    logger.success(f"MAE moyen : {np.mean(maes):,.0f} MWh")
    logger.success(f"MAPE moyen : {np.mean(mapes):.1f}%")

    return np.mean(maes), np.mean(mapes), best_params


def train_final_model(X, y, best_params: dict):
    """Entraîne le modèle final avec les meilleurs paramètres."""
    logger.info(f"Entraînement final avec : {best_params}")

    model = GradientBoostingRegressor(**best_params, random_state=42)
    model.fit(X, y)
    logger.success("Modèle entraîné !")
    return model


def plot_feature_importance(model, feature_names: list):
    """Affiche et sauvegarde l'importance des features."""
    importance = pd.Series(
        model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=True)

    plt.figure(figsize=(8, 5))
    importance.plot(kind="barh", color="steelblue")
    plt.title("Importance des features")
    plt.xlabel("Importance relative")
    plt.tight_layout()

    output_path = Path("ml/feature_importance.png")
    plt.savefig(output_path)
    logger.success(f"Graphique sauvegardé : {output_path}")
    plt.close()


def save_model(model, mae: float, mape: float):
    """Sauvegarde le modèle et ses métriques."""
    Path("ml/models").mkdir(parents=True, exist_ok=True)

    model_path = Path("ml/models/energy_predictor.pkl")
    joblib.dump(model, model_path)
    logger.success(f"Modèle sauvegardé : {model_path}")

    metrics = {
        "mae": mae,
        "mape": mape,
        "features": FEATURES,
        "target": TARGET,
        "best_params": getattr(model, 'get_params', lambda: {})()
    }
    metrics_path = Path("ml/models/metrics.json")
    import json
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    logger.success(f"Métriques sauvegardées : {metrics_path}")


def predict_sample(model, X):
    """Fait une prédiction sur les 3 derniers jours pour vérifier."""
    last_rows = X.tail(3)
    preds = model.predict(last_rows)
    print("\n--- Prédictions sur les 3 derniers jours ---")
    for i, pred in enumerate(preds):
        print(f"  Jour -{3-i} : {pred:,.0f} MWh prédit")


if __name__ == "__main__":
    df = load_data()
    X, y, df_clean = prepare_data(df)

    mae, mape, best_params = train_with_cross_validation(X, y)
    model = train_final_model(X, y, best_params)

    plot_feature_importance(model, FEATURES)
    save_model(model, mae, mape)
    predict_sample(model, X)

    logger.success("Pipeline ML terminé !")