import pandas as pd
import numpy as np
import psycopg2
import joblib
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, GridSearchCV
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error
from xgboost import XGBRegressor
from loguru import logger
from pathlib import Path
import os
import json
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
    "conso_semaine_precedente",
    "temp_ressentie",
    "conso_j1",
    "conso_j2",
    "conso_j14",
]

TARGET = "conso_totale_mwh"

# Grilles de paramètres pour chaque modèle
PARAM_GRIDS = {
    "GradientBoosting": {
        "n_estimators":  [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth":     [3, 4, 5],
        "subsample":     [0.8, 1.0],
    },
    "XGBoost": {
        "n_estimators":  [100, 200, 300],
        "learning_rate": [0.01, 0.05, 0.1],
        "max_depth":     [3, 4, 5],
        "subsample":     [0.8, 1.0],
    },
    "RandomForest": {
        "n_estimators": [100, 200, 300],
        "max_depth":    [3, 5, None],
        "max_features": ["sqrt", "log2"],
    },
    "Ridge": {
        "alpha": [0.1, 1.0, 10.0, 100.0],
    },
    "LinearRegression": {},
}

BASE_MODELS = {
    "LinearRegression": LinearRegression(),
    "Ridge":            Ridge(),
    "RandomForest":     RandomForestRegressor(random_state=42, n_jobs=-1),
    "GradientBoosting": GradientBoostingRegressor(random_state=42),
    "XGBoost":          XGBRegressor(random_state=42, verbosity=0),
}


def load_data() -> pd.DataFrame:
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
    """Prépare les features avec lag et features dérivées."""

    # Feature dérivée : température ressentie
    df["temp_ressentie"] = df["temp_moy"] - (df["vent_moy"] * 0.3)

    # Features de lag
    df["conso_j1"]  = df["conso_totale_mwh"].shift(1)
    df["conso_j2"]  = df["conso_totale_mwh"].shift(2)
    df["conso_j14"] = df["conso_totale_mwh"].shift(14)

    df = df.dropna(subset=FEATURES + [TARGET])

    X = df[FEATURES]
    y = df[TARGET]

    logger.info(f"Features : {FEATURES}")
    logger.info(f"Données : {len(X)} lignes après nettoyage")
    return X, y, df


def compare_models(X, y):
    """
    Étape 1 — Compare rapidement tous les modèles avec
    leurs paramètres par défaut pour identifier le meilleur.
    """
    tscv = TimeSeriesSplit(n_splits=5)
    results = {}

    logger.info("Étape 1 — Comparaison rapide des modèles...")
    print("\n" + "="*55)
    print(f"{'Modèle':<22} {'MAE':>12} {'MAPE':>8}")
    print("="*55)

    for name, model in BASE_MODELS.items():
        scores_mae = cross_val_score(
            model, X, y, cv=tscv,
            scoring="neg_mean_absolute_error",
            n_jobs=-1
        )
        scores_mape = cross_val_score(
            model, X, y, cv=tscv,
            scoring="neg_mean_absolute_percentage_error",
            n_jobs=-1
        )
        mae  = -scores_mae.mean()
        mape = -scores_mape.mean() * 100
        results[name] = {"mae": mae, "mape": mape}
        print(f"{name:<22} {mae:>10,.0f}   {mape:>6.1f}%")

    print("="*55)

    best_name = min(results, key=lambda k: results[k]["mae"])
    logger.success(f"Meilleur modèle : {best_name} — MAE: {results[best_name]['mae']:,.0f} MWh")
    return results, best_name


def tune_best_model(X, y, best_name: str):
    """
    Étape 2 — GridSearchCV sur le meilleur modèle uniquement.
    """
    tscv = TimeSeriesSplit(n_splits=5)
    param_grid = PARAM_GRIDS[best_name]

    if not param_grid:
        logger.info(f"{best_name} n'a pas de paramètres à tuner")
        model = BASE_MODELS[best_name]
        model.fit(X, y)
        return model, {}, 0

    n_combos = 1
    for v in param_grid.values():
        n_combos *= len(v)

    logger.info(f"Étape 2 — GridSearchCV sur {best_name}")
    logger.info(f"{n_combos} combinaisons × 5 folds = {n_combos * 5} fits")

    grid_search = GridSearchCV(
        BASE_MODELS[best_name],
        param_grid,
        cv=tscv,
        scoring="neg_mean_absolute_error",
        n_jobs=-1,
        verbose=1
    )
    grid_search.fit(X, y)

    best_params = grid_search.best_params_
    best_mae    = -grid_search.best_score_

    logger.success(f"Meilleurs paramètres : {best_params}")
    logger.success(f"MAE après tuning : {best_mae:,.0f} MWh")

    return grid_search.best_estimator_, best_params, best_mae


def plot_comparison(results: dict, best_name: str):
    """Graphique de comparaison des modèles."""
    names  = list(results.keys())
    maes   = [results[n]["mae"] / 1000 for n in names]
    colors = ["#22c55e" if n == best_name else "#64748b" for n in names]

    plt.figure(figsize=(9, 4))
    bars = plt.barh(names, maes, color=colors)
    plt.xlabel("MAE (GWh)")
    plt.title("Comparaison des modèles ML (vert = meilleur)")
    for bar, val in zip(bars, maes):
        plt.text(val + 0.5, bar.get_y() + bar.get_height() / 2,
                 f"{val:.0f} GWh", va="center", fontsize=9)
    plt.tight_layout()
    path = Path("ml/model_comparison.png")
    plt.savefig(path)
    logger.success(f"Graphique comparaison sauvegardé : {path}")
    plt.close()


def plot_feature_importance(model, feature_names: list, model_name: str):
    if not hasattr(model, "feature_importances_"):
        logger.info(f"{model_name} ne supporte pas feature_importances_")
        return

    importance = pd.Series(
        model.feature_importances_,
        index=feature_names
    ).sort_values(ascending=True)

    plt.figure(figsize=(8, 5))
    importance.plot(kind="barh", color="steelblue")
    plt.title(f"Importance des features — {model_name}")
    plt.xlabel("Importance relative")
    plt.tight_layout()
    path = Path("ml/feature_importance.png")
    plt.savefig(path)
    logger.success(f"Graphique features sauvegardé : {path}")
    plt.close()


def save_model(model, model_name: str, mae: float,
               mape: float, best_params: dict):
    Path("ml/models").mkdir(parents=True, exist_ok=True)

    joblib.dump(model, Path("ml/models/energy_predictor.pkl"))
    logger.success("Modèle sauvegardé : ml/models/energy_predictor.pkl")

    metrics = {
        "model_name":  model_name,
        "mae":         mae,
        "mape":        mape,
        "best_params": best_params,
        "features":    FEATURES,
        "target":      TARGET
    }
    with open(Path("ml/models/metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    logger.success("Métriques sauvegardées : ml/models/metrics.json")


def predict_sample(model, X):
    preds = model.predict(X.tail(3))
    print("\n--- Prédictions sur les 3 derniers jours ---")
    for i, pred in enumerate(preds):
        print(f"  Jour -{3-i} : {pred:,.0f} MWh prédit")


if __name__ == "__main__":
    df = load_data()
    X, y, df_clean = prepare_data(df)

    # Étape 1 : comparaison rapide
    results, best_name = compare_models(X, y)

    # Étape 2 : tuning du meilleur
    model, best_params, tuned_mae = tune_best_model(X, y, best_name)

    # Métriques finales
    final_mae  = tuned_mae if tuned_mae > 0 else results[best_name]["mae"]
    final_mape = results[best_name]["mape"]

    # Graphiques
    plot_comparison(results, best_name)
    plot_feature_importance(model, FEATURES, best_name)

    # Sauvegarde
    save_model(model, best_name, final_mae, final_mape, best_params)
    predict_sample(model, X)

    logger.success(f"Pipeline ML terminé — {best_name} "
                   f"MAE: {final_mae:,.0f} MWh")