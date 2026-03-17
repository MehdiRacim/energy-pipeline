# ⚡ Energy Pipeline — Prédiction de la consommation électrique française

Pipeline de données complet de bout en bout : collecte automatique des données météo et électriques françaises, transformation, modélisation SQL, prédiction ML et visualisation.

![Dashboard](https://img.shields.io/badge/Dashboard-Metabase-blue)
![API](https://img.shields.io/badge/API-FastAPI-green)
![Orchestration](https://img.shields.io/badge/Orchestration-Airflow-red)
![ML](https://img.shields.io/badge/ML-GradientBoosting-orange)

---

## Architecture
```
API Open-Meteo ──┐
                 ├──→ Airflow DAG (quotidien 6h) ──→ MinIO / S3 (Parquet)
API RTE ─────────┘                                        │
                                                          ↓
                                              PySpark (feature engineering)
                                                          │
                                                          ↓
                                              PostgreSQL + dbt (warehouse)
                                                          │
                                              ┌───────────┴───────────┐
                                              ↓                       ↓
                                     GradientBoosting ML       Metabase Dashboard
                                              │
                                              ↓
                                       FastAPI + Interface web
```

---

## Stack technique

| Couche | Technologies |
|---|---|
| Ingestion | Python, httpx, pandas |
| Orchestration | Apache Airflow 2.9 |
| Data Lake | MinIO (S3), Parquet |
| Transformation | PySpark 3.5, dbt 1.8 |
| Warehouse | PostgreSQL 15 |
| Qualité | dbt tests |
| Machine Learning | Scikit-learn, GradientBoosting, TimeSeriesSplit |
| API | FastAPI, uvicorn |
| Dashboard | Metabase |
| Infrastructure | Docker Compose, WSL2 |

---

## Résultats

- **28 128** points météo horaires collectés (2023-2026)
- **18 043** lignes après jointure et feature engineering
- **1 037** jours agrégés dans le warehouse
- **MAE moyen : ~141 000 MWh** sur validation temporelle 5 folds
- Corrélation température/consommation clairement visible dans le dashboard

---

## Structure du projet
```
energy-pipeline/
├── dags/               # DAG Airflow (pipeline quotidien)
├── ingestion/          # Scripts de collecte API
├── storage/            # Upload/download MinIO
├── transform/          # Jobs PySpark + chargement PostgreSQL
├── dbt/                # Modèles SQL (staging + mart) + tests
├── ml/                 # Entraînement + métriques + modèle
├── api/                # FastAPI + interface web
├── docker/             # JARs Spark
├── docker-compose.yml  # Infrastructure complète
└── README.md
```

---

## Démarrage rapide

### Prérequis
- Docker Desktop
- Python 3.11
- Java 17+ (pour Spark)

### Installation
```bash
# 1. Cloner le projet
git clone https://github.com/TON_USERNAME/energy-pipeline.git
cd energy-pipeline

# 2. Environnement Python
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# 3. Lancer l'infrastructure
docker compose up -d

# 4. Collecter les données
python ingestion/fetch_weather.py
python ingestion/fetch_rte.py
python storage/upload_to_minio.py

# 5. Transformer avec Spark
python transform/spark_transform.py
python transform/load_to_postgres.py

# 6. Modéliser avec dbt
cd dbt && dbt run && dbt test && cd ..

# 7. Entraîner le modèle ML
python ml/train.py

# 8. Lancer l'API
uvicorn api.main:app --port 8000
```

---

## Interfaces disponibles

| Service | URL | Credentials |
|---|---|---|
| Airflow | http://localhost:8080 | admin / admin |
| MinIO | http://localhost:9001 | minioadmin / minioadmin123 |
| API docs | http://localhost:8000/docs | — |
| Interface web | api/interface.html | — |
| Metabase | http://localhost:3000 | admin@energy.com / admin123 |

---

## Exemple de prédiction API
```bash
curl "http://localhost:8000/predict?temp_moy=5&temp_min=1&temp_max=9\
&vent_moy=20&nuages_moy=70&est_weekend=0&mois=1\
&conso_semaine_precedente=1100000"
```

Réponse :
```json
{
  "prediction_mwh": 1227810.0,
  "prediction_gwh": 1227.8,
  "date_prediction": "2026-03-18",
  "modele_mae_mwh": 140858.0,
  "message": "Consommation prédite pour le 2026-03-18"
}
```

---

## Compétences démontrées

- Ingestion de données via APIs REST publiques
- Orchestration de pipelines avec Airflow (DAGs, scheduling, retry)
- Data lake avec MinIO et format Parquet columaire
- Traitement distribué avec PySpark et window functions
- Modélisation SQL avec dbt (staging, mart, tests de qualité)
- Machine Learning avec validation temporelle correcte (TimeSeriesSplit)
- Exposition d'un modèle ML via API REST FastAPI
- Interface web de prédiction interactive
- Infrastructure as Code avec Docker Compose

---

## Auteur

Mehdi — Étudiant M1 Data Engineering