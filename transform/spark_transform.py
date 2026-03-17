from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window
from loguru import logger
import os

def create_spark_session() -> SparkSession:
    jars_path = os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../docker/jars")
    )
    jars = ",".join([
        f"{jars_path}/hadoop-aws-3.3.4.jar",
        f"{jars_path}/aws-java-sdk-bundle-1.12.262.jar"
    ])

    spark = SparkSession.builder \
        .appName("EnergyWeatherTransform") \
        .config("spark.jars", jars) \
        .config("spark.hadoop.fs.s3a.endpoint", "http://localhost:9000") \
        .config("spark.hadoop.fs.s3a.access.key", "minioadmin") \
        .config("spark.hadoop.fs.s3a.secret.key", "minioadmin123") \
        .config("spark.hadoop.fs.s3a.path.style.access", "true") \
        .config("spark.hadoop.fs.s3a.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem") \
        .config("spark.sql.session.timeZone", "Europe/Paris") \
        .config("spark.sql.parquet.int96RebaseModeInRead", "CORRECTED") \
        .config("spark.sql.parquet.datetimeRebaseModeInRead", "CORRECTED") \
        .config("spark.sql.legacy.parquet.nanosAsLong", "true") \
        .getOrCreate()

    spark.sparkContext.setLogLevel("WARN")
    return spark


def load_weather(spark: SparkSession):
    logger.info("Chargement des données météo...")
    df = spark.read.parquet("s3a://raw/weather/weather.parquet")
    logger.success(f"Météo : {df.count()} lignes")
    return df


def load_consumption(spark: SparkSession):
    logger.info("Chargement des données de consommation...")
    df = spark.read.parquet("s3a://raw/consumption/consumption.parquet")
    logger.success(f"Consommation : {df.count()} lignes")
    return df


def transform(weather_df, consumption_df):
    logger.info("Transformation en cours...")

    # time est déjà un TIMESTAMP_NTZ — tronque directement à l'heure
    weather_h = weather_df.withColumn(
        "date_heure",
        F.date_trunc("hour", F.col("time"))
    )

    # Conso : toutes les 30min → moyenne par heure
    consumption_h = consumption_df.withColumn(
        "date_heure",
        F.date_trunc("hour", F.col("date_heure"))
    ).groupBy("date_heure").agg(
        F.avg("consommation").alias("consommation"),
        F.avg("prevision_j1").alias("prevision_j1")
    )

    # Jointure météo + conso sur date_heure
    joined = weather_h.join(consumption_h, on="date_heure", how="inner")

    # Window function : moyenne glissante sur 24h
    window_24h = Window.orderBy("date_heure").rowsBetween(-23, 0)

    features = joined \
        .withColumn("hour",         F.hour("date_heure")) \
        .withColumn("dayofweek",    F.dayofweek("date_heure")) \
        .withColumn("month",        F.month("date_heure")) \
        .withColumn("is_weekend",   (F.dayofweek("date_heure").isin(1, 7)).cast("int")) \
        .withColumn("temp_24h_avg", F.avg("temperature_2m").over(window_24h)) \
        .withColumn("temp_24h_min", F.min("temperature_2m").over(window_24h)) \
        .withColumn("temp_24h_max", F.max("temperature_2m").over(window_24h)) \
        .dropna(subset=["consommation", "temperature_2m"])

    logger.success(f"Features créées : {features.count()} lignes, {len(features.columns)} colonnes")
    return features


def save_to_minio(df):
    logger.info("Sauvegarde dans MinIO bucket 'processed'...")
    df.write \
        .mode("overwrite") \
        .parquet("s3a://processed/features/features.parquet")
    logger.success("Sauvegardé dans s3a://processed/features/features.parquet")


def show_sample(df):
    print("\n--- Aperçu des features ---")
    df.select(
        "date_heure", "temperature_2m", "consommation",
        "hour", "dayofweek", "is_weekend", "temp_24h_avg"
    ).show(5, truncate=False)

    print("--- Colonnes disponibles ---")
    for col in df.columns:
        print(f"  - {col}")


if __name__ == "__main__":
    spark = create_spark_session()

    weather = load_weather(spark)
    consumption = load_consumption(spark)

    features = transform(weather, consumption)
    show_sample(features)
    save_to_minio(features)

    spark.stop()
    logger.success("Transformation terminée !")