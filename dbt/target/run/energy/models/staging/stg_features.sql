
  create view "energy_db"."public"."stg_features__dbt_tmp"
    
    
  as (
    WITH source AS (
    SELECT * FROM public.raw_features
),

cleaned AS (
    SELECT
        date_heure::TIMESTAMP                    AS date_heure,
        temperature_2m::FLOAT                    AS temperature,
        windspeed_10m::FLOAT                     AS vent,
        cloudcover::FLOAT                        AS nuages,
        shortwave_radiation::FLOAT               AS radiation_solaire,
        consommation::FLOAT                      AS consommation_mw,
        prevision_j1::FLOAT                      AS prevision_j1_mw,
        hour::INT                                AS heure,
        dayofweek::INT                           AS jour_semaine,
        month::INT                               AS mois,
        is_weekend::INT                          AS est_weekend,
        temp_24h_avg::FLOAT                      AS temp_moy_24h,
        temp_24h_min::FLOAT                      AS temp_min_24h,
        temp_24h_max::FLOAT                      AS temp_max_24h
    FROM source
    WHERE consommation IS NOT NULL
      AND temperature_2m IS NOT NULL
      AND date_heure IS NOT NULL
)

SELECT * FROM cleaned
  );