WITH hourly AS (
    SELECT * FROM "energy_db"."public"."stg_features"
),

daily AS (
    SELECT
        DATE_TRUNC('day', date_heure)       AS jour,
        AVG(temperature)                    AS temp_moy,
        MIN(temperature)                    AS temp_min,
        MAX(temperature)                    AS temp_max,
        AVG(vent)                           AS vent_moy,
        AVG(nuages)                         AS nuages_moy,
        SUM(consommation_mw)                AS conso_totale_mwh,
        AVG(consommation_mw)                AS conso_moy_mw,
        MAX(consommation_mw)                AS conso_max_mw,
        MIN(consommation_mw)                AS conso_min_mw,
        MAX(est_weekend)                    AS est_weekend,
        MAX(est_ferie)                      AS est_ferie,
        MAX(mois)                           AS mois,
        LAG(SUM(consommation_mw), 7)
            OVER (ORDER BY DATE_TRUNC('day', date_heure))
                                            AS conso_semaine_precedente
    FROM hourly
    GROUP BY 1
)

SELECT * FROM daily
ORDER BY jour