
    
    

select
    date_heure as unique_field,
    count(*) as n_records

from "energy_db"."public"."stg_features"
where date_heure is not null
group by date_heure
having count(*) > 1


