select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
    

select
    date_heure as unique_field,
    count(*) as n_records

from "energy_db"."public"."stg_features"
where date_heure is not null
group by date_heure
having count(*) > 1



      
    ) dbt_internal_test