select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
    



select date_heure
from "energy_db"."public"."stg_features"
where date_heure is null



      
    ) dbt_internal_test