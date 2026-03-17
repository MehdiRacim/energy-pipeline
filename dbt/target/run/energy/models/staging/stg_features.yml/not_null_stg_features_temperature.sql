select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
    



select temperature
from "energy_db"."public"."stg_features"
where temperature is null



      
    ) dbt_internal_test