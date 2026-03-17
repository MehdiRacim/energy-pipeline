select
      count(*) as failures,
      count(*) != 0 as should_warn,
      count(*) != 0 as should_error
    from (
      
    
    



select consommation_mw
from "energy_db"."public"."stg_features"
where consommation_mw is null



      
    ) dbt_internal_test