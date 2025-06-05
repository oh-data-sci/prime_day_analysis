-- DROP VIEW IF EXISTS top_domains;
-- DROP VIEW IF EXISTS user_overview;
-- DROP VIEW IF EXISTS all_user_behavior_events;

CREATE OR REPLACE VIEW top_domains AS
(
    SELECT
        b.domain AS domain,
        COUNT(*) AS num_observed_events
    FROM all_user_behavior_events b
    GROUP BY b.domain
    ORDER BY num_observed_events DESC
    LIMIT 20
);


SELECT
    is_amazon_user          AS is_amazon_user,
    COUNT(*)                AS num_events,
    COUNT(DISTINCT user_id) AS num_users
FROM all_user_behavior_events
GROUP BY is_amazon_user
;

-- seems legit
-- ┌────────────────┬────────────┬───────────┐
-- │ is_amazon_user │ num_events │ num_users │
-- ├────────────────┼────────────┼───────────┤
-- │ true           │    743,413 │       493 │
-- │ false          │    453,544 │       500 │
-- └────────────────┴────────────┴───────────┘

CREATE OR REPLACE VIEW user_search_terms AS
(
    SELECT
        b.user_id                                             AS user_id,
        b.is_amazon_user                                      AS is_amazon_user,
        SUM(                                                  
            CASE                                              
                WHEN b.search_term<>'none' THEN 1             
                ELSE 0                                        
            END                                               
        )                                                     AS num_search_terms,
        STRING_AGG(                                           
            CASE                                              
                WHEN b.search_term<>'none' THEN b.search_term 
                ELSE NULL                                     
            END, ', '                                         
        )                                                     AS all_searches_combined
    FROM all_user_behavior_events b
    GROUP BY
        b.user_id,
        b.is_amazon_user
);



CREATE OR REPLACE VIEW user_summary AS
(
    SELECT
        b.user_id                                                           AS user_id,
        b.is_amazon_user                                                    AS is_amazon_user,
        COUNT(*)                                                            AS num_events,
        COUNT(DISTINCT b.app_name)                                          AS num_apps,
        COUNT(DISTINCT b.event)                                             AS num_event_types,
        COUNT(DISTINCT b.domain)                                            AS num_domains,
        COUNT(DISTINCT b.country)                                           AS num_countries,
        COUNT(DISTINCT b.gender)                                            AS num_genders,
        COUNT(DISTINCT b.age_group)                                         AS num_age_groups,
        MIN(b.event_datetime)                                               AS earliest_event,
        MAX(b.event_datetime)                                               AS latest_event,
        DATEDIFF('DAYS', earliest_event, latest_event)                      AS range_active_days,
        COUNT(DISTINCT b.event_datetime::DATE)                              AS num_dates_active,
        SUM(                                                                
            CASE                                                            
                WHEN b.domain IN (SELECT domain FROM top_domains) THEN 1    
                ELSE 0                                                      
            END                                                             
        )/num_events::FLOAT                                                AS prop_top_domains,
        SUM(                                                                
            CASE                                                            
                WHEN b.domain IN (SELECT domain FROM top_domains) THEN 0    
                ELSE 1                                                      
            END                                                             
        )/num_events::FLOAT                                                 AS prop_rare_domains,
        SUM(                                                                
            CASE                                                            
                WHEN 'search_term' <> 'none' THEN 1                         
                ELSE 0                                                      
            END                                                             
        )/num_events::FLOAT                                                 AS prop_search_events,
        STRING_AGG(                                                         
            CASE                                                            
                WHEN b.search_term<>'none' THEN b.search_term               
                ELSE NULL                                                   
            END, ', '                                                       
        )                                                                   AS all_searches_combined, -- for natural language processing in python
        MAX(b.country)                                                      AS country, -- all 'US'
        ANY_VALUE(b.gender)                                                 AS the_gender,
        ANY_VALUE(b.age_group)                                              AS the_age_group,
        CASE the_age_group                                                  
            WHEN '16_17'   THEN 16                                          
            WHEN '18_24'   THEN 18                                          
            WHEN '25_34'   THEN 25                                          
            WHEN '35_44'   THEN 35                                          
            WHEN '45_54'   THEN 45                                          
            WHEN '55_64'   THEN 55                                          
            WHEN 'over_64' THEN 64                                          
            ELSE 0                                                          
        END                                                                 AS age_bracket_start
        
    FROM all_user_behavior_events b
    GROUP BY
        user_id,
        is_amazon_user
);


