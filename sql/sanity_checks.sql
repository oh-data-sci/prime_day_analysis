-- checks on in
-- check data contents:
SELECT
    COUNT(*)                    AS num_events,
    COUNT(DISTINCT user_id)     AS num_non_users,
    COUNT(DISTINCT event)       AS num_event_types,
    COUNT(DISTINCT country)     AS num_event_countries,
    COUNT(DISTINCT age_group)   AS num_event_types,
    COUNT(DISTINCT gender)      AS num_event_genders
FROM behavior_non_user
;
-- ┌────────────┬───────────────┬─────────────────┬─────────────────────┬─────────────────┬───────────────────┐
-- │ num_events │ num_non_users │ num_event_types │ num_event_countries │ num_event_types │ num_event_genders │
-- ├────────────┼───────────────┼─────────────────┼─────────────────────┼─────────────────┼───────────────────┤
-- │     453544 │           500 │               3 │                   1 │               7 │                 4 │
-- └────────────┴───────────────┴─────────────────┴─────────────────────┴─────────────────┴───────────────────┘
-- check data contents:
SELECT
    COUNT(*)                    AS num_events,
    COUNT(DISTINCT user_id)     AS num_non_users,
    COUNT(DISTINCT event)       AS num_event_types,
    COUNT(DISTINCT country)     AS num_event_countries,
    COUNT(DISTINCT age_group)   AS num_event_types,
    COUNT(DISTINCT gender)      AS num_event_genders
FROM behavior_user
;
--
-- ┌────────────┬───────────────┬─────────────────┬─────────────────────┬─────────────────┬───────────────────┐
-- │ num_events │ num_non_users │ num_event_types │ num_event_countries │ num_event_types │ num_event_genders │
-- ├────────────┼───────────────┼─────────────────┼─────────────────────┼─────────────────┼───────────────────┤
-- │     743413 │           493 │               3 │                   1 │               6 │                 4 │
-- └────────────┴───────────────┴─────────────────┴─────────────────────┴─────────────────┴───────────────────┘
SELECT 743413/(743413+453544); -- 62% of listed users are amazon users

-- verify that there is no overlap between the two sets
SELECT u.user_id, nu.user_id
FROM behavior_user u INNER JOIN behavior_non_user nu ON u.user_id=nu.user_id
;
-- good. no user id overlaps. users are either with amazon or not.
-- ┌─────────┬─────────┐
-- │ user_id │ user_id │
-- │ varchar │ varchar │
-- ├───────────────────┤
-- │      0 rows       │
-- └───────────────────┘
-- verify overlap with purchases: every purchase should be traced to one of the behavior tables
SELECT 
    COUNT(*) AS num_unmatched_purchases, 
    COUNT(DISTINCT p.user_id) AS num_unmatched_purchasers
FROM 
    prime_day_purchases          p
    LEFT JOIN behavior_user      u ON p.user_id=u.user_id  -- amazon users only
    LEFT JOIN behavior_non_user nu ON p.user_id=nu.user_id -- amazon non-users only
WHERE (u.user_id IS NULL AND nu.user_id IS NULL) -- purchaser match not found in either behavior tables
;
-- 
-- ┌─────────────────────────┬──────────────────────────┐
-- │ num_unmatched_purchases │ num_unmatched_purchasers │
-- ├─────────────────────────┼──────────────────────────┤
-- │                    5848 │                      922 │
-- └─────────────────────────┴──────────────────────────┘
SELECT COUNT(*) AS num_purchases, COUNT(DISTINCT user_id) AS num_purchasers FROM prime_day_purchases;
-- ┌───────────────┬─────────────────────────┐
-- │ num_purchases │ count(DISTINCT user_id) │
-- ├───────────────┼─────────────────────────┤
-- │          8707 │                    1413 │
-- └───────────────┴─────────────────────────┘

-- each user should only have one country, age group, gender?
SELECT * FROM user_summary WHERE num_age_groups>1; -- good
SELECT * FROM user_summary WHERE num_genders>1;    -- good
SELECT * FROM user_summary WHERE num_countries>1;  -- good
SELECT * FROM user_purchase_summaries WHERE num_genders>1;    -- good
SELECT * FROM user_purchase_summaries WHERE num_age_groups>1; -- good

-- 
SELECT
    order_status AS status,
    COUNT(*)     AS num_orders
FROM prime_day_purchases p
GROUP BY order_status
ORDER BY num_orders DESC
;

-- ┌─────────────────────────┬────────────┐
-- │         status          │ num_orders │
-- ├─────────────────────────┼────────────┤
-- │ Closed                  │       8395 │ -- include
-- │ Authorized              │        299 │ -- ? 
-- │ On Hold Pending Payment │         11 │ -- ignore
-- │ Payment Confirmed       │          2 │ -- ?
-- └─────────────────────────┴────────────┘



-- how many of the users made a purchase on prime day?
SELECT
    u.is_amazon_user                       AS is_amazon_user,
    COUNT(*)                               AS num_users,
    SUM(                                   
        CASE                               
            WHEN p.user_id IS NULL THEN 1  
            ELSE 0                         
        END                                
    )                                      AS num_non_buyers,
    SUM(                                   
        CASE                               
            WHEN p.user_id IS NULL THEN 0  
            ELSE 1                         
        END                                
    )                                      AS num_buyers
FROM
    user_summary                       u
    LEFT JOIN user_purchase_summaries p ON u.user_id=p.user_id
GROUP BY u.is_amazon_user
ORDER BY num_users DESC
;

-- ┌────────────────┬───────────┬────────────────┬────────────┐
-- │ is_amazon_user │ num_users │ num_non_buyers │ num_buyers │
-- ├────────────────┼───────────┼────────────────┼────────────┤
-- │ false          │       500 │            500 │          0 │
-- │ true           │       493 │              7 │        486 │
-- └────────────────┴───────────┴────────────────┴────────────┘
-- ok. this indicates that whether a user is an amazon user is an almost perfect predictor of whether they make a prime day purchase!
-- 
-- this changes things! 
-- only identified amazon users are capable of partaking in amazon prime day deals. 
-- so focus on them?
-- can we learn what separates out the 486 who did from the 7 who did not?

