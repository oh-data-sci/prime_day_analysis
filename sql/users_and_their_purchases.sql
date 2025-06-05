-- training data: one row per user, Y=is_purchaser, X=features
DROP TABLE IF EXISTS training_data;
DROP TABLE IF EXISTS training_data_only_users;

CREATE OR REPLACE VIEW training_data AS
(
    SELECT 
        us.*,
        CASE                                    
            WHEN ps.user_id IS NULL THEN FALSE  
            ELSE TRUE                           
        END                                     AS is_purchaser,
        COALESCE(num_successful_orders      ,0) AS num_successful_orders,
        COALESCE(total_spend                ,0) AS total_spend,
        COALESCE(num_categories_purchased   ,0) AS num_categories_purchased,
        COALESCE(num_subcategories_purchased,0) AS num_subcategories_purchased,
        COALESCE(num_brands_purchased       ,0) AS num_brands_purchased
    FROM
        user_summary                      us
        LEFT JOIN user_purchase_summaries ps ON us.user_id=ps.user_id
);


-- training data: one row per user, Y=is_purchaser, X=features
CREATE OR REPLACE VIEW training_data_only_users AS
(
    SELECT 
        us.*,
        CASE                                    
            WHEN ps.user_id IS NULL THEN FALSE  
            ELSE TRUE                           
        END                                     AS is_purchaser,
        COALESCE(num_successful_orders      ,0) AS num_successful_orders,
        COALESCE(total_spend                ,0) AS total_spend,
        COALESCE(num_categories_purchased   ,0) AS num_categories_purchased,
        COALESCE(num_subcategories_purchased,0) AS num_subcategories_purchased,
        COALESCE(num_brands_purchased       ,0) AS num_brands_purchased
    FROM
        user_summary                      us
        LEFT JOIN user_purchase_summaries ps ON us.user_id=ps.user_id
    WHERE us.is_amazon_user -- only amazon users this time, since we know the others do not purchase at all.
);


