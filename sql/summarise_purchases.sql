CREATE OR REPLACE VIEW user_purchase_summaries AS
(
    SELECT
        p.user_id                     AS user_id,
        COUNT(*)                      AS num_successful_orders,
        MIN(p.order_datetime)         AS earliest_purchase,
        MAX(p.order_datetime)         AS latest_purchase,
        SUM(p.unit_price*p.quantity)  AS total_spend,
        COUNT(DISTINCT category_1)    AS num_categories_purchased,
        COUNT(DISTINCT category_2)    AS num_subcategories_purchased,
        COUNT(DISTINCT brand)         AS num_brands_purchased,
        COUNT(DISTINCT gender)        AS num_genders,
        COUNT(DISTINCT age_group)     AS num_age_groups,
    FROM prime_day_purchases p
    WHERE order_status IN ('Closed', 'Payment Confirmed') -- ignore incomplete purchases
    GROUP BY user_id
);

