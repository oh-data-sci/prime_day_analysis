-- DROP TYPE genders;
-- DROP TYPE events;
-- DROP TYPE apps;
-- DROP TYPE countries;
-- DROP TYPE age_groups;
-- DROP TYPE categories;
CREATE TYPE genders    AS ENUM ('female', 'male', 'other', 'non_binary');
CREATE TYPE events     AS ENUM ('browsed_page','google_search','youtube_search');
CREATE TYPE apps       AS ENUM ('chrome','safari');
CREATE TYPE countries  AS ENUM ('US','UK');
CREATE TYPE age_groups AS ENUM ('16_17','18_24','25_34','35_44','45_54','55_64','over_64');
CREATE TYPE categories AS ENUM (
    'All Departments',
    'Grocery & Gourmet Food',
    'Health',
    'Beauty & Personal Care',
    'Home & Kitchen',
    'Electronics',
    'Toys & Games',
    'Pet Supplies',
    'Tools & Home Improvement',
    'Books',
    'Industrial & Scientific',
    'Office Products',
    'Sports & Outdoors',
    'Movies & TV',
    'Automotive Parts & Accessories',
    'Garden & Outdoor',
    'Arts',
    'Baby',
    'Video Games',
    'Amazon Devices',
    'Appliances',
    'CDs & Vinyl',
    'Musical Instruments',
    'Beauty',
    'Electronics & Photo',
    'Health & Personal Care',
    'Computers & Accessories',
    'Gift Cards',
    'Kindle Store',
    'Stationery & Office Supplies',
    'Premium Beauty',
    'Grocery',
    'DIY & Tools',
    'Lighting',
    'PC & Video Games',
    'None'
);


CREATE OR REPLACE TABLE behavior_non_user AS
(
    SELECT *
    FROM read_csv(
        'amazon_non_users_behavior.csv',
        delim = ',',
        header = true,
        columns = {
            "user_id"        : 'VARCHAR',
            "app_name"       : apps,
            "event_datetime" : 'DATETIME',
            "event"          : events,
            "search_term"    : 'VARCHAR',
            "title"          : 'VARCHAR',
            "page_url"       : 'VARCHAR',
            "domain"         : 'VARCHAR',
            "country"        : countries,
            "age_group"      : age_groups,
            "gender"         : genders
        }
        )
);


CREATE OR REPLACE TABLE behavior_user AS
(
    SELECT *
    FROM read_csv(
        'amazon_user_behaviors.csv',
        delim = ',',
        header = true,
        columns = {
            "user_id"        : 'VARCHAR',
            "app_name"       : apps,
            "event_datetime" : 'DATETIME',
            "event"          : events,
            "search_term"    : 'VARCHAR',
            "title"          : 'VARCHAR',
            "page_url"       : 'VARCHAR',
            "domain"         : 'VARCHAR',
            "country"        : countries,
            "age_group"      : age_groups,
            "gender"         : genders
        }
        )
);


-- combined users in single view
CREATE OR REPLACE VIEW all_user_behavior_events AS
(
    SELECT
        nu.*,
        FALSE  AS is_amazon_user
    FROM behavior_non_user nu
    UNION ALL
    SELECT 
        u.*,
        TRUE AS is_amazon_user
    FROM behavior_user u
);


-- purchase data
+(
    FROM read_csv(
        'prime_day_purchases_2024.csv',
        delim = ',',
        header = true,
        columns = {
            "user_id"        : 'VARCHAR',
            "order_datetime" : DATETIME,
            "unit_price"     : 'FLOAT',
            "asin"           : 'VARCHAR',
            "quantity"       : 'INTEGER',
            "order_status"   : 'VARCHAR',
            "product_name"   : 'VARCHAR',
            "category_1"     : categories,
            "category_2"     : 'VARCHAR',
            "brand"          : 'VARCHAR',
            "age_group"      : age_groups,
            "gender"         : genders
        }
    )
);
