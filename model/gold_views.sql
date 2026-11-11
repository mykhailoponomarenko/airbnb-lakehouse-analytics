-- Views and comments for the dashboard and Genie. Run after 02_price_prediction
-- (the notebook runs this file at the end, and it is also safe to run in the SQL editor).
-- Do not put semicolons inside comments: the notebook splits statements on them.
-- Comments matter: Genie reads table and column comments to pick tables and filters.

CREATE OR REPLACE VIEW airbnb_lakehouse.gold.v_model_metrics
COMMENT 'Latest test-set MAE per model run, city and variant. One row per run_id x city x variant. Currency EUR. Use this table for any question about model accuracy or the baseline.'
AS SELECT
  run_id,
  CASE run_id WHEN 'france_v1' THEN 'v1 (week 1)' WHEN 'france_v2' THEN 'v2 (week 2)' ELSE run_id END AS model,
  city,
  COALESCE(variant, CASE evaluation_stage WHEN 'train_ab' THEN 'holdout' ELSE 'unchanged' END) AS variant,
  evaluation_stage,
  n AS labeled_listings,
  ROUND(model_mae, 2) AS model_mae_eur,
  ROUND(baseline_mae, 2) AS baseline_mae_eur,
  ROUND(baseline_mae - model_mae, 2) AS improvement_eur,
  created_at
FROM airbnb_lakehouse.gold.model_metrics
WHERE city <> 'combined' OR evaluation_stage = 'train_ab'
QUALIFY ROW_NUMBER() OVER (
  PARTITION BY run_id, city, COALESCE(variant, evaluation_stage), evaluation_stage ORDER BY created_at DESC) = 1;

CREATE OR REPLACE VIEW airbnb_lakehouse.gold.v_test_predictions
COMMENT 'Held-out predictions only: Paris and Lyon test listings and city C (Bordeaux) evaluation listings that have an actual price. One row per run_id x listing. residual = actual - predicted: positive = the model UNDER-prices the listing, negative = it OVER-prices it. Currency EUR.'
AS SELECT
  p.run_id,
  CASE p.run_id WHEN 'france_v1' THEN 'v1 (week 1)' WHEN 'france_v2' THEN 'v2 (week 2)' ELSE p.run_id END AS model,
  p.city,
  INITCAP(p.city) AS city_name,
  CASE WHEN p.evaluation_set = 'c_unseen' THEN 'new city (C)' ELSE 'training city (A/B)' END AS city_role,
  p.listing_id,
  f.listing_url,
  f.name,
  f.price_flag,
  p.neighbourhood,
  p.room_type,
  p.accommodates,
  p.price AS actual_price_eur,
  ROUND(p.predicted_price, 2) AS predicted_price_eur,
  ROUND(p.baseline_price, 2) AS baseline_price_eur,
  ROUND(p.residual, 2) AS residual_eur,
  ROUND(p.absolute_error, 2) AS absolute_error_eur,
  ROUND(p.baseline_absolute_error, 2) AS baseline_absolute_error_eur,
  CASE WHEN p.residual > 0 THEN 'under-priced' ELSE 'over-priced' END AS error_direction
FROM airbnb_lakehouse.gold.price_predictions p
LEFT JOIN airbnb_lakehouse.gold.listing_features f
  ON f.city = p.city AND f.listing_id = p.listing_id
WHERE p.evaluation_set IN ('ab_test', 'c_unseen') AND p.price IS NOT NULL
QUALIFY ROW_NUMBER() OVER (PARTITION BY p.run_id, p.city, p.listing_id ORDER BY p.created_at DESC) = 1;

COMMENT ON TABLE airbnb_lakehouse.gold.city_summary IS 'One row per city: listings, priced listings, median/p25/p75 nightly price (EUR), availability, review scores, hosts, superhost share, room-type share, number of neighbourhoods. Cities: paris (A), lyon (B), bordeaux (C, new client).';
COMMENT ON TABLE airbnb_lakehouse.gold.neighbourhood_summary IS 'One row per city x neighbourhood: listings, median nightly price (EUR), review score, availability, entire-home share, centroid. price_rank 1 = most expensive neighbourhood in the city, size_rank 1 = most listings.';
COMMENT ON TABLE airbnb_lakehouse.gold.room_type_summary IS 'One row per city x room_type: listings, share of the city (share_pct), median nightly price (EUR), median guests.';
COMMENT ON TABLE airbnb_lakehouse.gold.listing_features IS 'One row per listing: cleaned attributes and the advertised nightly price in EUR (NULL when the listing showed no price). Use for listing-level questions such as the median price of a room type in a city.';
COMMENT ON TABLE airbnb_lakehouse.gold.price_predictions IS 'Raw model output for every listing, all runs. Prefer v_test_predictions for accuracy questions.';
COMMENT ON TABLE airbnb_lakehouse.gold.model_metrics IS 'Raw metric log, appended per run. Prefer v_model_metrics.';
ALTER TABLE airbnb_lakehouse.gold.listing_features ALTER COLUMN price COMMENT 'Advertised price per night in EUR on the snapshot date. NULL = listing not bookable at scrape time';
ALTER TABLE airbnb_lakehouse.gold.listing_features ALTER COLUMN room_type COMMENT 'One of: Entire home/apt, Private room, Hotel room, Shared room. "Entire home" means Entire home/apt.';
ALTER TABLE airbnb_lakehouse.gold.listing_features ALTER COLUMN city COMMENT 'Lower-case city key: paris, lyon, bordeaux';
ALTER TABLE airbnb_lakehouse.gold.listing_features ALTER COLUMN neighbourhood COMMENT 'Neighbourhood (arrondissement in Paris and Lyon, commune in Bordeaux metro)';
ALTER TABLE airbnb_lakehouse.gold.listing_features ALTER COLUMN split COMMENT 'train or test (fixed 20% hash split). The model is evaluated on test only';
ALTER TABLE airbnb_lakehouse.gold.city_summary ALTER COLUMN median_price COMMENT 'Median advertised nightly price in EUR over priced listings';
