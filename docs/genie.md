# Genie Agent: setup and test

Genie Agents (formerly Genie spaces) are configured in the UI. Bundles cannot create them yet, so this file is the configuration to copy into the UI. The setup takes about 10 minutes, after `02_price_prediction` has run for `france_v2` (that run creates the views and table comments).

## 1. Create the agent

**Genie → New**, then:

* **Title:** Airbnb France pricing assistant
* **Description:** Ask about Airbnb listings, prices and the price model for Paris, Lyon and Bordeaux (EUR).
* **Warehouse:** Serverless Starter Warehouse
* **Tables** (add only these six, because fewer tables give fewer wrong joins):
  * `airbnb_lakehouse.gold.city_summary`
  * `airbnb_lakehouse.gold.neighbourhood_summary`
  * `airbnb_lakehouse.gold.room_type_summary`
  * `airbnb_lakehouse.gold.listing_features`
  * `airbnb_lakehouse.gold.v_model_metrics`
  * `airbnb_lakehouse.gold.v_test_predictions`

## 2. Instructions (paste into *Instructions → General instructions*)

```text
You answer questions about short-term-rental listings in three French cities from Inside Airbnb snapshots of September 2026.
City keys are lower-case: paris (city A), lyon (city B), bordeaux (city C, the new client). "City A" means paris, "City B" means lyon, "City C" or "the new city" means bordeaux.
All prices are advertised nightly prices in EUR. Always say "EUR" and round to whole euros.
"Entire home" or "entire apartment" means room_type = 'Entire home/apt'.
Prices: listings with price IS NULL showed no price (inactive listing). Never count them in price statistics, and when you give a median price also give how many priced listings it is based on.
Use MEDIAN with percentile(price, 0.5) on listing_features for listing-level questions, or the pre-computed median_price columns in the summary tables when no filter beyond city / room type / neighbourhood is needed.
Neighbourhood questions: use neighbourhood_summary (listings, median_price). Paris and Lyon neighbourhoods are arrondissements, Bordeaux neighbourhoods are communes of the metro area.
Model questions:
- Accuracy and baseline comparisons come from v_model_metrics. Use model = 'v2 (week 2)' unless the user asks about week 1 / v1 / the first run. For Bordeaux v2, the default variant is 'calibrated'.
- Listing-level errors come from v_test_predictions, which holds only held-out listings. Default to run_id = 'france_v2'.
- residual_eur = actual - predicted. "Under-priced" means the model predicted LESS than the actual price: residual_eur > 0. "Over-priced" means residual_eur < 0.
- "Where does the model under-price most" means: group by city and neighbourhood, keep groups with at least 15 listings, order by AVG(residual_eur) descending.
- Never report errors on training listings.
If a question asks for a city that is not paris, lyon or bordeaux, say it is not in the data.
```

## 3. Example SQL (add under *Instructions → SQL queries*, so the answers are certified)

**What is the median price of entire homes in City A?**
```sql
SELECT city, PERCENTILE(price, 0.5) AS median_price_eur, COUNT(price) AS priced_listings
FROM airbnb_lakehouse.gold.listing_features
WHERE city = 'paris' AND room_type = 'Entire home/apt' AND price IS NOT NULL
GROUP BY city
```

**Which neighbourhoods in City B have the most listings?**
```sql
SELECT neighbourhood, listings, median_price
FROM airbnb_lakehouse.gold.neighbourhood_summary
WHERE city = 'lyon'
ORDER BY listings DESC
LIMIT 10
```

**Where does the model under-price listings the most?**
```sql
SELECT city, neighbourhood, COUNT(*) AS listings,
       ROUND(AVG(residual_eur), 0) AS avg_underpricing_eur,
       ROUND(100 * AVG(CASE WHEN residual_eur > 0 THEN 1 ELSE 0 END), 0) AS pct_listings_underpriced
FROM airbnb_lakehouse.gold.v_test_predictions
WHERE run_id = 'france_v2'
GROUP BY city, neighbourhood
HAVING COUNT(*) >= 15
ORDER BY avg_underpricing_eur DESC
LIMIT 10
```

**How much better is the model than the baseline in each city?**
```sql
SELECT city, variant, labeled_listings, model_mae_eur, baseline_mae_eur, improvement_eur
FROM airbnb_lakehouse.gold.v_model_metrics
WHERE model = 'v2 (week 2)'
ORDER BY city, variant
```

**What happened on the first Bordeaux run?**
```sql
SELECT model, variant, labeled_listings, model_mae_eur, baseline_mae_eur
FROM airbnb_lakehouse.gold.v_model_metrics
WHERE city = 'bordeaux'
ORDER BY model, variant
```

## 4. Sample questions (add under *Settings → Sample questions*)

* What is the median price of entire homes in City A?
* Which neighbourhoods in City B have the most listings?
* Where does the model under-price listings the most?
* How does the model's error on Bordeaux compare with Paris and Lyon?
* Which city has the highest share of listings with no price?
* Compare the room-type mix of Paris, Lyon and Bordeaux.

## 5. Expected answers (test before the demo)

Values from Databricks (`france_v2`). Genie returned all three correctly when tested on 2026-11-12.

| Question | Expected answer |
|---|---|
| Median price of entire homes in City A | **EUR 226** (38,131 priced Paris entire homes). Lyon: EUR 120, Bordeaux: EUR 124 |
| Neighbourhoods in City B with the most listings | 3e Arrondissement 1,599, 7e 1,380, 1er 1,356, 2e 1,180, 5e 890 |
| Where the model under-prices most | Paris central arrondissements: Palais-Bourbon (+EUR 171 on average, 279 test listings, 70% of them under-priced), Élysée (+EUR 141), Luxembourg (+EUR 101). The model has no neighbourhood feature, only distance to the city's centre point, so the most expensive central areas are pulled towards the city average |

## 6. Tricks we used (for the presentation)

1. **Views shaped for questions.** `v_test_predictions` already restricts to held-out listings, adds `error_direction` and uses explicit `_eur` column names. `v_model_metrics` deduplicates the append-only metric log and turns run IDs into readable model names. Genie never has to know about `evaluation_set` or `run_id` history.
2. **Table and column comments** (`model/gold_views.sql`). Genie reads them when it picks tables. For example, the `price` comment says NULL means "not bookable", and `room_type` lists its values and that "entire home" means `Entire home/apt`.
3. **Definitions in the instructions.** "Under-priced" is ambiguous unless it is defined (residual > 0 means the model predicted too little). *(Before the demo, note what Genie answered without this definition.)*
4. **Aliases.** "City A/B/C" maps to the lower-case city keys.
5. **Certified example SQL** for the questions in the brief, so those answers are exact and marked as trusted.
6. **Small table list:** six tables, with no raw bronze or silver tables, so there are fewer ambiguous joins.
