# Price prediction evaluation report

## Scope

Region: France. Currency: EUR. Paris (A) and Lyon (B) are the development cities; Bordeaux (C) is the new-client city evaluated after A/B. Results are derived from Databricks Gold table queries. Report date: 2026-11-11.

Sections up to "First Bordeaux run" describe the week-1 model, run ID `france_v1`, and are kept unchanged as first-run evidence. The week-2 model `france_v2` is described in [Week 2: france_v2](#week-2-france_v2); the full Bordeaux story is in [city_c.md](city_c.md). Model code: `model/price_model.py`; execution notebook: `notebooks/02_price_prediction.py`. The frozen artifact manifest records model configuration and dependency versions.

## Data and evaluation population

Source: Inside Airbnb detailed listings, CC BY 4.0. One Gold feature row represents one listing in one city at its snapshot date. Dates below were returned by `gold.listing_features`, rather than inferred from download time.

| City | Role | Snapshot | Gold listings | Training rows with price | Evaluation rows with price |
|---|---|---|---:|---:|---:|
| Paris | A | 2026-09-16 | 77,149 | 34,803 | 8,734 |
| Lyon | B | 2026-09-17 | 9,201 | 4,072 | 997 |
| Bordeaux | C | 2026-09-17 | 12,063 | Not used for training | 7,722 |

The existing deterministic listing-level hash split assigns approximately 20% of A/B listings to test before fitting. Paris has 61,794 train and 15,355 test rows; Lyon has 7,418 train and 1,783 test rows. Fitting uses 38,875 labeled training rows; A/B evaluation uses 9,731 labeled test rows. Bordeaux evaluation uses all 7,722 labeled listings, ignoring its ETL train/test label because the whole city is unseen by the fitted model.

Only positive finite prices are eligible for fitting and MAE. Listings without a valid target receive predictions but cannot contribute to accuracy. A/B contains 48,606 priced rows out of 86,350 (56.29%); Bordeaux evaluation covers 7,722 of 12,063 rows (64.01%). These coverage differences may affect comparability. Missingness causes have not been established in this evaluation.

## Features, baseline and regression

Target: advertised nightly price in EUR, not realized booking revenue.

Numeric predictors: accommodates, bedrooms, beds, bathrooms, amenities_count, minimum_nights, host_is_superhost, host_listings_count, host_years and instant_bookable. They represent capacity, facilities, booking conditions and host characteristics. Categorical predictors: room_type, property_type and bathroom_type, representing accommodation and privacy differences.

Numeric medians and missingness indicators are fitted on training rows only. Missing categorical values receive an explicit category; one-hot encoding tolerates previously unseen categories. Identifiers, names, price, log_price, price_flag, has_price and revenue-derived fields are excluded. City, neighbourhood, coordinates, distance, reviews and availability are also excluded in this initial model. Avoiding city-specific inputs supports a common feature contract but removes local-market information, which is a material limitation.

Required baseline: the pooled A/B training median for the same room_type and accommodates. Unseen combinations fall back to the training room-type median, then the global training median. Bordeaux labels never enter the baseline.

Model: RandomForestRegressor with 120 trees, max_depth=16, min_samples_leaf=5, squared-error criterion and seed 42. It fits untransformed EUR prices. Preprocessing and the model are persisted together; Bordeaux evaluation loads the frozen artifact without fitting. This evaluation covers the original model only.

All accuracy below is from held-out A/B rows or unseen-city Bordeaux rows. Combined A/B MAE weights individual listings equally; it is not the simple average of city MAEs. Shared hosts can cross the listing split, so this is not a new-host or future-snapshot test.

## Main results

| Evaluation | Labeled n | Model MAE EUR | Baseline MAE EUR | Baseline minus model EUR |
|---|---:|---:|---:|---:|
| Paris test | 8,734 | 142.55 | 156.65 | 14.10 |
| Lyon test | 997 | 157.94 | 104.61 | -53.33 |
| A/B combined test | 9,731 | 144.13 | 151.32 | 7.19 |
| Bordeaux unchanged model | 7,722 | 223.16 | 118.51 | -104.65 |

The model improves combined A/B MAE by approximately 4.75%, but loses substantially to the baseline in Lyon. Paris accounts for about 90% of labeled A/B test rows, so the aggregate improvement does not establish good performance in both cities.

Bordeaux model MAE is EUR 79.03 higher than A/B combined MAE, a 54.8% deterioration. Its frozen baseline outperforms the model by EUR 104.65 MAE. The evaluate_c combined row is identical to Bordeaux because this stage contains only one city; it is not an additional independent result.

## Error diagnosis

All figures below refer only to labeled evaluation rows. Residual = actual minus predicted; a negative mean residual indicates overprediction.

| City | Room type | n | Mean actual EUR | Mean prediction EUR | Model MAE EUR | Baseline MAE EUR | Mean residual EUR | Median absolute error EUR |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Bordeaux | Entire home/apt | 5,882 | 175.39 | 412.06 | 254.24 | 132.92 | -236.67 | 150.06 |
| Bordeaux | Shared room | 17 | 43.01 | 269.22 | 226.22 | 40.39 | -226.22 | 177.46 |
| Bordeaux | Hotel room | 11 | 259.57 | 279.79 | 150.01 | 163.62 | -20.22 | 123.28 |
| Bordeaux | Private room | 1,812 | 76.62 | 192.45 | 122.69 | 72.19 | -115.83 | 78.05 |
| Lyon | Entire home/apt | 823 | 144.97 | 303.63 | 164.84 | 111.20 | -158.65 | 114.01 |
| Lyon | Hotel room | 4 | 252.50 | 277.08 | 129.67 | 175.50 | -24.58 | 41.42 |
| Lyon | Private room | 167 | 87.59 | 197.10 | 126.75 | 72.04 | -109.51 | 59.57 |
| Lyon | Shared room | 3 | 43.67 | 80.91 | 37.25 | 13.17 | -37.25 | 34.13 |
| Paris | Hotel room | 51 | 444.72 | 357.72 | 189.00 | 205.29 | 87.00 | 115.78 |
| Paris | Entire home/apt | 7,673 | 328.16 | 332.73 | 144.71 | 150.60 | -4.57 | 66.23 |
| Paris | Private room | 986 | 318.75 | 309.30 | 125.79 | 204.19 | 9.45 | 48.85 |
| Paris | Shared room | 24 | 89.55 | 98.21 | 44.47 | 35.79 | -8.66 | 20.51 |

The dominant observed failure is overprediction for entire homes and private rooms in Bordeaux and Lyon. Bordeaux entire homes constitute 5,882 of 7,722 labeled rows and have a median absolute error of EUR 150.06: errors are not confined to a few extremes. Close mean actual/predicted prices in Paris do not imply accurate individual predictions; opposing errors can cancel.

Plausible explanations include Paris-heavy training (34,803 of 38,875 labeled training rows), omission of local-market features, and sensitivity of squared-error training to expensive listings. These are hypotheses, not causal findings. Bordeaux entire-home predictions even exceed Paris predictions, so a simple city-price-level explanation is incomplete. Inspect feature missingness, property-type mix, numeric distributions and high-prediction examples before attributing the gap to a specific mechanism. Hotel/shared-room samples, especially outside Paris, are too small for strong conclusions.

## First Bordeaux run and change record

The evaluation sequence was: fit and evaluate on A/B, then run the Bordeaux ETL, then `evaluate_c` with the saved model. The ETL raised no errors. Gold contains 12,063 Bordeaux listings, and the evaluation produced 7,722 labeled predictions. A separate Gold-table query confirmed 12,063 rows, 12,063 distinct listing IDs and 12,063 non-null predictions for `france_v1` / `evaluate_c`, so prediction coverage is complete. Complete coverage does not mean good accuracy. The data-quality findings, the change log (pipeline fixes P1–P3, model changes M1–M4) and the gap explanation are in [city_c.md](city_c.md).

## Week 2: france_v2

Changes, all chosen on a validation slice of A/B training rows:
* log target;
* the target expressed relative to each city's median training price;
* `host_years` restored (pipeline fix P1);
* `instant_bookable` (empty) and `property_type` (city-specific meaning) removed;
* location and review features added: `dist_to_center_km`, `review_scores_location`, `review_scores_rating`, `number_of_reviews`, `availability_365`.

None is derived from price; see the leakage review in the EDA notebook. The forest settings are unchanged.

A new city needs a price level. v2 takes it from that city's ETL `train` split, the calibration sample, and is evaluated on its `test` split. It also reports the uncalibrated score, which uses the pooled A/B level.

| Evaluation (Databricks, run `france_v2`) | Labeled n | Model MAE EUR | Baseline MAE EUR |
|---|---:|---:|---:|
| Paris test | 8,734 | 109.89 | 156.65 |
| Lyon test | 997 | 38.45 | 104.61 |
| A/B combined test | 9,731 | 102.57 | 151.32 |
| Bordeaux test, calibrated (post-diagnosis) | 1,583 | 48.21 | 114.78 |
| Bordeaux test, uncalibrated (post-diagnosis) | 1,583 | 133.12 | 114.78 |
| Bordeaux test, `france_v1` on the same rows (SQL join of both runs' predictions) | 1,583 | 234.00 | 114.78 |

v2 beats the baseline in both training cities, including Lyon, which v1 lost by €53. Paris remains the hardest city: its MAE is still above €100, and v2 underprices Paris on average (mean residual about +€34 in our local rerun), because a log model predicts close to the typical price and Paris has a long expensive tail. Without a Bordeaux price level the model still loses to the baseline. That limitation is explained in [city_c.md](city_c.md#explaining-the-gap).

## Gold outputs and dashboard integration

Catalog: airbnb_lakehouse. Tables: gold.price_predictions and gold.model_metrics. Filter one run ID: `france_v1` (week 1) or `france_v2` (week 2). For evaluation charts use evaluation_set IN ('ab_test', 'c_unseen'); `c_calibration` rows (v2) supplied the city level and are not test rows; never report ab_train errors as test accuracy. Null actual prices cannot contribute to error metrics.

Prediction coverage is confirmed for all three cities in `france_v1`: Paris 77,149/77,149, Lyon 9,201/9,201 and Bordeaux 12,063/12,063 (12,063 distinct listing IDs, 12,063 non-null predictions). Accuracy is evaluated on the 7,722 Bordeaux listings with valid actual prices. From `france_v2` onwards the notebook asserts completeness itself.

Dashboard: show baseline versus model MAE, held-out actual versus predicted, errors by room type/neighbourhood, and largest under/overpricing examples. Suggested evidence-based title: “The model improves Paris accuracy but overprices Lyon and Bordeaux.” Predictions are estimates of advertised prices, not guaranteed income.

Dashboard queries are provided in `model/dashboard_queries.sql`. Access requires read permissions on both Gold tables. Reproducibility requires the executed notebook, frozen model manifest, code revision and source snapshot metadata.
