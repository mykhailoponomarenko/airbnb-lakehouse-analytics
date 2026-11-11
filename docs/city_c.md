# City C (Bordeaux): first run, fixes, report

> **Where the numbers come from.** Every result in section 4 is from Databricks: runs `france_v1`
> (week 1) and `france_v2` (week 2), table `gold.model_metrics`. Numbers marked *(local)* are diagnostics
> we computed only once, by running this repository's code on the same Inside Airbnb files with local
> PySpark 3.5 and scikit-learn 1.6. That local rerun matched Databricks exactly on the split and the
> baseline, and within €1 on model MAE (for example, v2 Bordeaux calibrated: 47.98 local, 48.21 Databricks).

## 1. Week 1 frozen

| Item | Value |
|---|---|
| City C | Bordeaux (`city=bordeaux`) |
| Why C | Bordeaux has a price level like Lyon's but different housing (a metropolitan area full of houses), so it tests whether a Paris-dominated model transfers. *(Check this against the sentence posted in Teams.)* |
| Snapshot date | 2026-09-17 |
| Pipeline and model commit (week 1) | `ef58e23`, tag `week1` |
| Model (week 1) | `france_v1`: frozen artifact `/Volumes/airbnb_lakehouse/gold/model_artifacts/france_v1/` (application run ID, not MLflow) |

## 2. First run, unchanged

The week-1 job `airbnb_city_etl` was run with `city=bordeaux` and no code changes. The week-1 model was then scored with `mode=evaluate_c`, `run_id=france_v1`, with no refitting.

| Step | Outcome | Evidence |
|---|---|---|
| bronze | ran | 12,063 rows, 90 columns, the same 90 columns as Paris and Lyon, snapshot inferred as 2026-09-17 |
| silver | ran, with a silent problem | 0 missing optional columns, 0 parse failures, 0 invalid IDs, 0 duplicates, 1 bathroom anomaly. **`host_since` and `instant_bookable` were 100% NULL, but the report did not show it** (see change P1) |
| gold | ran | 12,063 rows in `gold.listing_features`, 62 neighbourhoods |
| model scoring | ran, but the predictions were wrong | 12,063 predictions (a Gold query confirmed 12,063 distinct IDs, all non-null), 7,722 with a price to evaluate. Model MAE €223.16 against a baseline of €118.51 |

First-run numbers:

| | Paris | Lyon | Bordeaux |
|---|---:|---:|---:|
| Listings (bronze = silver) | 77,149 | 9,201 | 12,063 |
| Priced listings | 43,537 (56.4%) | 5,069 (55.1%) | 7,722 (64.0%) |
| Median price (EUR) | 218 | 110 | 107.5 |
| Price flags: high / extreme | 3,026 / 349 | 372 / 9 | 604 / 22 |
| Neighbourhoods | 20 | 9 | 62 |
| Entire home/apt share | 88.6% | 79.7% | 77.4% |
| Property type "Entire home": share, and its median price | 0.6%, €456 | 0.8%, €190 | 18.2%, €185 |
| Mean bedrooms | 1.26 | 1.28 | 1.60 |
| `dist_to_center_km`, median and p90 | 2.8 / 4.7 | 1.6 / 3.4 | 2.9 / 8.5 |
| Feature NULL %: `host_years`, `instant_bookable` | 100 / 100 | 100 / 100 | 100 / 100 |
| Property types never seen in Paris or Lyon | | | 10 types, 34 listings (Entire chalet, Farm stay, Shipping container, …) |

**The pipeline did not crash on Bordeaux, but it did not work either.** Bordeaux has the same schema as the training cities, so every check passed. The two real problems never raised an error:

1. Two of the 13 model features were empty. They had also been empty in Paris and Lyon since week 1; nothing in the quality report showed it.
2. The model priced Bordeaux as if it were Paris. On average it overpriced entire homes by €237 and private rooms by €116. It predicted Bordeaux entire homes *higher* than Paris ones (€412 against €333), because Bordeaux listings are bigger, and in Paris bigger means very expensive.

| Room type (first run) | n | Mean actual (EUR) | Mean prediction (EUR) | Model MAE | Baseline MAE |
|---|---:|---:|---:|---:|---:|
| Entire home/apt | 5,882 | 175.39 | 412.06 | 254.24 | 132.92 |
| Private room | 1,812 | 76.62 | 192.45 | 122.69 | 72.19 |
| Hotel room | 11 | 259.57 | 279.79 | 150.01 | 163.62 |
| Shared room | 17 | 43.01 | 269.22 | 226.22 | 40.39 |

## 3. Change log

Each change is labelled as a **pipeline fix** (the pipeline should have handled it from the start) or a **model change** (the model did not transfer). Changes P1–P3 also affect Paris and Lyon, so A/B is retrained as `france_v2` rather than reusing `france_v1`.

| # | Kind | What broke / what we saw | Change | Reason |
|---|---|---|---|---|
| P1 | pipeline fix | `host_since` is empty in every 2026 snapshot (Inside Airbnb moved it to `hosts_time_as_host_years`), so `host_years` was 100% NULL in all three cities | `hosts_time_as_host_years` added to the silver contract; `host_years = coalesce(days since host_since / 365.25, hosts_time_as_host_years)` | a schema change by the source; the pipeline must not depend on one column name for a feature |
| P2 | pipeline fix | The quality report only checked parse failures and a short list of features, so a column that exists but is empty everywhere passed silently | new `empty_columns` field in `silver.quality_report` (contract columns that are NULL in every row), a warning in the silver task, shown in `04_run_report`; the null-% list now covers every source column of a model feature | a column that is present but empty is a schema difference, and it must be visible on the first run |
| P3 | pipeline fix | The model notebook hard-coded `paris`, `lyon` and `bordeaux`, appends failed when new metric columns were added, and nothing checked that every C listing got a prediction | `ab_cities` / `c_city` parameters, `mergeSchema` on append, an assertion of one non-null prediction per listing | adding a city must not mean editing code; completeness must be checked, not assumed |
| M1 | model change | Squared-error forest on raw EUR: Paris prices go up to €100,666, so a few extremes drive the fit | target = log price | MAE rewards predicting the typical price; log stops €10k+ listings dominating the fit |
| M2 | model change | Price level: Paris is 90% of training and twice as expensive as Lyon and Bordeaux, so the model learned Paris prices | target = log(price / city price level). The level is the median advertised price of the city's training listings. For a new city it is the median of that city's ETL `train` split (6,139 priced listings); MAE is measured only on its `test` split (1,583) | a price level the model never saw cannot be learned from Paris and Lyon; a new client's market median is public information that is available before we price their listings |
| M3 | model change | `instant_bookable` is empty; `property_type` categories mean different things per city ("Entire home" costs a median €456 in Paris and €185 in Bordeaux) | removed both | dead and non-transferable features. Measured afterwards: removing `property_type` changed Bordeaux MAE by less than €0.10, so it was not a cause of the gap |
| M4 | model change | v1 had no location information | added `dist_to_center_km` (to the city's own median point), `review_scores_location`, `review_scores_rating`, `number_of_reviews`, `availability_365`. None is computed from price (see the leakage review) | location is the main price driver inside a city; all of these are defined the same way in every city |

How the changes were chosen: on a 20% validation slice of the A/B **training** rows only. A/B test rows were not used for model choice. Bordeaux had already been seen, so every v2 number on Bordeaux is a **post-diagnosis** result; the first-run €223.16 remains the honest unseen-city number.

## 4. Result

| Metric (EUR) | Value |
|---|---:|
| Baseline on Bordeaux: A/B training median for room type × accommodates, all 7,722 priced listings | 118.51 |
| Baseline on Bordeaux, test split (1,583) | 114.78 |
| Model MAE on Paris test: v1 / v2 | 142.55 / 109.89 |
| Model MAE on Lyon test: v1 / v2 | 157.94 / 38.45 |
| Model MAE on Paris + Lyon test: v1 / v2 (baseline 151.32) | 144.13 / 102.57 |
| **Model MAE on Bordeaux, first run (v1, all priced listings)** | **223.16** |
| v1 on the Bordeaux test split, for a like-for-like comparison (SQL join of both runs' predictions) | 234.00 |
| v2 on Bordeaux test, **without** any Bordeaux prices (pooled A/B level €203) | 133.12 |
| **v2 on Bordeaux test, calibrated with Bordeaux's level (€107)** | **48.21** |
| For reference: a baseline built from Bordeaux's own training split | 51.89 *(local)* |

How much each change closed on the Bordeaux test split (cumulative, post-diagnosis, *local* ablation: each step is a separately trained model, so the last row differs slightly from the Databricks v2):

| Step | A/B test MAE | Bordeaux MAE |
|---|---:|---:|
| Week-1 model (with P1) | 141.79 | 225.35 |
| + M1 log price | 119.57 | 144.07 |
| + M2 city price level (calibrated) | 115.00 | 50.38 |
| + M3/M4 feature changes = v2 | 103.02 | 47.98 |

### Explaining the gap

* **The first-run error was a price-level error, not noise.** The week-1 model overpriced 97% of Bordeaux listings, by €207 on average *(local rerun of `france_v1`)*. Two effects add up. Paris prices dominate the training data. Bordeaux has bigger, house-type listings, which in Paris only exist at the luxury end. A Bordeaux house with 3 bedrooms looks like a Paris house with 3 bedrooms, and those cost €450+.
* **Log price removed about a third of it (€225 → €144).** The raw-price forest was pulled up by Paris listings priced at €10k–€100k.
* **The price level removed most of the rest (€144 → €50).** Without any Bordeaux prices, v2 is still worse than the naive baseline (€133 vs €115). **Our model cannot price a new city from Paris and Lyon alone.** What transfers is the *relative* structure (rooms, capacity, location within the city, reviews). The level has to come from the new city.
* **After calibration, Bordeaux is not worse than the training cities.** In EUR its MAE (€48) is lower than A/B (€103), but only because Bordeaux is cheaper. Relative to the city median, the error is 0.45 in Bordeaux, against 0.50 in Paris and 0.35 in Lyon.
* **What the model adds over a local rule is small.** A room-type × capacity median computed on Bordeaux's own listings scores €51.89; v2 scores €48.21. Most of the improvement comes from knowing the market level, not from the forest. That is the honest product message: for a new client city we need a calibration sample, or the first prices will be roughly 2× too high.
* **Remaining risks.** Bordeaux is a metropolitan area (62 communes, p90 distance 8.5 km against 4.7 km in Paris), so `dist_to_center_km` extends beyond anything the model saw. 34 listings have property types never seen in training. The test split has only 3 hotel rooms and 2 shared rooms, too few to say anything about those room types.

## Where to look first, for any new city

* `silver.quality_report`: `missing_optional_columns`, `empty_columns`, `parse_failures`. Schema and format differences.
* `feature_null_pct` for the new city next to Paris and Lyon. A feature that is mostly NULL cannot carry the model.
* `gold.city_summary`: median price and room-type mix against the training cities. A different level means the model needs calibration.
* Property types and neighbourhoods not present in the training data (`04_run_report`, section 05).
* `dist_to_center_km`: it is measured to the median point of the city's own listings; for a region or island that point may not be a centre.
