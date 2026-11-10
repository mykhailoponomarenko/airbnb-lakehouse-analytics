# City C — first run, fixes, report

> Week 2 only. Until the pipeline and model for Paris and Lyon are frozen, nobody opens, profiles or loads the city C file.

## 1. Freeze week 1

1. Commit everything and tag it: `git tag week1 && git push --tags`.
2. Note the model version that is in production (MLflow run id / registered version) in the table below.
3. From now on, every change goes into the change log in section 4 before it is committed.

| Item | Value |
|---|---|
| City C | _fill in_ (`city` parameter: _slug_) |
| Why C | _one sentence from the Teams thread_ |
| Snapshot date | _from Inside Airbnb_ |
| Pipeline commit (week 1) | `week1` → _sha_ |
| Model version (week 1) | _fill in_ |

## 2. First run — unchanged

1. Upload `listings.csv.gz` to `/Volumes/airbnb_lakehouse/bronze/raw_files/<city>_listings.csv.gz` (or pass `source_url`).
2. Run job `airbnb_city_etl` with `city=<city>`, `snapshot_date=<date>`, `pipeline_version=week1`. Do not edit any code before this run.
3. If it runs through, score city C with the week-1 model, unchanged.
4. Open `pipeline/04_run_report` with `city=<city>` and copy the outputs into section 3.

Whatever happens is the result. A failed task is a finding, not something to fix before writing it down.

## 3. What happened

| Step | Outcome (ran / failed / ran with warnings) | Evidence |
|---|---|---|
| bronze | | rows, columns, error message |
| silver | | parse failures, missing columns, rows dropped |
| gold | | |
| model scoring | | |

First-run numbers:

| | Value |
|---|---|
| Listings in bronze / silver | |
| Priced listings | |
| Missing optional columns | |
| Parse failures | |
| Key features with very different NULL % than Paris / Lyon | |
| Median price C vs Paris / Lyon | |
| MAE on city C (week-1 model) | |

## 4. Change log

Every change, in order. Kind is **pipeline fix** (the pipeline should have handled it from the start: an unexpected value, a schema difference, a failed download) or **model change** (the model did not transfer: a feature that means something different in C, a price level it never saw).

| # | Kind | What broke / what we saw | Change | Reason | Commit | `pipeline_version` of re-run |
|---|---|---|---|---|---|---|
| 1 | | | | | | `week2-fix1` |

## 5. Result

| Metric | Value |
|---|---|
| Baseline MAE on C (median by room type × accommodates from the A+B training data) | |
| Model MAE on Paris test | |
| Model MAE on Lyon test | |
| Model MAE on Paris + Lyon test | |
| Model MAE on C — first run | |
| Model MAE on C — after changes | |

### Explaining the gap

_What part of the error comes from the price level, which features shift meaning (for example `dist_to_center_km` if C is a region rather than a city, or neighbourhoods the model has never seen), and which changes closed how much of it._

## Where to look first

These are generic checks the run report produces; they apply to any new city.

* `silver.quality_report.missing_optional_columns` and `parse_failures` — schema and format differences.
* `feature_null_pct` for C next to Paris and Lyon — a feature that is mostly NULL in C cannot carry the model.
* `gold.city_summary` — median price and room-type mix of C against the training cities.
* Neighbourhoods and property types in C that do not exist in the training data.
* `dist_to_center_km` — measured to the median point of C's own listings; for a region or island that point may not be a centre at all.
