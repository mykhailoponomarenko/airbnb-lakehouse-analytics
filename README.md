# Airbnb Lakehouse Analytics

Group Assignment 2 — Big Data

## Project Overview
Airbnb data analytics using Databricks.

## Objectives
- Exploratory Data Analysis
- Bronze, Silver, Gold lakehouse architecture
- ETL pipeline
- Price prediction model
- AI/BI dashboard
- Genie Agent
- Evaluation on an unseen city

## Region and cities

Region **France**, currency **EUR**. Data: [Inside Airbnb](https://insideairbnb.com/get-the-data/) detailed `listings.csv.gz` (CC BY 4.0).

| Role | City | `city` parameter | Snapshot date |
|---|---|---|---|
| A (training) | Paris | `paris` | 2026-09-16 |
| B (training) | Lyon | `lyon` | 2026-09-17 |
| C (new client) | _fill in_ | _fill in_ | _fill in in week 2_ |

## Architecture

Medallion lakehouse in Unity Catalog `airbnb_lakehouse`: raw files in **bronze**, cleaned and typed listings in **silver**, metrics, model features and predictions in **gold**. Details, table contracts and design decisions: [docs/architecture.md](docs/architecture.md).

![Architecture](docs/architecture.png)

## ETL pipeline

One pipeline, the city is a parameter. All logic is in `pipeline/airbnb_etl.py`; the notebooks `01_bronze_ingest` → `02_silver_listings` → `03_gold_tables` are the job tasks.

| Job | What it does |
|---|---|
| `airbnb_city_etl` | bronze → silver → gold for one `city` |
| `airbnb_etl_all_cities` | runs `airbnb_city_etl` for each city in `cities` (default `["paris", "lyon"]`) |

Run it:

1. Upload each city's file to `/Volumes/airbnb_lakehouse/bronze/raw_files/<city>_listings.csv.gz`.
2. `databricks bundle deploy`, then `databricks bundle run airbnb_etl_all_cities` (or create the jobs in the UI as described in [docs/architecture.md](docs/architecture.md#jobs)).
3. `pipeline/04_run_report` shows what each run produced.

Adding a city means uploading its file and adding its name to `cities`; no code is copied.

## City C

Procedure and report template: [docs/city_c.md](docs/city_c.md).

## Repository

```
notebooks/01_exploratory_analysis.ipynb  EDA on Paris and Lyon
pipeline/airbnb_etl.py                  shared ETL module
pipeline/01_bronze_ingest.py            job task 1
pipeline/02_silver_listings.py          job task 2
pipeline/03_gold_tables.py              job task 3
pipeline/04_run_report.py               read-only run report
resources/airbnb_etl.job.yml            job definitions
databricks.yml                          Databricks Asset Bundle root
docs/                                   architecture, city C report
```
