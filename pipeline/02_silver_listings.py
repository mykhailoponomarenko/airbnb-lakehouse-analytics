# Databricks notebook source
# MAGIC %md
# MAGIC # Silver · cleaned and typed listings
# MAGIC
# MAGIC One row per listing per city. The rules come from the EDA (`notebooks/01_exploratory_analysis`):
# MAGIC
# MAGIC | Step | Rule | Why |
# MAGIC |---|---|---|
# MAGIC | Contract | required columns must exist, optional ones become NULL | fail loudly on a schema the model cannot use, tolerate the rest |
# MAGIC | Types | `try_cast` everywhere, `t/f` → boolean, `95%` → 0.95, `$1,234.00` → 1234.0, amenities JSON → array | serverless runs in ANSI mode, a plain cast would kill the run on one bad value |
# MAGIC | Parse failures | counted per column in `silver.quality_report` | a value that did not parse becomes NULL, but never silently |
# MAGIC | Keys | rows without a numeric `id` dropped, duplicate ids → latest `last_scraped` | one row per listing |
# MAGIC | Ranges | out-of-range values → NULL (price ≤ 0, availability outside 0–365, scores outside 0–5, …) | impossible values are missing values |
# MAGIC | Bathrooms | number from `bathrooms`, else parsed from `bathrooms_text` (half-bath = 0.5); type Shared / Private / Standard; > 8 in a small or private-room listing → NULL + flag | the numeric column is often empty |
# MAGIC | Price flag | `missing`, `normal`, `high` (above the city IQR fence), `extreme` (> 10× or < 1/10 of the city median) | outliers are classified, not deleted |
# MAGIC
# MAGIC Leakage columns (`estimated_revenue_l365d`, `price_quote_*`) stay in silver for analysis but never reach `gold.listing_features`.

# COMMAND ----------

from datetime import datetime, timezone

from pyspark.sql import functions as F

import airbnb_etl as etl

dbutils.widgets.text("city", "paris")
dbutils.widgets.text("run_id", "")
dbutils.widgets.text("pipeline_version", "dev")

ctx = etl.run_context(dbutils.widgets)
city = ctx["city"]
print(ctx)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 01 · Source schema against the contract

# COMMAND ----------

last_ingest = (
    spark.table(etl.table("bronze", "ingestion_log"))
    .filter(F.col("city") == city)
    .orderBy(F.desc("ingested_at"))
    .first()
)
if last_ingest is None:
    raise ValueError(f"No bronze ingestion for '{city}'. Run 01_bronze_ingest first.")

missing_optional, extra_columns = etl.check_contract(last_ingest["column_names"])
print("Missing optional columns:", missing_optional or "none")
print("Source columns outside the contract:", len(extra_columns))

bronze = etl.with_contract_columns(
    spark.table(etl.table("bronze", "listings_raw")).filter(F.col("city") == city)
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 02 · Type, validate, deduplicate

# COMMAND ----------

failures = etl.parse_failures(bronze)
print("Values that did not parse:", failures or "none")

silver_table = etl.table("silver", "listings")
etl.overwrite_city(etl.to_silver(bronze), silver_table, city)
silver = spark.table(silver_table).filter(F.col("city") == city)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 03 · Quality report

# COMMAND ----------

bronze_rows = bronze.count()
valid_ids = bronze.filter(F.expr(etl.cast_sql("id", "long")).isNotNull()).count()
silver_rows = silver.count()
priced_rows = silver.filter("has_price").count()
flags = {r["price_flag"]: r["count"] for r in silver.groupBy("price_flag").count().collect()}

report = (
    ctx["run_id"], ctx["pipeline_version"], city, last_ingest["snapshot_date"],
    bronze_rows, bronze_rows - valid_ids, valid_ids - silver_rows, silver_rows,
    priced_rows, round(priced_rows / silver_rows, 4) if silver_rows else None, flags,
    missing_optional, extra_columns, failures, etl.null_pct(silver, etl.KEY_FEATURES),
    silver.filter("bathroom_anomaly").count(), datetime.now(timezone.utc),
)

quality_table = etl.table("silver", "quality_report")
etl.append_log(spark, [report], etl.QUALITY_SCHEMA, quality_table)

display(spark.table(quality_table).filter(F.col("city") == city).orderBy(F.desc("created_at")).limit(5))

if silver_rows == 0:
    raise ValueError(f"Silver is empty for '{city}'")

# COMMAND ----------

display(
    silver.select(
        "listing_id", "room_type", "accommodates", "bedrooms", "beds", "bathrooms", "bathroom_type",
        "neighbourhood_cleansed", "price", "price_flag", "availability_365", "review_scores_rating",
    ).limit(10)
)
